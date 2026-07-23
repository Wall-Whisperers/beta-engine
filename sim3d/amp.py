"""Adversarial Motion Priors (AMP) discriminator for the climbing agent.

Architecture follows Peng et al. 2021 "AMP: Adversarial Motion Priors for
Stylized Physics-Based Character Animation" with simplifications appropriate
for our dataset size (9 clips, ~5 min total).

Usage
-----
# 1. Build a motion library from retargeted .npz clips:
    python -m sim3d.amp build-library \
        data/video/moonboard/spike1_npz/ \
        data/amp/motion_library.npz

# 2. Train the discriminator (offline, ~10 min on CPU):
    python -m sim3d.amp train \
        --library data/amp/motion_library.npz \
        --out     data/amp/disc.pt \
        --epochs  200

# 3. Evaluate quality (optional sanity check):
    python -m sim3d.amp eval --disc data/amp/disc.pt --library data/amp/motion_library.npz

# 4. Use in training (passed via --amp-disc):
    python -m sim3d.imitation --train ... --amp-disc data/amp/disc.pt

State representation: (qpos[7:30], qvel[6:29]) = 23+23 = 46-dim vector.
The discriminator sees (s_t, s_{t+1}) → 92-dim input → scalar style reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── State encoding ───────────────────────────────────────────────────────────

N_ACT = 23  # actuated joints
STATE_DIM = N_ACT * 2  # qpos[7:] + qvel[6:] per timestep
PAIR_DIM  = STATE_DIM * 2  # (s_t, s_{t+1})


def encode_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Extract the 46-dim AMP state from a full qpos (30,) and qvel (29,).

    Uses only the actuated joints — the pelvis free-joint position/orientation
    is excluded so the discriminator judges *style of movement*, not where on
    the wall the climber is (which varies across clips and walls).
    """
    return np.concatenate([qpos[7:30], qvel[6:29]]).astype(np.float32)


def make_pairs(qpos_seq: np.ndarray, qvel_seq: np.ndarray) -> np.ndarray:
    """(T, 30) qpos + (T, 29) qvel → (T-1, 92) consecutive-state pairs."""
    states = np.stack([encode_state(qpos_seq[t], qvel_seq[t])
                       for t in range(len(qpos_seq))])
    return np.concatenate([states[:-1], states[1:]], axis=1).astype(np.float32)


# ─── Discriminator network ───────────────────────────────────────────────────

class AMPDiscriminator(nn.Module):
    """Shallow MLP: (s, s') → scalar logit (positive = real climbing motion).

    Deliberately small: our dataset is ~5 min of video × 25 fps × 9 clips ≈
    67 k state pairs. A deep network would overfit.

    Inputs are normalised internally by ``in_mean``/``in_std`` buffers
    (per-dim stats of the REAL pair distribution, set via
    ``set_input_stats``). Without this the raw joint-velocity dims (range
    ~±6 rad/s) dwarf the pose dims and dominate every gradient. The buffers
    live in ``state_dict`` so checkpoints and cross-process broadcasts are
    self-contained.
    """

    def __init__(self, input_dim: int = PAIR_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("in_mean", torch.zeros(input_dim))
        self.register_buffer("in_std", torch.ones(input_dim))
        # Weight initialisation: small weights keep gradients well-conditioned
        # at the start; the ELU ensures non-vanishing gradients throughout.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def set_input_stats(self, pairs: np.ndarray) -> None:
        """Set the normalisation buffers from an array of REAL pairs
        ((N, PAIR_DIM)); a 0.01 std floor guards near-constant dims."""
        self.in_mean.copy_(torch.tensor(pairs.mean(axis=0), dtype=torch.float32))
        self.in_std.copy_(torch.tensor(
            np.maximum(pairs.std(axis=0), 0.01), dtype=torch.float32))

    def state_numpy(self) -> dict:
        """Picklable numpy snapshot of the full state (for broadcast to
        SubprocVecEnv workers, which can't share torch modules)."""
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def load_state_numpy(self, state: dict) -> None:
        self.load_state_dict({k: torch.tensor(v) for k, v in state.items()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., PAIR_DIM) raw pair → logit (..., 1)."""
        return self.net((x - self.in_mean) / self.in_std)

    def reward(self, s: np.ndarray, s_next: np.ndarray) -> float:
        """Compute AMP style reward for a single (s, s') transition.

        Formula (AMP paper eq. 9): r_style = max(0, 1 − 0.25·(1 − D)²)
        where D = tanh(logit) ∈ (−1, +1). The LS-GAN trains D toward +1
        for real transitions and −1 for fake ones:
          D=+1 (real)  → r = 1.0
          D= 0 (unsure)→ r = 0.75
          D=−1 (fake)  → r = 0.0

        Returns a scalar float in [0, 1].
        """
        pair = np.concatenate([s, s_next], dtype=np.float32)
        with torch.no_grad():
            x = torch.tensor(pair).unsqueeze(0)
            d = torch.tanh(self.forward(x)).squeeze().item()
        return float(max(0.0, 1.0 - 0.25 * (1.0 - d) ** 2))

    def reward_batch(self, pairs: np.ndarray) -> np.ndarray:
        """pairs: (N, PAIR_DIM) → (N,) rewards in [0, 1]."""
        with torch.no_grad():
            x = torch.tensor(pairs, dtype=torch.float32)
            d = torch.tanh(self.forward(x)).squeeze(-1).numpy()
        return np.clip(1.0 - 0.25 * (1.0 - d) ** 2, 0.0, None).astype(np.float32)


# ─── Gradient penalty (WGAN-GP style) ────────────────────────────────────────

def _gradient_penalty(disc: AMPDiscriminator,
                      real: torch.Tensor, fake: torch.Tensor,
                      lambda_gp: float = 10.0) -> torch.Tensor:
    """Interpolated gradient penalty to regularise the discriminator."""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


# ─── Motion library ──────────────────────────────────────────────────────────

class MotionLibrary:
    """In-memory store of (s, s') pairs from retargeted motion clips.

    Built once from retargeted .npz files; saved as a single .npz for fast
    reloading. During discriminator training, samples mini-batches of real
    pairs. During rollout, the discriminator queries single pairs step-by-step.
    """

    def __init__(self, pairs: np.ndarray):
        """pairs: (N, PAIR_DIM) float32 — all consecutive state pairs."""
        self.pairs = pairs.astype(np.float32)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n random real (s, s') pairs → (n, PAIR_DIM)."""
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.pairs), size=n)
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)

    @classmethod
    def from_clips(cls, clip_paths: list[str],
                   max_violations: int = 5) -> "MotionLibrary":
        """Build from retargeted .npz files (output of sim3d/retarget.py).

        Each .npz must have keys: qpos (T,30), qvel (T,29), n_violations (T,).
        Frames with too many joint-limit violations are excluded.
        """
        all_pairs = []
        for path in clip_paths:
            d = np.load(path)
            qpos = d["qpos"]
            qvel = d["qvel"]
            n_viol = d.get("n_violations", np.zeros(len(qpos), dtype=int))
            # Exclude frames where either endpoint has too many violations.
            good = (n_viol[:-1] <= max_violations) & (n_viol[1:] <= max_violations)
            pairs = make_pairs(qpos, qvel)
            all_pairs.append(pairs[good])
            print(f"  {Path(path).name}: {good.sum()}/{len(good)} pairs kept")
        if not all_pairs:
            raise ValueError("No valid pairs found in the supplied clips.")
        return cls(np.concatenate(all_pairs, axis=0))

    def save(self, path: str):
        np.savez_compressed(path, pairs=self.pairs)
        print(f"Motion library: {len(self)} pairs → {path}")

    @classmethod
    def load(cls, path: str) -> "MotionLibrary":
        d = np.load(path)
        return cls(d["pairs"])


# ─── Discriminator training ───────────────────────────────────────────────────

def train_discriminator(
        disc: AMPDiscriminator,
        library: MotionLibrary,
        fake_pairs: np.ndarray,
        *,
        batch_size: int = 256,
        n_steps: int = 5,
        lr: float = 1e-4,
        lambda_gp: float = 10.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """One outer training iteration: update the discriminator on real vs fake.

    Called once per PPO update (after collecting a rollout). Returns loss info.

    Args:
        disc:       the AMPDiscriminator to update (in-place).
        library:    MotionLibrary of real (s, s') pairs.
        fake_pairs: (N, PAIR_DIM) pairs from the current policy rollout.
        n_steps:    gradient steps per call (AMP paper uses 1; more is fine
                    for small datasets where real data is the bottleneck).
        optimizer:  pass the same optimizer across calls to retain momentum.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(disc.parameters(), lr=lr)

    rng = np.random.default_rng()
    info = {"d_loss": [], "gp": [], "d_real": [], "d_fake": []}

    for _ in range(n_steps):
        real_np = library.sample(batch_size, rng)
        fake_np = fake_pairs[rng.integers(0, len(fake_pairs), size=batch_size)]

        real_t = torch.tensor(real_np, dtype=torch.float32)
        fake_t = torch.tensor(fake_np, dtype=torch.float32)

        d_real = disc(real_t)
        d_fake = disc(fake_t)

        # Least-squares GAN loss (stable, no mode collapse):
        # real → +1, fake → −1
        loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
        loss_fake = F.mse_loss(d_fake, -torch.ones_like(d_fake))
        gp = _gradient_penalty(disc, real_t, fake_t, lambda_gp)
        loss = 0.5 * (loss_real + loss_fake) + gp

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        optimizer.step()

        info["d_loss"].append(loss.item())
        info["gp"].append(gp.item())
        info["d_real"].append(torch.sigmoid(d_real).mean().item())
        info["d_fake"].append(torch.sigmoid(d_fake).mean().item())

    return {k: float(np.mean(v)) for k, v in info.items()}


# The ONLINE AMP integration (discriminator updated on real-vs-POLICY pairs
# every PPO rollout, weights broadcast to the env workers) lives in
# ``sim3d.imitation.AMPOnlineCallback`` — it needs SB3, which this module
# deliberately doesn't import. Use ``--amp-online`` on the trainer.


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_library(args):
    clips = sorted(Path(args.clips_dir).glob("*.npz"))
    if not clips:
        raise FileNotFoundError(f"No .npz files in {args.clips_dir}")
    print(f"Building motion library from {len(clips)} clips:")
    lib = MotionLibrary.from_clips([str(c) for c in clips],
                                   max_violations=args.max_violations)
    print(f"Total pairs: {len(lib)}  ({len(lib)*PAIR_DIM*4/1e6:.1f} MB)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lib.save(args.out)


def _train(args):
    """Offline smoke-training ONLY — the "fake" distribution here is Gaussian
    noise, so the resulting boundary is "climbing vs noise" and any coherent
    motion (including the policy's servo snaps) scores ~1: as an RL reward
    this is a constant offset, NOT a style signal. Production training is the
    online loop (``sim3d.imitation --train --amp-online``), which pits the
    library against actual policy rollouts each PPO iteration."""
    print(f"Loading motion library from {args.library} ...")
    lib = MotionLibrary.load(args.library)
    print(f"  {len(lib)} real pairs")

    disc = AMPDiscriminator()
    disc.set_input_stats(lib.pairs)
    opt  = torch.optim.Adam(disc.parameters(), lr=args.lr)
    rng  = np.random.default_rng(42)

    print(f"Training discriminator for {args.epochs} epochs (SMOKE ONLY — "
          f"noise fakes; use --amp-online for the real adversarial loop) ...")
    for epoch in range(1, args.epochs + 1):
        fake = rng.standard_normal((len(lib), PAIR_DIM)).astype(np.float32)
        info = train_discriminator(disc, lib, fake, optimizer=opt,
                                   batch_size=args.batch_size, n_steps=1)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}: loss={info['d_loss']:.3f}  "
                  f"d_real={info['d_real']:.3f}  d_fake={info['d_fake']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": disc.state_dict(),
                "input_dim": PAIR_DIM,
                "hidden": 256}, args.out)
    print(f"Saved discriminator → {args.out}")


def _eval(args):
    ckpt = torch.load(args.disc, map_location="cpu")
    disc = AMPDiscriminator(ckpt["input_dim"], ckpt["hidden"])
    # strict=False: pre-normalisation checkpoints lack the in_mean/in_std
    # buffers; they fall back to identity normalisation (old behaviour).
    disc.load_state_dict(ckpt["state_dict"], strict=False)
    disc.eval()

    lib  = MotionLibrary.load(args.library)
    rng  = np.random.default_rng(0)
    real = lib.sample(1000, rng)
    fake = rng.standard_normal((1000, PAIR_DIM)).astype(np.float32)

    r_real = disc.reward_batch(real).mean()
    r_fake = disc.reward_batch(fake).mean()
    print(f"Real style reward (mean): {r_real:.3f}  (target > 0.5)")
    print(f"Fake style reward (mean): {r_fake:.3f}  (target ≈ 0.0)")
    print(f"Separation: {r_real - r_fake:.3f}  (target > 0.4)")


def main():
    ap = argparse.ArgumentParser(description="AMP discriminator utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-library",
                              help="Build motion library from retargeted clips")
    p_build.add_argument("clips_dir", help="Dir of retargeted .npz files")
    p_build.add_argument("out",       help="Output library .npz")
    p_build.add_argument("--max-violations", type=int, default=5)

    p_train = sub.add_parser("train", help="Train discriminator offline")
    p_train.add_argument("--library",    required=True)
    p_train.add_argument("--out",        default="data/amp/disc.pt")
    p_train.add_argument("--epochs",     type=int, default=200)
    p_train.add_argument("--lr",         type=float, default=1e-4)
    p_train.add_argument("--batch-size", type=int, default=256)

    p_eval = sub.add_parser("eval", help="Sanity-check trained discriminator")
    p_eval.add_argument("--disc",    required=True)
    p_eval.add_argument("--library", required=True)

    args = ap.parse_args()
    {"build-library": _build_library, "train": _train, "eval": _eval}[args.cmd](args)


if __name__ == "__main__":
    main()
