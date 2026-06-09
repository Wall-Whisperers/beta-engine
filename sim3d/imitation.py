"""Imitation training loop — RSI + bounded DeepMimic reward + termination curriculum.

This is the "learn a robust controller" half of the plan, on top of a reference
authored by ``sim3d.reference``. It replaces the reverse-curriculum / warm-start
approaches that all failed (NEXT_STEPS A1b): instead of asking PPO to *discover*
the climb, it asks PPO to *track a known-good reference*, which the field treats
as the only way long-horizon physical skills are learned.

``ImitationEnv`` wraps a ``Climbing3DEnv`` (``task_mode="imitate"``) and supplies:

* **Reference State Initialization** — every episode starts RSI'd into a random
  frame of the reference (not the bottom), so every move in the sequence gets
  on-policy gradient. This is the chaining fix: a fixed-start policy never
  reaches later states, so they never train.
* **Bounded imitation reward** — ``sim3d.reference.imitation_reward`` ∈ [0,1],
  un-farmable by oscillation (maximised only by matching the reference).
* **Termination Curriculum** — terminate when the instantaneous imitation
  reward drops below ``R_min``; anneal ``R_min`` 0.75 → 0.50 over training. Early
  on, drifting off the reference ends the episode fast (sim budget stays near the
  good trajectory); later the agent may venture further. The principled version
  of "seed high, walk down" — it hugs the *reference*, not wall geometry.

Run a smoke check / short train:
    python -m sim3d.imitation --smoke
    python -m sim3d.imitation --train --steps 60000 --n-envs 4
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("sim3d.imitation requires gymnasium (pip install -r requirements.txt)") from e

from sim3d.body import LIMBS, ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig
from sim3d.reference import (ENV_SUBSTEPS, ImitationCoeffs, Reference,
                             author_weight_shift_move, imitation_reward)


@dataclass
class ImitationConfig:
    coeffs: ImitationCoeffs = field(default_factory=ImitationCoeffs)
    w_task: float = 0.0               # weight on the task term (0 = pure imitation, v1)
    r_min_start: float = 0.75
    r_min_end: float = 0.50
    r_min_decay_steps: int = 150_000  # linear R_min anneal horizon (per-env steps)
    min_episode_frames: int = 4       # RSI start phase ≤ T − this
    settle_frames: int = 4            # RSI weld-transient settle
    inner_max_steps: int = 200        # inner-env backstop (phase-end ends it first)
    # Cap the RSI start phase so every episode includes the *active* part of the
    # move. A short single-move reference is mostly post-grip settle frames, so
    # uniform RSI would start ~80% of episodes in the trivial tail and a null
    # policy would "succeed" — masking learning. Capping to the pre-/at-reach
    # region makes success a real measure of executing the move. None = uniform
    # (the right choice for a full multi-move reference, where every phase is
    # active).
    rsi_phase_max: Optional[int] = None
    # Anneal the RSI cap from len(ref) (uniform — every move gets gradient) down
    # to rsi_phase_max (forced near-bottom — the full chain) over this many
    # per-env steps. Hardens the full climb: phase-averaged success hides that the
    # hardest start (the bottom, crossing every boundary) is undertrained under
    # pure uniform RSI; annealing focuses late training there. 0 = fixed cap.
    rsi_anneal_steps: int = 0


class ImitationEnv(gym.Env):
    """RSI + bounded-imitation-reward + termination-curriculum wrapper around one
    ``Climbing3DEnv`` tracking a single ``Reference``. The reference is sampled at
    the env control rate, so phase advances 1 frame/step."""

    metadata = Climbing3DEnv.metadata

    def __init__(self, reference: Reference, wall, profile: Optional[ClimberProfile] = None,
                 imitation_config: Optional[ImitationConfig] = None,
                 env_config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.ref = reference
        self.icfg = imitation_config or ImitationConfig()
        base = env_config or EnvConfig()
        base = replace(base, task_mode="imitate", max_steps=self.icfg.inner_max_steps)
        self.env = Climbing3DEnv(wall, profile=profile, config=base, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._total_steps = 0
        self._phase = 0

    def r_min(self) -> float:
        frac = min(1.0, self._total_steps / max(1, self.icfg.r_min_decay_steps))
        return self.icfg.r_min_start + frac * (self.icfg.r_min_end - self.icfg.r_min_start)

    def rsi_cap(self) -> Optional[int]:
        """Current RSI start-phase cap. None = uniform. With annealing it walks
        from len(ref) (uniform) down to rsi_phase_max (forced near-bottom)."""
        tgt = self.icfg.rsi_phase_max
        if tgt is None or self.icfg.rsi_anneal_steps <= 0:
            return tgt
        frac = min(1.0, self._total_steps / self.icfg.rsi_anneal_steps)
        return int(round(len(self.ref) + frac * (tgt - len(self.ref))))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        T = len(self.ref)
        hi = max(1, T - self.icfg.min_episode_frames)
        cap = self.rsi_cap()
        if cap is not None:
            hi = min(hi, cap + 1)
        self._phase = int(self.np_random.integers(0, hi))   # RSI start phase
        t = self._phase
        obs, info = self.env.reset_to_reference(
            self.ref.qpos[t], self.ref.qvel[t], self.ref.frame_grips(t),
            settle_frames=self.icfg.settle_frames,
        )
        info.update(self._info(r_imit=1.0))
        return obs, info

    def step(self, action):
        # v1 — drive the 4 grip intents from the reference's contacts at the
        # frame we're tracking toward (+1 hold, −1 release), so the policy learns
        # joint tracking without the std≈0.22 grip-intent noise randomly dropping
        # anchors (the NEXT_STEPS 0.4 foot-gun). The inner env's engage logic
        # no-ops on already-held anchors and releases the limb the reference
        # frees. Policy-controlled grips (rewarded for matching the contact
        # schedule) are a follow-up. The policy's own grip outputs are ignored.
        action = np.asarray(action, dtype=np.float32).copy()
        t_next = min(self._phase + 1, len(self.ref) - 1)
        ref_grips = self.ref.frame_grips(t_next)
        for i, limb in enumerate(LIMBS):
            action[len(action) - 4 + i] = 1.0 if ref_grips[limb] is not None else -1.0

        # Apply the action; the inner env advances physics + handles obs, grips,
        # and the physical-fall backstop. Its reward is ignored.
        obs, _r_inner, fell, _trunc_inner, info = self.env.step(action)

        self._phase += 1
        self._total_steps += 1
        t = min(self._phase, len(self.ref) - 1)
        r_imit, comp = imitation_reward(self.env.world, self.ref, t, self.icfg.coeffs)
        reward = (1.0 - self.icfg.w_task) * r_imit + self.icfg.w_task * self._task_reward(t)

        terminated = bool(fell)
        outcome = info.get("outcome", "")
        rmin = self.r_min()
        if not terminated and r_imit < rmin:
            terminated = True
            outcome = "off-reference"            # termination curriculum cut
        if not terminated and self._phase >= len(self.ref) - 1:
            terminated = True
            outcome = "completed"                # tracked to the end of the reference

        info["outcome"] = outcome
        info["is_success"] = (outcome == "completed")
        info.update(self._info(r_imit=r_imit, comp=comp, rmin=rmin))
        return obs, float(reward), terminated, False, info

    def _task_reward(self, t: int) -> float:
        # v1: pure imitation. (Hook for a top-out/progress term, w_task > 0.)
        return 0.0

    def _info(self, *, r_imit: float, comp: Optional[dict] = None,
              rmin: Optional[float] = None) -> dict:
        d = {"phase": self._phase, "ref_len": len(self.ref), "r_imit": round(float(r_imit), 3)}
        if rmin is not None:
            d["r_min"] = round(float(rmin), 3)
        if comp is not None:
            d.update({f"r_{k}": round(float(v), 3) for k, v in comp.items()})
        return d

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ─── Training ────────────────────────────────────────────────────────────────

def _build_wall(seed: int):
    """Rebuild the exact wall the reference was authored on (suppressing the
    seed-overbrace chatter)."""
    import contextlib
    import io
    from sim3d.probe_transitions import build_wall_and_moves
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        wall, profile, _feas = build_wall_and_moves(seed=seed)
    return wall, profile


def _load_wall_for_ref(ref: Reference, wall_json: Optional[str] = None,
                       ref_path: Optional[str] = None):
    """The exact wall the reference was authored/discovered on. A CMA-ES
    reference comes from a non-default ``reach_frac`` wall that can't be rebuilt
    from seed alone, so it's persisted as JSON next to the .npz.

    Resolution order:
      1. Explicit ``wall_json`` argument.
      2. Auto-detected sibling ``<ref_stem>.wall.json`` (when ``ref_path`` given).
      3. Rebuild from ``ref.wall_gen_seed`` (fallback — only correct for walls
         whose hold layout matches the generator defaults; WRONG for CMA-ES refs
         that used a different wall parametrisation).
    """
    import contextlib
    import io
    from solver.wall import load_wall

    candidates = []
    if wall_json:
        candidates.append(wall_json)
    if ref_path:
        sibling = Path(ref_path).with_suffix(".wall.json")
        if sibling.exists():
            candidates.append(str(sibling))

    for cand in candidates:
        if Path(cand).exists():
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                return load_wall(cand), ClimberProfile()

    return _build_wall(ref.wall_gen_seed)


def make_env(ref_path: str, icfg: ImitationConfig, rank: int = 0,
             wall_json: Optional[str] = None):
    def _init():
        ref = Reference.load(ref_path)
        wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
        env = ImitationEnv(ref, wall, profile=profile, imitation_config=icfg)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return _init


def smoke(ref_path: str, icfg: Optional[ImitationConfig] = None,
          wall_json: Optional[str] = None) -> None:
    """Single-env sanity: RSI works, reward stays in [0,1], episodes end via the
    termination curriculum / phase-end, and zero-action vs random differ."""
    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    env = ImitationEnv(ref, wall, profile=profile, imitation_config=icfg)
    print(f"reference: {len(ref)} frames | obs {env.observation_space.shape} "
          f"action {env.action_space.shape}")
    for label, policy in (("zero", lambda: np.zeros(env.action_space.shape, dtype=np.float32)),
                          ("random", lambda: env.action_space.sample())):
        outcomes, rewards, lengths = [], [], []
        for ep in range(20):
            _o, _i = env.reset(seed=ep)
            ep_r, ep_l, done = 0.0, 0, False
            while not done:
                o, r, term, trunc, info = env.step(policy())
                assert 0.0 <= r <= 1.0 + 1e-6, f"reward out of [0,1]: {r}"
                ep_r += r
                ep_l += 1
                done = term or trunc
            outcomes.append(info["outcome"])
            rewards.append(ep_r)
            lengths.append(ep_l)
        comp = {k: outcomes.count(k) for k in set(outcomes)}
        print(f"  {label:6s}: ep_rew {np.mean(rewards):.2f}  ep_len {np.mean(lengths):.1f}  "
              f"r/step {np.mean(rewards)/max(1,np.mean(lengths)):.2f}  outcomes {comp}")
    env.close()


def train(ref_path: str, *, steps: int, n_envs: int, run_id: str, icfg: ImitationConfig,
          wall_json: Optional[str] = None, load_run: Optional[str] = None) -> None:
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    out = Path("data/runs/sim3d") / run_id
    out.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec = vec_cls([make_env(ref_path, icfg, i, wall_json) for i in range(n_envs)])
    # Warm-start: reuse the prior run's VecNormalize stats + policy weights so we
    # continue (and harden) a trained policy instead of relearning from scratch.
    vn_path = Path(load_run) / "vecnormalize.pkl" if load_run else None
    if vn_path is not None and vn_path.exists():
        vec = VecNormalize.load(str(vn_path), vec)
        vec.training, vec.norm_reward = True, False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    class ProgressCallback(BaseCallback):
        """Log success rate + mean tracking quality (r_imit) + R_min per rollout.
        r_imit is R_min-independent, so it shows learning even when the success
        baseline is high."""
        def __init__(self):
            super().__init__()
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                self.rimit_sum += float(info.get("r_imit", 0.0))
                self.rimit_n += 1
                if "episode" in info:  # Monitor end-of-episode
                    self.ep_done += 1
                    self.ep_succ += int(info.get("is_success", False))
            return True

        def _on_rollout_end(self) -> None:
            sr = self.ep_succ / max(1, self.ep_done)
            rimit = self.rimit_sum / max(1, self.rimit_n)
            try:
                rmin = float(np.mean(self.training_env.env_method("r_min")))
                cap = self.training_env.env_method("rsi_cap")[0]
            except Exception:  # noqa: BLE001
                rmin, cap = float("nan"), None
            cap_s = "uniform" if cap is None else f"cap{cap}"
            print(f"  [{self.num_timesteps:>7}] success {sr*100:5.1f}%  "
                  f"({self.ep_succ}/{self.ep_done} eps)  r_imit {rimit:.3f}  "
                  f"R_min {rmin:.3f}  RSI {cap_s}")
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0

    model_path = Path(load_run) / "model.zip" if load_run else None
    if model_path is not None and model_path.exists():
        model = sb3.PPO.load(str(model_path), env=vec)
        print(f"Warm-started from {model_path}")
    else:
        model = sb3.PPO(
            "MlpPolicy", vec, verbose=0,
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.005,
            target_kl=0.03, policy_kwargs={"log_std_init": -1.5},
        )
    # Checkpoint every 200k global steps (vec-normalize stats saved alongside).
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
        verbose=0,
    )
    callbacks = CallbackList([ProgressCallback(), ckpt_cb])

    print(f"Training imitation: {steps} steps, {n_envs} envs → {out}")
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(str(out / "model.zip"))
    vec.save(str(out / "vecnormalize.pkl"))
    print(f"Saved {out/'model.zip'}")


def record_video(model_path: str, ref_path: str, out_path: str, *,
                 vecnorm: Optional[str] = None, n_episodes: int = 4,
                 fps: int = 10, size: int = 480, wall_json: Optional[str] = None,
                 rsi_phase_max: Optional[int] = 0) -> None:
    """Roll out the trained policy in its ImitationEnv and render to mp4.

    Critically applies the saved VecNormalize obs stats — without them the
    policy gets unnormalised observations and flails (which is why the standard
    browser viewer can't replay an imitation model). Starts every episode at
    phase 0 (RSI cap 0) so the clip shows the full release→reach→regrip, with a
    body-tracking camera."""
    import mujoco
    import imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    # Record at the trained R_min floor (0.50), not the 0.75 default — otherwise
    # the termination curriculum cuts a policy that tracks the (harder) full climb
    # at r_imit ~0.74 right at the start, which looks like total failure.
    icfg = ImitationConfig(rsi_phase_max=rsi_phase_max, r_min_start=0.5, r_min_end=0.5)
    env = ImitationEnv(ref, wall, profile, imitation_config=icfg)
    model = PPO.load(model_path)
    vn = None
    if vecnorm and Path(vecnorm).exists():
        vn = VecNormalize.load(vecnorm, DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, icfg)]))
        vn.training = False

    m, d = env.env.world.model, env.env.world.data
    # Tame the default over-exposure so the wall isn't blown to white and the
    # coloured holds + climber read clearly.
    m.vis.headlight.diffuse[:] = 0.35
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.specular[:] = 0.0
    for _i in range(m.nlight):
        m.light_diffuse[_i] *= 0.35
        m.light_specular[_i] *= 0.0
    renderer = mujoco.Renderer(m, height=size, width=size)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    # Front view (az 270): the climber's back against the wall FACE, so the
    # coloured holds are visible and upward progress along them is legible. A
    # side view hides the holds (they lie flat on the face) and reads as "leaning".
    cam.azimuth, cam.elevation, cam.distance = 270.0, -10.0, 3.6

    frames, n_done, n_succ = [], 0, 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(len(ref) + 2):
            o = vn.normalize_obs(obs) if vn is not None else obs
            action, _ = model.predict(o, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            com = env.env.world.com()
            cam.lookat[:] = [float(com[0]), 0.2, float(com[2])]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
            if term or trunc:
                n_done += 1
                n_succ += int(info.get("is_success", False))
                break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Wrote {out_path}  ({len(frames)} frames, {n_episodes} eps, "
          f"{n_succ}/{n_done} reached the regrip)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=str, default="data/runs/sim3d/imitation/ref_move.npz")
    ap.add_argument("--author", action="store_true", help="author the reference first")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--move-index", type=int, default=14, help="feasible move to author")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--record", type=str, default=None,
                    help="render a trained model to this mp4 (needs --model + --ref)")
    ap.add_argument("--model", type=str, default=None, help="model.zip for --record")
    ap.add_argument("--vecnorm", type=str, default=None, help="vecnormalize.pkl for --record")
    ap.add_argument("--wall-json", type=str, default=None,
                    help="exact wall JSON (for CMA-ES refs on a non-default wall); "
                         "auto-detected as <ref>.wall.json if present")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--run-id", type=str, default="imitation/smoke")
    ap.add_argument("--load", type=str, default=None,
                    help="warm-start from a prior run dir (loads model.zip + vecnormalize.pkl)")
    ap.add_argument("--r-min-start", type=float, default=0.75,
                    help="initial R_min termination threshold (default 0.75)")
    ap.add_argument("--r-min-end", type=float, default=0.50,
                    help="final R_min after decay (default 0.50)")
    ap.add_argument("--r-min-decay", type=int, default=150_000,
                    help="per-env steps to anneal R_min start→end (default 150k)")
    ap.add_argument("--rsi-phase-max", type=int, default=None,
                    help="final RSI start-phase cap (low = forced near-bottom)")
    ap.add_argument("--rsi-anneal", type=int, default=0,
                    help="anneal the RSI cap len(ref)→rsi-phase-max over this many "
                         "per-env steps (hardens the full climb). 0 = fixed cap")
    args = ap.parse_args()

    icfg = ImitationConfig(r_min_start=args.r_min_start,
                           r_min_end=args.r_min_end,
                           r_min_decay_steps=args.r_min_decay,
                           rsi_phase_max=args.rsi_phase_max,
                           rsi_anneal_steps=args.rsi_anneal)

    if args.author:
        from sim3d.probe_transitions import build_wall_and_moves
        wall, profile, feas = build_wall_and_moves(seed=args.seed)
        move = feas[args.move_index]
        ref, diag = author_weight_shift_move(wall, profile, move, balance_kp=250.0,
                                             wall_gen_seed=args.seed)
        ref.save(args.ref)
        print(f"Authored move {move['move_k']} → {args.ref}  {diag}")

    # Auto-detect the sibling wall JSON a CMA-ES reference saves next to itself.
    wall_json = args.wall_json
    if wall_json is None:
        sib = Path(args.ref).with_suffix(".wall.json")
        if sib.exists():
            wall_json = str(sib)

    if args.record:
        record_video(args.model, args.ref, args.record, vecnorm=args.vecnorm,
                     wall_json=wall_json)
    if args.smoke:
        smoke(args.ref, icfg, wall_json)
    if args.train:
        train(args.ref, steps=args.steps, n_envs=args.n_envs, run_id=args.run_id,
              icfg=icfg, wall_json=wall_json, load_run=args.load)


if __name__ == "__main__":
    main()
