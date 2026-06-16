"""RL algorithm bake-off — PPO vs off-policy (SAC, TQC) on the SAME env.

Motivation (2026-06-08): PPO is a generalist; off-policy actor-critics (SAC,
TQC) are usually more *sample-efficient* for continuous motor control. This
harness measures that empirically WITHOUT touching the production trainers
(`train.py`, `imitation.py`). It builds each algorithm at its sensible defaults
on a shared, identically-wrapped env and reports **steps-to-threshold**.

Two tasks (pick with --task):

  tracking  — the imitation/RSI task on a single reference (default the
              landing move `ref_land.npz`, which PPO took 0->88%). This is a
              *learnable* task with a known baseline, so it's where an
              algorithm swap can genuinely win: if SAC/TQC reach the success
              threshold in fewer steps, that compounds across the many moves a
              full climb needs. THIS is the informative comparison.

  climb     — the full climb-curriculum env. Honest expectation: every
              algorithm stays near-flat, because the blocker is reference
              quality / discovery (NEXT_STEPS A1d), not the optimizer. Running
              it confirms the diagnosis with data rather than asserting it.

Usage:
  python -m sim3d.bakeoff --task tracking --algos ppo,sac,tqc --steps 60000
  python -m sim3d.bakeoff --task climb    --algos ppo,sac,tqc --steps 60000

Off-policy algos run on 1 env (replay buffer); PPO uses --n-envs (default 4 on
the on-policy side). All CPU. Results print as a table and save to
data/runs/sim3d/bakeoff/<task>_<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


# ─── per-algorithm builders ──────────────────────────────────────────────────
# Each builder returns (model, n_envs, is_offpolicy). Hyperparameters are each
# algorithm's reasonable defaults for MuJoCo-style continuous control — the
# point is to compare algorithms as you'd actually reach for them, not to
# hand-tune one to win.

def _build_ppo(vec, *, seed, tb, log_std_init):
    import stable_baselines3 as sb3
    return sb3.PPO(
        "MlpPolicy", vec, verbose=0, device="cpu", seed=seed,
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.005,
        target_kl=0.03, policy_kwargs={"log_std_init": log_std_init},
        tensorboard_log=tb,
    )


def _build_sac(vec, *, seed, tb, log_std_init):
    import stable_baselines3 as sb3
    return sb3.SAC(
        "MlpPolicy", vec, verbose=0, device="cpu", seed=seed,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
        learning_starts=1000, tensorboard_log=tb,
    )


def _build_tqc(vec, *, seed, tb, log_std_init):
    from sb3_contrib import TQC
    return TQC(
        "MlpPolicy", vec, verbose=0, device="cpu", seed=seed,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
        learning_starts=1000, top_quantiles_to_drop_per_net=2,
        tensorboard_log=tb,
    )


_ALGOS = {
    "ppo": (_build_ppo, False),
    "sac": (_build_sac, True),
    "tqc": (_build_tqc, True),
}


# ─── shared, algorithm-agnostic progress callback ────────────────────────────

def _make_callback(base_cls, threshold: float, window: int, print_freq: int,
                   success_key: str, quality_key: Optional[str]):
    """One callback for on- AND off-policy. Everything happens in _on_step
    (off-policy has no PPO-style rollout boundary), reading the raw step infos.
    Records the first num_timesteps where rolling success over `window` eps
    crosses `threshold`."""

    class _Cb(base_cls):
        def __init__(self):
            super().__init__()
            self.succ_window = deque(maxlen=window)
            self.ep_done = 0
            self.steps_to_threshold = None
            self.q_sum, self.q_n = 0.0, 0
            self.best_sr = 0.0
            self._last_print = 0

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if quality_key is not None and quality_key in info:
                    self.q_sum += float(info[quality_key]); self.q_n += 1
                if "episode" in info:  # Monitor end-of-episode marker
                    self.ep_done += 1
                    self.succ_window.append(int(bool(info.get(success_key, False))))
            if len(self.succ_window) >= min(window, 20):
                sr = sum(self.succ_window) / len(self.succ_window)
                self.best_sr = max(self.best_sr, sr)
                if self.steps_to_threshold is None and sr >= threshold:
                    self.steps_to_threshold = int(self.num_timesteps)
            if self.num_timesteps - self._last_print >= print_freq:
                self._last_print = self.num_timesteps
                sr = (sum(self.succ_window) / len(self.succ_window)
                      if self.succ_window else 0.0)
                q = self.q_sum / max(1, self.q_n)
                qs = f"  {quality_key} {q:.3f}" if quality_key else ""
                print(f"    [{self.num_timesteps:>7}] success {sr*100:5.1f}% "
                      f"(n={self.ep_done}){qs}", flush=True)
            return True

    return _Cb


# ─── env factories (reuse production env-construction, not the trainers) ──────

def _tracking_vec(ref_path, n_envs, wall_json):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from sim3d.imitation import ImitationConfig, make_env
    icfg = ImitationConfig()
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec = vec_cls([make_env(ref_path, icfg, i, wall_json) for i in range(n_envs)])
    return VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)


def _climb_vec(n_envs, steps):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from sim3d.train import TrainConfig, _make_env_factory

    cfg = TrainConfig(climb_curriculum=True, total_timesteps=steps,
                      max_episode_steps=1000)
    factory = _make_env_factory(cfg)

    def _mk(rank):
        def _init():
            return Monitor(factory())
        return _init

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec = vec_cls([_mk(i) for i in range(n_envs)])
    return VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)


# ─── runner ──────────────────────────────────────────────────────────────────

def run(task: str, algos, steps: int, n_envs: int, ref_path: str,
        wall_json: Optional[str], threshold: float, out_dir: Path) -> dict:
    from stable_baselines3.common.callbacks import BaseCallback

    # The climb env has no real "success" channel during early training, so we
    # rank on the imitation/quality signal there; tracking ranks on is_success.
    if task == "tracking":
        success_key, quality_key = "is_success", "r_imit"
    else:
        success_key, quality_key = "is_success", "r_height"

    results = {}
    for name in algos:
        if name not in _ALGOS:
            raise SystemExit(f"unknown algo {name!r}; choose from {sorted(_ALGOS)}")
        build, offpolicy = _ALGOS[name]
        a_envs = 1 if offpolicy else n_envs

        print(f"\n=== {name.upper()}  ({'off' if offpolicy else 'on'}-policy, "
              f"{a_envs} env{'s' if a_envs > 1 else ''}, {steps} steps) ===", flush=True)
        if task == "tracking":
            vec = _tracking_vec(ref_path, a_envs, wall_json)
        else:
            vec = _climb_vec(a_envs, steps)

        tb = str(out_dir / "tb" / name)
        # log_std_init only consumed by PPO; off-policy builders ignore it.
        model = build(vec, seed=0, tb=tb, log_std_init=-1.5)

        cb_cls = _make_callback(BaseCallback, threshold, window=50,
                                print_freq=max(2000, steps // 40),
                                success_key=success_key, quality_key=quality_key)
        cb = cb_cls()
        t0 = time.time()
        model.learn(total_timesteps=steps, callback=cb, progress_bar=False)
        wall_s = time.time() - t0

        results[name] = {
            "offpolicy": offpolicy,
            "n_envs": a_envs,
            "steps": steps,
            "wall_seconds": round(wall_s, 1),
            "best_success_rate": round(cb.best_sr, 3),
            "steps_to_threshold": cb.steps_to_threshold,
            "episodes": cb.ep_done,
            f"mean_{quality_key}": round(cb.q_sum / max(1, cb.q_n), 4),
        }
        vec.close()
        print(f"    done in {wall_s:.0f}s | best success {cb.best_sr*100:.1f}% | "
              f"steps→{threshold:.0%}: {cb.steps_to_threshold}", flush=True)

    return results


def _print_table(task, threshold, results):
    print(f"\n{'='*64}\nBAKE-OFF RESULTS — task={task}, success threshold={threshold:.0%}")
    print(f"{'='*64}")
    hdr = f"{'algo':<6}{'best SR':>9}{'steps→thr':>12}{'eps':>7}{'wall(s)':>9}"
    print(hdr); print("-" * len(hdr))
    # Rank: reached threshold first wins; else higher best-SR wins.
    def _key(kv):
        r = kv[1]
        s = r["steps_to_threshold"]
        return (0, s) if s is not None else (1, -r["best_success_rate"])
    for name, r in sorted(results.items(), key=_key):
        stt = r["steps_to_threshold"]
        stt_s = f"{stt:,}" if stt is not None else "—"
        print(f"{name:<6}{r['best_success_rate']*100:>8.1f}%{stt_s:>12}"
              f"{r['episodes']:>7}{r['wall_seconds']:>9.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["tracking", "climb"], default="tracking")
    ap.add_argument("--algos", default="ppo,sac,tqc",
                    help="comma list from: ppo, sac, tqc")
    ap.add_argument("--steps", type=int, default=60_000,
                    help="per-algorithm training budget")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="parallel envs for the ON-policy algo (off-policy use 1)")
    ap.add_argument("--ref", default="data/runs/sim3d/imitation/ref_land.npz",
                    help="reference .npz for --task tracking")
    ap.add_argument("--wall-json", default=None,
                    help="wall JSON for a CMA-ES reference (else rebuilt from seed)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="success rate that defines steps-to-threshold")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("data/runs/sim3d/bakeoff") / f"{args.task}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run(args.task, algos, args.steps, args.n_envs, args.ref,
                  args.wall_json, args.threshold, out_dir)

    payload = {
        "task": args.task, "steps": args.steps, "threshold": args.threshold,
        "ref": args.ref if args.task == "tracking" else None,
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    _print_table(args.task, args.threshold, results)
    print(f"\nSaved → {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
