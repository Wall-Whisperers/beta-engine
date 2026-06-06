"""RL training entry point for the 3D climbing simulator.

Wraps Stable-Baselines3 PPO around `Climbing3DEnv`, logs episode
stats to CSV + TensorBoard, and saves the trained policy to a run
directory you can later replay in the viewer.

Stable-Baselines3 is pinned in the project requirements because this module
is the supported PPO training entry point. If it is missing, this module raises
a clear ImportError with the install command.

Run directory layout:

    data/runs/sim3d/<run_id>/
    ├── config.json          # training config (wall, profile, hyperparams)
    ├── episode_stats.csv    # one row per finished episode (reward, length, outcome, ...)
    ├── tb/                  # TensorBoard event files
    └── model.zip            # the trained PPO policy
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from solver.wall import load_wall
from sim3d.body import ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig


def _require_sb3():
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        return sb3, BaseCallback, DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise ImportError(
            "sim3d.train requires stable-baselines3. "
            "Install project dependencies with:  pip install -r requirements.txt"
        ) from e


@dataclass
class TrainConfig:
    # Default to the smallest HANGABLE demo route. example-v2-boulder (the old
    # default) seeds the body at COM 0.28 m over-braced and falls in 3 steps —
    # training on it produces ~100% falls regardless of the reward. baby-v1
    # hangs cleanly at COM 1.24 m (~1 m climb, 5 holds). See _warn_if_unhangable.
    wall: str = "baby-v1"
    moonboard_file: Optional[str] = None
    moonboard_problem_id: Optional[int] = None
    moonboard_split: str = "train"            # train | validation | test when sampling a corpus
    split_seed: int = 42
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    moonboard_vertical_projection: bool = False
    height_cm: float = 175.0
    wingspan_cm: float = 175.0
    mass_kg: float = 70.0
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 42
    move_mode: str = "reach"            # snap is faster but less realistic
    start_mode: str = "seed"            # seed | ground-reach
    move_frames: int = 24               # shorter for snap, longer for reach
    # Continuous-joint control runs at ~7.8 Hz, so 30 steps = 3.84 s — far
    # too short to climb (a MoonBoard problem takes 10-60 s). 1000 ≈ 128 s.
    # (30 was a discrete-move holdover where 1 step = 1 full limb move.)
    max_episode_steps: int = 1000
    enable_slip: bool = True
    fall_z: float = 0.20
    body_intersection_penalty: float = 20.0
    out_dir: str = "data/runs/sim3d"
    run_id: Optional[str] = None        # auto-generated from timestamp if None
    n_envs: int = 1                     # parallel CPU env workers
    device: str = "auto"                # SB3/PyTorch device: auto | cpu | cuda
    # Normalise observations with a running mean/std (VecNormalize). Standard
    # practice for MuJoCo PPO — raw obs span wildly different scales (joint
    # angles in rad, world positions in m, binary grip flags). Disable only
    # to replay old models trained without normalisation.
    use_vec_normalize: bool = True
    # ── PPO algorithm knobs ──────────────────────────────────────────
    # Policy init — lower log_std_init shrinks the initial action std so
    # grips aren't randomly released on the first step.  -1.5 → std≈0.22.
    # Default 0.0 (std 1.0) releases ~2 grips on step 1 → instant fall.
    log_std_init: float = -1.5
    # Grip release deadband (asymmetric). Engage fires on intent > 0 (50%
    # chance with std≈0.22). Release fires on intent < -db — so db=0.5 keeps
    # release a deliberate ~2.3σ event (P≈1%), protecting existing grips
    # while still allowing new grip acquisition. A symmetric 0.5 deadband
    # (week8) blocked both sides: agent reached feet to holds but could never
    # grip them because engage also needed to clear 0.5.
    grip_intent_deadband: float = 0.5
    # PPO clip range. Lowered 0.2 → 0.1 (2026-06-05): the staged run blew the
    # trust region (clip_fraction 0.70, approx_kl 0.20) and a post-peak update
    # destroyed a learned 48%-success reach skill. 0.1 tightens each update.
    clip_range: float = 0.1
    # Entropy coefficient. 0.005 is enough for joint exploration while keeping
    # the policy stable enough to hang. 0.01 caused constant grip-noise and
    # prevented the policy from ever learning to hang (week3 result: 0 hangs).
    ent_coef: float = 0.005
    # Number of PPO optimisation epochs per rollout batch. Lowered 10 → 5: with
    # 10 epochs the policy over-fit each batch and drifted far past the trust
    # region (see clip_range note).
    n_epochs: int = 5
    # Early-stop the epoch loop once the policy has moved approx_kl past this —
    # the direct guard against the trust-region blowout that collapsed the
    # staged run (observed approx_kl ~0.20; healthy PPO is ~0.01–0.02).
    target_kl: float = 0.03
    # Linearly decay the learning rate to 0 over training. Constant 3e-4 let a
    # late update overshoot and forget a learned skill; decaying shrinks step
    # size as the policy nears competence.
    lr_decay: bool = True
    # ── Reward coefficients (matched to EnvConfig) ───────────────────
    # CLEAN-RESTART DEFAULTS (2026-06-05). The reward is height_progress
    # (primary) + hold_match (sparse grip nudge) + terminals + physics gates.
    # Every shaping term below is OFF by default — each was added in a past
    # week and either got farmed or blocked something else. Re-enable one at a
    # time, diagnostics-driven (see episode_stats.csv r_* columns), never all
    # at once.
    #
    # Dense finish-hold shaping (potential-based). Off: the primary up-signal
    # is now height_progress; this duplicated it and had a reference-point
    # jump (B2) that could leak non-telescoping reward. First candidate to
    # re-add (with B2 fixed) if the agent climbs but wanders off-route.
    finish_approach_coeff: float = 0.0
    # Per-limb reach shaping. Off: caused the fall-and-swing exploit (release
    # all grips, collect reach reward while limbs swing toward holds on the
    # way down).
    reach_approach_coeff: float = 0.0
    # Per-step survival bonus. Off: any positive value is a floor-hanging
    # attractor (grip lowest holds forever beats climbing).
    survival_bonus_coeff: float = 0.0
    # No release reward — week4 showed that release reward + big lump-sum
    # jackpot caused the agent to farm: grab jackpot → fall → repeat in
    # 8-35 step micro-episodes. Release must be neutral (0) so the only
    # incentive to release is reaching something worthwhile.
    grip_release_penalty: float = 0.0
    # GATED on HWM gain — now equivalent to extra hwm_height_scale, kept
    # for backwards compatibility with old CLIs.
    upward_velocity_coeff: float = 0.0
    # Primary dense signal: potential-based height progress, K·(com_z −
    # prev_com_z) per step. Symmetric (up pays, down costs), telescopes ⇒
    # un-farmable by oscillation. This is the canonical "reward up" term.
    height_progress_scale: float = 60.0
    # Legacy high-water-mark — inert by default, superseded by height_progress
    # (the one-way ratchet couldn't punish sliding back down).
    hwm_height_scale: float = 0.0
    # Rising-edge bonus for gripping any new hold (first touch per episode).
    # The ONLY grip incentive. Kept low so grab-then-fall stays negative-EV
    # (a handful of fresh matches < the 50 fall penalty).
    hold_match_bonus: float = 10.0
    # Lump-sum reward when a grip raises episode_max_grip_z. Off: this +50
    # jackpot was the main magnet for grab→fall→repeat micro-episode farming.
    new_high_grip_bonus: float = 0.0
    # Restore fall penalty to 50 — reducing it to 30 made short jackpot
    # episodes too cheap. Falling must cost more than a single grip bonus.
    fall_penalty: float = 50.0
    # Warm-start: path to an existing model.zip whose policy weights are
    # copied into the new model at construction. Hyperparameters (lr,
    # clip_range, ent_coef, etc.) and reward coefficients come from the
    # current run's config — only the network weights transfer.
    init_from: str = ""
    # Real resume: path to a model.zip to continue training from. Restores
    # policy weights AND optimizer state AND timestep counter. CSV is
    # appended. Use this to recover from a crash; use init_from to start a
    # new experiment from an old policy with different hyperparameters.
    resume: str = ""
    # Env-step interval at which model_latest.zip is overwritten by the
    # RollingBestCheckpointCallback (crash-safe atomic write).
    save_freq: int = 25_000
    # Energy penalty. Default env=0.001 (lowered from 0.005 which
    # swamped the HWM height signal).
    energy_penalty_coeff: float = 0.001
    video_freq: int = 100_000           # env-steps between mp4 rollouts (0 = disable)
    checkpoint_first_at: int = 1024     # env-step at which to save model_first.zip
    # Curriculum training (procedural wall generator + auto difficulty)
    curriculum: bool = False
    curriculum_start_difficulty: float = 0.0
    curriculum_min_difficulty: float = 0.0
    curriculum_max_difficulty: float = 1.0
    curriculum_window: int = 20
    curriculum_up_threshold: float = 0.60
    curriculum_down_threshold: float = 0.20
    curriculum_difficulty_step: float = 0.05
    curriculum_gen_cols: int = 12
    curriculum_gen_rows: int = 20  # see GeneratorConfig.rows: start sits 2 rows up
    curriculum_fallback_wall: str = "baby-v1"
    # ── Task-stage curriculum (A1: hang → reach-one → climb) ─────────
    # Breaks the "hangs but won't climb" exploration trap by handing the agent
    # the reach primitive on a fixed generated wall before the full climb.
    staged_curriculum: bool = False
    staged_wall: Optional[str] = None     # None → generate a wall (staged_gen_seed)
    staged_gen_seed: int = 7
    staged_difficulty: float = 0.0
    staged_window: int = 30               # rolling success window per sub-task
    staged_up_threshold: float = 0.60     # advance when success_rate ≥ this
    hang_target_steps: int = 60
    reach_one_coeff: float = 30.0
    reach_regrip_bonus: float = 50.0
    reach_episode_steps: int = 200
    # Full-climb reverse curriculum (seed progressively lower toward the start,
    # always targeting the finish). Reuses staged_wall/gen_seed/difficulty/window.
    climb_curriculum: bool = False
    climb_up_threshold: float = 0.40   # top-out rate to drop the start lower


class _EpisodeStatsCallback:
    """SB3-compatible callback that streams episode stats to a CSV.

    SB3 already logs to TensorBoard out of the box. This adds a flat
    CSV that's easy to grep / plot externally — one row per completed
    episode, columns: episode, total_steps, reward, length, outcome.
    """

    def __init__(self, csv_path: Path, BaseCallback, *, append: bool = False) -> None:
        self._csv_path = csv_path
        self._fh = None
        self._writer = None
        self._episode = 0
        self._BaseCallback = BaseCallback
        self._impl = None  # The real BaseCallback subclass
        self._append = bool(append)

    def build(self):
        outer = self

        class _Cb(self._BaseCallback):
            def _on_training_start(self) -> None:
                if outer._append and outer._csv_path.exists():
                    # Resume: append without re-writing the header. Also pick
                    # up the prior episode count so 'episode' column stays
                    # monotonic across the crash.
                    with open(outer._csv_path, "r", encoding="utf-8") as rf:
                        prior_lines = sum(1 for _ in rf)
                    outer._episode = max(0, prior_lines - 1)
                    outer._fh = open(outer._csv_path, "a", newline="", encoding="utf-8")
                    outer._writer = csv.writer(outer._fh)
                else:
                    outer._fh = open(outer._csv_path, "w", newline="", encoding="utf-8")
                    outer._writer = csv.writer(outer._fh)
                    outer._writer.writerow([
                        "episode", "total_steps", "reward", "length",
                        "outcome", "final_com_z", "n_slips", "body_intersections",
                        "curriculum_difficulty", "stage", "rc_pos",
                        # Per-term reward decomposition (sums to 'reward').
                        # Watch r_height dominate; any other column creeping up
                        # is the next exploit surfacing.
                        "r_height", "r_match", "r_reach", "r_finish", "r_fall",
                        "r_slip", "r_intersect", "r_energy", "r_other",
                    ])

            def _on_step(self) -> bool:
                # SB3 stuffs episode info into self.locals when an
                # episode finishes. Each parallel env emits its own.
                infos = self.locals.get("infos")
                dones = self.locals.get("dones")
                if infos is None:
                    infos = []
                if dones is None:
                    dones = []
                if hasattr(dones, "__len__") and not isinstance(dones, list):
                    dones = list(dones)
                for done, info in zip(dones, infos):
                    if not done:
                        continue
                    outer._episode += 1
                    # Episode return / length are stuffed under "episode" by SB3's Monitor.
                    ep = info.get("episode", {}) or {}
                    outcome = info.get("outcome", "?")
                    com = info.get("com", (0, 0, 0))
                    rt = info.get("rew_terms", {}) or {}
                    outer._writer.writerow([
                        outer._episode,
                        self.num_timesteps,
                        round(float(ep.get("r", 0.0)), 3),
                        int(ep.get("l", 0)),
                        outcome,
                        round(float(com[2]), 3),
                        int(info.get("slips", 0)),
                        int(info.get("body_intersections", 0)),
                        round(float(info.get("curriculum_difficulty", 0.0)), 4),
                        info.get("stage", ""),
                        info.get("rc_pos", ""),
                        round(float(rt.get("height", 0.0)), 3),
                        round(float(rt.get("match", 0.0)), 3),
                        round(float(rt.get("reach", 0.0)), 3),
                        round(float(rt.get("finish", 0.0)), 3),
                        round(float(rt.get("fall", 0.0)), 3),
                        round(float(rt.get("slip", 0.0)), 3),
                        round(float(rt.get("intersect", 0.0)), 3),
                        round(float(rt.get("energy", 0.0)), 3),
                        round(float(rt.get("other", 0.0)), 3),
                    ])
                outer._fh.flush()
                return True

            def _on_training_end(self) -> None:
                if outer._fh:
                    outer._fh.close()

        outer._impl = _Cb()
        return outer._impl


def _moonboard_splits_for_config(cfg: TrainConfig):
    if not cfg.moonboard_file or cfg.moonboard_problem_id is not None:
        return None
    from sim3d.moonboard import load_moonboard_corpus, split_moonboard_problems

    corpus = load_moonboard_corpus(cfg.moonboard_file)
    splits = split_moonboard_problems(
        corpus,
        seed=cfg.split_seed,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
    )
    if cfg.moonboard_split not in splits:
        raise ValueError(
            f"unknown MoonBoard split {cfg.moonboard_split!r}; "
            f"expected one of {sorted(splits)}"
        )
    if not splits[cfg.moonboard_split]:
        raise ValueError(f"MoonBoard split {cfg.moonboard_split!r} is empty")
    return splits


def _make_env_factory(cfg: TrainConfig, moonboard_splits=None):
    """Returns a Gymnasium env factory for SB3's vec-env wrappers."""

    profile = ClimberProfile(
        height_cm=cfg.height_cm,
        wingspan_cm=cfg.wingspan_cm,
        mass_kg=cfg.mass_kg,
    )
    env_cfg = EnvConfig(
        move_mode=cfg.move_mode,
        start_mode=cfg.start_mode,
        move_frames=cfg.move_frames,
        max_steps=cfg.max_episode_steps,
        enable_slip=cfg.enable_slip,
        fall_z=cfg.fall_z,
        body_intersection_penalty=cfg.body_intersection_penalty,
        grip_intent_deadband=cfg.grip_intent_deadband,
        finish_approach_coeff=cfg.finish_approach_coeff,
        reach_approach_coeff=cfg.reach_approach_coeff,
        survival_bonus_coeff=cfg.survival_bonus_coeff,
        grip_release_penalty=cfg.grip_release_penalty,
        upward_velocity_coeff=cfg.upward_velocity_coeff,
        energy_penalty_coeff=cfg.energy_penalty_coeff,
        height_progress_scale=cfg.height_progress_scale,
        hwm_height_scale=cfg.hwm_height_scale,
        hold_match_bonus=cfg.hold_match_bonus,
        new_high_grip_bonus=cfg.new_high_grip_bonus,
        fall_penalty=cfg.fall_penalty,
    )

    def _factory():
        if cfg.climb_curriculum:
            from sim3d.staged_curriculum import (
                ClimbCurriculumEnv, ClimbCurriculumConfig,
            )
            ccfg = ClimbCurriculumConfig(
                wall=cfg.staged_wall,
                gen_seed=cfg.staged_gen_seed,
                gen_difficulty=cfg.staged_difficulty,
                window=cfg.staged_window,
                up_threshold=cfg.climb_up_threshold,
                finish_approach_coeff=cfg.finish_approach_coeff,
                climb_episode_steps=cfg.max_episode_steps,
            )
            return ClimbCurriculumEnv(ccfg, profile=profile, env_config=env_cfg)

        if cfg.staged_curriculum:
            from sim3d.staged_curriculum import (
                StagedCurriculumEnv, StagedCurriculumConfig,
            )
            scfg = StagedCurriculumConfig(
                wall=cfg.staged_wall,
                gen_seed=cfg.staged_gen_seed,
                gen_difficulty=cfg.staged_difficulty,
                window=cfg.staged_window,
                up_threshold=cfg.staged_up_threshold,
                hang_target_steps=cfg.hang_target_steps,
                reach_one_coeff=cfg.reach_one_coeff,
                reach_regrip_bonus=cfg.reach_regrip_bonus,
                reach_episode_steps=cfg.reach_episode_steps,
                climb_episode_steps=cfg.max_episode_steps,
            )
            return StagedCurriculumEnv(scfg, profile=profile, env_config=env_cfg)

        if cfg.moonboard_file:
            from sim3d.moonboard import (
                load_moonboard_problems, moonboard_problem_to_wall, find_problem,
            )
            if cfg.moonboard_problem_id is not None:
                problems = load_moonboard_problems(cfg.moonboard_file)
                problem = find_problem(problems, id=cfg.moonboard_problem_id)
                if problem is None:
                    raise ValueError(
                        f"problem id {cfg.moonboard_problem_id} not found in "
                        f"{cfg.moonboard_file}"
                    )
                wall = moonboard_problem_to_wall(
                    problem,
                    vertical_projection=cfg.moonboard_vertical_projection,
                )
                return Climbing3DEnv(wall, profile=profile, config=env_cfg)

            from sim3d.moonboard_env import MoonboardClimbing3DEnv

            assert moonboard_splits is not None
            return MoonboardClimbing3DEnv(
                moonboard_splits[cfg.moonboard_split],
                profile=profile,
                config=env_cfg,
                vertical_projection=cfg.moonboard_vertical_projection,
            )

        if cfg.curriculum:
            from sim3d.curriculum import CurriculumEnv, CurriculumConfig
            from solver.wall import DEFAULT_CELL_SIZE_CM
            cur_cfg = CurriculumConfig(
                start_difficulty=cfg.curriculum_start_difficulty,
                min_difficulty=cfg.curriculum_min_difficulty,
                max_difficulty=cfg.curriculum_max_difficulty,
                window=cfg.curriculum_window,
                up_threshold=cfg.curriculum_up_threshold,
                down_threshold=cfg.curriculum_down_threshold,
                difficulty_step=cfg.curriculum_difficulty_step,
                gen_cols=cfg.curriculum_gen_cols,
                gen_rows=cfg.curriculum_gen_rows,
                gen_cell_size_cm=DEFAULT_CELL_SIZE_CM,
                fallback_wall=cfg.curriculum_fallback_wall,
            )
            return CurriculumEnv(cur_cfg, profile=profile, env_config=env_cfg)

        wall = load_wall(cfg.wall)
        return Climbing3DEnv(wall, profile=profile, config=env_cfg)

    return _factory


def _warn_if_unhangable(factory, seed: int, steps: int = 20) -> None:
    """One-time startup sanity check: if the seed pose can't even hang (the body
    falls under a zero action), print a loud warning.

    An unhangable seed makes training produce ~100% falls no matter how good the
    reward is — the single most common silent foot-gun here (most hand-designed
    demo walls place the start holds too close to the feet for this body, so the
    hands settle above their slip cap and shed on the first steps). Non-fatal: we
    warn and continue, because the user may be deliberately debugging.
    """
    try:
        env = factory()
    except Exception:  # noqa: BLE001 — never let a sanity check kill training
        return
    try:
        _obs, info = env.reset(seed=seed)
        seed_z = float(info.get("com", (0.0, 0.0, 0.0))[2])
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        fell_at = None
        for i in range(steps):
            _obs, _r, term, _trunc, info = env.step(zero)
            if term and info.get("outcome") == "fell":
                fell_at = i + 1
                break
        if fell_at is not None:
            bar = "=" * 72
            print(
                f"\n{bar}\n"
                f"  ⚠  UNHANGABLE SEED POSE: the body falls in {fell_at} steps under a\n"
                f"     zero action (seed COM {seed_z:.2f} m). Training on this wall will\n"
                f"     produce ~100% falls regardless of the reward. Use a hangable wall\n"
                f"     (curriculum walls are hang-tuned; baby-v1 / climb-v1 hang) or widen\n"
                f"     the hand–foot vertical gap on the start holds.\n{bar}\n"
            )
        else:
            print(f"[hang-check] seed pose holds (COM {seed_z:.2f} m, no fall in "
                  f"{steps} zero-action steps).")
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def train(cfg: TrainConfig) -> Path:
    sb3, BaseCallback, DummyVecEnv, SubprocVecEnv = _require_sb3()
    from stable_baselines3.common.monitor import Monitor

    if cfg.run_id is None:
        cfg.run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the config for reproducibility.
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    moonboard_splits = _moonboard_splits_for_config(cfg)
    if moonboard_splits is not None:
        from sim3d.moonboard import moonboard_split_manifest

        with open(out_dir / "moonboard_splits.json", "w", encoding="utf-8") as f:
            json.dump(moonboard_split_manifest(moonboard_splits), f, indent=2)

    factory = _make_env_factory(cfg, moonboard_splits)

    # Loud, non-fatal warning if the chosen wall's seed pose can't even hang —
    # catches the "trained 1 M steps on an unhangable wall" foot-gun up front.
    _warn_if_unhangable(factory, cfg.seed)

    def _make_monitored_env(rank: int):
        def _init():
            env = factory()
            env.reset(seed=cfg.seed + rank)
            return Monitor(env)
        return _init

    n_envs = max(1, int(cfg.n_envs))
    env_fns = [_make_monitored_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    if cfg.use_vec_normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        if cfg.resume:
            _vn_path = Path(cfg.resume).parent / "vec_normalize.pkl"
            if _vn_path.exists():
                vec_env = VecNormalize.load(str(_vn_path), vec_env)
                print(f"  Loaded VecNormalize stats from {_vn_path}")
            else:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # TensorBoard is an optional sub-dep. Enable logging only if it's
    # importable; otherwise SB3 errors out at .learn() time.
    tb_log = None
    try:
        import tensorboard  # noqa: F401
        (out_dir / "tb").mkdir(exist_ok=True)
        tb_log = str(out_dir / "tb")
    except ImportError:
        print("(tensorboard not installed — skipping TB logging. "
              "Install with `pip install tensorboard` to enable.)")

    policy_kwargs = {}
    if cfg.log_std_init != 0.0:
        policy_kwargs["log_std_init"] = cfg.log_std_init

    if cfg.resume and cfg.init_from:
        raise SystemExit(
            "--resume and --init-from are mutually exclusive. "
            "--resume continues a crashed run (restores optimizer + step "
            "counter); --init-from copies weights into a new experiment."
        )

    reset_num_timesteps = True
    if cfg.resume:
        from pathlib import Path as _P
        resume_path = _P(cfg.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume path does not exist: {resume_path}")
        print(f"Resuming training from {resume_path}")
        # PPO.load restores policy weights, optimizer moments, and the
        # num_timesteps counter. We pass custom_objects to override any
        # hyperparameter that may have changed in the new CLI invocation —
        # without this, the saved values would be silently used and tuning
        # via CLI between resumes would be a no-op.
        custom_objects = {
            "learning_rate": cfg.learning_rate,
            "clip_range":    cfg.clip_range,
            "ent_coef":      cfg.ent_coef,
            "n_epochs":      cfg.n_epochs,
            "n_steps":       cfg.n_steps,
            "batch_size":    cfg.batch_size,
            "gamma":         cfg.gamma,
        }
        model = sb3.PPO.load(
            str(resume_path),
            env=vec_env,
            device=cfg.device,
            tensorboard_log=tb_log,
            custom_objects=custom_objects,
        )
        reset_num_timesteps = False
        print(f"  resumed at step {model.num_timesteps}, "
              f"target = {cfg.total_timesteps}")
    else:
        # Linear LR decay (SB3 calls the schedule with progress_remaining 1→0).
        base_lr = cfg.learning_rate
        lr = (lambda pr: pr * base_lr) if cfg.lr_decay else base_lr
        model = sb3.PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            target_kl=cfg.target_kl,
            seed=cfg.seed,
            tensorboard_log=tb_log,
            device=cfg.device,
            verbose=1,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )

        if cfg.init_from:
            from pathlib import Path as _P
            init_path = _P(cfg.init_from)
            if not init_path.exists():
                raise SystemExit(f"--init-from path does not exist: {init_path}")
            print(f"Warm-starting policy from {init_path}")
            loaded = sb3.PPO.load(str(init_path), device=cfg.device)
            # Copy weights only — keep the new model's hyperparameters and env.
            model.set_parameters(loaded.get_parameters(), exact_match=False)

    csv_cb = _EpisodeStatsCallback(
        out_dir / "episode_stats.csv",
        BaseCallback,
        append=bool(cfg.resume),
    ).build()

    # Compose the callback list: CSV stats + first/mid/last checkpoints +
    # crash-safe rolling-best checkpoints + (optionally) periodic mp4.
    from sim3d.callbacks import (
        FirstMidLastCheckpointCallback,
        RollingBestCheckpointCallback,
        VideoRolloutCallback,
    )
    callbacks = [
        csv_cb,
        FirstMidLastCheckpointCallback(
            out_dir=str(out_dir),
            total_timesteps=cfg.total_timesteps,
            first_at=cfg.checkpoint_first_at,
            verbose=1,
        ),
        RollingBestCheckpointCallback(
            out_dir=str(out_dir),
            save_freq=cfg.save_freq,
            window=100,
            verbose=1,
        ),
    ]
    if cfg.video_freq > 0:
        # Use a fresh env (not the vec_env) for deterministic eval rollouts.
        eval_env = factory()
        eval_env.reset(seed=cfg.seed + 9999)
        callbacks.append(
            VideoRolloutCallback(
                eval_env=eval_env,
                video_dir=str(out_dir / "videos"),
                eval_freq=cfg.video_freq,
                verbose=1,
            )
        )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(out_dir / "model.zip")
    from stable_baselines3.common.vec_env import VecNormalize as _VN
    if isinstance(vec_env, _VN):
        vec_env.save(str(out_dir / "vec_normalize.pkl"))

    print(f"\nTraining complete. Run dir: {out_dir}")
    print(f"  model:         {out_dir / 'model.zip'}")
    print(f"  checkpoints:   {out_dir / 'model_first.zip'}, "
          f"{out_dir / 'model_mid.zip'}, {out_dir / 'model_last.zip'}")
    print(f"  episode CSV:   {out_dir / 'episode_stats.csv'}")
    if cfg.video_freq > 0:
        print(f"  videos:        {out_dir / 'videos'}/*.mp4")
    print(f"  env workers:   {n_envs}")
    print(f"  torch device:  {model.device}")
    if tb_log:
        print(f"  tensorboard:   tensorboard --logdir {out_dir / 'tb'}")
    replay_cmd = f"python -m sim3d --play {out_dir / 'model.zip'}"
    if cfg.moonboard_file:
        replay_cmd += f" --moonboard {cfg.moonboard_file}"
        if cfg.moonboard_problem_id is not None:
            replay_cmd += f" --problem {cfg.moonboard_problem_id}"
        elif moonboard_splits is not None:
            first = moonboard_splits[cfg.moonboard_split][0]
            replay_cmd += f" --problem {first.id} --moonboard-full-board"
        if cfg.moonboard_vertical_projection:
            replay_cmd += " --vertical-projection"
        if cfg.start_mode != "seed":
            replay_cmd += f" --start-mode {cfg.start_mode}"
    else:
        replay_cmd += f" --wall {cfg.wall}"
    replay_cmd += (
        f" --height {cfg.height_cm}"
        f" --wingspan {cfg.wingspan_cm}"
        f" --mass {cfg.mass_kg}"
        f" --move-mode {cfg.move_mode}"
        f" --play-frames {cfg.move_frames}"
    )
    print(f"  replay:       {replay_cmd}")
    return out_dir / "model.zip"


def play(model_path: str | Path, env: Climbing3DEnv, *, deterministic: bool = True) -> dict:
    """Run a single deterministic episode in the given env using a saved
    SB3 model. Returns the final info dict."""
    sb3, _BaseCallback, _DummyVecEnv, _SubprocVecEnv = _require_sb3()
    model = sb3.PPO.load(str(model_path), device="auto")
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            return info


# ─── CLI ──────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d.train")
    p.add_argument("--wall", default="baby-v1",
                   help="Wall id for single-wall training. Default baby-v1 (smallest "
                        "hangable demo route). NB: example-v2-boulder/infinity-labyrinth/"
                        "moonlit-snake have unhangable start geometry for this body.")
    p.add_argument("--moonboard", dest="moonboard_file")
    p.add_argument("--problem", dest="moonboard_problem_id", type=int,
                   help="Train on one MoonBoard problem id. Omit to sample a deterministic split.")
    p.add_argument("--moonboard-split", default="train", choices=("train", "validation", "test"),
                   help="MoonBoard corpus split to sample when --problem is omitted.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Deterministic MoonBoard train/validation/test split seed.")
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--vertical-projection", action="store_true",
                   help="Use MoonBoard vertical-projection geometry.")
    p.add_argument("--height", type=float, default=175.0)
    p.add_argument("--wingspan", type=float, default=175.0)
    p.add_argument("--mass", type=float, default=70.0)
    p.add_argument("--steps", type=int, default=100_000,
                   help="Total environment steps (PPO default budget).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024,
                   help="PPO rollout length per update.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move-mode", default="reach", choices=("snap", "reach", "dyno"))
    p.add_argument("--start-mode", default="seed", choices=("seed", "ground-reach"),
                   help="seed starts welded on route holds; ground-reach starts on the floor and reaches to start hands.")
    p.add_argument("--move-frames", type=int, default=24)
    p.add_argument("--episode-steps", type=int, default=1000,
                   help="Max env steps per episode. Continuous control is ~7.8 Hz, "
                        "so 1000 ≈ 128 s. Use ~30 only for discrete-move.")
    p.add_argument("--no-slip", "--no_slip", dest="no_slip", action="store_true")
    p.add_argument("--fall-z", type=float, default=0.20,
                   help="Pelvis height below which the episode terminates as a fall. "
                        "Default 0.20 m. Raise to 0.40 m to prevent floor-hanging "
                        "on low footholds — forces the agent to maintain start-hold "
                        "height or die.")
    p.add_argument("--body-intersection-penalty", type=float, default=20.0,
                   help="Reward penalty per limb-vs-torso/pelvis intersection contact (gating coefficient).")
    p.add_argument("--video-freq", type=int, default=100_000,
                   help="Env-steps between mp4 rollouts. 0 disables video saving.")
    p.add_argument("--checkpoint-first-at", type=int, default=1024,
                   help="Env-step at which to save model_first.zip.")
    p.add_argument("--out-dir", default="data/runs/sim3d")
    p.add_argument("--run-id", default=None,
                   help="Custom run id; default = run_<timestamp>.")
    p.add_argument("--init-from", default="",
                   help="Warm-start: path to an existing model.zip whose "
                        "policy weights are copied into the new model. "
                        "Hyperparameters and reward come from the new config; "
                        "only network weights transfer.")
    p.add_argument("--resume", default="",
                   help="Resume training from a saved model.zip. Restores "
                        "policy weights, optimizer state, and the timestep "
                        "counter. Appends to episode_stats.csv. Use this to "
                        "recover from a crash. Mutually exclusive with "
                        "--init-from.")
    p.add_argument("--save-freq", type=int, default=25_000,
                   help="Env-step interval at which model_latest.zip is "
                        "overwritten by the crash-safe checkpoint callback.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel environment workers. Use >1 to speed up CPU-bound MuJoCo rollouts.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="PyTorch device for PPO policy updates. Env simulation still runs on CPU.")
    p.add_argument("--log-std-init", type=float, default=-1.5,
                   help="Initial log-std for the Gaussian policy. -1.5 -> std~0.22, "
                        "keeps early actions small so grips aren't randomly dropped on step 1.")
    p.add_argument("--grip-deadband", type=float, default=0.5,
                   help="Grip intent deadband. Intents in (-db, +db) hold current grip state. "
                        "At db=0.5 + log_std_init=-1.5 (std≈0.22), release is a 2.3σ event "
                        "(P≈1%%) — deliberate, not random noise.")
    p.add_argument("--clip-range", type=float, default=0.1,
                   help="PPO clip range. 0.1 (default) keeps updates inside the trust "
                        "region; 0.2 blew it (clip_fraction 0.70) and collapsed a learned skill.")
    p.add_argument("--ent-coef", type=float, default=0.005,
                   help="PPO entropy coefficient. 0.005 gives joint exploration while "
                        "staying stable enough to hang; 0.01 caused constant grip-noise "
                        "and 0 hangs (week3). Matches the TrainConfig default.")
    p.add_argument("--n-epochs", type=int, default=5,
                   help="PPO optimisation epochs per rollout. 5 (default) avoids the "
                        "per-batch over-fitting that drove approx_kl to 0.20 at 10 epochs.")
    p.add_argument("--target-kl", type=float, default=0.03,
                   help="Early-stop the epoch loop once approx_kl exceeds this. Direct "
                        "guard against the trust-region blowout (healthy ~0.01–0.02).")
    p.add_argument("--no-lr-decay", action="store_true",
                   help="Disable the linear learning-rate decay to 0 (on by default).")
    p.add_argument("--finish-approach-coeff", type=float, default=0.0,
                   help="OFF by default (clean restart). Dense finish-hold shaping: "
                        "coeff × (prev_dist − cur_dist) per step. Duplicated the "
                        "height-progress signal and had a reference-jump seam (B2); "
                        "first candidate to re-add if the agent wanders off-route.")
    p.add_argument("--reach-approach-coeff", type=float, default=0.0,
                   help="OFF by default (clean restart). Per-step per-limb reach "
                        "shaping — caused the fall-and-swing exploit.")
    p.add_argument("--survival-bonus-coeff", type=float, default=0.0,
                   help="OFF by default (clean restart). Per-step grip-fraction bonus "
                        "— any positive value is a floor-hanging attractor.")
    p.add_argument("--grip-release-penalty", type=float, default=0.0,
                   help="Per-limb penalty when a grip releases. 0 = neutral (default). "
                        "Negative = reward for releasing, which caused jackpot-farming "
                        "in week4 (8-step micro-episodes). Keep at 0.")
    p.add_argument("--upward-velocity-coeff", type=float, default=0.0,
                   help="GATED on HWM gain — effectively an extra coefficient "
                        "on hwm_height_scale. Kept for backwards compatibility; "
                        "the original path-dependent semantics was exploitable. "
                        "Default 0; the default --hwm-height-scale 50 is the "
                        "primary upward signal.")
    p.add_argument("--height-progress-scale", type=float, default=60.0,
                   help="PRIMARY dense signal: reward per metre of COM-z progress, "
                        "K·(com_z − prev_com_z) per step. Symmetric (up pays, down "
                        "costs) and telescoping — un-farmable by oscillation. "
                        "Default 60 (≈+90–120 over a full 1.5–2 m climb).")
    p.add_argument("--hwm-height-scale", type=float, default=0.0,
                   help="OFF by default — superseded by --height-progress-scale. "
                        "Legacy high-water-mark (one-way ratchet; can't punish "
                        "sliding back down). Additive lever only.")
    p.add_argument("--hold-match-bonus", type=float, default=10.0,
                   help="Rising-edge bonus for gripping any new hold per episode. "
                        "The only grip incentive. Keep well under fall-penalty so "
                        "grab-then-fall stays negative-EV.")
    p.add_argument("--new-high-grip-bonus", type=float, default=0.0,
                   help="OFF by default (clean restart). Lump-sum on raising "
                        "max-grip-z — was the main grab→fall→repeat farming magnet.")
    p.add_argument("--fall-penalty", type=float, default=50.0,
                   help="Terminal penalty on fall. Must exceed any single lump-sum "
                        "grip bonus so micro-episode farming is negative-EV.")
    p.add_argument("--energy-penalty-coeff", type=float, default=0.001,
                   help="Per-step energy penalty: coeff * sum(ctrl^2). "
                        "Default 0.001 (was 0.005 which swamped the height signal).")
    p.add_argument("--curriculum", action="store_true",
                   help="Train on procedurally generated walls with automatic difficulty scheduling.")
    p.add_argument("--curriculum-start-difficulty", type=float, default=0.0,
                   help="Starting difficulty level (0.0=easy jugs, 1.0=hard crimps). Default: 0.0")
    p.add_argument("--curriculum-max-difficulty", type=float, default=1.0,
                   help="Difficulty ceiling. Default: 1.0")
    p.add_argument("--curriculum-window", type=int, default=20,
                   help="Rolling window of episodes for success-rate computation. Default: 20")
    p.add_argument("--curriculum-up-threshold", type=float, default=0.60,
                   help="Success rate above which difficulty increases. Default: 0.60")
    p.add_argument("--curriculum-down-threshold", type=float, default=0.20,
                   help="Success rate below which difficulty decreases. Default: 0.20")
    p.add_argument("--curriculum-difficulty-step", type=float, default=0.05,
                   help="Difficulty increment per up-tick (halved for down-ticks). Default: 0.05")
    p.add_argument("--curriculum-cols", type=int, default=12,
                   help="Grid columns for generated walls. Default: 12")
    p.add_argument("--curriculum-rows", type=int, default=20,
                   help="Grid rows for generated walls. Default: 20")
    p.add_argument("--curriculum-fallback-wall", default="baby-v1",
                   help="Wall id to use if generation fails. Default: baby-v1 (hangable)")
    # ── Task-stage curriculum (A1: hang → reach-one → climb) ─────────
    p.add_argument("--staged-curriculum", action="store_true",
                   help="Train the hang→reach-one→climb task curriculum: hands the "
                        "agent the reach primitive before the full climb. Breaks "
                        "the 'hangs but won't climb' exploration trap.")
    p.add_argument("--climb-curriculum", action="store_true",
                   help="Reverse curriculum on the FULL climb: seed the body "
                        "progressively lower down the route (start near the finish, "
                        "walk back), always targeting the finish. For chaining a "
                        "full top-out. Reuses --staged-wall/--staged-gen-seed/etc.")
    p.add_argument("--climb-up-threshold", type=float, default=0.40,
                   help="Top-out rate at which the climb curriculum drops the start "
                        "one stance lower. Default 0.40 (climbing is hard).")
    p.add_argument("--staged-wall", default=None,
                   help="Static wall id for the staged curriculum. Default: generate "
                        "a wall (--staged-gen-seed) with a clean L/R hand spine.")
    p.add_argument("--staged-gen-seed", type=int, default=7,
                   help="Generator seed for the staged-curriculum wall. Default: 7.")
    p.add_argument("--staged-difficulty", type=float, default=0.0,
                   help="Generated wall difficulty for the staged curriculum (0=jugs).")
    p.add_argument("--staged-window", type=int, default=30,
                   help="Rolling success window per sub-task before advancing. Default 30.")
    p.add_argument("--staged-up-threshold", type=float, default=0.60,
                   help="Success rate at which a sub-task advances. Default 0.60.")
    p.add_argument("--hang-target-steps", type=int, default=60,
                   help="Hang stage: steps survived without falling = success. Default 60.")
    p.add_argument("--reach-one-coeff", type=float, default=30.0,
                   help="reach-one: dense signed-potential reward × (prev_d − d) toward "
                        "the target hold, gated on the other 3 limbs anchored. Default 30.")
    p.add_argument("--reach-regrip-bonus", type=float, default=50.0,
                   help="reach-one: one-shot bonus + success when the mover grips the "
                        "target hold. Default 50.")
    p.add_argument("--reach-episode-steps", type=int, default=200,
                   help="reach-one episode truncation cap (steps). Default 200.")
    p.add_argument("--no-vec-normalize", action="store_true",
                   help="Disable VecNormalize observation normalisation. "
                        "Only use to replay old models trained without it.")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        wall=args.wall,
        moonboard_file=args.moonboard_file,
        moonboard_problem_id=args.moonboard_problem_id,
        moonboard_split=args.moonboard_split,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        moonboard_vertical_projection=args.vertical_projection,
        height_cm=args.height,
        wingspan_cm=args.wingspan,
        mass_kg=args.mass,
        total_timesteps=args.steps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        move_mode=args.move_mode,
        start_mode=args.start_mode,
        move_frames=args.move_frames,
        max_episode_steps=args.episode_steps,
        enable_slip=not args.no_slip,
        fall_z=args.fall_z,
        body_intersection_penalty=args.body_intersection_penalty,
        out_dir=args.out_dir,
        run_id=args.run_id,
        init_from=args.init_from,
        resume=args.resume,
        save_freq=args.save_freq,
        n_envs=args.n_envs,
        device=args.device,
        video_freq=args.video_freq,
        checkpoint_first_at=args.checkpoint_first_at,
        log_std_init=args.log_std_init,
        grip_intent_deadband=args.grip_deadband,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        lr_decay=not args.no_lr_decay,
        finish_approach_coeff=args.finish_approach_coeff,
        reach_approach_coeff=args.reach_approach_coeff,
        survival_bonus_coeff=args.survival_bonus_coeff,
        grip_release_penalty=args.grip_release_penalty,
        upward_velocity_coeff=args.upward_velocity_coeff,
        height_progress_scale=args.height_progress_scale,
        hwm_height_scale=args.hwm_height_scale,
        hold_match_bonus=args.hold_match_bonus,
        new_high_grip_bonus=args.new_high_grip_bonus,
        fall_penalty=args.fall_penalty,
        energy_penalty_coeff=args.energy_penalty_coeff,
        curriculum=args.curriculum,
        curriculum_start_difficulty=args.curriculum_start_difficulty,
        curriculum_max_difficulty=args.curriculum_max_difficulty,
        curriculum_window=args.curriculum_window,
        curriculum_up_threshold=args.curriculum_up_threshold,
        curriculum_down_threshold=args.curriculum_down_threshold,
        curriculum_difficulty_step=args.curriculum_difficulty_step,
        curriculum_gen_cols=args.curriculum_cols,
        curriculum_gen_rows=args.curriculum_rows,
        curriculum_fallback_wall=args.curriculum_fallback_wall,
        staged_curriculum=args.staged_curriculum,
        staged_wall=args.staged_wall,
        staged_gen_seed=args.staged_gen_seed,
        staged_difficulty=args.staged_difficulty,
        staged_window=args.staged_window,
        staged_up_threshold=args.staged_up_threshold,
        hang_target_steps=args.hang_target_steps,
        reach_one_coeff=args.reach_one_coeff,
        reach_regrip_bonus=args.reach_regrip_bonus,
        reach_episode_steps=args.reach_episode_steps,
        climb_curriculum=args.climb_curriculum,
        climb_up_threshold=args.climb_up_threshold,
        use_vec_normalize=not args.no_vec_normalize,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
