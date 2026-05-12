# RL — Gymnasium environment

A [Gymnasium](https://gymnasium.farama.org/) (the maintained successor
to OpenAI Gym) environment wrapping the `physics/` simulation. Plug a
PPO/SAC/etc. agent in and train it to climb.

> Not to be confused with `solver/rl_qlearn.py`, which is a tabular
> Q-learning toy on the static reachability graph. This `rl/` env runs
> the actual pymunk physics underneath, so the agent sees forces,
> body settling, and (eventually) dynamic moves.

---

## Quickstart

```bash
# Random-policy smoke test
docker compose exec beta-engine python -m rl --wall example-v2-boulder

# 5 random-policy episodes + GIF of the last one
docker compose exec beta-engine python -m rl --wall example-v2-boulder \
  --episodes 5 --gif

# Pure-random (no legality mask) — useful for testing reward shaping
docker compose exec beta-engine python -m rl --wall example-v2-boulder \
  --policy random --episodes 10
```

```bash
pip install stable-baselines3 tensorboard

# Train
python -m sim3d.train --steps 200_000 --run-id mb_v4 \
    --moonboard data/moonboard/sample-problems.json --problem 19215

# Watch loss curves
tensorboard --logdir data/runs/sim3d/mb_v4/tb

# Read episode stats
column -ts, data/runs/sim3d/mb_v4/episode_stats.csv | head -20

# Replay in the native viewer
python -m sim3d --play data/runs/sim3d/mb_v4/model.zip \
    --moonboard data/moonboard/sample-problems.json --problem 19215
```

GIF lands in `data/runs/<wall_id>-rl.gif`.

---

## Action / observation spaces

| | Type | Shape | Meaning |
|---|---|---|---|
| **Action** | `Discrete(4·N)` | `4 × n_holds` | flat index encoding `(limb, hold)`. `limb_idx = action // n_holds`; `hold_idx = action % n_holds`. |
| **Observation** | `Box(float32)` | `4 + 4·N + 4` | COM x, y, vx, vy (m, m·s⁻¹), then per-limb one-hot occupancy (4 blocks of `n_holds`), then per-limb force fraction (4 floats, 0 if in flight). |

For a 13-hold wall: `52` actions, `60`-dim observation. Both spaces
scale linearly with `n_holds` and are compatible with `MlpPolicy` in
Stable-Baselines3.

`env.legal_actions()` returns the currently-legal indices — useful
for debugging or for masked policies. The default reward signal
penalises (but doesn't block) illegal actions, so the agent has to
*learn* legality.

---

## Reward shaping

Defined at the top of `env.py` — tweak as needed.

| Signal | Default | Why |
|---|---|---|
| Efficiency (each step) | −0.5 | encourages short paths |
| Progress (per cm closer to nearest finish) | +0.05 | dense signal so the agent gets feedback on every move |
| Illegal-move penalty | −5 | tried to grab a foothold with a hand, etc. |
| Slip penalty | −10 | any attached limb maxed-out (force ≥ 100 % of its budget) |
| Completion bonus | +100 | a hand landed on a `is_finish` hold |

`reset()` settles the body for `settle_frames` (default 8) physics
frames after seeding the start pose; `step()` runs the same number
after each move so each transition has time to converge.

---

## How to plug in Stable-Baselines3 / RLlib

The env follows the Gymnasium 1.0 contract exactly:

```python
from stable_baselines3 import PPO
from rl.env import ClimbingEnv
from solver.wall import load_wall

env = ClimbingEnv(load_wall("example-v2-boulder"))
agent = PPO("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=100_000)
```

Add `gymnasium.wrappers.TimeLimit`, `Monitor`, `VecEnv` etc. as
usual; the env doesn't do anything weird that would break those
wrappers.

Stable-Baselines3 is pinned in the project `requirements.txt` now, so the
snippet above works after `pip install -r requirements.txt`. If you want SB3's
extra Atari/video tooling, install `stable-baselines3[extra]` in your own venv.

---

## Module reference

| File | Responsibility |
|------|----------------|
| `env.py` | `ClimbingEnv` — Gymnasium env wrapping `ClimbWorld` |
| `random_policy.py` | `random_policy`, `masked_random_policy`, `rollout` — for smoke tests |
| `__main__.py` | CLI: `python -m rl --wall <id>` |

---

## Future work

- **Train** something real: the env is SB3-compatible, but this package still
  ships only smoke-test/random-policy helpers. Use the snippet above as the
  starting point for a 2D PPO experiment.
- **Continuous actions**: today the action space is discrete
  `(limb, hold)`. Real climbing is continuous (target velocity, grip
  force, body shift). Switch to `Box` observations once dynamic moves
  are modelled.
- **Procedural walls**: the agent currently overfits to one wall.
  Train on synthetic or MoonBoard-derived walls for a policy that
  generalises to unseen routes.
- **Self-play / curriculum**: increasing wall difficulty, varying
  climber profiles (different reach, strength).
