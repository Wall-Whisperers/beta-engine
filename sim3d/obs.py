"""Observation builder for the 3D climbing environment.

The observation is wall-size-independent: it always reports K=8 nearest holds
in the climber's local frame plus per-limb goal vectors to the (single)
finish hold. This makes the same policy applicable to MoonBoard 11×18, the
generic editor walls, and any synthetic wall.

Layout (fixed dimension, regardless of wall size):

    [0  : 3 )   pelvis world pos                       (3)
    [3  : 9 )   pelvis rot6d (cols 0 and 1 of R)       (6)
    [9  : 12)   centre-of-mass world pos               (3)
    [12 : 12 + n_act)   joint qpos[7:]                 (n_act)
    [..        + n_act) joint qvel[6:]                 (n_act)
    [..  : .. + 4)      per-limb grip flag             (4)
    [..  : .. + 56)     K=8 nearest holds × 7          (56)
                          per hold: rel_pos_in_pelvis (3),
                                    role_onehot (3, start/mid/finish),
                                    is_gripping (1)
    [..  : .. + 12)     per-limb anchor/goal vector    (12)
                          zero if gripped, else (finish_world - tip_world)
    [..  : .. + 1)      euclid dist (highest gripped hand → nearest finish)

Per-stream NaN/Inf guard: each stream is checked at the end and replaced
with zeros, with a one-line warning printed on first occurrence (so it does
not raise from inside a vectorised env).
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim3d.env import EnvConfig
    from sim3d.world import Climb3DWorld


K_NEAREST_HOLDS: int = 8
HOLD_OBS_DIM: int = 7   # 3 rel_pos + 3 role onehot + 1 gripping flag

_LIMBS = ("LH", "RH", "LF", "RF")


def observation_dim(world: "Climb3DWorld") -> int:
    """Total observation length for the given (already constructed) world.

    Depends on `world.model.nu` (number of actuated joints). It is the same
    for every wall as long as the climber body is unchanged, which is the
    invariant we want for MoonBoard sampling.
    """
    n_act = int(world.model.nu)
    return (
        3 + 6 + 3                # pelvis pos, rot6d, com
        + n_act + n_act          # qpos[7:], qvel[6:]
        + 4                      # grip flags
        + K_NEAREST_HOLDS * HOLD_OBS_DIM
        + 4 * 3                  # anchor/goal vectors per limb
        + 1                      # finish distance scalar
    )


_WARNED_STREAMS: set[str] = set()


def _guard(arr: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(arr)):
        if name not in _WARNED_STREAMS:
            _WARNED_STREAMS.add(name)
            print(
                f"[sim3d.obs] WARNING: NaN/Inf in stream '{name}' — zeroed.",
                file=sys.stderr,
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _rot6d_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a (w,x,y,z) quaternion to the 6-D rotation encoding
    (first two columns of the rotation matrix, flattened)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R[:, :2].flatten()


def build_observation(world: "Climb3DWorld", env_config: "EnvConfig") -> np.ndarray:
    """Assemble the flat observation vector.

    Args:
        world: live Climb3DWorld (post mj_forward / step).
        env_config: env config — currently unused but kept for symmetry so
            future curricula can vary K, encoding, etc.

    Returns:
        float32 1-D ndarray of length `observation_dim(world)`.
    """
    del env_config  # reserved for future use

    d = world.data
    m = world.model
    n_act = int(m.nu)

    # ── Stream 1 — proprioception ────────────────────────────────────────
    pelvis = np.array(d.qpos[0:3], dtype=np.float64)
    pelvis_quat = np.array(d.qpos[3:7], dtype=np.float64)
    rot6d = _rot6d_from_quat(pelvis_quat)
    com = world.com()
    joint_pos = np.array(d.qpos[7: 7 + n_act], dtype=np.float64)
    joint_vel = np.array(d.qvel[6: 6 + n_act], dtype=np.float64)

    # Grip flags
    grip_flags = np.array(
        [1.0 if world.on_hold(l) is not None else 0.0 for l in _LIMBS],
        dtype=np.float64,
    )

    stream1 = _guard(
        np.concatenate([pelvis, rot6d, com, joint_pos, joint_vel, grip_flags]),
        "proprioception",
    )

    # ── Stream 2 — K-nearest holds in pelvis-relative coords ─────────────
    # Build the rotation matrix once for the pelvis frame.
    w_, x_, y_, z_ = pelvis_quat
    R = np.array([
        [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_),     2 * (x_ * z_ + y_ * w_)],
        [2 * (x_ * y_ + z_ * w_),     1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
        [2 * (x_ * z_ - y_ * w_),     2 * (y_ * z_ + x_ * w_),     1 - 2 * (x_ * x_ + y_ * y_)],
    ], dtype=np.float64)
    R_T = R.T  # world → pelvis-local

    gripping_ids = {world.on_hold(l) for l in _LIMBS}
    gripping_ids.discard(None)

    hold_items = list(world._hold_meta_by_id.values())
    sorted_holds = sorted(
        hold_items,
        key=lambda meta: float(np.linalg.norm(np.array(meta["world_pos"]) - pelvis)),
    )[:K_NEAREST_HOLDS]

    stream2 = np.zeros(K_NEAREST_HOLDS * HOLD_OBS_DIM, dtype=np.float64)
    for i, meta in enumerate(sorted_holds):
        rel_world = np.array(meta["world_pos"]) - pelvis
        rel_pelvis = R_T @ rel_world

        # Role one-hot: [start, mid, finish]. "mid" = neither flag set.
        if meta["is_finish"]:
            role = (0.0, 0.0, 1.0)
        elif meta["is_start"]:
            role = (1.0, 0.0, 0.0)
        else:
            role = (0.0, 1.0, 0.0)

        is_grip = 1.0 if meta["hold_id"] in gripping_ids else 0.0

        base = i * HOLD_OBS_DIM
        stream2[base: base + 3] = rel_pelvis
        stream2[base + 3: base + 6] = role
        stream2[base + 6] = is_grip
    stream2 = _guard(stream2, "exteroception")

    # ── Stream 3 — per-limb anchor/goal vectors to the finish hold ──────
    finish_metas = [
        m_ for m_ in world._hold_meta_by_id.values() if m_["is_finish"]
    ]
    if not finish_metas:
        # No marked finish: treat the highest hold as the finish.
        finish_metas = [
            max(world._hold_meta_by_id.values(), key=lambda mm: mm["world_pos"][2])
        ]
    # Use the highest finish hold as the canonical target for the goal vector.
    finish_meta = max(finish_metas, key=lambda mm: mm["world_pos"][2])
    finish_world = np.array(finish_meta["world_pos"], dtype=np.float64)

    stream3 = np.zeros(12, dtype=np.float64)
    for li, limb in enumerate(_LIMBS):
        if world.on_hold(limb) is not None:
            # Limb is anchored — zero vector signals "nothing to reach for".
            continue
        tip_world = world.limb_tip_pos(limb)
        stream3[li * 3: li * 3 + 3] = finish_world - tip_world
    stream3 = _guard(stream3, "goal-vectors")

    # ── Stream 4 — distance from highest gripped hand to nearest finish ─
    hand_zs = []
    for hand in ("LH", "RH"):
        hid = world.on_hold(hand)
        if hid is not None:
            hand_zs.append((world.limb_tip_pos(hand), hid))
    if hand_zs:
        highest = max(hand_zs, key=lambda hz: hz[0][2])[0]
    else:
        # Fall back to whichever hand is highest (gripped or not).
        lh = world.limb_tip_pos("LH")
        rh = world.limb_tip_pos("RH")
        highest = lh if lh[2] >= rh[2] else rh
    nearest_finish = min(
        (np.array(m_["world_pos"]) for m_ in finish_metas),
        key=lambda p: float(np.linalg.norm(p - highest)),
    )
    finish_dist = np.array(
        [float(np.linalg.norm(nearest_finish - highest))], dtype=np.float64,
    )
    stream4 = _guard(finish_dist, "finish-dist")

    obs = np.concatenate([stream1, stream2, stream3, stream4]).astype(np.float32)
    return obs
