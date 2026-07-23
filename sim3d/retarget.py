"""SMPL → 29-DOF MuJoCo retargeting pipeline.

Converts per-frame SMPL body pose parameters (from 4D-Humans / WHAM or similar)
into a Reference-compatible qpos/qvel sequence for the beta-engine climber.

Pipeline:
  1. Extract SMPL .npz from Colab (4D-Humans / WHAM on spike1 clips)
  2. python -m sim3d.retarget <smpl.npz> <out.npz> [--wall <wall.json>]
  3. Inspect with python -m sim3d --play data/runs/sim3d/imitation/<out.npz>
     (after Reference.load wraps the qpos/qvel)

SMPL joint indices (SMPL-H / SMPL-X body, 24 joints, local axis-angle):
  0 pelvis  1 L_hip   2 R_hip   3 spine1  4 L_knee   5 R_knee
  6 spine2  7 L_ankle 8 R_ankle 9 spine3 10 L_foot  11 R_foot
 12 neck   13 L_collar 14 R_collar 15 head  16 L_shoulder 17 R_shoulder
 18 L_elbow 19 R_elbow 20 L_wrist  21 R_wrist

Our 29-DOF qpos layout (qpos[0:7] = free joint, qpos[7:30] = actuated):
  7  spine_lean   (X hinge, +ve = forward toward wall)
  8  spine_lat    (Y hinge, +ve = lean left)
  9  spine_twist  (Z hinge, +ve = rotate right)
 10  l_shoulder_az  (Y hinge)   11 l_shoulder_el  (X)  12 l_shoulder_roll (Z)
 13  l_elbow (X)  14 l_wrist (X)
 15  r_shoulder_az  (−Y hinge)  16 r_shoulder_el  (X)  17 r_shoulder_roll (Z)
 18  r_elbow (X)  19 r_wrist (X)
 20  l_hip_flex (−X)  21 l_hip_abduct (Y)  22 l_hip_rot (Z)
 23  l_knee (X)  24 l_ankle (X)
 25  r_hip_flex (−X)  26 r_hip_abduct (Y)  27 r_hip_rot (Z)
 28  r_knee (X)  29 r_ankle (X)

Coordinate conventions:
  SMPL canonical:   +X right, +Y up, +Z toward camera (right-hand)
  MuJoCo world:     +X along wall left→right, +Y away from wall, +Z up
  Our climber faces the wall: pelvis −Y toward wall, +Z up.

The SMPL→MuJoCo frame transform is:
  R_smpl2mj = Rz(−90°) · Rx(−90°)   (rotate Y-up into Z-up, face wall)
  (validated against seed-pose qpos visually — adjust if needed)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ─── SMPL joint index constants ──────────────────────────────────────────────

SMPL_PELVIS      = 0
SMPL_L_HIP       = 1;  SMPL_R_HIP      = 2
SMPL_SPINE1      = 3
SMPL_L_KNEE      = 4;  SMPL_R_KNEE     = 5
SMPL_SPINE2      = 6
SMPL_L_ANKLE     = 7;  SMPL_R_ANKLE    = 8
SMPL_SPINE3      = 9
SMPL_L_FOOT      = 10; SMPL_R_FOOT     = 11
SMPL_NECK        = 12
SMPL_L_COLLAR    = 13; SMPL_R_COLLAR   = 14
SMPL_HEAD        = 15
SMPL_L_SHOULDER  = 16; SMPL_R_SHOULDER = 17
SMPL_L_ELBOW     = 18; SMPL_R_ELBOW    = 19
SMPL_L_WRIST     = 20; SMPL_R_WRIST    = 21

N_SMPL_JOINTS = 22  # body-only (no hand joints)
N_ACT = 23          # our actuated joints


# ─── Rotation helpers ─────────────────────────────────────────────────────────

def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle (3,) → rotation matrix (3,3)."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → axis-angle (3,)."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(theta) < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def _project_angle(R: np.ndarray, axis: np.ndarray) -> float:
    """Extract the scalar rotation angle about `axis` from matrix R.

    Uses the component of the axis-angle vector along `axis`. For hinge joints
    this is exact when the motion is truly 1-DOF; for ball joints it is the
    projection of the full rotation onto the hinge axis — a reasonable first-
    order approximation that the physics IK will then refine.
    """
    aa = _rotmat_to_aa(R)
    return float(np.dot(aa, axis / (np.linalg.norm(axis) + 1e-12)))


# SMPL-world → MuJoCo-world orientation (validated against seed pose visually).
# SMPL: Y-up, Z toward viewer.  MuJoCo: Z-up, Y away from wall (climber faces −Y).
# Transform: swap axes so SMPL-Y→MJ-Z, SMPL-Z→MJ-Y, SMPL-X→MJ-X, then mirror Y.
_SMPL_TO_MJ = np.array([
    [ 1,  0,  0],   # MJ-X  ←  SMPL-X  (both point right along wall)
    [ 0,  0,  1],   # MJ-Y  ←  SMPL-Z  (toward camera → away from wall)
    [ 0,  1,  0],   # MJ-Z  ←  SMPL-Y  (up)
], dtype=float)


def _smpl_pose_to_rotmats(pose: np.ndarray) -> np.ndarray:
    """(72,) or (N_joints*3,) SMPL pose → (N_joints, 3, 3) local rotation matrices."""
    n = len(pose) // 3
    return np.stack([_aa_to_rotmat(pose[3*i:3*i+3]) for i in range(n)])


# ─── Per-joint extraction ─────────────────────────────────────────────────────

# MuJoCo hinge axes for each of our 23 actuated joints (in local body frame).
# Sign matches the builder.py axis definitions.
_HINGE_AXES = {
    "spine_lean":     np.array([ 1, 0, 0]),  # forward lean
    "spine_lat":      np.array([ 0, 1, 0]),  # lateral
    "spine_twist":    np.array([ 0, 0, 1]),  # axial
    "l_shoulder_az":  np.array([ 0, 1, 0]),
    "l_shoulder_el":  np.array([ 1, 0, 0]),
    "l_shoulder_roll":np.array([ 0, 0,-1]),  # mirrored vs r_shoulder_roll's
                                              # [0,0,1] — same left/right mirroring
                                              # (measured: l always negative-saturated
                                              # while r tracked 0..80 cleanly)
    "l_elbow":        np.array([ 1, 0, 0]),
    "l_wrist":        np.array([ 1, 0, 0]),
    "r_shoulder_az":  np.array([ 0,-1, 0]),  # mirrored for right
    "r_shoulder_el":  np.array([ 1, 0, 0]),
    "r_shoulder_roll":np.array([ 0, 0, 1]),
    "r_elbow":        np.array([-1, 0, 0]),  # mirrored vs l_elbow's [1,0,0] — same
                                              # left/right mirroring (measured across
                                              # all 5 spike1 clips: r_elbow mean always
                                              # -5..-13 deg vs l_elbow's +9..+16 deg for
                                              # comparable arm motion)
    "r_wrist":        np.array([ 1, 0, 0]),
    "l_hip_flex":     np.array([ 1, 0, 0]),  # mirrored vs builder's "-1 0 0":
                                              # SMPL's left-hip local frame reads
                                              # flexion with opposite sign from the
                                              # right hip (measured: l_hip_flex was
                                              # always negative while r_hip_flex was
                                              # always positive for the same physical
                                              # motion — see sim3d/retarget.py probe)
    "l_hip_abduct":   np.array([ 0, 1, 0]),
    "l_hip_rot":      np.array([ 0, 0, 1]),
    "l_knee":         np.array([-1, 0, 0]),  # mirrored vs r_knee's [1,0,0] — same
                                              # left/right SMPL local-frame mirroring
    "l_ankle":        np.array([ 1, 0, 0]),
    "r_hip_flex":     np.array([-1, 0, 0]),
    "r_hip_abduct":   np.array([ 0, 1, 0]),
    "r_hip_rot":      np.array([ 0, 0, 1]),
    "r_knee":         np.array([ 1, 0, 0]),
    "r_ankle":        np.array([ 1, 0, 0]),
}

# qpos index for each joint name (qpos[7:] ordering from the live model dump).
JOINT_QPOS_IDX = {
    "spine_lean": 7, "spine_lat": 8, "spine_twist": 9,
    "l_shoulder_az": 10, "l_shoulder_el": 11, "l_shoulder_roll": 12,
    "l_elbow": 13, "l_wrist": 14,
    "r_shoulder_az": 15, "r_shoulder_el": 16, "r_shoulder_roll": 17,
    "r_elbow": 18, "r_wrist": 19,
    "l_hip_flex": 20, "l_hip_abduct": 21, "l_hip_rot": 22,
    "l_knee": 23, "l_ankle": 24,
    "r_hip_flex": 25, "r_hip_abduct": 26, "r_hip_rot": 27,
    "r_knee": 28, "r_ankle": 29,
}


def _smpl_rotmats_to_qpos(Rs: np.ndarray) -> np.ndarray:
    """(N_smpl_joints, 3, 3) SMPL local rotmats → qpos[7:30] (23,).

    All SMPL rotations are expressed in their local parent frame. We convert
    them into MuJoCo-world frame first, then project onto each hinge axis.

    This is an analytic first-pass. Run IK refinement afterwards to correct
    residual mismatches (joint limits, multi-DOF ball-joint approximations).
    """
    # Reframe the SMPL rotation matrices into our coordinate system.
    S = _SMPL_TO_MJ
    St = S.T

    def reframe(R):
        return S @ R @ St

    qpos = np.zeros(30)  # full qpos, qpos[7:] will be filled

    # Spine: compose spine2+spine3 local rotations in MJ frame (SMPL_SPINE1
    # deliberately excluded — measured separately across all 5 spike1 clips,
    # spine1's axis-angle reading carries a large, almost entirely-negative
    # structural offset (mean -45..-85 deg, never crossing positive) that is
    # NOT real torso motion: spine2 and spine3 alone are well-behaved
    # (small, physically plausible values) on every clip, while adding
    # spine1 saturated spine_lean at our +-15 deg limit on 79%+ of frames.
    # This points to spine1's local rest-pose frame using a different bone
    # axis convention than spine2/spine3 in SMPL's rig — not something our
    # simple reframe-and-project corrects for. Dropping it loses some real
    # lower-back motion but produces plausible, non-saturating torso angles;
    # revisit if a proper per-joint rest-pose correction is derived later.
    R_spine = reframe(Rs[SMPL_SPINE2]) @ reframe(Rs[SMPL_SPINE3])
    qpos[7] = _project_angle(R_spine, _HINGE_AXES["spine_lean"])
    qpos[8] = _project_angle(R_spine, _HINGE_AXES["spine_lat"])
    qpos[9] = _project_angle(R_spine, _HINGE_AXES["spine_twist"])

    # Left arm.
    R_ls = reframe(Rs[SMPL_L_COLLAR]) @ reframe(Rs[SMPL_L_SHOULDER])
    qpos[10] = _project_angle(R_ls, _HINGE_AXES["l_shoulder_az"])
    qpos[11] = _project_angle(R_ls, _HINGE_AXES["l_shoulder_el"])
    qpos[12] = _project_angle(R_ls, _HINGE_AXES["l_shoulder_roll"])
    qpos[13] = _project_angle(reframe(Rs[SMPL_L_ELBOW]), _HINGE_AXES["l_elbow"])
    qpos[14] = _project_angle(reframe(Rs[SMPL_L_WRIST]),  _HINGE_AXES["l_wrist"])

    # Right arm.
    R_rs = reframe(Rs[SMPL_R_COLLAR]) @ reframe(Rs[SMPL_R_SHOULDER])
    qpos[15] = _project_angle(R_rs, _HINGE_AXES["r_shoulder_az"])
    qpos[16] = _project_angle(R_rs, _HINGE_AXES["r_shoulder_el"])
    qpos[17] = _project_angle(R_rs, _HINGE_AXES["r_shoulder_roll"])
    qpos[18] = _project_angle(reframe(Rs[SMPL_R_ELBOW]), _HINGE_AXES["r_elbow"])
    qpos[19] = _project_angle(reframe(Rs[SMPL_R_WRIST]),  _HINGE_AXES["r_wrist"])

    # Left leg.
    R_lh = reframe(Rs[SMPL_L_HIP])
    qpos[20] = _project_angle(R_lh, _HINGE_AXES["l_hip_flex"])
    qpos[21] = _project_angle(R_lh, _HINGE_AXES["l_hip_abduct"])
    qpos[22] = _project_angle(R_lh, _HINGE_AXES["l_hip_rot"])
    qpos[23] = _project_angle(reframe(Rs[SMPL_L_KNEE]),   _HINGE_AXES["l_knee"])
    qpos[24] = _project_angle(reframe(Rs[SMPL_L_ANKLE]),  _HINGE_AXES["l_ankle"])

    # Right leg.
    R_rh = reframe(Rs[SMPL_R_HIP])
    qpos[25] = _project_angle(R_rh, _HINGE_AXES["r_hip_flex"])
    qpos[26] = _project_angle(R_rh, _HINGE_AXES["r_hip_abduct"])
    qpos[27] = _project_angle(R_rh, _HINGE_AXES["r_hip_rot"])
    qpos[28] = _project_angle(reframe(Rs[SMPL_R_KNEE]),   _HINGE_AXES["r_knee"])
    qpos[29] = _project_angle(reframe(Rs[SMPL_R_ANKLE]),  _HINGE_AXES["r_ankle"])

    return qpos[7:]  # return only actuated slice


# ─── Joint-limit clipping ─────────────────────────────────────────────────────

# From CLAUDE.md / config.py (radians).
_LIMITS_DEG = {
    "spine_lean":      (-15,  15), "spine_lat":      (-25,  25),
    "spine_twist":     (-30,  30),
    "l_shoulder_az":   (-50, 180), "l_shoulder_el":  (  0, 180),
    "l_shoulder_roll": (-80,  80), "l_elbow":        (  0, 150), "l_wrist": (-70, 70),
    "r_shoulder_az":   (-50, 180), "r_shoulder_el":  (  0, 180),
    "r_shoulder_roll": (-80,  80), "r_elbow":        (  0, 150), "r_wrist": (-70, 70),
    "l_hip_flex": (-20, 140), "l_hip_abduct": (-30, 85), "l_hip_rot": (-60, 60),
    "l_knee": (0, 150), "l_ankle": (-25, 45),
    "r_hip_flex": (-20, 140), "r_hip_abduct": (-30, 85), "r_hip_rot": (-60, 60),
    "r_knee": (0, 150), "r_ankle": (-25, 45),
}
_LIMITS_RAD = {k: (np.deg2rad(lo), np.deg2rad(hi))
               for k, (lo, hi) in _LIMITS_DEG.items()}

_JOINT_ORDER = list(JOINT_QPOS_IDX.keys())  # matches qpos[7:30] order


def clip_to_limits(act_qpos: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip a (23,) actuated-joint array to anatomical limits.

    Returns (clipped, n_violations) where n_violations counts how many joints
    were out of range (used for quality filtering).
    """
    out = act_qpos.copy()
    n_viol = 0
    for i, name in enumerate(_JOINT_ORDER):
        lo, hi = _LIMITS_RAD[name]
        if out[i] < lo or out[i] > hi:
            n_viol += 1
            out[i] = np.clip(out[i], lo, hi)
    return out, n_viol


# ─── Root-position extraction ─────────────────────────────────────────────────

def _smpl_trans_to_pelvis(
        trans: np.ndarray,     # (3,) SMPL global translation (metres)
        global_orient: np.ndarray,  # (3,) SMPL global orientation axis-angle
        wall_dist: float = 0.40,    # initial pelvis-to-wall distance (m)
) -> np.ndarray:
    """Convert SMPL global translation to our pelvis position (qpos[0:3]).

    SMPL translation is in SMPL world frame (Y-up, Z toward camera). We
    convert to MuJoCo frame and offset so the first frame's wall-normal
    component aligns to wall_dist.

    Returns pelvis xyz in MuJoCo world frame.
    """
    # Reframe: SMPL-Y → MJ-Z, SMPL-Z → MJ-Y, SMPL-X → MJ-X
    t_mj = _SMPL_TO_MJ @ trans
    # The SMPL translation origin is the mean body centre at floor level.
    # We'll anchor the first frame to a wall position later in retarget_clip.
    return t_mj


# ─── Global orientation → pelvis quaternion ──────────────────────────────────

def _smpl_orient_to_quat(global_orient: np.ndarray) -> np.ndarray:
    """SMPL global_orient axis-angle (3,) → MuJoCo quaternion (wxyz, 4,)."""
    R_smpl = _aa_to_rotmat(global_orient)
    R_mj = _SMPL_TO_MJ @ R_smpl @ _SMPL_TO_MJ.T
    # Convert rotation matrix to quaternion (MuJoCo uses wxyz).
    # Using Shepperd's method.
    trace = R_mj[0, 0] + R_mj[1, 1] + R_mj[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_mj[2, 1] - R_mj[1, 2]) * s
        y = (R_mj[0, 2] - R_mj[2, 0]) * s
        z = (R_mj[1, 0] - R_mj[0, 1]) * s
    else:
        i = np.argmax(np.diag(R_mj))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R_mj[0, 0] - R_mj[1, 1] - R_mj[2, 2])
            w = (R_mj[2, 1] - R_mj[1, 2]) / s
            x = 0.25 * s
            y = (R_mj[0, 1] + R_mj[1, 0]) / s
            z = (R_mj[0, 2] + R_mj[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R_mj[1, 1] - R_mj[0, 0] - R_mj[2, 2])
            w = (R_mj[0, 2] - R_mj[2, 0]) / s
            x = (R_mj[0, 1] + R_mj[1, 0]) / s
            y = 0.25 * s
            z = (R_mj[1, 2] + R_mj[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_mj[2, 2] - R_mj[0, 0] - R_mj[1, 1])
            w = (R_mj[1, 0] - R_mj[0, 1]) / s
            x = (R_mj[0, 2] + R_mj[2, 0]) / s
            y = (R_mj[1, 2] + R_mj[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ─── Main retargeting function ────────────────────────────────────────────────

def retarget_clip(
        smpl_poses: np.ndarray,       # (T, 72) or (T, N_joints*3) axis-angle
        smpl_trans: np.ndarray,       # (T, 3) global translation
        smpl_betas: Optional[np.ndarray] = None,  # (10,) shape params (unused for now)
        fps: float = 30.0,
        wall_x: float = 0.0,          # wall X offset (hold cluster centre)
        wall_y_pelvis: float = 0.40,  # initial pelvis-to-wall distance
        wall_z_floor: float = 1.0,    # pelvis height in first frame
        max_violations: int = 5,      # reject frames with more out-of-range joints
) -> dict:
    """Retarget an SMPL clip to our 29-DOF MuJoCo model.

    Args:
        smpl_poses: (T, 72) SMPL body_pose in axis-angle format.
                    Can also be (T, 24*3) if global_orient is concatenated at [0:3].
        smpl_trans: (T, 3) global translation in metres (SMPL frame).
        smpl_betas: (10,) shape coefficients (reserved for body-shape adaptation).
        fps: clip frame rate (30 Hz typical for WHAM/4D-Humans output).
        wall_x / wall_y_pelvis / wall_z_floor: anchor the first frame to this
            wall-facing position in MuJoCo world.
        max_violations: quality filter — skip clips where any frame has more
            than this many joints out of anatomical range.

    Returns dict with keys:
        qpos: (T, 30) full qpos including free joint (7) + actuated (23)
        qvel: (T, 29) finite-difference velocities (same shape as MuJoCo nv)
        n_violations: per-frame joint violation counts (T,)
        dt: control timestep (1/fps)
        ok: True if quality filter passed
    """
    T = len(smpl_poses)
    dt = 1.0 / fps
    qpos_seq = np.zeros((T, 30))
    n_viol = np.zeros(T, dtype=int)

    # Compute anchor offset: align first frame to wall position.
    t0_mj = _SMPL_TO_MJ @ smpl_trans[0]
    anchor_offset = np.array([wall_x - t0_mj[0],
                              wall_y_pelvis - t0_mj[1],
                              wall_z_floor - t0_mj[2]])

    for t in range(T):
        pose = smpl_poses[t]
        # global_orient is joints[0] (first 3 floats), body_pose is joints[1:]
        global_orient = pose[:3]
        body_pose = pose[3:]  # (69,) = 23 joints * 3

        # Pelvis position.
        t_mj = _SMPL_TO_MJ @ smpl_trans[t] + anchor_offset
        qpos_seq[t, 0:3] = t_mj

        # Pelvis quaternion (wxyz).
        qpos_seq[t, 3:7] = _smpl_orient_to_quat(global_orient)

        # Actuated joints.
        Rs = _smpl_pose_to_rotmats(body_pose)  # (23, 3, 3) local rotmats
        act_raw = _smpl_rotmats_to_qpos(Rs)    # (23,)
        act_clipped, viol = clip_to_limits(act_raw)
        qpos_seq[t, 7:] = act_clipped
        n_viol[t] = viol

    # Finite-difference velocities (central differences, edge = forward/backward).
    qvel_seq = np.zeros((T, 29))
    for t in range(T):
        t_prev = max(0, t - 1)
        t_next = min(T - 1, t + 1)
        dq = qpos_seq[t_next] - qpos_seq[t_prev]
        dq[3:7] = np.zeros(4)  # quaternion diff needs special handling — zero for now
        qvel_seq[t, :3] = dq[:3] / (2 * dt if t > 0 and t < T - 1 else dt)
        qvel_seq[t, 6:] = dq[7:] / (2 * dt if t > 0 and t < T - 1 else dt)

    ok = bool(np.all(n_viol <= max_violations))
    return {
        "qpos": qpos_seq,
        "qvel": qvel_seq,
        "n_violations": n_viol,
        "dt": dt,
        "ok": ok,
    }


# ─── Contact detection ────────────────────────────────────────────────────────

def detect_contacts(
        smpl_joints_3d: np.ndarray,  # (T, N_joints, 3) in MuJoCo world frame
        hold_positions: np.ndarray,  # (H, 3) hold positions
        threshold_m: float = 0.12,   # grip proximity (m) — wider than 8 cm for video
) -> dict:
    """Label frames where hands/feet are near holds.

    Uses the wrist positions (joints 20, 21) as hand proxies and ankle
    positions (joints 7, 8) as foot proxies.

    Returns dict:
        grips: (T, 4) bool array [LH, RH, LF, RF] — True if near a hold
        grip_hold_idx: (T, 4) int array — index into hold_positions (-1 = no grip)
    """
    T = len(smpl_joints_3d)
    limb_smpl_idx = {
        "LH": SMPL_L_WRIST, "RH": SMPL_R_WRIST,
        "LF": SMPL_L_ANKLE, "RF": SMPL_R_ANKLE,
    }
    grips = np.zeros((T, 4), dtype=bool)
    grip_hold_idx = np.full((T, 4), -1, dtype=int)

    for li, limb in enumerate(["LH", "RH", "LF", "RF"]):
        ji = limb_smpl_idx[limb]
        tip = smpl_joints_3d[:, ji, :]  # (T, 3)
        for t in range(T):
            dists = np.linalg.norm(hold_positions - tip[t], axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < threshold_m:
                grips[t, li] = True
                grip_hold_idx[t, li] = nearest

    return {"grips": grips, "grip_hold_idx": grip_hold_idx}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _load_smpl_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load SMPL .npz from 4D-Humans / WHAM output.

    Supported key layouts:
      4D-Humans: 'body_pose' (T,69), 'global_orient' (T,3), 'transl' (T,3)
      WHAM:      'pose' (T,72), 'trans' (T,3)
      Generic:   'smpl_poses' (T,72), 'smpl_trans' (T,3)
    """
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())

    if "body_pose" in keys and "global_orient" in keys:
        pose = np.concatenate([d["global_orient"], d["body_pose"]], axis=1)
        trans = d["transl"] if "transl" in keys else np.zeros((len(pose), 3))
    elif "pose" in keys:
        pose = d["pose"]
        trans = d["trans"] if "trans" in keys else np.zeros((len(pose), 3))
    elif "smpl_poses" in keys:
        pose = d["smpl_poses"]
        trans = d["smpl_trans"] if "smpl_trans" in keys else np.zeros((len(pose), 3))
    else:
        raise ValueError(f"Unrecognised SMPL .npz layout. Keys: {keys}")

    return pose.astype(np.float32), trans.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Retarget an SMPL clip to the 29-DOF MuJoCo climber.")
    ap.add_argument("smpl_npz", help="Input SMPL .npz (from 4D-Humans / WHAM)")
    ap.add_argument("out_npz",  help="Output retargeted .npz")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--wall-x",       type=float, default=0.0,
                    help="Wall X offset for first-frame anchor (m)")
    ap.add_argument("--wall-y",       type=float, default=0.40,
                    help="Initial pelvis-to-wall distance (m)")
    ap.add_argument("--wall-z",       type=float, default=1.0,
                    help="Pelvis height at first frame (m)")
    ap.add_argument("--max-violations", type=int, default=5,
                    help="Max joint-limit violations per frame for quality pass")
    ap.add_argument("--visualise", action="store_true",
                    help="Launch MuJoCo viewer to inspect the retargeted clip")
    args = ap.parse_args()

    print(f"Loading {args.smpl_npz} ...")
    poses, trans = _load_smpl_npz(args.smpl_npz)
    print(f"  {len(poses)} frames @ {args.fps} fps  ({len(poses)/args.fps:.1f} s)")

    print("Retargeting ...")
    result = retarget_clip(
        poses, trans, fps=args.fps,
        wall_x=args.wall_x,
        wall_y_pelvis=args.wall_y,
        wall_z_floor=args.wall_z,
        max_violations=args.max_violations,
    )

    ok = result["ok"]
    vmax = int(result["n_violations"].max())
    vmean = float(result["n_violations"].mean())
    print(f"  Quality: {'PASS' if ok else 'FAIL'}  "
          f"(max violations/frame={vmax}, mean={vmean:.1f})")

    out = Path(args.out_npz)
    np.savez(out,
             qpos=result["qpos"],
             qvel=result["qvel"],
             n_violations=result["n_violations"],
             fps=args.fps)
    print(f"Saved → {out}")

    if not ok:
        print(f"WARNING: {(result['n_violations'] > args.max_violations).sum()} frames "
              f"exceeded the joint-violation threshold. Inspect before training.")

    if args.visualise:
        _visualise(result["qpos"], args.fps)


def _visualise(qpos_seq: np.ndarray, fps: float):
    """Play back the retargeted clip in the MuJoCo viewer (requires display)."""
    import time
    from sim3d.probe_transitions import build_wall_and_moves
    from sim3d.builder import build_mjcf_xml
    import mujoco
    import mujoco.viewer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wall, profile, _ = build_wall_and_moves(seed=11)
    xml, _ = build_mjcf_xml(wall, profile)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dt = 1.0 / fps

    with mujoco.viewer.launch_passive(m, d) as v:
        for qpos in qpos_seq:
            d.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(dt)
        # Hold last frame.
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
