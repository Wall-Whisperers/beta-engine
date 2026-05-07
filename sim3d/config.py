"""Tunable constants for the 3D climbing simulator.

Anything you might want to tweak from a notebook lives here. Numbers
are SI (metres, seconds, kilograms, newtons) unless noted.
"""
from __future__ import annotations

# ─── Time integration ───────────────────────────────────────────────────────
# MuJoCo prefers small dt for stiff contacts. 2 ms keeps the implicit
# integrator stable through climb-loaded constraints; 60 Hz render
# means 1 frame = 8 sub-steps.
PHYS_DT = 0.002
RENDER_HZ = 60
SUBSTEPS_PER_FRAME = int(round(1.0 / PHYS_DT / RENDER_HZ))  # 8

# ─── Gravity / wall ─────────────────────────────────────────────────────────
GRAVITY_M_S2 = 9.81

# Wall plate dimensions are derived from the Wall (cols × rows × cell_size_cm)
# at build time. These are just safety paddings around the playable area:
WALL_PADDING_M = 0.30
WALL_THICKNESS_M = 0.05

# Floor height (metres below the wall's bottom edge). Pure cosmetic /
# safety net so a fallen climber doesn't sink to -inf.
FLOOR_DROP_M = 0.05

# ─── Hold geometry ──────────────────────────────────────────────────────────
# Holds are visualised as a small puck poking out of the wall. The
# shape doesn't drive physics (grabbing is a constraint, not a contact),
# but the size affects what the climber sees and what foot-friction
# patches look like.
HOLD_PROTRUDE_M = 0.04
HOLD_RADIUS_BY_SIZE_M = {
    "small":  0.030,
    "medium": 0.050,
    "large":  0.075,
}

# ─── Climber defaults ───────────────────────────────────────────────────────
# Average-climber dimensions. Mirror solver/body defaults so betas line
# up across 2D and 3D solvers.
DEFAULT_HEIGHT_CM = 175.0
DEFAULT_WINGSPAN_CM = 175.0
DEFAULT_MASS_KG = 70.0

# Per-segment fraction of total height (anatomy approximation).
THIGH_FRAC = 0.245
SHIN_FRAC = 0.245
FOOT_HEIGHT_FRAC = 0.04
PELVIS_TO_CHEST_FRAC = 0.30   # spine length
NECK_HEAD_FRAC = 0.18
SHOULDER_WIDTH_FRAC = 0.23
PELVIS_WIDTH_FRAC = 0.18

# Per-segment fraction of total mass (Winter, "Biomechanics & Motor
# Control of Human Movement", rounded). Numbers are total per segment,
# not per side.
MASS_FRAC = {
    "pelvis":   0.30,    # pelvis + lower torso
    "chest":    0.20,    # upper torso
    "head":     0.08,
    "upper_arm":0.028,   # ×2 for both sides
    "forearm":  0.016,   # ×2
    "hand":     0.006,   # ×2
    "thigh":    0.10,    # ×2
    "shin":     0.046,   # ×2
    "foot":     0.014,   # ×2
}
# Note: the dict above sums to ~0.94 — rounding error gets absorbed into
# the pelvis at runtime so total mass exactly matches DEFAULT_MASS_KG.

# ─── Joint limits (radians). Climbing-realistic, slightly generous.
# Reference: average human range of motion, biased loose so the
# solver/RL agent isn't fighting a clipping limit on every reach.
import math
DEG = math.pi / 180.0

JOINT_LIMITS_RAD = {
    # Spine: forward flex / side bend (single hinge, "lean")
    "spine_lean":      (-30 * DEG, 30 * DEG),
    # Shoulders — 3-axis ball decomposed into 3 hinges (azimuth, elevation, roll)
    "shoulder_az":     (-90 * DEG, 180 * DEG),    # forward/back
    "shoulder_el":     (-30 * DEG, 180 * DEG),    # abduction (arm overhead = 180)
    "shoulder_roll":   (-90 * DEG, 90 * DEG),     # internal/external rotation
    # Elbow: flexion only (no hyperextension)
    "elbow":           (0 * DEG, 150 * DEG),
    # Hips — 3-axis decomposed
    "hip_flex":        (-30 * DEG, 130 * DEG),    # bring knee to chest = +130
    "hip_abduct":      (-30 * DEG, 60 * DEG),     # leg out to side
    "hip_rot":         (-45 * DEG, 45 * DEG),
    # Knee: flexion only
    "knee":            (0 * DEG, 150 * DEG),
    # Ankle: dorsi/plantar
    "ankle":           (-30 * DEG, 50 * DEG),
}

# ─── Actuation ──────────────────────────────────────────────────────────────
# Each non-free joint gets a position actuator. The solver (and later RL
# policy) writes ctrl values; the actuators servo the joint to that
# angle with a stiff PD. We don't model muscle physiology — the user's
# brief said to "think about muscles" and we explicitly chose to model
# them as torque-limited PD servos rather than Hill-type muscles, which
# would 5x the simulator complexity for marginal RL benefit.
ACTUATOR_KP = 200.0
ACTUATOR_KV = 20.0
# Per-joint torque cap (Nm). A real climber is roughly 50-150 Nm at the
# shoulder, 80 Nm at the hip; values below are intentionally generous so
# unsolvable poses fail because of geometry, not because the actuator
# saturated. Tune later when sim2real matters.
TORQUE_CAP_NM = {
    "shoulder": 150.0,
    "elbow":    100.0,
    "spine":     80.0,
    "hip":      200.0,
    "knee":     150.0,
    "ankle":     80.0,
}

# ─── Hold attachment ────────────────────────────────────────────────────────
# When a limb is "on" a hold we activate an equality/connect between
# the limb's tip site and a per-limb mocap target body. The mocap is
# moved to the hold's centre on attach; on release we just deactivate
# the constraint. solref/solimp control how stiff the catch is.
HOLD_CONSTRAINT_SOLREF = (0.01, 1.0)   # (timeconst, dampratio)
HOLD_CONSTRAINT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)

# Friction on the bare wall surface (used for slab smearing). Holds
# carry their own per-hold friction patch.
DEFAULT_WALL_FRICTION = (0.7, 0.005, 0.001)  # (slide, spin, roll)
