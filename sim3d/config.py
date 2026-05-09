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
# at build time. We pad above and to the sides only — never below — so the
# plate bottom can sit exactly at z=0 alongside the floor.
WALL_PADDING_M = 0.30
WALL_THICKNESS_M = 0.05

# Floor sits at z=0 (top surface). The MJCF plane geom's "thickness" is
# implicit; we render it as a thin slab visually. No floor penetration is
# allowed; holds must sit at z >= floor_z + small clearance.
FLOOR_Z = 0.0
HOLD_FLOOR_CLEARANCE = 0.02

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

# Anatomically motivated. Tighter than what we had — looser limits let the
# RL agent explore physically silly poses, which then makes the policy
# fragile when transferred. Numbers below are biased toward the "trained
# climber" end of the human range of motion. Refs: Norkin & White, "Joint
# Range of Motion"; ACSM exercise physiology tables.
JOINT_LIMITS_RAD = {
    # Spine — forward flex (climber tucks for high feet); we keep it
    # single-axis to avoid the under-actuated lumbar mess.
    "spine_lean":      (-15 * DEG, 60 * DEG),

    # Shoulder (3-axis Euler around hinges). Order matters: az is
    # forward/back, then el (abduction), then roll (internal/external).
    # No hyper-extension behind the back (humans clear ~50° max).
    "shoulder_az":     (-50 * DEG, 180 * DEG),
    "shoulder_el":     (  0 * DEG, 180 * DEG),
    "shoulder_roll":   (-80 * DEG,  80 * DEG),

    # Elbow — strict flexion, NO hyperextension. 150° is "fist near
    # shoulder", 0° is locked-out arm.
    "elbow":           (  0 * DEG, 150 * DEG),

    # Wrist — 1 DOF flexion/extension. Modest range is enough to wrap a
    # hold; we don't model radial/ulnar deviation.
    "wrist":           (-70 * DEG,  70 * DEG),

    # Hip flex (knee to chest), abduction (legs apart), rotation.
    # Bias hip_flex high so high-step betas are reachable.
    "hip_flex":        (-20 * DEG, 140 * DEG),
    "hip_abduct":      (-20 * DEG,  70 * DEG),
    "hip_rot":         (-40 * DEG,  40 * DEG),

    # Knee — flexion only, no hyperextension. 0° = locked-out, 150° = heel-to-butt.
    "knee":            (  0 * DEG, 150 * DEG),

    # Ankle — dorsi (toe up) / plantar (toe down).
    "ankle":           (-25 * DEG,  45 * DEG),
}

# Per-joint passive stiffness (Nm / rad) and damping (Nm·s / rad).
# These give every joint a slight "spring back to neutral" feel, which
# stops the climber from flopping into pretzels when the actuator is
# under-driving. Numbers chosen so the joints feel taut but not stiff —
# climbers DO have passive elastic torque from tendons / ligaments.
JOINT_PASSIVE = {
    "spine_lean":    (15.0, 2.0),
    "shoulder_az":   ( 5.0, 0.8),
    "shoulder_el":   ( 5.0, 0.8),
    "shoulder_roll": ( 3.0, 0.5),
    "elbow":         ( 4.0, 0.6),
    "wrist":         ( 2.0, 0.3),
    "hip_flex":      (10.0, 1.5),
    "hip_abduct":    (10.0, 1.5),
    "hip_rot":       ( 6.0, 1.0),
    "knee":          ( 8.0, 1.2),
    "ankle":         ( 4.0, 0.6),
}

# ─── Actuation ──────────────────────────────────────────────────────────────
# Each non-free joint gets a position actuator. The solver (and later RL
# policy) writes ctrl values; the actuators servo the joint to that
# angle with a stiff PD. We don't model muscle physiology — the user's
# brief said to "think about muscles" and we explicitly chose to model
# them as torque-limited PD servos rather than Hill-type muscles, which
# would 5× the simulator complexity for marginal RL benefit.
ACTUATOR_KP = 120.0
ACTUATOR_KV = 8.0

# Per-joint torque cap (Nm). Real climber peak ≈ 60–150 Nm shoulder,
# 80 Nm spine, 200 Nm hip, 150 Nm knee. We keep these generous so failure
# modes are geometric (out-of-reach), not actuator saturation.
TORQUE_CAP_NM = {
    "shoulder": 180.0,
    "elbow":    120.0,
    "wrist":     30.0,
    "spine":    120.0,
    "hip":      220.0,
    "knee":     180.0,
    "ankle":     90.0,
}

# ─── Hold attachment ────────────────────────────────────────────────────────
# When a limb is "on" a hold we activate a weld equality between the
# limb's hand/foot body and a per-limb mocap target body. The mocap is
# moved to the hold's centre on attach; on release we deactivate the
# constraint. solref/solimp control how stiff the catch is — climbers
# describe "snapping onto a hold" as a quick lock-in, so we tune for a
# fast catch with low spring-back.
HOLD_CONSTRAINT_SOLREF = (0.008, 1.0)
HOLD_CONSTRAINT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2)

# ─── Continuous-reach controller ────────────────────────────────────────────
# When a limb is mid-flight (released from one hold, reaching toward
# another), we apply a Cartesian PD on the limb tip to drive it through
# space. This is what makes the climber actually MOVE between holds
# rather than teleport. Tuned by hand: enough force to overcome gravity
# on the limb segment plus the actuator stiffness; not so much that the
# tip overshoots.
REACH_KP_HAND = 600.0      # N / m  — proportional gain pulling hand to target
REACH_KD_HAND = 60.0       # N·s / m — velocity damping
REACH_KP_FOOT = 800.0
REACH_KD_FOOT = 80.0

# Distance (m) at which the reach controller engages the weld. ~5 cm
# matches the visual hold radius — when the hand is "on" the hold.
REACH_ATTACH_RADIUS = 0.05
# Hard timeout (s) on a reach. Past this we give up and weld at the
# closest approach. Without a timeout, an unreachable target makes the
# limb dangle forever.
REACH_TIMEOUT_S = 1.5

# Dyno: explosive whole-body extension when the moving limb is too far
# for a static reach. We boost legs / hips toward extension and fly the
# limb forward.
DYNO_KP_BOOST = 2.5         # multiplier on REACH_KP for moving limb
DYNO_LEG_PUSH_NM = 100.0    # extra Nm on knee+hip during dyno
DYNO_DURATION_S = 0.35

# Slip model. We don't trust raw weld constraints to model breakaway —
# the weld is rigid until released. Instead, every step we read the
# constraint force on each active weld, and if it exceeds the hold's
# capacity (max_force_n × this slack factor) we deactivate the weld.
# Setting > 1.0 gives the climber more grip than the hold's rated
# capacity (real climbers do peak above hold-rated forces in dynos);
# setting < 1.0 makes them slip more easily.
SLIP_FORCE_SLACK = 1.25

# Friction on the bare wall surface (used for slab smearing). Holds
# carry their own per-hold friction patch.
DEFAULT_WALL_FRICTION = (0.7, 0.005, 0.001)  # (slide, spin, roll)

# When the limb's tip body collides with the wall plate (mid-flight),
# we want enough friction to enable smearing. These geom-level frictions
# are multiplied with the wall friction by MuJoCo via geometric mean.
LIMB_TIP_FRICTION = {
    "hand": (1.6, 0.01, 0.005),    # chalk + skin
    "foot": (1.4, 0.01, 0.005),    # rubber sole
}
