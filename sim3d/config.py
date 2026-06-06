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
DEFAULT_WINGSPAN_CM = 181.0   # 5'9" male: fingertip-to-fingertip
DEFAULT_MASS_KG = 70.0

# Per-segment fraction of total height — derived from measured joint heights
# for a 175 cm male: ankle=8 cm, knee=51 cm, hip=94 cm, shoulder=145 cm.
THIGH_FRAC = 0.246            # hip–knee: 43/175
SHIN_FRAC = 0.246             # knee–ankle: 43/175
FOOT_HEIGHT_FRAC = 0.046      # ankle joint from ground: 8/175
PELVIS_TO_CHEST_FRAC = 0.291  # hip–shoulder: 51/175
NECK_HEAD_FRAC = 0.171        # shoulder–crown: 30/175
SHOULDER_WIDTH_FRAC = 0.269   # biacromial: 47/175
PELVIS_WIDTH_FRAC = 0.160     # hip-joint separation ≈28 cm: 28/175
HEAD_RADIUS_FRAC = 0.052      # from head circumference 57 cm → r=9.1 cm: 9.1/175
NECK_RADIUS_M    = 0.059      # from neck circumference 37 cm → r=37/(2π)/100

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
    # single-axis to avoid the under-actuated lumbar mess. Upper bound
    # tightened from 30° → 15° because the upper-body gravity torque
    # (~60 Nm at rest) used to pin the joint at +28° under the old
    # actuator gains, which threw the head through the wall on overhangs.
    # With the per-group spine gains in ACTUATOR_GAINS_BY_GROUP the
    # equilibrium is now near 0° so this cap is a safety bound, not the
    # operating point.
    "spine_lean":      (-15 * DEG, 15 * DEG),

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
# Passive (stiffness Nm/rad, damping Nm·s/rad).
# Upper body (shoulders, elbow, wrist): unchanged — arms are actively loaded
# against the wall and should feel taut.
# Lower body and spine: stiffness reduced so gravity can sag the body into a
# natural hang; damping raised to overdamped so the settle phase converges
# without ringing (ζ = d / (2√(k·I)) > 1 for each joint).
JOINT_PASSIVE = {
    "spine_lean":    (40.0, 8.0),   # raised: was (6, 5). Keeps the spine near 0° under upper-body gravity instead of pinned at the limit.
    "shoulder_az":   ( 5.0, 0.8),
    "shoulder_el":   ( 5.0, 0.8),
    "shoulder_roll": ( 3.0, 0.5),
    "elbow":         ( 4.0, 0.6),
    "wrist":         ( 2.0, 0.3),
    "hip_flex":      ( 3.0, 3.5),   # was (10.0, 1.5) — hips sag freely
    "hip_abduct":    ( 3.0, 3.5),   # was (10.0, 1.5)
    "hip_rot":       ( 2.0, 2.0),   # was (6.0, 1.0)
    "knee":          ( 2.0, 3.0),   # was (8.0, 1.2) — knees bend freely
    "ankle":         ( 2.0, 1.5),   # was (4.0, 0.6)
}

# ─── Actuation ──────────────────────────────────────────────────────────────
# Each non-free joint gets a position actuator. The solver (and later RL
# policy) writes ctrl values; the actuators servo the joint to that
# angle with a stiff PD. We don't model muscle physiology — the user's
# brief said to "think about muscles" and we explicitly chose to model
# them as torque-limited PD servos rather than Hill-type muscles, which
# would 5× the simulator complexity for marginal RL benefit.
#
# Gains are *per joint group*, sized so each group is near-critically damped
# under the segment it carries. Picked from ζ = kv / (2·√(kp·I)) with
# segment inertia approximations for the default 70 kg / 175 cm climber.
# A flat (kp=120, kv=8) made hips and shoulders ring (ζ≈0.3–0.6), which is
# the "twisting weirdly" failure mode you see under random actions.
ACTUATOR_GAINS_BY_GROUP: dict[str, tuple[float, float]] = {
    # group:    (kp,   kv)
    "shoulder": (220.0, 20.0),   # ζ ≈ 1.0 for ~3 kg arm at 0.35 m
    "elbow":    ( 80.0,  6.0),
    "wrist":    ( 15.0,  1.0),
    "spine":    (250.0, 22.0),   # carries upper body — must be stiff
    "hip":      (450.0, 45.0),   # ζ ≈ 1.0 for ~6 kg leg at 0.45 m
    "knee":     (200.0, 17.0),
    "ankle":    ( 40.0,  3.0),
}
# Legacy uniform defaults kept for backwards compat / debug experiments.
ACTUATOR_KP = 120.0
ACTUATOR_KV = 8.0

# Per-joint armature (reflected motor inertia, kg·m²). Higher armature
# damps high-frequency joint chatter without affecting steady-state pose.
# A flat 0.01 is the MuJoCo default for tiny robots; for a 70 kg climber
# the load-bearing joints need an order of magnitude more.
JOINT_ARMATURE_BY_GROUP: dict[str, float] = {
    "shoulder": 0.05,
    "elbow":    0.02,
    "wrist":    0.005,
    "spine":    0.08,
    "hip":      0.10,
    "knee":     0.04,
    "ankle":    0.01,
}

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
# ─── Pose seeding ───────────────────────────────────────────────────────────
# Ramp: gravity rises from 0→full over SEED_RAMP_S so the welds aren't hit
# by the body's full weight in one step. Hold: free-physics settle with
# actuators disabled. Must be ≥ 2–3× the longest joint period (≈ 2 s at the
# new lower stiffness values) to guarantee overdamped convergence.
SEED_RAMP_S = 0.5
SEED_HOLD_S = 2.0

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

# ─── Continuous-joint grip control ──────────────────────────────────────────
# In continuous-joint mode the agent emits a per-limb grip intent in addition
# to the joint targets. A grip can only engage when the limb tip is within
# GRIP_PROXIMITY_M of an unoccupied valid hold. There is no auto-grip — the
# agent must explicitly raise the intent above 0.0.
# Raised 0.05 → 0.08 (2026-06-05): the reach-one curriculum showed the agent
# learning to reach toward a target hold but rarely landing inside a 5 cm radius
# to complete the grip — near-misses never converted to grabs. 8 cm turns those
# learned near-reaches into successful regrips. (Long-flagged in the post-mortem.)
GRIP_PROXIMITY_M = 0.08

# Feet push on overhangs; hands pull. Per-limb capacity multiplier so feet can
# generate more reaction force than the hands' rated grip strength.
# Raised 1.5 → 3.0 (2026-06-05) alongside HAND_FORCE_MULTIPLIER: when a hand
# releases for a move, its load redistributes onto the anchor hand AND both
# feet, so the feet need headroom too. With foot 1.5 the feet slipped during the
# transition and the stance still collapsed; 3.0 holds a one-hand stance solidly
# (verified: anchors survive 40/40 steps through a hand release, even under
# joint jitter). Same training-phase rationale as HAND_FORCE_MULTIPLIER.
FOOT_FORCE_MULTIPLIER = 3.0

# Hand grip-strength multiplier (2026-06-05). Raised from an implicit 1.0 after
# a decisive finding: with the default grip cap the body CANNOT hold a one-hand
# stance — releasing either hand collapses the (already ~1.3× over-braced) grips
# and the climber drops. That is the physical reason every run learned to cling
# and never climb: with weak hands, "let go and reach" is a losing move. 2.5×
# lets the body bear the load transfer a hand move requires (verified: anchors
# survive the release, the reach reward fires). This is a training-phase choice
# — the roadmap defers realistic grip force to the torque/muscle phase; tighten
# it back toward 1.0 once the agent reliably climbs.
HAND_FORCE_MULTIPLIER = 2.5

# ─── Kickboard ─────────────────────────────────────────────────────────────
# A "kickboard" is a secondary near-vertical plate below the main wall with a
# small handful of foot-only holds. MoonBoard uses this for the canonical
# starting foot position. Sizes are SI metres; foothold positions are in
# kickboard-local (X-along, Z-up) coordinates centred on the kickboard.
KICKBOARD_WIDTH_M = 1.30
KICKBOARD_HEIGHT_M = 0.30
KICKBOARD_THICKNESS_M = 0.04
# Vertical gap between top of kickboard and bottom of main wall.
KICKBOARD_GAP_M = 0.02
# Approx near-vertical (slight backward tilt so feet press into it).
KICKBOARD_ANGLE_DEG = -5.0
# Two upper foot holds, narrowed inward to match the climber's hip range.
KICKBOARD_FOOTHOLDS = [
    ("KB_LU", -0.244, 0.20),
    ("KB_RU",  0.244, 0.20),
]

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
