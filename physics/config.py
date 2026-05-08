"""Physics tuning constants, all in one place.

Every magic number the simulation cares about should live here, with a
short comment explaining what it represents and which direction to
push it. Keep the rest of the package free of numeric literals.
"""
from __future__ import annotations

# ─── World ─────────────────────────────────────────────────────────────────
GRAVITY_M_S2 = 9.81           # standard gravity, m/s²
PHYS_DT = 1.0 / 240.0          # physics step. Small for stable joints.
SUBSTEPS_PER_FRAME = 8         # 8 × 1/240 ≈ 30 fps render rate
DEFAULT_DAMPING = 0.4          # global linear/angular damping. High to
                               # keep the body from oscillating wildly —
                               # this is a quasi-static climbing model,
                               # not a ragdoll. Lower (→0) for dynos.

# Unit conversion: wall + hold coordinates live in cm; pymunk runs in m.
CM_PER_M = 100.0


# ─── Default climber strength (the "average climber") ─────────────────────
# Tuneable per-climber via ClimberProfile. Numbers are loosely calibrated
# to peak forces a recreational climber can sustain for one move.
DEFAULT_MASS_KG = 70.0
DEFAULT_GRIP_FORCE_N = 600.0       # max pull force a hand can exert on a
                                    # perfect hold (jug, positivity 1.0).
                                    # Real per-hold cap is grip × positivity.
DEFAULT_FOOT_PUSH_FORCE_N = 1500.0 # legs are far stronger than arms.
DEFAULT_FRICTION_FOOT = 0.9        # shoe rubber on plastic.
DEFAULT_FRICTION_HAND = 0.7        # skin/chalk on plastic.


# ─── Joint angle limits (radians, applied via pymunk RotaryLimitJoint) ────
# Soft-ish; mostly to keep elbows/knees from hyper-extending or folding
# the wrong way. They don't need to be anatomically perfect — they need
# to keep the simulator from generating clearly impossible poses.
import math

ELBOW_MIN = math.radians(0)        # straight
ELBOW_MAX = math.radians(160)      # nearly fully bent
KNEE_MIN = math.radians(-160)      # nearly fully bent (knee bends backward)
KNEE_MAX = math.radians(0)         # straight
SHOULDER_MIN = math.radians(-180)  # very permissive — climbers reach all over
SHOULDER_MAX = math.radians(180)
HIP_MIN = math.radians(-150)
HIP_MAX = math.radians(150)


# ─── Limb segment masses (sum should equal DEFAULT_MASS_KG) ───────────────
# Distribution drawn from rough biomechanics tables. The torso carries
# most of the mass; arms are light.
TORSO_MASS_FRAC = 0.50              # head + chest + pelvis
UPPER_ARM_MASS_FRAC = 0.03
LOWER_ARM_MASS_FRAC = 0.02          # incl hand
UPPER_LEG_MASS_FRAC = 0.10
LOWER_LEG_MASS_FRAC = 0.07          # incl foot
# 0.50 + 2·(0.03+0.02) + 2·(0.10+0.07) = 1.00 ✓


# ─── Passive muscle stiffness ─────────────────────────────────────────────
# Without these, the multi-segment body chain folds under gravity (no
# joint has a "preferred angle"). DampedRotarySpring acts like a passive
# muscle holding each joint near its rest angle. The rest angle is set
# at seed time so the climber holds whatever pose the kinematic seeder
# gave them.
#
# Stiffness in N·m/rad. Tuned by feel:
#   - elbows/knees stiffer than shoulders/hips, because in real climbing
#     the elbow/knee acts more "locked" while the shoulder/hip is mobile.
#   - high damping to suppress oscillations (we're modelling quasi-static
#     climbing, not a ragdoll).
ELBOW_KNEE_STIFFNESS = 80.0
ELBOW_KNEE_DAMPING = 60.0
SHOULDER_HIP_STIFFNESS = 50.0
SHOULDER_HIP_DAMPING = 50.0

# "Core tension" — passive spring keeping the torso upright (zero rotation).
# Without this the torso spins freely on its leashes and the climber ends
# up upside down. Climbers actively keep their torso oriented; this is
# the cheapest passive analogue of that effort.
TORSO_UPRIGHT_STIFFNESS = 200.0
TORSO_UPRIGHT_DAMPING = 40.0

# "Active posture" force. Each frame we pull the torso toward its
# kinematic ideal position (the static-pose COM the solver would
# compute from the attached holds). Without this the body hangs
# passively from its leashes — a physically valid but visually
# unrealistic pose. Climbers actively stand up using their leg
# extensors; this term is the cheapest passive analogue.
#
# Gain in N/m. Tuned so the body tracks the target without overshoot
# at the simulation step rate.
POSTURE_GAIN_N_PER_M = 4000.0
POSTURE_DAMPING = 800.0


# ─── Hold attachment ──────────────────────────────────────────────────────
# A "hand on hold" is modelled as a PivotJoint between the limb tip and
# a static body at the hold position. max_force on the joint encodes
# grip strength × positivity — exceed it and the hand visibly slips off.
ATTACH_BIAS_COEF = 0.1            # how fast pymunk corrects positional
                                    # drift in the joint. Lower = looser
                                    # (hand can drift slightly).
ATTACH_MAX_FORCE_SCALE = 1.0      # multiplier on the per-hold max force.

# Default radius (cm) for collision/render of an end-effector contact.
END_EFFECTOR_RADIUS_CM = 4.0
