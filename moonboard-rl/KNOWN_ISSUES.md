# Known Issues — Pre-RL Audit

Issues documented after the Section 7 audit. Do not fix yet — these are Week 2/3
work items.

---

## ISSUE 1: Foot routing during climb

Feet currently stay on kickboard holds for the entire episode. A real climber
moves their feet to higher holds as they climb; keeping feet pinned at z=0.27 m
while the body rises creates an increasingly unrealistic posture and limits how
high the agent can climb before the body geometry becomes invalid.

The Week 2 RSI (Reference State Initialization) will include explicit foot
positions per climbing stance. The foot target sequencer in
`_advance_foot_target()` already has the two-step-lag logic ready; it just
needs to activate when the hand pointer crosses a row threshold.

---

## ISSUE 2: Heel and toe hooks

On overhanging terrain, heel hooks (foot hooked over a hold from above) and
toe hooks (toe pressed into a hold from below) are common advanced techniques.
The current grip system only models "press into hold from the front" (the
standard climbing contact). Heel hooks require an inverted normal alignment
check — the foot approaches from the opposite side — and a different anchor
geometry.

Defer to Month 2. Will require a per-hold surface-normal attribute and a new
`GripMode` enum (FRONT, HEEL, TOE) that modifies the alignment check and
force thresholds in `try_grip()`.

---

## ISSUE 3: Dynamic foot placement by policy

The policy currently decides WHEN to grip or release a foot (binary grip intent
action) but not WHERE. The scripted fallback in `step()` searches all registered
holds and grips the nearest reachable one. In Month 2, the foot target should
be a continuous latent chosen by a higher-level policy, or added to the action
space as a discrete hold-index selection (one integer per foot slot from
`get_available_holds_for_slot()`). This requires a new observation encoding for
foot-reachable hold candidates and a modified action space.

---

## ISSUE 4: Kickboard contact during swing

When the agent builds momentum for a deadpoint or dynamic move, the torso or
upper legs may contact the kickboard geom. This is physically realistic —
climbers often press knees or hips against the kickboard. However, the current
reward function penalises all non-end-effector contacts via the energy penalty
and slip detection. Torso/leg contact with the kickboard specifically should
probably be allowed (or at least not penalised), whereas contact with the main
wall face should remain penalised (it indicates a fall).

Fix: add a contact-masking table that exempts `kickboard_panel` geom contacts
with non-grip body geoms from penalties.

---

## ISSUE 5: No momentum / dynamic moves yet

V4 routes require controlled quasi-static movement. V7+ routes require dynamic
moves: deadpoints (jumping to a hold with both hands), dynos (both feet leave
the wall), and flagging (extending a free limb for balance). The current RSI
produces quasi-static reference stances. AMP (Adversarial Motion Priors,
Week 3) will push the agent toward human-like motion quality but does not
enforce specific momentum profiles needed for dynos.

True dynos will likely require an explicit sub-policy or curriculum entry point
that grants a large reward bonus for successfully completing a dynamic move,
paired with a motion library of human-capture data for the AMP discriminator.
