"""A* baseline solver over the pose graph.

Nodes  = poses `(LH, RH, LF, RF)` of hold IDs.
Edges  = legal one-limb moves (see `reachability.reachable_moves`).
Goal   = at least one hand on a finish hold.
Heuristic = vertical distance from highest hand to nearest finish hold.

This is the "mathematically shortest" baseline the user's plan describes,
the same one the RL agent is compared against.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count
from typing import Optional

import numpy as np

from solver.body import BodyModel, HAND_LIMBS, Limb
from solver.reachability import Pose, is_stable, reachable_moves
from solver.wall import Wall, foot_holds, hand_holds


@dataclass
class SolveResult:
    """Output of any solver."""

    poses: list[Pose]               # pose 0 is the start, pose N is finish
    moves: list[tuple[Limb, str]]   # length N (one per transition)
    expanded: int                    # nodes/poses popped during search
    method: str

    def text_steps(self) -> list[str]:
        out = []
        for i, (limb, hold) in enumerate(self.moves, 1):
            out.append(f"Step {i:>2}: {limb} → {hold}")
        return out


def _heuristic(wall: Wall, pose: Pose) -> float:
    finishes = wall.finishes() or [h for h in hand_holds(wall.holds)]
    if not finishes:
        return 0.0
    hand_ys = [
        wall.by_id(h).y_cm
        for h in (pose.LH, pose.RH)
        if h is not None
    ]
    cur_y = max(hand_ys) if hand_ys else 0.0
    target_y = max(f.y_cm for f in finishes)
    return max(0.0, target_y - cur_y)


def _is_goal(wall: Wall, pose: Pose) -> bool:
    finish_ids = {h.hold_id for h in wall.finishes()}
    if not finish_ids:
        return False
    return (pose.LH in finish_ids) or (pose.RH in finish_ids)


def starting_poses(wall: Wall, body: BodyModel) -> list[Pose]:
    """Enumerate plausible 4-limb starting matchups across the start
    holds plus any nearby footholds. We prefer poses where:

      - both hands are on `is_start` holds (or one hand if only one start),
      - both feet are on the closest reachable footholds,
      - the pose is stable.
    """
    starts = wall.starts()
    foots = foot_holds(wall.holds)
    if not foots:
        return []

    # Pick hand assignments.
    hand_assignments: list[tuple[Optional[str], Optional[str]]] = []
    if len(starts) >= 2:
        # Sort starts by x to assign LH ↔ leftmost, RH ↔ rightmost.
        sorted_starts = sorted(starts, key=lambda h: h.x_cm)
        hand_assignments.append((sorted_starts[0].hold_id, sorted_starts[-1].hold_id))
    elif len(starts) == 1:
        hand_assignments.append((starts[0].hold_id, starts[0].hold_id))
    else:
        # No start markers — use the lowest two hand-usable holds.
        low_hands = sorted(hand_holds(wall.holds), key=lambda h: h.y_cm)[:2]
        if len(low_hands) >= 2:
            l, r = sorted(low_hands[:2], key=lambda h: h.x_cm)
            hand_assignments.append((l.hold_id, r.hold_id))

    poses: list[Pose] = []
    for lh, rh in hand_assignments:
        # Pick the two foot holds with lowest y that are below the hands and stable.
        candidate_feet = sorted(foots, key=lambda h: h.y_cm)
        for i, lf in enumerate(candidate_feet):
            for rf in candidate_feet[i + 1:]:
                lf_id, rf_id = (
                    (lf.hold_id, rf.hold_id) if lf.x_cm <= rf.x_cm
                    else (rf.hold_id, lf.hold_id)
                )
                pose = Pose(LH=lh, RH=rh, LF=lf_id, RF=rf_id)
                if is_stable(wall, pose):
                    poses.append(pose)
                    break
            else:
                continue
            break

    # De-dupe while preserving order.
    seen = set()
    unique: list[Pose] = []
    for p in poses:
        key = p.as_tuple()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def solve_astar(
    wall: Wall,
    body: BodyModel | None = None,
    max_expansions: int = 20_000,
) -> Optional[SolveResult]:
    """Find a beta from a starting pose to a finish hold."""
    body = body or BodyModel()
    starts = starting_poses(wall, body)
    if not starts:
        return None

    # Multi-source A*: push every plausible starting pose with cost 0.
    counter = count()
    open_heap: list[tuple[float, int, Pose]] = []
    came_from: dict[tuple, tuple[Pose, tuple[Limb, str]] | None] = {}
    g_score: dict[tuple, float] = {}

    for s in starts:
        key = s.as_tuple()
        g_score[key] = 0.0
        came_from[key] = None
        heapq.heappush(open_heap, (_heuristic(wall, s), next(counter), s))

    expanded = 0
    while open_heap and expanded < max_expansions:
        _, _, current = heapq.heappop(open_heap)
        expanded += 1
        ckey = current.as_tuple()

        if _is_goal(wall, current):
            return _reconstruct(current, came_from, expanded, "astar")

        for limb, target_id in reachable_moves(body, wall, current):
            neighbor = current.with_limb(limb, target_id)
            nkey = neighbor.as_tuple()
            tentative = g_score[ckey] + _move_cost(wall, current, limb, target_id)
            if tentative < g_score.get(nkey, float("inf")):
                g_score[nkey] = tentative
                came_from[nkey] = (current, (limb, target_id))
                heapq.heappush(
                    open_heap,
                    (tentative + _heuristic(wall, neighbor), next(counter), neighbor),
                )

    return None


def _move_cost(wall: Wall, pose: Pose, limb: Limb, target_id: str, body: BodyModel | None = None) -> float:
    """Cost per move. Worse holds (slopers etc.) cost slightly more."""
    target = wall.by_id(target_id)
    return 1.0 + 0.5 * (1.0 - target.positivity)


def _reconstruct(
    end: Pose,
    came_from: dict[tuple, tuple[Pose, tuple[Limb, str]] | None],
    expanded: int,
    method: str,
) -> SolveResult:
    poses: list[Pose] = [end]
    moves: list[tuple[Limb, str]] = []
    cur = end
    while True:
        link = came_from.get(cur.as_tuple())
        if link is None:
            break
        prev, move = link
        poses.append(prev)
        moves.append(move)
        cur = prev
    poses.reverse()
    moves.reverse()
    return SolveResult(poses=poses, moves=moves, expanded=expanded, method=method)
