"""Beta Engine solver — 2D IK + reachability + A*/RL beta finder.

Consumes wall JSON written by the grid editor and produces a move
sequence (`(LH, RH, LF, RF)` tuples per step) that takes a stick-figure
climber from start holds to a finish hold.

Entry point: `python -m solver --wall <wall_id>`.
"""

from solver.body import BodyModel
from solver.wall import Hold, Wall, load_wall
from solver.reachability import Pose, reachable_moves
from solver.astar import solve_astar
from solver.rl_qlearn import solve_qlearn

__all__ = [
    "BodyModel",
    "Hold",
    "Wall",
    "load_wall",
    "Pose",
    "reachable_moves",
    "solve_astar",
    "solve_qlearn",
]
