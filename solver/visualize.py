"""Matplotlib visualization of a solved beta.

Renders the wall, the holds, and a per-step stick-figure overlay. Saves
either a single PNG (last pose), a multi-panel PNG (one frame per move),
or an MP4/GIF animation.

Headless-safe: uses the Agg backend by default. Imports `matplotlib`
lazily so importing the solver package itself stays light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from solver.astar import SolveResult
from solver.body import BodyModel, FOOT_LIMBS, HAND_LIMBS, LIMBS, resolve_skeleton
from solver.reachability import Pose, estimate_com
from solver.wall import Wall

# Color per hold type — mirrors the editor defaults but keeps this module
# independent of any external palette.
HOLD_COLORS = {
    "jug": "#22c55e",
    "crimp": "#ef4444",
    "sloper": "#f59e0b",
    "pinch": "#3b82f6",
    "foothold": "#a855f7",
}

LIMB_COLORS = {
    "LH": "#ef4444",
    "RH": "#22c55e",
    "LF": "#3b82f6",
    "RF": "#f59e0b",
}


def _setup_axes(ax, wall: Wall) -> None:
    ax.set_xlim(-10, wall.width_cm + 10)
    ax.set_ylim(-10, wall.height_cm + 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")

    # Grid lines.
    for x in range(0, wall.cols + 1):
        ax.axvline(x * wall.cell_size_cm, color="#1e293b", lw=0.5, zorder=0)
    for y in range(0, wall.rows + 1):
        ax.axhline(y * wall.cell_size_cm, color="#1e293b", lw=0.5, zorder=0)


def _draw_holds(ax, wall: Wall) -> None:
    radii = {"small": 6.0, "medium": 9.0, "large": 12.0}
    for h in wall.holds:
        color = HOLD_COLORS.get(h.hold_type, "#cbd5e1")
        r = radii.get(h.size, 8.0)
        ax.scatter(h.x_cm, h.y_cm, s=r * r * 2, c=color, edgecolors="white",
                   linewidths=0.6, zorder=2)
        if h.is_start:
            ax.scatter(h.x_cm, h.y_cm, s=r * r * 4, facecolors="none",
                       edgecolors="#22c55e", linewidths=2, zorder=1)
        if h.is_finish:
            ax.scatter(h.x_cm, h.y_cm, s=r * r * 5.5, facecolors="none",
                       edgecolors="#ef4444", linewidths=2, zorder=1)
        ax.text(h.x_cm, h.y_cm - r - 4, h.hold_id, color="#cbd5e1",
                ha="center", va="top", fontsize=6, zorder=3)


def _draw_skeleton(ax, wall: Wall, body: BodyModel, pose: Pose) -> None:
    com = estimate_com(wall, pose, body)
    if not np.any(com):
        return

    targets = {}
    for limb in LIMBS:
        hid = pose.get(limb)
        if hid is None:
            anchor = com + body.anchor_offset(limb)
            targets[limb] = anchor + np.array([0.0, 0.0])
        else:
            h = wall.by_id(hid)
            targets[limb] = np.array([h.x_cm, h.y_cm])

    skel = resolve_skeleton(body, com, targets)

    # Torso line: average shoulder to average hip.
    sh_mid = np.mean([skel.shoulders[l] for l in HAND_LIMBS], axis=0)
    hip_mid = np.mean([skel.hips[l] for l in FOOT_LIMBS], axis=0)
    ax.plot([sh_mid[0], hip_mid[0]], [sh_mid[1], hip_mid[1]],
            color="#e2e8f0", lw=2.5, zorder=4)
    # Shoulder + hip bars.
    ax.plot([skel.shoulders["LH"][0], skel.shoulders["RH"][0]],
            [skel.shoulders["LH"][1], skel.shoulders["RH"][1]],
            color="#e2e8f0", lw=2, zorder=4)
    ax.plot([skel.hips["LF"][0], skel.hips["RF"][0]],
            [skel.hips["LF"][1], skel.hips["RF"][1]],
            color="#e2e8f0", lw=2, zorder=4)

    # Limbs.
    for limb in HAND_LIMBS:
        anchor = skel.shoulders[limb]
        elbow = skel.elbows[limb]
        ee = skel.end_effectors[limb]
        ax.plot([anchor[0], elbow[0], ee[0]], [anchor[1], elbow[1], ee[1]],
                color=LIMB_COLORS[limb], lw=2.2, zorder=5)
        ax.scatter(*ee, s=40, color=LIMB_COLORS[limb], edgecolor="white",
                   linewidths=0.6, zorder=6)
        ax.text(ee[0] + 4, ee[1] + 4, limb, color=LIMB_COLORS[limb],
                fontsize=7, zorder=7)

    for limb in FOOT_LIMBS:
        anchor = skel.hips[limb]
        knee = skel.knees[limb]
        ee = skel.end_effectors[limb]
        ax.plot([anchor[0], knee[0], ee[0]], [anchor[1], knee[1], ee[1]],
                color=LIMB_COLORS[limb], lw=2.2, zorder=5)
        ax.scatter(*ee, s=40, color=LIMB_COLORS[limb], edgecolor="white",
                   linewidths=0.6, zorder=6)
        ax.text(ee[0] + 4, ee[1] + 4, limb, color=LIMB_COLORS[limb],
                fontsize=7, zorder=7)

    # Head + COM.
    head_y = sh_mid[1] + 0.10 * body.height_cm
    ax.scatter(sh_mid[0], head_y, s=180, color="#e2e8f0", edgecolor="#94a3b8",
               linewidths=0.6, zorder=7)
    ax.scatter(*com, s=30, color="#fde047", edgecolor="black",
               linewidths=0.5, zorder=8)


def render_panels(
    wall: Wall,
    result: SolveResult,
    out_path: Path,
    body: BodyModel | None = None,
    cols: int = 4,
) -> Path:
    """One subplot per pose in the solve. Saves a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = body or BodyModel()
    n = len(result.poses)
    cols = min(cols, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5.5 * rows))
    fig.patch.set_facecolor("#020617")
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue
        _setup_axes(ax, wall)
        _draw_holds(ax, wall)
        _draw_skeleton(ax, wall, body, result.poses[i])
        if i == 0:
            title = "Start"
        else:
            limb, hold_id = result.moves[i - 1]
            title = f"Step {i}: {limb} → {hold_id}"
        ax.set_title(title, color="#e2e8f0", fontsize=10)

    fig.suptitle(
        f"{wall.name} — beta via {result.method} ({len(result.moves)} moves)",
        color="#f8fafc", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_frame(
    wall: Wall,
    result: SolveResult,
    index: int,
    body: BodyModel | None = None,
) -> bytes:
    """Render a single pose frame as PNG bytes (no file written)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    body = body or BodyModel()
    fig, ax = plt.subplots(figsize=(5, 7))
    fig.patch.set_facecolor("#020617")
    _setup_axes(ax, wall)
    _draw_holds(ax, wall)
    _draw_skeleton(ax, wall, body, result.poses[index])

    if index == 0:
        title = "Start"
    else:
        limb, hold_id = result.moves[index - 1]
        title = f"Step {index}: {limb} → {hold_id}"
    ax.set_title(title, color="#e2e8f0", fontsize=11, pad=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def render_all_frames(
    wall: Wall,
    result: SolveResult,
    body: BodyModel | None = None,
) -> list[bytes]:
    """Return one PNG bytes object per pose in the solution."""
    return [render_frame(wall, result, i, body) for i in range(len(result.poses))]


def render_animation(
    wall: Wall,
    result: SolveResult,
    out_path: Path,
    body: BodyModel | None = None,
    fps: int = 2,
) -> Optional[Path]:
    """Animated GIF (or MP4 if ffmpeg present). Returns the actual path
    written, or None if all writers failed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    body = body or BodyModel()
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("#020617")

    def draw_frame(i: int) -> None:
        ax.clear()
        _setup_axes(ax, wall)
        _draw_holds(ax, wall)
        _draw_skeleton(ax, wall, body, result.poses[i])
        if i == 0:
            ax.set_title("Start", color="#e2e8f0")
        else:
            limb, hold_id = result.moves[i - 1]
            ax.set_title(f"Step {i}: {limb} → {hold_id}", color="#e2e8f0")

    anim = FuncAnimation(fig, draw_frame, frames=len(result.poses), interval=1000 // fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(".gif")
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
