"""Matplotlib renderer for the physics simulation.

Headless-safe (uses MPLBACKEND=Agg). Reads pose state from a
`ClimbWorld` and produces:

    render_frame(world, ax)   — draw one snapshot onto an axis
    render_animation(world, frames_iter, out_path)
                              — drive the simulator forward, rendering
                                a GIF along the way

The look matches `solver/visualize.py` so existing example outputs are
visually consistent. Per-limb force is shown via a colour ramp on the
limb segments — green = comfortable, red = at slip threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from physics import config as cfg
from physics.body import FOOT_LIMBS, HAND_LIMBS, LIMBS
from physics.world import ClimbWorld
from solver.wall import Wall

# Hold colours mirror solver/visualize.py.
HOLD_COLORS = {
    "jug": "#22c55e",
    "crimp": "#ef4444",
    "sloper": "#f59e0b",
    "pinch": "#3b82f6",
    "foothold": "#a855f7",
}

# Limb colours: warm = arms, cool = legs.
LIMB_BASE_COLORS = {
    "LH": "#fca5a5",
    "RH": "#86efac",
    "LF": "#93c5fd",
    "RF": "#fcd34d",
}


def _setup_axes(ax, wall: Wall) -> None:
    ax.set_xlim(-10, wall.width_cm + 10)
    ax.set_ylim(-60, wall.height_cm + 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    # Grid.
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


def _force_color(fraction: Optional[float], base: str) -> str:
    """Tint a base limb colour by how loaded the limb is.

    `fraction` is force / max_force on the SlideJoint (None if the limb
    is in flight). 0 = unloaded → base colour; 1 = at slip threshold →
    red. Smooth interpolation between.
    """
    if fraction is None:
        return base + "55"  # heavy alpha for in-flight limbs
    f = float(np.clip(fraction, 0.0, 1.0))
    # Linear interpolate from base RGB toward red.
    base_rgb = _hex_to_rgb(base)
    red_rgb = (0.94, 0.27, 0.27)
    out = tuple(b * (1 - f) + r * f for b, r in zip(base_rgb, red_rgb))
    return "#%02x%02x%02x" % tuple(int(c * 255) for c in out)


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    hs = hex_str.lstrip("#")
    return tuple(int(hs[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _draw_skeleton(ax, world: ClimbWorld) -> None:
    body = world.body
    com = world.com_cm()

    # Torso line: hip midpoint to shoulder midpoint.
    bm = body.profile.body
    # Body-local hip / shoulder y, expressed in cm relative to COM.
    hip_local_y = -0.5 * bm.shoulder_height
    shoulder_local_y = +0.5 * bm.shoulder_height
    torso_angle = body.torso.angle
    cos_a, sin_a = np.cos(torso_angle), np.sin(torso_angle)

    def _torso_local_to_world_cm(lx_cm: float, ly_cm: float) -> np.ndarray:
        """Transform a torso-local point (cm) to world (cm), respecting
        the torso's current rotation."""
        rotated = np.array([cos_a * lx_cm - sin_a * ly_cm,
                            sin_a * lx_cm + cos_a * ly_cm])
        return com + rotated

    hip_mid = _torso_local_to_world_cm(0.0, hip_local_y)
    shoulder_mid = _torso_local_to_world_cm(0.0, shoulder_local_y)
    ax.plot([hip_mid[0], shoulder_mid[0]],
            [hip_mid[1], shoulder_mid[1]],
            color="#e2e8f0", lw=3.5, zorder=4)

    # Shoulder bar + hip bar.
    sh_l = _torso_local_to_world_cm(-bm.shoulder_offset, shoulder_local_y)
    sh_r = _torso_local_to_world_cm(+bm.shoulder_offset, shoulder_local_y)
    hp_l = _torso_local_to_world_cm(-bm.hip_offset, hip_local_y)
    hp_r = _torso_local_to_world_cm(+bm.hip_offset, hip_local_y)
    ax.plot([sh_l[0], sh_r[0]], [sh_l[1], sh_r[1]], color="#e2e8f0", lw=2, zorder=4)
    ax.plot([hp_l[0], hp_r[0]], [hp_l[1], hp_r[1]], color="#e2e8f0", lw=2, zorder=4)

    # Limbs: anchor → elbow/knee → end_effector. End-effector = hold for
    # attached limbs, else "hangs straight down at max reach" stub.
    for limb in LIMBS:
        anchor_cm = world.shoulder_or_hip_cm(limb)
        end_cm = world.end_effector_cm(limb)
        joint_cm = world.joint_cm(limb)
        frac = body.per_limb_force_fraction(limb)
        color = _force_color(frac, LIMB_BASE_COLORS[limb])
        lw = 3.0 if frac is not None else 1.5
        ax.plot([anchor_cm[0], joint_cm[0]],
                [anchor_cm[1], joint_cm[1]],
                color=color, lw=lw, zorder=5)
        ax.plot([joint_cm[0], end_cm[0]],
                [joint_cm[1], end_cm[1]],
                color=color, lw=lw, zorder=5)
        # End-effector dot.
        ax.scatter(end_cm[0], end_cm[1], s=40,
                   c=color, edgecolors="white", linewidths=0.5, zorder=6)

    # COM marker.
    ax.scatter(com[0], com[1], s=50, c="#fbbf24", marker="x", zorder=7)


def _draw_status(ax, world: ClimbWorld, t: float) -> None:
    body = world.body
    forces = []
    for limb in LIMBS:
        f = body.per_limb_force(limb)
        forces.append(f"{limb}={f:5.0f}N" if f is not None else f"{limb}= --  ")
    ax.set_title(
        f"t={t:5.2f}s   wall_angle={world.wall.wall_angle_deg:.1f}°   "
        + "  ".join(forces) + f"   stable={'Y' if world.is_stable() else 'N'}",
        color="#e2e8f0", fontsize=9,
    )


def render_frame(world: ClimbWorld, ax, *, t: float = 0.0) -> None:
    """Draw one snapshot of the world onto a matplotlib axis."""
    ax.clear()
    _setup_axes(ax, world.wall)
    _draw_holds(ax, world.wall)
    _draw_skeleton(ax, world)
    _draw_status(ax, world, t)


def render_animation(
    world: ClimbWorld,
    out_path: str | Path,
    *,
    n_frames: int = 60,
    frames_per_step: int = 1,
    on_frame: Optional[Callable[[ClimbWorld, int], None]] = None,
    fps: int = 30,
) -> Path:
    """Run the simulation forward and write an animated GIF.

    Args:
        world: a ClimbWorld with a seeded pose.
        out_path: target .gif path.
        n_frames: number of rendered frames.
        frames_per_step: physics frames per rendered frame.
            (Each physics frame is `cfg.SUBSTEPS_PER_FRAME` substeps.)
        on_frame: optional callback `(world, frame_idx) -> None` invoked
            BEFORE each physics step, e.g. to issue move_limb commands.
        fps: render rate.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter, FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 10), dpi=110)
    fig.patch.set_facecolor("#0f172a")

    state = {"t": 0.0}
    dt_per_frame = cfg.PHYS_DT * cfg.SUBSTEPS_PER_FRAME * frames_per_step

    def _draw(idx):
        if on_frame is not None:
            on_frame(world, idx)
        if idx > 0:
            world.step(frames_per_step)
            state["t"] += dt_per_frame
        render_frame(world, ax, t=state["t"])
        return ()

    anim = FuncAnimation(fig, _draw, frames=n_frames, interval=1000 / fps, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def render_still(world: ClimbWorld, out_path: str | Path) -> Path:
    """Render a single PNG of the current world state (no simulation)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10), dpi=110)
    fig.patch.set_facecolor("#0f172a")
    render_frame(world, ax, t=0.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_key_frames(
    world: ClimbWorld,
    moves: list[tuple[str, str]],
    settle_steps: int = 80,
) -> list[bytes]:
    """Run the physics through a move sequence and return one PNG per pose.

    One frame is captured after the initial settle, then one more after each
    move settles — matching the solver visualiser's frame count so the UI
    can reuse the same navigator.

    Args:
        world:        A ClimbWorld with a pose already seeded.
        moves:        Sequence of (limb, hold_id) pairs.
        settle_steps: Physics frames to run between renders.

    Returns:
        List of PNG bytes — frame 0 is the start pose, frame i+1 is after
        move i.
    """
    import io
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frames: list[bytes] = []

    def _snap() -> bytes:
        fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        t = len(frames) * settle_steps * cfg.PHYS_DT * cfg.SUBSTEPS_PER_FRAME
        render_frame(world, ax, t=t)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # Initial settle.
    world.step(settle_steps)
    frames.append(_snap())

    for limb, hold_id in moves:
        world.move_limb(limb, hold_id, mode="snap")
        world.step(settle_steps)
        frames.append(_snap())

    return frames
