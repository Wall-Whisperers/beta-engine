"""Interactive matplotlib viewer for the physics climbing simulation.

Right-click an end-effector (hand or foot) to grab it; drag to move it.
The other three limbs stay attached to their holds with full physics.
Release near a hold to snap on; release in empty space to leave it free.

Usage:
    python -m physics --wall example-v2-boulder --interactive
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pymunk

from physics import config as cfg
from physics.body import LIMBS
from physics.render import LIMB_BASE_COLORS, render_frame
from physics.world import ClimbWorld

# How close (cm) the cursor must be to an end-effector to grab it.
_GRAB_RADIUS_CM = 12.0
# How close (cm) the released limb must be to a hold to snap onto it.
_SNAP_RADIUS_CM = 18.0
# Force cap for the drag joint — large enough to feel "locked" to the cursor.
_DRAG_MAX_FORCE = 8_000.0  # N — firm but lets the body shift naturally


class _DragState:
    """Tracks one active right-click drag."""

    def __init__(
        self,
        limb: str,
        world: ClimbWorld,
        x_m: float,
        y_m: float,
    ) -> None:
        self.limb = limb
        self._world = world

        # Kinematic anchor body that follows the cursor.
        self._anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self._anchor.position = (x_m, y_m)
        world.space.add(self._anchor)

        anchor_local = world.body._anchor_local[limb]
        max_reach = world.body._max_reach[limb]

        # SlideJoint: torso shoulder/hip → cursor anchor, capped at limb length.
        self._joint = pymunk.SlideJoint(
            world.body.torso,
            self._anchor,
            anchor_local,
            (0.0, 0.0),
            0.0,
            max_reach * 0.999,
        )
        self._joint.max_force = _DRAG_MAX_FORCE
        world.space.add(self._joint)

        # Detach from whichever hold it was on.
        world.release_limb(limb)

    def move(self, x_m: float, y_m: float) -> None:
        self._anchor.position = (x_m, y_m)

    def tip_cm(self) -> np.ndarray:
        return np.array(self._anchor.position) * cfg.CM_PER_M

    def finish(self, x_m: float, y_m: float) -> None:
        """Remove the drag constraint; snap to the nearest hold if close enough."""
        world = self._world
        if self._joint in world.space.constraints:
            world.space.remove(self._joint)
        if self._anchor in world.space.bodies:
            world.space.remove(self._anchor)

        # Snap to nearest hold within radius.
        x_cm, y_cm = x_m * cfg.CM_PER_M, y_m * cfg.CM_PER_M
        best_id: Optional[str] = None
        best_dist = _SNAP_RADIUS_CM
        for hold_id, ha in world.holds.items():
            hx = ha.position_m[0] * cfg.CM_PER_M
            hy = ha.position_m[1] * cfg.CM_PER_M
            d = float(np.hypot(hx - x_cm, hy - y_cm))
            if d < best_dist:
                best_dist = d
                best_id = hold_id

        if best_id is not None:
            world.move_limb(self.limb, best_id, mode="snap")


class InteractiveViewer:
    """Runs the physics sim in a live matplotlib window with right-click dragging."""

    def __init__(self, world: ClimbWorld) -> None:
        self.world = world
        self._drag: Optional[_DragState] = None
        self._last_cursor_m: tuple[float, float] = (0.0, 0.0)
        self._t = 0.0

    def _pick_limb(self, x_cm: float, y_cm: float) -> Optional[str]:
        best_limb: Optional[str] = None
        best_dist = _GRAB_RADIUS_CM
        for limb in LIMBS:
            ee = self.world.end_effector_cm(limb)
            d = float(np.hypot(ee[0] - x_cm, ee[1] - y_cm))
            if d < best_dist:
                best_dist = d
                best_limb = limb
        return best_limb

    def _on_press(self, event) -> None:
        if event.button != 3 or event.xdata is None or event.ydata is None:
            return
        limb = self._pick_limb(event.xdata, event.ydata)
        if limb is None:
            return
        x_m = event.xdata / cfg.CM_PER_M
        y_m = event.ydata / cfg.CM_PER_M
        self._last_cursor_m = (x_m, y_m)
        self._drag = _DragState(limb, self.world, x_m, y_m)

    def _on_motion(self, event) -> None:
        if self._drag is None:
            return
        if event.xdata is not None and event.ydata is not None:
            x_m = event.xdata / cfg.CM_PER_M
            y_m = event.ydata / cfg.CM_PER_M
            self._last_cursor_m = (x_m, y_m)
        self._drag.move(*self._last_cursor_m)

    def _on_release(self, event) -> None:
        if self._drag is None or event.button != 3:
            return
        if event.xdata is not None and event.ydata is not None:
            x_m = event.xdata / cfg.CM_PER_M
            y_m = event.ydata / cfg.CM_PER_M
            self._last_cursor_m = (x_m, y_m)
        drag, self._drag = self._drag, None
        drag.finish(*self._last_cursor_m)

    def run(self, fps: int = 30) -> None:
        import matplotlib
        # Pick the best available interactive backend on this platform.
        for backend in ("MacOSX", "TkAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("button_release_event", self._on_release)

        # Suppress the default right-click context menu so it doesn't interfere.
        try:
            fig.canvas.toolbar.mode = ""
        except Exception:
            pass

        dt_per_frame = cfg.PHYS_DT * cfg.SUBSTEPS_PER_FRAME

        def _animate(_frame):
            self.world.step(1)
            self._t += dt_per_frame
            render_frame(self.world, ax, t=self._t)

            # Draw crosshair cursor + limb label while dragging.
            if self._drag is not None:
                tip = self._drag.tip_cm()
                color = LIMB_BASE_COLORS.get(self._drag.limb, "#ffffff")
                ax.scatter(tip[0], tip[1], s=200, c=color, marker="*",
                           edgecolors="white", linewidths=0.8, zorder=10)
                ax.text(tip[0] + 3, tip[1] + 3, self._drag.limb,
                        color=color, fontsize=8, zorder=11)

            return ()

        # Suppress right-click context menu at the canvas level.
        try:
            fig.canvas.callbacks.callbacks.pop("button_press_event_canvas", None)
        except Exception:
            pass

        _anim = FuncAnimation(
            fig, _animate,
            interval=max(1, 1000 // fps),
            blit=False,
            cache_frame_data=False,
        )

        print("Right-click an end-effector (hand or foot) and drag to move it.")
        print("Release near a hold to snap on; release in empty space to leave free.")
        plt.tight_layout()
        plt.show()


def launch(world: ClimbWorld, *, fps: int = 30) -> None:
    """Entry point called from physics/__main__.py --interactive."""
    InteractiveViewer(world).run(fps=fps)
