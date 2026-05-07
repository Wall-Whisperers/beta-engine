"""Native MuJoCo viewer wrapper.

`mujoco.viewer.launch_passive` is the supported "embed an interactive
3D window in a Python script" entry point. We layer a small loop on top
that drives the simulation forward and respects the user's chosen
playback speed.

This file is import-safe in headless environments (the Docker image,
the test runner, CI). The viewer itself only opens a window when
`run()` is called, and the GLFW import is deferred until then so the
mujoco package can be used for headless training without a display.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Optional

from sim3d.world import Climb3DWorld


@contextmanager
def native_viewer(world: Climb3DWorld):
    """Yield a passive viewer attached to `world.model` / `world.data`.

    Use as:

        with native_viewer(world) as viewer:
            while viewer.is_running():
                world.step()
                viewer.sync()
    """
    import mujoco.viewer  # deferred import — needs an OpenGL display
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        # A reasonable default camera: in front of the wall, looking at
        # the climber's chest.
        viewer.cam.lookat[:] = (0.0, 0.0, world.profile.segments.height_m * 0.6)
        viewer.cam.distance = max(3.5, world.wall.height_cm / 100.0 * 0.9)
        viewer.cam.azimuth = 90.0       # camera on the +Y side
        viewer.cam.elevation = -10.0
        yield viewer


def run_demo(
    world: Climb3DWorld,
    *,
    duration_s: float = 30.0,
    on_frame: Optional[Callable[[Climb3DWorld, float], None]] = None,
) -> None:
    """Spin up the native viewer and step `world` in real time for
    `duration_s`. Handy for `python -m sim3d`.

    `on_frame(world, t)` runs once per render frame — use it to script
    moves over time without having to write a viewer loop yourself.
    """
    from sim3d import config as cfg
    with native_viewer(world) as viewer:
        sim_start = time.monotonic()
        next_render = sim_start
        frame_dt = 1.0 / cfg.RENDER_HZ
        while viewer.is_running():
            t = time.monotonic() - sim_start
            if t >= duration_s:
                break
            # Step one render frame
            world.step()
            if on_frame is not None:
                on_frame(world, t)
            viewer.sync()

            # Real-time pacing: sleep until the next render slot.
            next_render += frame_dt
            sleep_for = next_render - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Drifting behind real time — reset the clock so we
                # don't accumulate a lag debt.
                next_render = time.monotonic()
