"""Unified MuJoCo viewer entry point for all MoonBoard RL scripts.

All scripts that need a viewer (day1_viewer.py, test_grip.py,
interactive_grip.py) call ``launch()`` from this module instead of
duplicating viewer boilerplate.

Key design decisions
--------------------
- ``on_key`` receives a *string* (single character or special name like
  "space") rather than a raw GLFW integer keycode, so callers never
  import or know about GLFW.
- Physics steps run inside the viewer loop at the model's own timestep.
  Callers that want to inject logic each step pass ``on_step``.
- If ``launch_passive`` raises (no display, wrong Python runtime on macOS),
  the function prints a clear message and falls back to running the step
  loop headlessly so automated tests still work.

macOS note
----------
MuJoCo's passive viewer requires ``mjpython`` on macOS::

    mjpython scripts/day1_viewer.py
    mjpython scripts/interactive_grip.py

Without ``mjpython`` the viewer falls back to headless mode automatically.
"""

from __future__ import annotations
from typing import Callable, Optional

# GLFW keycode → single-character string mapping for printable ASCII keys.
# Non-printable keys (arrows, F-keys, etc.) get the name "key_<int>".
_GLFW_PRINTABLE_OFFSET = 0   # GLFW key codes for A-Z are 65-90 (uppercase)

_SPECIAL_KEYS: dict[int, str] = {
    256: "escape",
    257: "enter",
    258: "tab",
    259: "backspace",
    32:  "space",
    # Number row (GLFW_KEY_0..9 = 48..57, same as ASCII).
}


def _keycode_to_str(keycode: int) -> str:
    """Convert a GLFW integer keycode to a lowercase string.

    Args:
        keycode: Raw integer keycode from MuJoCo's key_callback.

    Returns:
        A single lowercase character for printable keys (letters, digits,
        punctuation), or a descriptive string like ``"escape"`` for special keys.
    """
    if keycode in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[keycode]
    if 32 <= keycode < 128:
        return chr(keycode).lower()
    return f"key_{keycode}"


def launch(
    model,
    data,
    on_key: Optional[Callable[[str, object, object], None]] = None,
    on_step: Optional[Callable[[object, object], None]] = None,
    title: str = "MoonBoard RL",
    headless_steps: int = 0,
) -> None:
    """Launch the MuJoCo passive viewer and run the simulation loop.

    Registers an optional key callback and an optional per-step callback.
    Falls back to headless simulation if the viewer cannot be opened.

    Args:
        model: A loaded ``mujoco.MjModel`` instance.
        data:  The corresponding ``mujoco.MjData`` instance.
        on_key: Optional callable invoked on every key press.
            Signature: ``on_key(key: str, model, data) -> None``.
            ``key`` is a lowercase string (e.g. ``"1"``, ``"g"``, ``"space"``).
        on_step: Optional callable invoked once per simulation step *after*
            ``mujoco.mj_step``.  Signature: ``on_step(model, data) -> None``.
            Useful for injecting grip logic into the step loop without
            modifying this module.
        title: Window title shown in the viewer (informational only).
        headless_steps: If the viewer cannot open, run this many simulation
            steps in headless mode before returning.  Pass 0 (default) for
            indefinite headless running (Ctrl-C to stop).
    """
    import mujoco
    import mujoco.viewer as mj_viewer

    def _key_cb(keycode: int) -> None:
        if on_key is not None:
            on_key(_keycode_to_str(keycode), model, data)

    try:
        with mj_viewer.launch_passive(
            model,
            data,
            key_callback=_key_cb if on_key else None,
        ) as viewer:
            print(f"[viewer] '{title}' open — close window or press Ctrl-C to exit.")
            step = 0
            while viewer.is_running():
                mujoco.mj_step(model, data)
                if on_step is not None:
                    on_step(model, data)
                viewer.sync()
                step += 1

    except Exception as exc:
        print(f"[viewer] UNAVAILABLE ({type(exc).__name__}: {exc})")
        print("[viewer] Running in headless mode.")
        _run_headless(model, data, on_step, headless_steps, mujoco)


def _run_headless(model, data, on_step, max_steps: int, mujoco) -> None:
    """Run the simulation loop without a viewer window.

    Args:
        model: MjModel.
        data:  MjData.
        on_step: Optional per-step callback.
        max_steps: Number of steps to run (0 = indefinite until Ctrl-C).
        mujoco: The mujoco module (passed in to avoid re-importing).
    """
    step = 0
    try:
        while max_steps == 0 or step < max_steps:
            mujoco.mj_step(model, data)
            if on_step is not None:
                on_step(model, data)
            step += 1
    except KeyboardInterrupt:
        print(f"[viewer] Headless loop stopped after {step} steps.")
