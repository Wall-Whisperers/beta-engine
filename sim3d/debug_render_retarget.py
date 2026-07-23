"""Headless diagnostic renderer for retargeted SMPL clips.

Renders an 8-frame contact sheet from a retargeted .npz (output of
`sim3d.retarget`), with the pelvis pinned upright at a fixed position so
only the actuated-joint motion is visible. No display / mjpython needed —
uses `mujoco.Renderer` offscreen.

Usage:
    python -m sim3d.debug_render_retarget data/video/moonboard/spike1_npz/clip01_v3a.npz \
        --out /tmp/clip01_sheet.png
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np


def render_sheet(qpos_seq: np.ndarray, out_path: str, n_frames: int = 8,
                  pin_root: bool = True):
    import mujoco
    from PIL import Image

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sim3d.probe_transitions import build_wall_and_moves
    from sim3d.builder import build_mjcf_xml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wall, profile, _ = build_wall_and_moves(seed=11)
    xml, _ = build_mjcf_xml(wall, profile)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 420)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.6, 1.1]
    cam.distance = 2.2
    cam.azimuth = -110
    cam.elevation = -15

    T = len(qpos_seq)
    idxs = np.linspace(0, T - 1, n_frames).astype(int)

    frames = []
    for t in idxs:
        d.qpos[:] = 0
        qp = qpos_seq[t]
        d.qpos[:len(qp)] = qp
        if pin_root:
            d.qpos[0:3] = [0.0, 0.4, 1.0]
            d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        img = r.render()
        frames.append(img)

    # Tile into a 2-row contact sheet.
    n_cols = (n_frames + 1) // 2
    h, w, c = frames[0].shape
    sheet = np.zeros((h * 2, w * n_cols, c), dtype=np.uint8)
    for i, img in enumerate(frames):
        row, col = divmod(i, n_cols)
        sheet[row*h:(row+1)*h, col*w:(col+1)*w] = img

    Image.fromarray(sheet).save(out_path)
    print(f"Saved {n_frames}-frame contact sheet -> {out_path}")


def _has_cam(m, name) -> bool:
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="Retargeted .npz with a 'qpos' array (T,30)")
    ap.add_argument("--out", default="/tmp/retarget_sheet.png")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--no-pin", action="store_true",
                     help="Don't pin the root — show true world trajectory")
    args = ap.parse_args()

    d = np.load(args.npz)
    qpos = d["qpos"]
    print(f"Loaded qpos {qpos.shape} from {args.npz}")
    render_sheet(qpos, args.out, n_frames=args.frames, pin_root=not args.no_pin)


if __name__ == "__main__":
    main()
