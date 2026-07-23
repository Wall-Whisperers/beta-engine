# spike1 — first MoonBoard video dataset (2026-06-11)

Source: single YouTube video (`moonboard.mp4`, 1280×720 @ 25 fps, static
head-on tripod shot, whole board in frame, grade overlay bottom-left).
Cut with ffmpeg (libx264 crf 18, audio stripped); timestamps are from the
source video.

| clip | src time (s) | grade | problem (from overlay) |
|---|---|---|---|
| clip01_v3a | 0–14 | V3 | The Warm up Problem — Set: RussK |
| clip02_v3b | 15–30 | V3 | (read overlay) |
| clip03_v4a | 30–50 | V4 | (read overlay) |
| clip04_v4b | 50–62 | V4 | (read overlay) |
| clip05_v5a | 62–90 | V5 | Black Muffler — Set: Koala Climbing |
| clip06_v5b | 90–122 | V5 | (read overlay) |
| clip07_v6a | 122–136 | V6 | (read overlay) |
| clip08_v6b | 136–157 | V6 | (read overlay) |
| clip09_v7 | 157–177 | V7 | (read overlay) |

TODO: fill problem names from the overlay text per clip and resolve each to
its hold set via the MoonBoard problem database (`data/moonboard/`); note
board version + angle for grid registration.

## 2D pose feasibility (YOLO11s-pose, MPS)

| clip | person detected | wrists | ankles | hips | shoulders |
|---|---|---|---|---|---|
| clip01_v3a | 349/350 | 0.90 | 0.89 | 0.99 | 0.98 |
| clip02_v3b | 372/375 | 0.90 | 0.92 | 0.99 | 0.98 |
| clip05_v5a | 700/700 | 0.93 | 0.87 | 0.99 | 0.99 |
| clip09_v7  | 493/500 | 0.95 | 0.85 | 0.99 | 0.98 |

(mean per-frame keypoint confidence of the top-confidence person)

Verdict: footage is highly extractable — ≥99% person detection at every
grade tested, drop-knee/turned-hip poses tracked cleanly. Known caveats:
crash pads occlude feet for the first ~1 s of each climb (low starts);
720p means the climber box is ~250 px tall (modern estimators upsample the
crop, confirmed fine). Next: SMPL extraction (4D-Humans) for 3D — upload
`smpl_extract_colab.ipynb` (in this directory) to Colab with a GPU runtime,
feed it these clips, download `spike1_smpl.zip` back here.
