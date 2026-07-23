#!/usr/bin/env bash
# Build the AMP style prior from extracted SMPL clips.
# Run AFTER you've unzipped spike1_smpl.zip into data/video/moonboard/.
#
#   bash build_amp.sh
#
# Produces:
#   data/amp/clips/*.npz        retargeted 29-DOF motion (one per video clip)
#   data/amp/motion_library.npz pooled (s,s') pairs
#   data/amp/disc.pt            trained discriminator (use with --amp-disc)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

SMPL_DIR=data/video/moonboard/spike1_npz
CLIP_DIR=data/amp/clips
mkdir -p "$CLIP_DIR"

if [ ! -d "$SMPL_DIR" ]; then
  echo "ERROR: $SMPL_DIR not found."
  echo "Run the Colab notebook, then: unzip -o ~/Downloads/spike1_smpl.zip -d data/video/moonboard/"
  exit 1
fi

echo "=== Retargeting SMPL → 29-DOF (per clip) ==="
for f in "$SMPL_DIR"/*.npz; do
  name=$(basename "$f")
  echo "--- $name ---"
  # --max-violations rejects frames where retargeting hit joint limits hard.
  # --fps 10: spike1 clips were downsampled to 10fps before SMPL extraction
  # (see data/video/moonboard/spike1/smpl_extract_colab.ipynb); the default
  # of 30 would silently mis-scale the finite-difference qvel by 3x.
  $PY -m sim3d.retarget "$f" "$CLIP_DIR/$name" --fps 10 --max-violations 6 || \
    echo "  (skipped $name — retarget failed)"
done

echo ""
echo "=== Building motion library ==="
$PY -m sim3d.amp build-library "$CLIP_DIR" data/amp/motion_library.npz

echo ""
echo "=== Training discriminator (~10 min CPU) ==="
$PY -m sim3d.amp train --library data/amp/motion_library.npz --out data/amp/disc.pt --epochs 200

echo ""
echo "=== Sanity check ==="
$PY -m sim3d.amp eval --disc data/amp/disc.pt --library data/amp/motion_library.npz

echo ""
echo "Done. Train a policy with the style prior:"
echo "  $PY -m sim3d.imitation --train --ref data/runs/sim3d/imitation/ref_adaptive_s14.npz \\"
echo "      --amp-disc data/amp/disc.pt --amp-coeff 0.5 --steps 1_000_000 --n-envs 8 \\"
echo "      --run-id imitation/amp_test"
