"""Clean generated output files from training runs.

Usage:
    python3 scripts/clean_output.py           # list what would be deleted
    python3 scripts/clean_output.py --yes     # delete everything
    python3 scripts/clean_output.py --videos  # delete only mp4s
    python3 scripts/clean_output.py --models  # delete only .zip model files
    python3 scripts/clean_output.py --logs    # delete only TensorBoard logs
"""

import argparse
import os
import shutil
import sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUT_ROOT    = os.path.join(_PROJECT_ROOT, "output")

_TARGETS = {
    "videos": {
        "path":    os.path.join(_OUT_ROOT, "videos"),
        "pattern": ".mp4",
        "label":   "evaluation videos",
    },
    "models": {
        "path":    os.path.join(_OUT_ROOT, "checkpoints"),
        "pattern": ".zip",
        "label":   "model checkpoints",
    },
    "logs": {
        "path":    os.path.join(_OUT_ROOT, "tb_logs"),
        "pattern": None,   # whole directory
        "label":   "TensorBoard logs",
    },
}


def _list_files(root: str, suffix: str | None) -> list[str]:
    """Return all files under root matching suffix (or all files if suffix is None)."""
    result = []
    if not os.path.isdir(root):
        return result
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if suffix is None or fname.endswith(suffix):
                result.append(os.path.join(dirpath, fname))
    return result


def _human_size(path: str) -> str:
    try:
        total = 0
        if os.path.isfile(path):
            total = os.path.getsize(path)
        else:
            for dp, _, fns in os.walk(path):
                for fn in fns:
                    total += os.path.getsize(os.path.join(dp, fn))
        if total < 1024:
            return f"{total} B"
        elif total < 1024 ** 2:
            return f"{total/1024:.1f} KB"
        else:
            return f"{total/1024**2:.1f} MB"
    except Exception:
        return "? B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MoonBoard RL output files.")
    parser.add_argument("--yes",    action="store_true", help="Actually delete (no dry-run).")
    parser.add_argument("--videos", action="store_true", help="Target evaluation videos only.")
    parser.add_argument("--models", action="store_true", help="Target model checkpoints only.")
    parser.add_argument("--logs",   action="store_true", help="Target TensorBoard logs only.")
    args = parser.parse_args()

    # If no specific target, apply to all.
    targets = []
    if args.videos: targets.append("videos")
    if args.models: targets.append("models")
    if args.logs:   targets.append("logs")
    if not targets: targets = list(_TARGETS.keys())

    grand_total_files = 0

    for key in targets:
        cfg = _TARGETS[key]
        root    = cfg["path"]
        pattern = cfg["pattern"]
        label   = cfg["label"]

        files = _list_files(root, pattern)
        grand_total_files += len(files)

        if not files:
            print(f"  {label}: nothing to delete")
            continue

        print(f"  {label}  ({_human_size(root)}):")
        for f in files:
            rel = os.path.relpath(f, _PROJECT_ROOT)
            print(f"    {rel}  ({_human_size(f)})")

    if grand_total_files == 0:
        print("Nothing to clean.")
        return

    if not args.yes:
        print(f"\n{grand_total_files} file(s) would be deleted.  "
              "Re-run with --yes to confirm.")
        return

    # Perform deletion.
    for key in targets:
        cfg  = _TARGETS[key]
        root = cfg["path"]
        if not os.path.isdir(root):
            continue
        if cfg["pattern"] is None:
            # Remove entire directory tree.
            shutil.rmtree(root, ignore_errors=True)
            print(f"  Removed {os.path.relpath(root, _PROJECT_ROOT)}/")
        else:
            for f in _list_files(root, cfg["pattern"]):
                os.remove(f)
                print(f"  Deleted {os.path.relpath(f, _PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()
