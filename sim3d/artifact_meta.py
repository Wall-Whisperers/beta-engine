"""Embedded provenance metadata for training artifacts + load-time validation.

Three artifact families write/read this metadata:

  - Reference trajectories (``sim3d/reference.py``, ``sim3d/discover.py``) —
    embedded as a JSON string under the existing npz ``meta`` key.
  - Policy/VecNormalize checkpoints (``sim3d/imitation.py``, ``sim3d/train.py``)
    — a sidecar ``<name>.meta.json`` next to the ``.zip`` (SB3 zips are awkward
    to extend).
  - DAgger landing banks (``sim3d/imitation.py --landing-bank``) — embedded as
    a JSON string under a npz ``meta`` key.

At save time we record wall identity (id + content hash), cell size, obs/action
dims, env mode, git commit, a timestamp, and a LINEAGE pointer (parent artifact
path + that parent's own recorded eval, if known). At load time ``validate()``
HARD-ERRORS on a mismatch of the fields that matter for correctness (cell size,
wall hash, obs/action dims, env mode) — these are exactly the silent
incompatibilities that have cost real time: a hardcoded 20cm cell size
rescaling holds 4x on a 5cm wall, a warm-start built on a regressed parent, a
``--record`` run in the wrong env mode. Lineage is informational only — printed
at load, never enforced.

Legacy artifacts saved before this module existed carry no metadata; loading
them prints a loud warning instead of erroring.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GIT_COMMIT_CACHE: Optional[str] = None

# Fields hard-validated when present (non-None) on BOTH sides. A field absent
# from either the loaded artifact or the caller's `current` dict is skipped —
# so a validator only checks what it actually knows how to compare.
VALIDATE_FIELDS = ("cell_size_cm", "wall_hash", "obs_dim", "action_dim", "env_mode")


def git_commit() -> str:
    """Short git commit hash of the repo, cached for the process lifetime."""
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            _GIT_COMMIT_CACHE = out or "unknown"
        except Exception:  # noqa: BLE001 — never let provenance block a save
            _GIT_COMMIT_CACHE = "unknown"
    return _GIT_COMMIT_CACHE


def wall_fingerprint(wall: Any) -> dict:
    """``{wall_id, wall_hash, cell_size_cm}`` for a ``solver.wall.Wall``.

    ``wall_hash`` is a content hash of the grid + every hold's placement/type,
    so two walls that differ only in e.g. cell size (the 20cm-vs-5cm bug this
    module exists to catch) hash differently even if ``wall_id`` matches.
    """
    holds = sorted(
        (h.hold_id, h.grid_x, h.grid_y, h.hold_type, round(float(h.orientation_deg), 1),
         h.size, bool(h.is_start), bool(h.is_finish))
        for h in wall.holds
    )
    payload = {
        "wall_id": wall.wall_id, "cols": wall.cols, "rows": wall.rows,
        "cell_size_cm": float(wall.cell_size_cm),
        "wall_angle_deg": float(getattr(wall, "wall_angle_deg", 0.0)),
        "holds": holds,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return {"wall_id": wall.wall_id, "wall_hash": digest, "cell_size_cm": float(wall.cell_size_cm)}


def build_meta(*, artifact_type: str, wall: Any = None,
               obs_dim: Optional[int] = None, action_dim: Optional[int] = None,
               env_mode: Optional[str] = None, extra: Optional[dict] = None,
               parent: Optional[str] = None, parent_eval: Optional[float] = None) -> dict:
    """Build the metadata dict embedded/sidecar-written at save time."""
    fp = wall_fingerprint(wall) if wall is not None else \
        {"wall_id": None, "wall_hash": None, "cell_size_cm": None}
    meta = {
        "artifact_type": artifact_type,
        "wall_id": fp["wall_id"], "wall_hash": fp["wall_hash"], "cell_size_cm": fp["cell_size_cm"],
        "obs_dim": obs_dim, "action_dim": action_dim, "env_mode": env_mode,
        "git_commit": git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lineage": {"parent": parent, "parent_eval": parent_eval},
    }
    if extra:
        meta.update(extra)
    return meta


def validate(loaded_meta: Optional[dict], current: dict, *, name: str, path: Any) -> None:
    """Hard-error on a mismatch of any ``VALIDATE_FIELDS`` entry present
    (non-None) in both ``loaded_meta`` and ``current``. Print lineage if
    present. Warn (don't error) when ``loaded_meta`` is missing/legacy."""
    if not loaded_meta:
        print(f"[artifact_meta] WARNING: {name} at {path} has no embedded metadata "
              f"(legacy artifact, predates provenance tracking) — skipping validation.")
        return
    mismatches = []
    for field in VALIDATE_FIELDS:
        want = current.get(field)
        got = loaded_meta.get(field)
        if want is None or got is None:
            continue
        if isinstance(want, float) or isinstance(got, float):
            if abs(float(want) - float(got)) > 1e-6:
                mismatches.append((field, got, want))
        elif got != want:
            mismatches.append((field, got, want))
    if mismatches:
        lines = "\n".join(f"    {f}: artifact={g!r}  current={w!r}" for f, g, w in mismatches)
        raise ValueError(
            f"[artifact_meta] {name} at {path} is INCOMPATIBLE with the current run:\n{lines}\n"
            f"  Loading a mismatched artifact silently corrupts training/authoring. "
            f"Fix the run config or point at the right artifact."
        )
    lineage = loaded_meta.get("lineage") or {}
    if lineage.get("parent"):
        pe = lineage.get("parent_eval")
        pe_s = f"{pe * 100:.1f}%" if isinstance(pe, (int, float)) else "unknown"
        print(f"[artifact_meta] {name} at {path}: lineage parent={lineage['parent']}  "
              f"parent_eval={pe_s}")


def validate_reference(ref: Any, wall: Any, *, path: Any) -> None:
    """Validate a loaded ``Reference``'s embedded metadata against the wall it
    is about to be used with. The check that would have caught the cell-size
    rescale bug: same ``wall_hash``/``cell_size_cm`` as the wall it was
    authored on."""
    validate(getattr(ref, "artifact_meta", None), wall_fingerprint(wall),
             name="reference", path=path)


def validate_landing_bank(bank_meta: Optional[dict], *, wall: Any = None, path: Any) -> None:
    current = wall_fingerprint(wall) if wall is not None else {}
    validate(bank_meta, current, name="landing bank", path=path)


def validate_checkpoint(model_path: Any, *, wall: Any = None, obs_dim: Optional[int] = None,
                        action_dim: Optional[int] = None,
                        env_mode: Optional[str] = None) -> Optional[dict]:
    """Load + validate a checkpoint's sidecar meta against the current run.
    Returns the loaded meta (or ``None`` if legacy) so callers can chain
    lineage (e.g. read the parent's own recorded eval)."""
    meta = read_checkpoint_meta(model_path)
    current: dict = {"obs_dim": obs_dim, "action_dim": action_dim, "env_mode": env_mode}
    if wall is not None:
        current.update(wall_fingerprint(wall))
    validate(meta, current, name="checkpoint", path=model_path)
    return meta


# ─── .npz embedding (Reference, landing banks) ───────────────────────────────

def npz_meta_value(artifact_meta: Optional[dict], user_meta: Optional[dict] = None) -> np.ndarray:
    """JSON-encode ``{artifact_meta, user_meta}`` as the 0-d object array
    ``np.savez`` expects for the npz ``meta`` key."""
    payload = {"artifact_meta": artifact_meta or {}, "user_meta": user_meta or {}}
    return np.array(json.dumps(payload, default=str), dtype=object)


def parse_npz_meta(raw: Any) -> tuple[Optional[dict], dict]:
    """Returns ``(artifact_meta_or_None, user_meta)``. Falls back to the
    legacy plain-``repr``'d-dict format (predates this module): returned as
    ``(None, that_dict)`` so callers see it as a legacy artifact."""
    if raw is None:
        return None, {}
    s = str(raw)
    try:
        payload = json.loads(s)
        if isinstance(payload, dict) and "artifact_meta" in payload:
            return payload.get("artifact_meta") or None, payload.get("user_meta") or {}
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        legacy = eval(s)  # noqa: S307 — our own repr, trusted (pre-metadata format)
        return None, legacy if isinstance(legacy, dict) else {}
    except Exception:  # noqa: BLE001
        return None, {}


# ─── checkpoint sidecar (<name>.meta.json) ───────────────────────────────────

def checkpoint_meta_path(zip_path: Any) -> Path:
    p = Path(zip_path)
    if p.suffix == ".zip":
        return p.with_name(p.stem + ".meta.json")
    return p.with_name(p.name + ".meta.json")


def write_checkpoint_meta(zip_path: Any, meta: dict) -> None:
    checkpoint_meta_path(zip_path).write_text(json.dumps(meta, indent=2, default=str))


def read_checkpoint_meta(zip_path: Any) -> Optional[dict]:
    p = checkpoint_meta_path(zip_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def parent_eval_of(zip_path: Optional[Any]) -> Optional[float]:
    """The frame-0 success the given checkpoint recorded for ITSELF at save
    time — used as ``lineage.parent_eval`` when a child artifact is saved."""
    if not zip_path:
        return None
    meta = read_checkpoint_meta(zip_path)
    if not meta:
        return None
    return (meta.get("eval") or {}).get("frame0_success")
