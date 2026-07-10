"""Tests for sim3d.artifact_meta: embedded provenance + load-time validation
on the three artifact families (reference .npz, checkpoints, landing banks).

Run with:  python -m unittest tests.test_artifact_meta -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim3d import artifact_meta as am
from sim3d.reference import Reference
from solver.wall import load_wall

_WALL_PATH = Path(__file__).resolve().parent.parent / "data" / "examples" / "baby-v1.json"


def _tiny_reference(nq: int = 30, nv: int = 29, n_frames: int = 3) -> Reference:
    rng = np.random.default_rng(0)
    return Reference(
        qpos=rng.standard_normal((n_frames, nq)),
        qvel=rng.standard_normal((n_frames, nv)),
        eef=rng.standard_normal((n_frames, 4, 3)),
        com=rng.standard_normal((n_frames, 3)),
        grips=np.array([["h1", "h2", "", ""]] * n_frames, dtype="<U24"),
        wall_gen_seed=7,
    )


class ReferenceMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "ref.npz"
        self.wall = load_wall(str(_WALL_PATH))

    def test_roundtrip_embeds_wall_meta(self):
        ref = _tiny_reference()
        ref.save(self.out, wall=self.wall, env_mode="discover")
        loaded = Reference.load(self.out)
        self.assertIsNotNone(loaded.artifact_meta)
        self.assertEqual(loaded.artifact_meta["wall_id"], self.wall.wall_id)
        self.assertEqual(loaded.artifact_meta["cell_size_cm"], self.wall.cell_size_cm)
        self.assertEqual(loaded.artifact_meta["env_mode"], "discover")
        self.assertEqual(loaded.artifact_meta["artifact_type"], "reference")
        # user diagnostics (`meta`) survive alongside the artifact metadata.
        np.testing.assert_array_equal(loaded.qpos, ref.qpos)

    def test_validate_passes_for_matching_wall(self):
        ref = _tiny_reference()
        ref.save(self.out, wall=self.wall)
        loaded = Reference.load(self.out)
        am.validate_reference(loaded, self.wall, path=self.out)  # must not raise

    def test_cell_size_mismatch_raises(self):
        """The bug this module exists to catch: a reference authored on one
        cell size, loaded/continued against a wall resolved at a different
        cell size (e.g. the discover.py 20cm-default-vs-5cm-wall bug)."""
        ref = _tiny_reference()
        ref.save(self.out, wall=self.wall)
        loaded = Reference.load(self.out)
        mismatched_wall = load_wall(str(_WALL_PATH), cell_size_cm=self.wall.cell_size_cm * 4)
        with self.assertRaises(ValueError) as cm:
            am.validate_reference(loaded, mismatched_wall, path=self.out)
        self.assertIn("cell_size_cm", str(cm.exception))

    def test_legacy_artifact_warns_not_raises(self):
        """A pre-metadata npz (plain repr'd dict under `meta`, the old
        format) must still load, and validation must warn, not error."""
        ref = _tiny_reference()
        np.savez(
            self.out, qpos=ref.qpos, qvel=ref.qvel, eef=ref.eef, com=ref.com,
            grips=ref.grips, wall_gen_seed=ref.wall_gen_seed,
            meta=np.array(repr({"move_k": 3}), dtype=object),
        )
        loaded = Reference.load(self.out)
        self.assertIsNone(loaded.artifact_meta)
        self.assertEqual(loaded.meta, {"move_k": 3})
        am.validate_reference(loaded, self.wall, path=self.out)  # warns, does not raise

    def test_no_meta_key_at_all_loads(self):
        """An even older npz with no `meta` key whatsoever."""
        ref = _tiny_reference()
        np.savez(self.out, qpos=ref.qpos, qvel=ref.qvel, eef=ref.eef, com=ref.com,
                 grips=ref.grips, wall_gen_seed=ref.wall_gen_seed)
        loaded = Reference.load(self.out)
        self.assertIsNone(loaded.artifact_meta)
        self.assertEqual(loaded.meta, {})


class CheckpointMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.zip_path = Path(self._tmp.name) / "model.zip"
        self.zip_path.write_bytes(b"")  # sidecar path derivation only; not a real zip
        self.wall = load_wall(str(_WALL_PATH))

    def test_sidecar_path_naming(self):
        self.assertEqual(am.checkpoint_meta_path(self.zip_path).name, "model.meta.json")

    def test_roundtrip_and_matching_validate(self):
        meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                             obs_dim=131, action_dim=27, env_mode="imitate:milestone")
        am.write_checkpoint_meta(self.zip_path, meta)
        loaded = am.read_checkpoint_meta(self.zip_path)
        self.assertEqual(loaded["obs_dim"], 131)
        am.validate_checkpoint(self.zip_path, wall=self.wall, obs_dim=131, action_dim=27,
                               env_mode="imitate:milestone")  # must not raise

    def test_obs_dim_mismatch_raises(self):
        meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                             obs_dim=131, action_dim=27, env_mode="imitate:milestone")
        am.write_checkpoint_meta(self.zip_path, meta)
        with self.assertRaises(ValueError) as cm:
            am.validate_checkpoint(self.zip_path, wall=self.wall, obs_dim=132, action_dim=27,
                                   env_mode="imitate:milestone")
        self.assertIn("obs_dim", str(cm.exception))

    def test_env_mode_mismatch_raises(self):
        meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                             obs_dim=131, action_dim=27, env_mode="imitate:milestone")
        am.write_checkpoint_meta(self.zip_path, meta)
        with self.assertRaises(ValueError):
            am.validate_checkpoint(self.zip_path, wall=self.wall, obs_dim=131, action_dim=27,
                                   env_mode="imitate:dense")

    def test_env_mode_mismatch_warns_when_not_strict(self):
        """Warm-start path: an env_mode mismatch is a warning, not an error, when
        strict_env_mode=False (transferring a parent from a different obs/reward
        regime is intentional; e.g. milestone -> milestone+postureobs)."""
        meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                             obs_dim=131, action_dim=27, env_mode="imitate:milestone")
        am.write_checkpoint_meta(self.zip_path, meta)
        # must NOT raise
        am.validate_checkpoint(self.zip_path, wall=self.wall, obs_dim=131, action_dim=27,
                               env_mode="imitate:milestone+postureobs", strict_env_mode=False)

    def test_non_env_mode_field_still_hard_errors_when_not_strict(self):
        """strict_env_mode=False relaxes ONLY env_mode — a dim/wall mismatch still
        hard-errors (those would silently corrupt a weight load)."""
        meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                             obs_dim=131, action_dim=27, env_mode="imitate:milestone")
        am.write_checkpoint_meta(self.zip_path, meta)
        with self.assertRaises(ValueError) as cm:
            am.validate_checkpoint(self.zip_path, wall=self.wall, obs_dim=132, action_dim=27,
                                   env_mode="imitate:milestone+postureobs", strict_env_mode=False)
        self.assertIn("obs_dim", str(cm.exception))

    def test_lineage_chain(self):
        parent_meta = am.build_meta(artifact_type="checkpoint", wall=self.wall,
                                    obs_dim=131, action_dim=27, env_mode="imitate:dense",
                                    extra={"eval": {"frame0_success": 0.85, "n_episodes": 40}})
        am.write_checkpoint_meta(self.zip_path, parent_meta)
        self.assertAlmostEqual(am.parent_eval_of(self.zip_path), 0.85)

    def test_missing_sidecar_is_legacy_warning(self):
        no_sidecar = Path(self._tmp.name) / "legacy_model.zip"
        no_sidecar.write_bytes(b"")
        am.validate_checkpoint(no_sidecar, wall=self.wall, obs_dim=131,
                               action_dim=27)  # warns, does not raise
        self.assertIsNone(am.parent_eval_of(no_sidecar))


class LandingBankMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "bank.npz"
        self.wall = load_wall(str(_WALL_PATH))

    def _save_bank(self, wall) -> None:
        meta = am.build_meta(artifact_type="landing_bank", wall=wall, env_mode="imitate:milestone",
                             extra={"focus_stance": 2})
        np.savez(self.out, qpos=np.zeros((2, 30)), qvel=np.zeros((2, 29)),
                 grips=np.array([["h1", "", "", ""]] * 2, dtype="<U24"), focus_stance=2,
                 meta=am.npz_meta_value(meta))

    def test_roundtrip_and_validate(self):
        self._save_bank(self.wall)
        d = np.load(self.out, allow_pickle=True)
        bank_meta, _user = am.parse_npz_meta(d["meta"])
        self.assertEqual(bank_meta["artifact_type"], "landing_bank")
        am.validate_landing_bank(bank_meta, wall=self.wall, path=self.out)  # must not raise

    def test_wall_hash_mismatch_raises(self):
        self._save_bank(self.wall)
        d = np.load(self.out, allow_pickle=True)
        bank_meta, _user = am.parse_npz_meta(d["meta"])
        mismatched_wall = load_wall(str(_WALL_PATH), cell_size_cm=self.wall.cell_size_cm * 4)
        with self.assertRaises(ValueError):
            am.validate_landing_bank(bank_meta, wall=mismatched_wall, path=self.out)


if __name__ == "__main__":
    unittest.main()
