"""Offline contract checks; these do not claim full-model numerical validation."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from nap_llm.backends import backend_root
from nap_llm.common import Job, TASKS, atomic_json, stable_hash, stage_done, stage_record

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_python310_syntax(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(ROOT)),
                      feature_version=(3, 10))

    def test_job_environment_expansion(self):
        with patch.dict(os.environ, {"MODEL_PATH": "/path/to/test-model"}):
            with (ROOT / "configs/weight_only.example.csv").open(newline="") as handle:
                jobs = [Job.from_row(row) for row in csv.DictReader(handle)]
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(job.model_path == "/path/to/test-model" for job in jobs))
        self.assertTrue(all(job.bits == 3 and job.group_size == 128 for job in jobs))
        self.assertEqual(len({job.job_id for job in jobs}), 4)

    def test_fingerprint_stable(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))
        self.assertNotEqual(stable_hash({"a": 1}), stable_hash({"a": 2}))

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "nested/state.json"
            atomic_json(target, {"complete": False})
            atomic_json(target, {"complete": True})
            self.assertEqual(json.loads(target.read_text()), {"complete": True})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_stage_failure_not_done(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                with stage_record(root, "test"):
                    raise RuntimeError("expected failure")
            self.assertFalse(stage_done(root, "test"))

    def test_stage_success_done(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with stage_record(root, "test"):
                pass
            self.assertTrue(stage_done(root, "test"))

    def test_backend_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SQ_RESCOMP_ROOT": temp}):
            self.assertEqual(backend_root("rescomp"), Path(temp).resolve())

    def test_backend_missing_fails(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SQ_RESCOMP_ROOT": ""}):
            with self.assertRaisesRegex(FileNotFoundError, "SQ_RESCOMP_ROOT"):
                backend_root("rescomp", Path(temp))

    def test_backend_unknown_fails(self):
        with self.assertRaises(ValueError):
            backend_root("unregistered")

    def test_default_qep_path(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SQ_QEP_ROOT": ""}):
            expected = Path(temp) / "third_party/qep"
            expected.mkdir(parents=True)
            self.assertEqual(backend_root("qep", Path(temp)), expected.resolve())

    def test_partial_avg6_is_not_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = root / "jobs.csv"
            matrix.write_text(
                "job_id,model_key,model_path,method,bits,run_nap\n"
                "test,model,/path/to/model,rtn,3,0\n", encoding="utf-8")
            result = root / "test/results/baseline"
            result.mkdir(parents=True)
            for task in TASKS[:-1]:
                atomic_json(result / f"task_{task}.json", {"value": 0.5})
            cmd = [sys.executable, str(ROOT / "summarize.py"), "--matrix", str(matrix),
                   "--runs-root", str(root)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            with (root / "summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["avg"], "")
            self.assertEqual(row["complete"], "False")
            atomic_json(result / f"task_{TASKS[-1]}.json", {"value": 0.5})
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            with (root / "summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["avg"]), 0.5)

    def test_order_source_path_is_explicit(self):
        tree = ast.parse((ROOT / "run_order_w3a16.py").read_text())
        constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
        self.assertIn("--source-run", constants)
        self.assertNotIn("--paper-runs-root", constants)
        self.assertIn("save_preconditioned_hf", [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)])

    def test_model_selection_freezes_before_affine(self):
        # Static contract check only; accelerator behavior needs integration tests.
        tree = ast.parse((ROOT / "nap_llm/modeling.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "select_norm_parameters")
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "requires_grad_"]
        self.assertEqual({n.args[0].value for n in calls}, {False, True})

    def test_no_vendored_backend(self):
        self.assertFalse((ROOT / "third_party").exists())
        self.assertFalse((ROOT / "monkeypatch.py").exists())

    def test_png_has_no_ancillary_metadata(self):
        data = (ROOT / "assets/framework.png").read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        offset = 8
        types = []
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            payload = data[offset+8:offset+8+length]
            crc = struct.unpack(">I", data[offset+8+length:offset+12+length])[0]
            self.assertEqual(zlib.crc32(kind + payload) & 0xffffffff, crc)
            types.append(kind)
            offset += length + 12
        self.assertEqual(offset, len(data))
        self.assertTrue(set(types) <= {b"IHDR", b"IDAT", b"IEND", b"PLTE", b"tRNS"})
        self.assertEqual(types[-1], b"IEND")

    def test_readme_local_links_exist(self):
        import re
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", readme):
            if "://" not in target and not target.startswith("#"):
                self.assertTrue((ROOT / target).is_file(), target)
        self.assertIn('src="assets/framework.png"', readme)

    def test_shell_files_are_lf(self):
        for path in (ROOT / "scripts").glob("*.sh"):
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertTrue(path.read_bytes().startswith(b"#!/usr/bin/env bash\n"))


if __name__ == "__main__":
    unittest.main()
