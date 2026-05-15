"""Tests for benchmark run_metadata helpers."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarking.run_metadata import BenchmarkRunContext, write_run_metadata


def _fake_freeze(output_dir: Path) -> dict:
    p = Path(output_dir) / "environment_freeze.txt"
    p.write_text("pkg==0\n", encoding="utf-8")
    return {"file": "environment_freeze.txt", "sha256": "aa" * 32, "error": None}


class RunMetadataHelpersTest(unittest.TestCase):
    def test_write_run_metadata_invocation_runtime_and_environment(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            repo = out / "repo"
            repo.mkdir()
            script = repo / "run_benchmark.py"
            script.write_text("", encoding="utf-8")
            (out / "preexisting.txt").write_text("x", encoding="utf-8")

            ctx = BenchmarkRunContext(
                argv=["--models", "logistic_regression"],
                args={"models": ["logistic_regression"]},
                config_path=None,
                repo_root=repo,
                data_dir=repo,
                output_dir=out,
                explainability={
                    "xai_enabled": False,
                    "xai_requested_by_cli": False,
                    "xai_enabled_in_config": False,
                    "xai_effective_source": "disabled",
                },
                script_path=script,
                cwd=repo,
                runtime={"started_at_utc": "2020-01-01T00:00:00+00:00", "stages": {"s1": 0.1}},
                run_start_perf=None,
                model_runtime_metadata={"tabnet": {"actual_device": "cpu:0"}},
            )
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "x", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={"root": str(repo), "combined_sha256": None, "files": {}},
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0"},
            ), mock.patch(
                "benchmarking.run_metadata.write_environment_freeze",
                side_effect=_fake_freeze,
            ), mock.patch(
                "benchmarking.run_metadata._hardware_metadata",
                return_value={
                    "cpu_count_logical": 2,
                    "gpus": [],
                    "warnings": [],
                },
            ):
                write_run_metadata(ctx)

            data = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("RunMetadataV1", data["schema_version"])
            inv = data["invocation"]
            self.assertEqual(str(repo.resolve()), inv["cwd"])
            self.assertIn(sys.executable, inv["command_tokens"])
            self.assertIn("--models", inv["command_tokens"])
            self.assertIn("command", inv)
            self.assertIn("hardware", data)
            self.assertIn("environment", data)
            self.assertEqual("environment_freeze.txt", data["environment"]["file"])
            self.assertEqual("tabnet", list(data["models_runtime"].keys())[0])
            rt = data["runtime"]
            self.assertEqual("2020-01-01T00:00:00+00:00", rt["started_at_utc"])
            self.assertIn("ended_at_utc", rt)
            self.assertNotIn("total_seconds", rt)
