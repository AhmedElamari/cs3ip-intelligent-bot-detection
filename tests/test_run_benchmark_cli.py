import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark


class _BenchmarkStub:
    def __init__(self, *args, **kwargs):
        self.results = {}

    def run_benchmark(self, *args, **kwargs):
        return {}

    def print_summary(self):
        return None

    def get_best_model(self, metric):
        return "stub_model", object(), {"f1": 1.0, "roc_auc": 1.0}


class RunBenchmarkCliTest(unittest.TestCase):
    def _write_config(self, directory: Path) -> Path:
        config_path = directory / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "explainability": {"enabled": True},
                    "robustness": {"enabled": True},
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def _run_main(self, *extra_args: str):
        with TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "results"
            argv = [
                "run_benchmark.py",
                "--output",
                str(output_root),
                "--models",
                "logistic_regression",
                "--config",
                str(self._write_config(tmp_path)),
                *extra_args,
            ]

            def _fake_prepare(*args: Any, **kwargs: Any):
                return (
                    np.zeros((4, 2)),
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.array([0, 1, 0, 1]),
                    np.array([0, 1]),
                    np.array([0, 1]),
                    ["a", "b"],
                )

            def _fake_resolve(model_name: str, config: Any, **kw: Any):
                return (
                    {
                        "schema_version": "HPOResultV1",
                        "status": "skipped",
                        "best_params": {},
                        "best_score": float("nan"),
                        "trial_count": 0,
                        "metric": "val_f1",
                        "seed": 2112,
                        "warnings": [],
                        "model_name": model_name,
                        "search_space_version": "none",
                    },
                    {
                        "cache_hit": False,
                        "artifact": None,
                        "search_space_version": None,
                        "trial_count": 0,
                        "best_score": None,
                        "skipped": True,
                    },
                )

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_benchmark,
                "load_data",
                return_value={"train": object(), "val": object(), "test": object()},
            ), mock.patch.object(
                run_benchmark,
                "prepare_data",
                side_effect=_fake_prepare,
            ), mock.patch.object(
                run_benchmark,
                "resolve_hpo",
                side_effect=_fake_resolve,
            ), mock.patch.object(
                run_benchmark,
                "merge_hpo_into_config_params",
            ), mock.patch.object(
                run_benchmark,
                "create_models",
                return_value={},
            ), mock.patch.object(
                run_benchmark,
                "ModelBenchmark",
                return_value=_BenchmarkStub(),
            ), mock.patch.object(
                run_benchmark,
                "save_final_outputs",
            ) as save_final_outputs, mock.patch.object(
                run_benchmark,
                "run_explainability_analysis",
            ) as run_explainability_analysis, mock.patch.object(
                run_benchmark,
                "run_robustness_analysis",
            ) as run_robustness_analysis:
                run_benchmark.main()

            return {
                "final_called": save_final_outputs.called,
                "xai_called": run_explainability_analysis.called,
                "robustness_called": run_robustness_analysis.called,
                "context": save_final_outputs.call_args.args[3],
            }

    def test_dissertation_core_keeps_final_outputs_and_skips_slow_extras(self):
        result = self._run_main("--dissertation-core")

        self.assertTrue(result["final_called"])
        self.assertFalse(result["xai_called"])
        self.assertFalse(result["robustness_called"])
        self.assertTrue(result["context"].args["dissertation_core"])
        self.assertEqual(
            "dissertation_core",
            result["context"].explainability["xai_effective_source"],
        )

    def test_scoreboard_only_alias_maps_to_dissertation_core_behavior(self):
        result = self._run_main("--scoreboard-only")

        self.assertTrue(result["final_called"])
        self.assertFalse(result["xai_called"])
        self.assertFalse(result["robustness_called"])
        self.assertTrue(result["context"].args["dissertation_core"])
        self.assertEqual(
            "dissertation_core",
            result["context"].explainability["xai_effective_source"],
        )


if __name__ == "__main__":
    unittest.main()
