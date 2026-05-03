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

            import pandas as pd

            def _fake_prepare(*args: Any, **kwargs: Any):
                meta = pd.DataFrame({"user_id": ["u"], "row_index": [0], "label": [0]})
                return (
                    np.zeros((4, 2)),
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.array([0, 1, 0, 1]),
                    np.array([0, 1]),
                    np.array([0, 1]),
                    ["a", "b"],
                    meta,
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
                "final_kwargs": save_final_outputs.call_args.kwargs,
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

    def test_threshold_analysis_flag_reaches_final_outputs(self):
        result = self._run_main("--threshold-analysis", "--threshold-precision-floor", "0.82")

        self.assertTrue(result["final_called"])
        self.assertTrue(result["final_kwargs"]["threshold_analysis_enabled"])
        self.assertEqual(0.82, result["final_kwargs"]["threshold_precision_floor"])

    def test_time_stratified_flag_sets_concept_drift_enabled_and_passes_benchmark(self):
        wrote = {"drift": None}

        def capture_save_final_outputs(bm, od, cfg, ctx, **kwargs):
            wrote["drift"] = kwargs.get("drift_benchmark")

        import pandas as pd

        def _fake_prepare(*args: Any, **kwargs: Any):
            meta = pd.DataFrame({"user_id": ["u"], "row_index": [0], "label": [0]})
            base = (
                np.zeros((4, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.array([0, 1, 0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                ["a", "b"],
            )
            if kwargs.get("return_metadata"):
                return (*base, meta)
            return base

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

        with TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            argv = [
                "run_benchmark.py",
                "--output",
                str(tmp_path / "results"),
                "--models",
                "logistic_regression",
                "--config",
                str(self._write_config(tmp_path)),
                "--time-stratified-results",
            ]

            class _StubWithDrift(_BenchmarkStub):
                def run_benchmark(self, *args, **kwargs):
                    return {}
                def print_summary(self):
                    return None
                def set_test_metadata(self, *_a, **_k):
                    return None

            calls = {"n": 0}

            def factory(*a, **k):
                calls["n"] += 1
                return _StubWithDrift()

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
                "build_temporal_split_dict",
                return_value={"train": object(), "val": object(), "test": object()},
            ), mock.patch.object(
                run_benchmark,
                "format_protocol_note",
                return_value="drift-protocol-note",
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
                side_effect=factory,
            ), mock.patch.object(
                run_benchmark,
                "save_final_outputs",
                side_effect=capture_save_final_outputs,
            ), mock.patch.object(
                run_benchmark,
                "run_explainability_analysis",
            ), mock.patch.object(
                run_benchmark,
                "run_robustness_analysis",
            ):
                run_benchmark.main()

            self.assertEqual(calls["n"], 2)
            self.assertIsNotNone(wrote["drift"])


if __name__ == "__main__":
    unittest.main()
