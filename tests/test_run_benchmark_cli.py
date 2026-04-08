import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest import mock

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


class RunBenchmarkCliContractTest(unittest.TestCase):
    def _write_config(self, directory: Path, enabled: bool) -> Path:
        config_path = directory / "config.json"
        config_path.write_text(
            json.dumps({"explainability": {"enabled": enabled}}),
            encoding="utf-8",
        )
        return config_path

    def _run_main(self, *, explain_flag: bool, config_enabled: Optional[bool]):
        with TemporaryDirectory(dir=ROOT) as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "results"
            argv = [
                "run_benchmark.py",
                "--output",
                str(output_root),
                "--models",
                "logistic_regression",
            ]
            if explain_flag:
                argv.append("--explain")
            if config_enabled is not None:
                argv.extend(["--config", str(self._write_config(tmp_path, config_enabled))])

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_benchmark,
                "load_data",
                return_value={"train": object(), "val": object(), "test": object()},
            ), mock.patch.object(
                run_benchmark,
                "prepare_data",
                return_value=([], [], [], [], [], [], []),
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
                "save_comparison_outputs",
            ), mock.patch.object(
                run_benchmark,
                "save_final_outputs",
            ) as save_final_outputs, mock.patch.object(
                run_benchmark,
                "run_explainability_analysis",
            ) as run_explainability_analysis, mock.patch.object(
                run_benchmark,
                "run_robustness_analysis",
            ):
                run_benchmark.main()

            return {
                "xai_called": run_explainability_analysis.called,
                "context": save_final_outputs.call_args.args[3],
            }

    def _assert_explainability_result(
        self,
        result,
        *,
        xai_called: bool,
        xai_enabled: bool,
        effective_source: str,
    ) -> None:
        self.assertEqual(xai_called, result["xai_called"])
        self.assertEqual(xai_enabled, result["context"].explainability["xai_enabled"])
        self.assertEqual(effective_source, result["context"].explainability["xai_effective_source"])

    def test_cli_only_explainability_is_enabled(self):
        result = self._run_main(explain_flag=True, config_enabled=False)
        self._assert_explainability_result(
            result,
            xai_called=True,
            xai_enabled=True,
            effective_source="cli",
        )

    def test_config_only_explainability_is_enabled(self):
        result = self._run_main(explain_flag=False, config_enabled=True)
        self._assert_explainability_result(
            result,
            xai_called=True,
            xai_enabled=True,
            effective_source="config",
        )

    def test_cli_and_config_explainability_records_combined_source(self):
        result = self._run_main(explain_flag=True, config_enabled=True)
        self._assert_explainability_result(
            result,
            xai_called=True,
            xai_enabled=True,
            effective_source="cli+config",
        )

    def test_explicitly_disabled_explainability_skips_xai(self):
        result = self._run_main(explain_flag=False, config_enabled=False)
        self._assert_explainability_result(
            result,
            xai_called=False,
            xai_enabled=False,
            effective_source="disabled",
        )


if __name__ == "__main__":
    unittest.main()
