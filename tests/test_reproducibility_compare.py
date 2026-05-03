import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_COMPARE_SCRIPT = ROOT / "tasks" / "reproducibility_compare.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("reproducibility_compare", _COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {_COMPARE_SCRIPT}")
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    _COMPARE_SCRIPT.is_file(),
    f"Optional helper missing: {_COMPARE_SCRIPT}",
)
class ReproducibilityCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._compare = _load_compare_module()

    def _write_run_artifacts(
        self,
        directory: Path,
        *,
        f1: float,
        training_time: float,
        experiment_name: str,
        output_dir: str,
    ) -> None:
        (directory / "config.json").write_text(
            json.dumps({"random_state": 2112}, indent=2),
            encoding="utf-8",
        )
        (directory / "model_comparison.csv").write_text(
            f"Model,Training Time (s),F1\nrandom_forest,{training_time:.2f},{f1:.4f}\n",
            encoding="utf-8",
        )
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "experiment_name": experiment_name,
                    "models": {
                        "random_forest": {
                            "training_time": training_time,
                            "test_metrics": {"f1": f1},
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "RunMetadataV1",
                    "python": {"version": "3.11.0"},
                    "platform": {"platform": "Windows"},
                    "git": {"commit": "abc123", "branch": "main", "dirty": False},
                    "invocation": {"xai_effective_source": "config"},
                    "dataset": {"combined_sha256": "dataset-hash"},
                    "packages": {"numpy": "1.0.0"},
                    "artifacts": {
                        "output_dir": output_dir,
                        "files": [
                            "config.json",
                            "model_comparison.csv",
                            "results.json",
                            "run_metadata.json",
                        ],
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _build_run_pair(self, *, f1_a: float, f1_b: float, training_time_a: float = 1.23, training_time_b: float = 1.23):
        tmp_a = TemporaryDirectory(dir=ROOT)
        tmp_b = TemporaryDirectory(dir=ROOT)
        self.addCleanup(tmp_a.cleanup)
        self.addCleanup(tmp_b.cleanup)
        run_a = Path(tmp_a.name)
        run_b = Path(tmp_b.name)
        self._write_run_artifacts(
            run_a,
            f1=f1_a,
            training_time=training_time_a,
            experiment_name="benchmark_a",
            output_dir=str(run_a),
        )
        self._write_run_artifacts(
            run_b,
            f1=f1_b,
            training_time=training_time_b,
            experiment_name="benchmark_b",
            output_dir=str(run_b),
        )
        return run_a, run_b

    def test_compare_runs_ignores_normalized_fields(self):
        run_a, run_b = self._build_run_pair(
            f1_a=0.9123,
            f1_b=0.9123,
            training_time_a=1.23,
            training_time_b=9.87,
        )
        matches, differences = self._compare.compare_runs(run_a, run_b)

        self.assertTrue(matches)
        self.assertEqual([], differences)

    def test_compare_runs_reports_real_metric_differences(self):
        run_a, run_b = self._build_run_pair(f1_a=0.9123, f1_b=0.5000)
        matches, differences = self._compare.compare_runs(run_a, run_b)

        self.assertFalse(matches)
        self.assertTrue(any("model_comparison.csv" in difference or "results.json" in difference for difference in differences))

    def test_compare_runs_ignores_plot_files(self):
        run_a, run_b = self._build_run_pair(f1_a=0.9123, f1_b=0.9123)
        (run_a / "pr_curves_comparison.png").write_bytes(b"plot-a")
        (run_b / "pr_curves_comparison.png").write_bytes(b"plot-b")
        matches, differences = self._compare.compare_runs(run_a, run_b)

        self.assertTrue(matches)
        self.assertEqual([], differences)


if __name__ == "__main__":
    unittest.main()
