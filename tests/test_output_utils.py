"""Tests for benchmark output utility helpers."""

import ast
import io
import importlib
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from config import Config
from benchmarking import ModelBenchmark
from benchmarking.run_metadata import BenchmarkRunContext

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "tests" / "test_output_utils.py"


def _output_utils():
    return importlib.import_module("benchmarking.output_utils")


def _poster_figures():
    return importlib.import_module("benchmarking.poster_figures")


def _make_benchmark(specs, experiment_name: str, seed: int = 2112, *, interpretable=None) -> ModelBenchmark:
    rng = np.random.default_rng(seed)
    y_test = np.concatenate([np.zeros(12), np.ones(12)]).astype(int)
    b = ModelBenchmark(models={}, experiment_name=experiment_name)
    b.y_test = y_test
    if interpretable is None:
        interpretable = lambda _name: True
    b.results = {}
    b.predictions = {}
    b.probabilities = {}
    for name, f1m, roc, pr_auc in specs:
        y_pred = rng.integers(0, 2, size=len(y_test))
        proba = rng.random((len(y_test), 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        b.predictions[name] = y_pred
        b.probabilities[name] = proba
        tm = b.metrics_calculator.compute_all_metrics(y_test, y_pred, proba)
        tm["f1_macro"] = f1m
        tm["roc_auc"] = roc
        tm["pr_auc"] = pr_auc
        b.results[name] = {
            "model": None,
            "training_time": 1.0,
            "val_metrics": {},
            "test_metrics": tm,
            "feature_importance": None,
            "is_interpretable": interpretable(name),
            "X_train": None,
            "X_val": None,
            "X_test": None,
            "feature_names": [],
            "scaler": None,
        }
    return b


def _make_three_model_benchmark() -> ModelBenchmark:
    """Minimal benchmark state so dissertation PR + CM figures can render."""
    return _make_benchmark(
        [
            ("m_a", 0.92, 0.81, 0.79),
            ("m_b", 0.88, 0.82, 0.78),
            ("m_c", 0.84, 0.83, 0.77),
        ],
        "stub_three",
    )


def _make_one_model_benchmark() -> ModelBenchmark:
    return _make_benchmark([("solo", 0.9, 0.85, 0.8)], "stub_one", seed=42)


def _make_named_best_model_benchmark(best_name: str = "xgboost") -> ModelBenchmark:
    return _make_benchmark(
        [(best_name, 0.91, 0.86, 0.84), ("decision_tree", 0.88, 0.82, 0.8)],
        "stub_named",
        interpretable=lambda name: name == "decision_tree",
    )


def _degradation_rows(
    models,
    scenario_values=(("baseline", 0.9), ("cheap_only", 0.85), ("realistic_mixed", 0.8)),
    pr_auc=0.7,
):
    return [
        {"model": model, "scenario": scenario, "macro_f1": f1, "pr_auc": pr_auc}
        for model in models
        for scenario, f1 in scenario_values
    ]


def _robustness_summary_rows(models):
    rows = []
    for model in models:
        rows.extend([
            {
                "model": model,
                "profile": "cheap_only",
                "attacked_true_bots": 120,
                "baseline_detected_bots": 118,
                "flip_rate": 0.0339,
                "confidence_drop_mean": 0.0711,
                "confidence_drop_median": 0.0680,
                "confidence_drop_std": 0.0500,
                "confidence_drop_non_flip_mean": 0.0692,
                "attacked_bot_recall_baseline": 0.9833,
                "attacked_bot_recall": 0.9500,
                "attacked_bot_recall_delta": -0.0333,
                "attacked_bot_mean_probability_baseline": 0.8400,
                "attacked_bot_mean_probability": 0.7700,
                "attacked_bot_mean_probability_delta": -0.0700,
            },
            {
                "model": model,
                "profile": "realistic_mixed",
                "attacked_true_bots": 120,
                "baseline_detected_bots": 118,
                "flip_rate": 0.0424,
                "confidence_drop_mean": 0.0752,
                "confidence_drop_median": 0.0710,
                "confidence_drop_std": 0.0520,
                "confidence_drop_non_flip_mean": 0.0734,
                "attacked_bot_recall_baseline": 0.9833,
                "attacked_bot_recall": 0.9417,
                "attacked_bot_recall_delta": -0.0416,
                "attacked_bot_mean_probability_baseline": 0.8400,
                "attacked_bot_mean_probability": 0.7600,
                "attacked_bot_mean_probability_delta": -0.0800,
            },
        ])
    return rows


def _with_robustness_data(
    bench: ModelBenchmark,
    models,
    scenario_values=(("baseline", 0.9), ("cheap_only", 0.85), ("realistic_mixed", 0.8)),
):
    import pandas as pd

    bench.robustness_degradation = pd.DataFrame(_degradation_rows(models, scenario_values))
    bench.robustness_summary = pd.DataFrame(_robustness_summary_rows(models))
    return bench


class _FinalOutputStub:
    results = {}

    def __init__(self, save_exc=None, report="report", write_results=False):
        self._save_exc = save_exc
        self._report = report
        self._write_results = write_results

    def plot_pr_curves_top(self, top_n=3):
        return None

    def plot_best_confusion_matrix(self, normalize="true"):
        return None

    def save_results(self, output_dir):
        if self._save_exc:
            raise self._save_exc
        if not self._write_results:
            return
        (Path(output_dir) / "model_comparison.csv").write_text("Model,F1\nstub,1.0\n", encoding="utf-8")
        (Path(output_dir) / "results.json").write_text("{}", encoding="utf-8")

    def generate_report(self):
        return self._report


class _FinalOutputStubWithScoreboard(_FinalOutputStub):
    """Stub with populated ``results`` so dissertation scoreboard files are emitted."""

    results = {
        "logistic_regression": {
            "training_time": 1.234,
            "test_metrics": {
                "precision": 0.7,
                "recall": 0.8,
                "f1_macro": 0.75,
                "f1_weighted": 0.76,
                "pr_auc": 0.77,
                "roc_auc": 0.78,
                "mcc": 0.12,
                "balanced_accuracy": 0.79,
            },
        }
    }


class OutputUtilsTest(unittest.TestCase):
    def _caption_case(
        self,
        *,
        attacked_recall=(0.8, 0.72),
        f1=(0.7, 0.65),
        pr=(0.7, 0.65),
        scenarios=("baseline", "cheap_only"),
        models=("m_a",),
    ):
        return _poster_figures().degradation_caption(
            list(models),
            scenarios,
            np.array([f1]),
            np.array([pr]),
            [],
            np.array([attacked_recall]),
        )

    def _assert_attack_targeted_caption(self, caption: str) -> None:
        self.assertIn("Attacked true-bot recall", caption)
        self.assertIn("Full-test Macro-F1", caption)

    def test_module_avoids_top_level_optional_plot_imports(self):
        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        top_level = {
            getattr(node, "module", None)
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("benchmarking.output_utils", top_level)
        self.assertNotIn("benchmarking.poster_figures", top_level)

    def test_benchmarking_uses_headless_matplotlib_backend(self):
        import matplotlib

        self.assertEqual("agg", matplotlib.get_backend().lower())

    def _build_run_context(self, output_dir: Path) -> BenchmarkRunContext:
        return BenchmarkRunContext(
            argv=["--models", "logistic_regression"],
            args={"models": ["logistic_regression"], "explain": False, "output": str(output_dir)},
            config_path=None,
            repo_root=output_dir,
            data_dir=output_dir,
            output_dir=output_dir,
            explainability={
                "xai_enabled": True,
                "xai_requested_by_cli": False,
                "xai_enabled_in_config": True,
                "xai_effective_source": "config",
            },
        )

    def test_save_final_outputs_propagates_required_export_failures(self):
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory() as tmp:
            config = Config()
            with self.assertRaises(OSError):
                save_final_outputs(
                    _FinalOutputStub(save_exc=OSError("disk full")),
                    Path(tmp),
                    config,
                    self._build_run_context(Path(tmp)),
                )

    def test_save_final_outputs_writes_markdown_report(self):
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory() as tmp:
            config = Config()
            output_dir = Path(tmp)
            save_final_outputs(
                _FinalOutputStub(report="# report"),
                output_dir,
                config,
                self._build_run_context(output_dir),
            )
            self.assertTrue((output_dir / "benchmark_report.md").exists())
            self.assertFalse((output_dir / "benchmark_report.txt").exists())
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())

    def test_save_final_outputs_writes_run_metadata_contract(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            benchmark = _make_three_model_benchmark()
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={
                    "root": str(output_dir),
                    "combined_sha256": "dataset-hash",
                    "files": {},
                },
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    benchmark,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                )

            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("RunMetadataV1", metadata["schema_version"])
            self.assertEqual("config", metadata["invocation"]["xai_effective_source"])
            self.assertEqual("dataset-hash", metadata["dataset"]["combined_sha256"])
            self.assertEqual("1.0.0", metadata["packages"]["numpy"])
            self.assertEqual("abc123", metadata["git"]["commit"])
            self.assertIn("run_metadata.json", metadata["artifacts"]["files"])
            self.assertIn("config.json", metadata["artifacts"]["files"])
            self.assertIn("model_comparison.csv", metadata["artifacts"]["files"])
            self.assertIn("pr_curves_comparison.png", metadata["artifacts"]["files"])
            self.assertIn("confusion_matrix_best_model_normalized.png", metadata["artifacts"]["files"])
            self.assertIn("confusion_matrix_best_model_raw.png", metadata["artifacts"]["files"])

    def test_save_final_outputs_writes_dissertation_scoreboard(self):
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory() as tmp:
            config = Config()
            output_dir = Path(tmp)
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={
                    "root": str(output_dir),
                    "combined_sha256": "dataset-hash",
                    "files": {},
                },
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    _FinalOutputStubWithScoreboard(report="# report", write_results=True),
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                )

            self.assertTrue((output_dir / "dissertation_scoreboard.csv").exists())
            self.assertTrue((output_dir / "dissertation_scoreboard.md").exists())
            self.assertTrue((output_dir / "dissertation_scoreboard.tex").exists())
            csv_text = (output_dir / "dissertation_scoreboard.csv").read_text(encoding="utf-8")
            tex_text = (output_dir / "dissertation_scoreboard.tex").read_text(encoding="utf-8")
            self.assertIn("logistic_regression", csv_text)
            self.assertIn("PR-AUC", csv_text)
            self.assertIn(r"\begin{tabular}{@{}rlrrrrrrrrr@{}}", tex_text)
            self.assertNotIn(r"\begin{tabular}{@rl", tex_text)

    def test_save_final_outputs_skips_threshold_analysis_by_default(self):
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            benchmark = _make_three_model_benchmark()
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={"root": str(output_dir), "combined_sha256": "dataset-hash", "files": {}},
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    benchmark,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                )

            self.assertFalse((output_dir / "threshold_analysis.csv").exists())
            self.assertFalse((output_dir / "threshold_analysis.md").exists())

    def test_save_final_outputs_writes_threshold_analysis_when_enabled(self):
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            benchmark = _make_three_model_benchmark()
            benchmark.y_val = benchmark.y_test.copy()
            benchmark.validation_probabilities = {
                name: proba.copy()
                for name, proba in benchmark.probabilities.items()
            }
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={"root": str(output_dir), "combined_sha256": "dataset-hash", "files": {}},
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    benchmark,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                    threshold_analysis_enabled=True,
                    threshold_precision_floor=0.8,
                )

            csv_text = (output_dir / "threshold_analysis.csv").read_text(encoding="utf-8")
            md_text = (output_dir / "threshold_analysis.md").read_text(encoding="utf-8")
            self.assertIn("precision_floor_0.80", csv_text)
            self.assertIn("validation-selected threshold", md_text)

    def test_save_final_outputs_writes_concept_drift_artifacts_when_enabled(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            main_b = _make_three_model_benchmark()
            drift_b = _make_three_model_benchmark()
            for res in drift_b.results.values():
                tm = dict(res["test_metrics"])
                for k in ("f1_macro", "roc_auc", "pr_auc"):
                    if k in tm and tm[k] is not None:
                        tm[k] = float(tm[k]) - 0.05
                res["test_metrics"] = tm
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={"root": str(output_dir), "combined_sha256": "dataset-hash", "files": {}},
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    main_b,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                    drift_benchmark=drift_b,
                    drift_protocol_note="Protocol line for test.",
                )

            self.assertTrue((output_dir / "time_stratified_scoreboard.csv").exists())
            md = (output_dir / "time_stratified_scoreboard.md").read_text(encoding="utf-8")
            self.assertIn("Protocol line for test.", md)
            self.assertTrue((output_dir / "concept_drift_delta.csv").exists())
            delta_md = (output_dir / "concept_drift_delta.md").read_text(encoding="utf-8")
            self.assertIn("Δ", delta_md)

    def test_save_dissertation_statistical_summary_writes_ci_tables(self):
        save_dissertation_statistical_summary = _output_utils().save_dissertation_statistical_summary
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            benchmark = _make_three_model_benchmark()
            benchmark.confidence_intervals = {
                "m_a": {
                    "f1_macro": {"lower": 0.90, "point": 0.92, "upper": 0.94},
                    "pr_auc": {"lower": 0.75, "point": 0.79, "upper": 0.82},
                },
                "m_b": {
                    "f1_macro": {"lower": 0.85, "point": 0.88, "upper": 0.90},
                    "pr_auc": {"lower": 0.76, "point": 0.78, "upper": 0.81},
                },
            }

            save_dissertation_statistical_summary(benchmark, output_dir)

            csv_path = output_dir / "dissertation_statistical_summary.csv"
            md_path = output_dir / "dissertation_statistical_summary.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            csv_text = csv_path.read_text(encoding="utf-8")
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("F1-Macro Point", csv_text)
            self.assertIn("PR-AUC CI Lower", csv_text)
            self.assertIn("**0.920 [0.900, 0.940]**", md_text)

    def test_save_final_outputs_writes_pr_and_cm_figures(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            benchmark = _make_three_model_benchmark()
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={
                    "root": str(output_dir),
                    "combined_sha256": "dataset-hash",
                    "files": {},
                },
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ):
                save_final_outputs(
                    benchmark,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                )
            self.assertTrue((output_dir / "pr_curves_comparison.png").exists())
            self.assertTrue((output_dir / "confusion_matrix_best_model_normalized.png").exists())
            self.assertTrue((output_dir / "confusion_matrix_best_model_raw.png").exists())

    def test_save_final_outputs_skips_pr_when_fewer_than_three_models(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        save_final_outputs = _output_utils().save_final_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            config = Config()
            output_dir = Path(tmp)
            benchmark = _make_one_model_benchmark()
            out = io.StringIO()
            with mock.patch(
                "benchmarking.run_metadata._git_metadata",
                return_value={"commit": "abc123", "branch": "main", "dirty": False},
            ), mock.patch(
                "benchmarking.run_metadata._dataset_metadata",
                return_value={
                    "root": str(output_dir),
                    "combined_sha256": "dataset-hash",
                    "files": {},
                },
            ), mock.patch(
                "benchmarking.run_metadata._package_versions",
                return_value={"numpy": "1.0.0"},
            ), redirect_stdout(out):
                save_final_outputs(
                    benchmark,
                    output_dir,
                    config,
                    self._build_run_context(output_dir),
                )
            self.assertFalse((output_dir / "pr_curves_comparison.png").exists())
            self.assertTrue((output_dir / "confusion_matrix_best_model_normalized.png").exists())
            self.assertTrue((output_dir / "confusion_matrix_best_model_raw.png").exists())
            self.assertIn("Skipping PR curve comparison", out.getvalue())

    def test_save_robustness_degradation_figure_no_data_is_no_op(self):
        save_robustness_degradation_figure = _output_utils().save_robustness_degradation_figure

        class _NoRobustness:
            pass

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_robustness_degradation_figure(_NoRobustness(), out)
            self.assertFalse((out / "robustness_profile_degradation.png").exists())

    def test_save_robustness_degradation_figure_writes_png(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_robustness_degradation_figure = _output_utils().save_robustness_degradation_figure
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _with_robustness_data(_make_three_model_benchmark(), ("m_a", "m_b", "m_c"))
            save_robustness_degradation_figure(bench, out)
            p = out / "robustness_profile_degradation.png"
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 100)

    def test_save_feature_vulnerability_outputs_top_n(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_feature_vulnerability_outputs = _output_utils().save_feature_vulnerability_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _make_three_model_benchmark()
            rows = []
            for i in range(15):
                rows.append({
                    "model": "m_a",
                    "feature": f"f{i}",
                    "attack_name": "atk",
                    "cost_tier": "cheap",
                    "flip_rate": 1.0 - i * 0.01,
                    "confidence_drop_mean": 0.05 + i * 0.001,
                })
            bench.feature_attack_results = pd.DataFrame(rows)
            save_feature_vulnerability_outputs(bench, out, top_n=10)
            csv_path = out / "top_feature_vulnerabilities.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 10)
            self.assertListEqual(
                list(df.columns),
                ["Feature", "Attack Name", "Cost Tier", "Flip Rate", "Confidence Drop Mean"],
            )
            self.assertEqual(df.iloc[0]["Feature"], "f0")
            self.assertAlmostEqual(float(df.iloc[0]["Flip Rate"]), 1.0)
            png = out / "feature_attack_flip_rates_best_model.png"
            self.assertTrue(png.exists())
            self.assertGreater(png.stat().st_size, 100)

    def test_degradation_figure_writes_pdf_and_caption(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_robustness_degradation_figure = _output_utils().save_robustness_degradation_figure
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _with_robustness_data(_make_three_model_benchmark(), ("m_a", "m_b", "m_c"))
            save_robustness_degradation_figure(bench, out)
            self.assertTrue((out / "robustness_profile_degradation.pdf").exists())
            cap = out / "robustness_profile_degradation_caption.md"
            self.assertTrue(cap.exists())
            text = cap.read_text(encoding="utf-8")
            self.assertIn("PR-AUC", text)
            self.assertIn("Macro-F1", text)
            self.assertIn("Attacked true-bot recall", text)
            self.assertIn("TwiBot-20 held-out test split", text)
            self.assertNotIn("primary evidence", text.lower())
            self.assertNotIn("primary robustness signal", text.lower())

    def test_robustness_degradation_writes_companion_delta_table(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_robustness_degradation_figure = _output_utils().save_robustness_degradation_figure
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _with_robustness_data(
                _make_three_model_benchmark(),
                ("m_a", "m_b", "m_c"),
                (("baseline", 0.90), ("cheap_only", 0.85), ("realistic_mixed", 0.81)),
            )
            save_robustness_degradation_figure(bench, out)

            csv_path = out / "robustness_profile_degradation_table.csv"
            md_path = out / "robustness_profile_degradation_table.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            table = pd.read_csv(csv_path)
            self.assertEqual(len(table), 6)
            self.assertListEqual(
                list(table.columns),
                [
                    "Model",
                    "Attack Profile",
                    "Attacked True Bots",
                    "Baseline Recall",
                    "Attacked Recall",
                    "Recall Delta",
                    "Recall Delta %",
                    "Baseline Macro-F1",
                    "Attacked Macro-F1",
                    "Macro-F1 Delta",
                    "Macro-F1 Delta %",
                    "Baseline PR-AUC",
                    "Attacked PR-AUC",
                    "PR-AUC Delta",
                    "PR-AUC Delta %",
                    "Flip Rate",
                    "Mean Confidence Drop",
                    "Mean Non-Flip Confidence Drop",
                ],
            )
            self.assertIn("Cheap attacks", set(table["Attack Profile"]))
            self.assertIn("Mixed realistic attacks", set(table["Attack Profile"]))
            first = table[table["Attack Profile"].eq("Cheap attacks")].iloc[0]
            self.assertAlmostEqual(float(first["Baseline Macro-F1"]), 0.90)
            self.assertAlmostEqual(float(first["Attacked Macro-F1"]), 0.85)
            self.assertAlmostEqual(float(first["Macro-F1 Delta"]), -0.05)
            self.assertAlmostEqual(float(first["Macro-F1 Delta %"]), -5.5556)
            self.assertAlmostEqual(float(first["Mean Confidence Drop"]), 0.0711)
            self.assertIn("Exact robustness deltas", md_path.read_text(encoding="utf-8"))

    def test_degradation_caption_filters_significance_to_plotted_models(self):
        cap = _poster_figures().degradation_caption(
            ["m_a", "m_b"],
            ("baseline", "cheap_only"),
            np.array([[0.9, 0.8], [0.88, 0.82]]),
            np.array([[0.7, 0.6], [0.69, 0.64]]),
            [
                {"metric": "f1", "model_a": "hidden_a", "model_b": "hidden_b", "bootstrap_p_corrected": 1e-9},
                {"metric": "f1", "model_a": "m_a", "model_b": "m_b", "bootstrap_p_corrected": 0.04},
            ],
        )

        self.assertIn("m_a vs m_b", cap)
        self.assertNotIn("hidden_a", cap)

    def test_degradation_caption_describes_only_plotted_scenarios(self):
        cap = _poster_figures().degradation_caption(
            ["m_a"],
            ("baseline", "cheap_only"),
            np.array([[0.9, 0.82]]),
            np.array([[0.7, 0.61]]),
            [],
        )

        self.assertIn("Baseline (clean)", cap)
        self.assertIn("Cheap attacks", cap)
        self.assertNotIn("realistic mixed", cap.lower())

    def test_degradation_caption_handles_missing_attack_targeted_deltas(self):
        cap = self._caption_case(
            attacked_recall=(0.8, np.nan),
            f1=(0.7, 0.65),
            pr=(0.7, 0.65),
        )

        self.assertIn("unavailable", cap.lower())

    def test_degradation_caption_mentions_attack_targeted_primary_metric(self):
        cap = self._caption_case()
        self.assertIn("top-1 model", cap)
        self.assertNotIn("top-3 models", cap)
        self._assert_attack_targeted_caption(cap)
        self.assertIn("TwiBot-20 held-out test split", cap)
        self.assertNotIn("primary evidence", cap.lower())
        self.assertNotIn("primary robustness signal", cap.lower())
        self.assertIn("delta", cap.lower())

    def test_degradation_caption_wording_follows_metric_shape(self):
        cases = [
            {
                "label": "stable",
                "kwargs": {
                    "attacked_recall": (0.98, 0.94, 0.93),
                    "f1": (0.8051, 0.8084, 0.8084),
                    "pr": (0.8384, 0.7933, 0.7873),
                    "scenarios": ("baseline", "cheap_only", "realistic_mixed"),
                    "models": ("xgboost",),
                },
                "contains": ("stable",),
                "excludes": (
                    "drops under realistic adversarial profiles",
                    "drop relative to",
                ),
            },
            {
                "label": "slightly lower",
                "kwargs": {
                    "attacked_recall": (0.99, 0.96, 0.96),
                    "f1": (0.8051, 0.7914, 0.7930),
                    "pr": (0.8384, 0.6336, 0.6278),
                    "scenarios": ("baseline", "cheap_only", "realistic_mixed"),
                    "models": ("xgboost",),
                },
                "contains": ("slightly lower", "mixed realistic attacks"),
                "excludes": ("macro-f1 declines",),
            },
        ]
        for case in cases:
            with self.subTest(case=case["label"]):
                cap = self._caption_case(**case["kwargs"])
                self._assert_attack_targeted_caption(cap)
                for text in case["contains"]:
                    self.assertIn(text, cap.lower())
                for text in case["excludes"]:
                    self.assertNotIn(text, cap.lower())

    def test_plot_degradation_titles_follow_metric_shape(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")
        plot_degradation = _poster_figures().plot_degradation

        cases = [
            ("stable", np.array([[0.8051, 0.8084, 0.8084], [0.8024, 0.8057, 0.8057]])),
            ("slightly lower", np.array([[0.8051, 0.7914, 0.7930], [0.8024, 0.8008, 0.8008]])),
        ]
        for expected, values in cases:
            with self.subTest(title=expected):
                fig = plot_degradation(
                    ["xgboost", "decision_tree"],
                    ("baseline", "cheap_only", "realistic_mixed"),
                    values,
                )
                ax = fig.axes[0]
                self.assertIn(expected, ax.get_title().lower())
                self.assertNotIn("drops under realistic adversarial profiles", ax.get_title().lower())
                if expected == "stable":
                    change_texts = [text for text in ax.texts if text.get_text().startswith("Change ")]
                    self.assertGreaterEqual(len(change_texts), 2)
                    self.assertGreater(len({text.get_position()[1] for text in change_texts}), 1)
                else:
                    self.assertNotIn("declines", ax.get_title().lower())
                plt.close(fig)

    def test_plot_degradation_supports_attack_targeted_dual_metric_layout(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")
        plot_degradation = _poster_figures().plot_degradation

        fig = plot_degradation(
            ["xgboost", "decision_tree"],
            ("baseline", "cheap_only", "realistic_mixed"),
            np.array([[0.8051, 0.7940, 0.7920], [0.8024, 0.8008, 0.8008]]),
            np.array([[0.98, 0.95, 0.94], [0.99, 0.97, 0.96]]),
        )
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("Attacked true-bot recall", fig.axes[0].get_title())
        self.assertIn("Full-test Macro-F1", fig.axes[1].get_title())
        self.assertNotIn("primary evidence", fig.axes[0].get_title().lower())
        self.assertEqual(fig.axes[0].get_xlabel(), "Model")
        self.assertEqual(fig.axes[1].get_xlabel(), "Model")
        plt.close(fig)

    def test_plot_best_confusion_matrix_uses_takeaway_title_and_pretty_model_name(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib/seaborn not installed")

        benchmark = _make_named_best_model_benchmark("xgboost")
        fig = benchmark.plot_best_confusion_matrix(normalize="true")
        ax = fig.axes[0]
        title = ax.get_title()
        self.assertIn("XGBoost", title)
        self.assertNotIn("xgboost", title)
        self.assertNotIn("Confusion Matrix (Normalized by True Label)", title)
        plt.close(fig)

    def test_vulnerability_figure_writes_pdf_and_caption(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_feature_vulnerability_outputs = _output_utils().save_feature_vulnerability_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _make_three_model_benchmark()
            rows = []
            for i in range(10):
                rows.append({
                    "model": "m_a",
                    "feature": f"f{i}",
                    "attack_name": "atk",
                    "cost_tier": "cheap" if i % 2 == 0 else "expensive",
                    "flip_rate": 1.0 - i * 0.01,
                    "confidence_drop_mean": 0.05,
                })
            bench.feature_attack_results = pd.DataFrame(rows)
            save_feature_vulnerability_outputs(bench, out, top_n=8)
            self.assertTrue((out / "feature_attack_flip_rates_best_model.pdf").exists())
            cap = out / "feature_attack_flip_rates_best_model_caption.md"
            self.assertTrue(cap.exists())
            self.assertIn("attack surface", cap.read_text(encoding="utf-8").lower())

    def test_vulnerability_defaults_to_top8(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        save_feature_vulnerability_outputs = _output_utils().save_feature_vulnerability_outputs
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _make_three_model_benchmark()
            rows = []
            for i in range(15):
                rows.append({
                    "model": "m_a",
                    "feature": f"f{i}",
                    "attack_name": "atk",
                    "cost_tier": "cheap",
                    "flip_rate": 1.0 - i * 0.01,
                    "confidence_drop_mean": 0.05,
                })
            bench.feature_attack_results = pd.DataFrame(rows)
            save_feature_vulnerability_outputs(bench, out)
            df = pd.read_csv(out / "top_feature_vulnerabilities.csv")
            self.assertEqual(len(df), 8)

    def test_vulnerability_cost_tier_legend(self):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        plot_vulnerability = _poster_figures().plot_vulnerability
        filtered = pd.DataFrame({
            "flip_rate": [0.9, 0.5],
            "cost_tier": ["cheap", "expensive"],
            "display_feature": ["FeatA", "FeatB"],
        })
        fig = plot_vulnerability(filtered, "solo_model")
        ax = fig.axes[0]
        leg = ax.get_legend()
        self.assertIsNotNone(leg)
        labels = [t.get_text() for t in leg.get_texts()]
        self.assertEqual(len(labels), 2)
        joined = " ".join(labels)
        self.assertIn("Cheap", joined)
        self.assertIn("Expensive", joined)
        plt.close(fig)

    def test_poster_style_does_not_leak_rcparams(self):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        output_utils = _output_utils()
        poster_style = _poster_figures().poster_style
        before = float(plt.rcParams["font.size"])
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            bench = _make_three_model_benchmark()
            _with_robustness_data(
                bench,
                ("m_a",),
                (("baseline", 0.9), ("cheap_only", 0.85)),
            )
            output_utils.save_robustness_degradation_figure(bench, out)
            rows = [{"model": "m_a", "feature": "f0", "attack_name": "a", "cost_tier": "cheap",
                     "flip_rate": 0.5, "confidence_drop_mean": 0.1}]
            bench.feature_attack_results = pd.DataFrame(rows)
            output_utils.save_feature_vulnerability_outputs(bench, out, top_n=1)
        self.assertEqual(float(plt.rcParams["font.size"]), before)
        with poster_style():
            _ = plt.rcParams["font.size"]
        self.assertEqual(float(plt.rcParams["font.size"]), before)

    def test_robustness_degradation_figure_uses_top_model_scenarios_only(self):
        try:
            import matplotlib  # noqa: F401
            import pandas as pd
        except ImportError:
            self.skipTest("matplotlib/pandas not installed")
        output_utils = _output_utils()
        bench = _make_three_model_benchmark()
        bench.robustness_degradation = pd.DataFrame(
            _degradation_rows(
                ("m_a", "m_b", "m_c"),
                (("baseline", 0.9), ("cheap_only", 0.84)),
            )
            + _degradation_rows(
                ("lower_ranked",),
                (("baseline", 0.7), ("cheap_only", 0.65), ("realistic_mixed", 0.6)),
            )
        )
        bench.robustness_summary = pd.DataFrame(_robustness_summary_rows(("m_a", "m_b", "m_c")))

        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp:
            out = Path(tmp)
            output_utils.save_robustness_degradation_figure(bench, out)
            cap = (out / "robustness_profile_degradation_caption.md").read_text(encoding="utf-8")

        self.assertIn("Cheap attacks", cap)
        self.assertNotIn("Mixed realistic attacks", cap)

if __name__ == "__main__":
    unittest.main()
