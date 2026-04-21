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


def _make_three_model_benchmark() -> ModelBenchmark:
    """Minimal benchmark state so dissertation PR + CM figures can render."""
    rng = np.random.default_rng(2112)
    n = 24
    y_test = np.concatenate([np.zeros(12), np.ones(12)]).astype(int)
    b = ModelBenchmark(models={}, experiment_name="stub_three")
    b.y_test = y_test
    specs = [
        ("m_a", 0.92, 0.81, 0.79),
        ("m_b", 0.88, 0.82, 0.78),
        ("m_c", 0.84, 0.83, 0.77),
    ]
    b.results = {}
    b.predictions = {}
    b.probabilities = {}
    for name, f1m, roc, pr_auc in specs:
        y_pred = rng.integers(0, 2, size=n)
        proba = rng.random((n, 2))
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
            "is_interpretable": True,
            "X_train": None,
            "X_val": None,
            "X_test": None,
            "feature_names": [],
            "scaler": None,
        }
    return b


def _make_one_model_benchmark() -> ModelBenchmark:
    rng = np.random.default_rng(42)
    n = 16
    y_test = np.concatenate([np.zeros(8), np.ones(8)]).astype(int)
    b = ModelBenchmark(models={}, experiment_name="stub_one")
    b.y_test = y_test
    name = "solo"
    y_pred = rng.integers(0, 2, size=n)
    proba = rng.random((n, 2))
    proba = proba / proba.sum(axis=1, keepdims=True)
    b.predictions[name] = y_pred
    b.probabilities[name] = proba
    tm = b.metrics_calculator.compute_all_metrics(y_test, y_pred, proba)
    tm["f1_macro"] = 0.9
    tm["roc_auc"] = 0.85
    b.results[name] = {
        "model": None,
        "training_time": 1.0,
        "val_metrics": {},
        "test_metrics": tm,
        "feature_importance": None,
        "is_interpretable": True,
        "X_train": None,
        "X_val": None,
        "X_test": None,
        "feature_names": [],
        "scaler": None,
    }
    return b


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
            bench = _make_three_model_benchmark()
            rows = []
            for m in ("m_a", "m_b", "m_c"):
                for s, f1 in (("baseline", 0.9), ("cheap_only", 0.85), ("realistic_mixed", 0.8)):
                    rows.append({"model": m, "scenario": s, "macro_f1": f1, "pr_auc": 0.7})
            bench.robustness_degradation = pd.DataFrame(rows)
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
            bench = _make_three_model_benchmark()
            rows = []
            for m in ("m_a", "m_b", "m_c"):
                for s, f1 in (("baseline", 0.9), ("cheap_only", 0.85), ("realistic_mixed", 0.8)):
                    rows.append({"model": m, "scenario": s, "macro_f1": f1, "pr_auc": 0.7})
            bench.robustness_degradation = pd.DataFrame(rows)
            save_robustness_degradation_figure(bench, out)
            self.assertTrue((out / "robustness_profile_degradation.pdf").exists())
            cap = out / "robustness_profile_degradation_caption.md"
            self.assertTrue(cap.exists())
            text = cap.read_text(encoding="utf-8")
            self.assertIn("PR-AUC", text)
            self.assertIn("Macro-F1", text)

    def test_degradation_caption_mentions_delta(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        degradation_caption = _poster_figures().degradation_caption

        models = ["m_a"]
        scenarios = ("baseline", "cheap_only")
        pr = np.array([[0.7, 0.65]])
        cap = degradation_caption(models, scenarios, pr, [])
        self.assertIn("top-1 model", cap)
        self.assertNotIn("top-3 models", cap)
        self.assertTrue("Δ" in cap or "delta" in cap.lower())

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
            bench.robustness_degradation = pd.DataFrame([
                {"model": "m_a", "scenario": "baseline", "macro_f1": 0.9, "pr_auc": 0.7},
                {"model": "m_a", "scenario": "cheap_only", "macro_f1": 0.85, "pr_auc": 0.65},
            ])
            output_utils.save_robustness_degradation_figure(bench, out)
            rows = [{"model": "m_a", "feature": "f0", "attack_name": "a", "cost_tier": "cheap",
                     "flip_rate": 0.5, "confidence_drop_mean": 0.1}]
            bench.feature_attack_results = pd.DataFrame(rows)
            output_utils.save_feature_vulnerability_outputs(bench, out, top_n=1)
        self.assertEqual(float(plt.rcParams["font.size"]), before)
        with poster_style():
            _ = plt.rcParams["font.size"]
        self.assertEqual(float(plt.rcParams["font.size"]), before)


if __name__ == "__main__":
    unittest.main()
