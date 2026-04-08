"""Tests for benchmark output utility helpers."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from config import Config
from benchmarking.output_utils import save_comparison_outputs, save_final_outputs
from benchmarking.run_metadata import BenchmarkRunContext


class _PlotStub:
    def __init__(self, plot_exc=None):
        self._plot_exc = plot_exc

    def plot_comparison(self):
        if self._plot_exc:
            raise self._plot_exc
        return None

    def plot_training_times(self):
        raise AssertionError("plot_training_times should not run when comparison plot fails")


class _FinalOutputStub:
    def __init__(self, save_exc=None, report="report", write_results=False):
        self._save_exc = save_exc
        self._report = report
        self._write_results = write_results

    def save_results(self, output_dir):
        if self._save_exc:
            raise self._save_exc
        if not self._write_results:
            return
        (Path(output_dir) / "model_comparison.csv").write_text("Model,F1\nstub,1.0\n", encoding="utf-8")
        (Path(output_dir) / "results.json").write_text("{}", encoding="utf-8")

    def generate_report(self):
        return self._report


class OutputUtilsTest(unittest.TestCase):
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

    def _run_plot_save(self, stub, save_plots=True):
        with TemporaryDirectory() as tmp:
            config = Config()
            config.set("output.save_plots", save_plots)
            out = io.StringIO()
            with redirect_stdout(out):
                save_comparison_outputs(stub, Path(tmp), config)
        return out.getvalue()

    def test_warns_when_plot_raises_keyerror(self):
        output = self._run_plot_save(_PlotStub(plot_exc=KeyError("model_a")))
        self.assertIn("Warning: Could not save plots", output)

    def test_warns_when_plot_raises_attribute_error(self):
        output = self._run_plot_save(_PlotStub(plot_exc=AttributeError("bad fig")))
        self.assertIn("Warning: Could not save plots", output)

    def test_warns_when_plot_raises_import_error(self):
        output = self._run_plot_save(_PlotStub(plot_exc=ImportError("No backend")))
        self.assertIn("Warning: Could not save plots", output)

    def test_save_final_outputs_propagates_required_export_failures(self):
        with TemporaryDirectory() as tmp:
            config = Config()
            with self.assertRaises(OSError):
                save_final_outputs(
                    _FinalOutputStub(save_exc=OSError("disk full")),
                    Path(tmp),
                    config,
                    self._build_run_context(Path(tmp)),
                )

    def test_save_final_outputs_writes_markdown_and_text_reports(self):
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
            self.assertTrue((output_dir / "benchmark_report.txt").exists())
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())

    def test_save_final_outputs_writes_run_metadata_contract(self):
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
                    _FinalOutputStub(report="# report", write_results=True),
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


if __name__ == "__main__":
    unittest.main()
