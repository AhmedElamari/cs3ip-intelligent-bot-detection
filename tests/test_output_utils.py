"""Tests for benchmark output utility helpers."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from config import Config
from benchmarking.output_utils import save_comparison_outputs, save_final_outputs


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
    def __init__(self, save_exc=None, report="report"):
        self._save_exc = save_exc
        self._report = report

    def save_results(self, _output_dir):
        if self._save_exc:
            raise self._save_exc

    def generate_report(self):
        return self._report


class OutputUtilsTest(unittest.TestCase):
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
                )

    def test_save_final_outputs_writes_markdown_and_text_reports(self):
        with TemporaryDirectory() as tmp:
            config = Config()
            output_dir = Path(tmp)
            save_final_outputs(_FinalOutputStub(report="# report"), output_dir, config)
            self.assertTrue((output_dir / "benchmark_report.md").exists())
            self.assertTrue((output_dir / "benchmark_report.txt").exists())
            self.assertTrue((output_dir / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
