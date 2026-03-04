"""
Tests for benchmark output utility helpers.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from config import Config
from benchmarking.output_utils import save_comparison_outputs


class _FakeTable:
    def to_csv(self, *_args, **_kwargs):
        return None


class _BenchmarkStub:
    def __init__(self, table_exc=None, plot_exc=None):
        self._table_exc = table_exc
        self._plot_exc = plot_exc

    def get_comparison_table(self):
        if self._table_exc:
            raise self._table_exc
        return _FakeTable()

    def plot_comparison(self):
        if self._plot_exc:
            raise self._plot_exc
        return None

    def plot_training_times(self):
        raise AssertionError("plot_training_times should not run")


class OutputUtilsTest(unittest.TestCase):

    def _run_save(self, stub, save_plots=True):
        with TemporaryDirectory() as tmp:
            config = Config()
            config.set('output.save_plots', save_plots)
            out = io.StringIO()
            with redirect_stdout(out):
                save_comparison_outputs(stub, Path(tmp), config)
        return out.getvalue()

    def test_warns_when_plot_raises_keyerror(self):
        output = self._run_save(_BenchmarkStub(plot_exc=KeyError("model_a")))
        self.assertIn("Warning: Could not save plots", output)

    def test_warns_when_table_raises_keyerror(self):
        output = self._run_save(_BenchmarkStub(table_exc=KeyError("test_metrics")))
        self.assertIn("Warning: Could not save comparison table", output)

    def test_warns_when_plot_raises_attribute_error(self):
        """AttributeError (e.g. None returned as fig) must be caught non-fatally."""
        output = self._run_save(_BenchmarkStub(plot_exc=AttributeError("'NoneType' has no savefig")))
        self.assertIn("Warning: Could not save plots", output)

    def test_warns_when_plot_raises_import_error(self):
        """ImportError from misconfigured matplotlib backend must be caught."""
        output = self._run_save(_BenchmarkStub(plot_exc=ImportError("No backend")))
        self.assertIn("Warning: Could not save plots", output)

    def test_warns_when_table_raises_type_error(self):
        """TypeError from malformed DataFrame must be caught non-fatally."""
        output = self._run_save(_BenchmarkStub(table_exc=TypeError("unhashable")))
        self.assertIn("Warning: Could not save comparison table", output)

    def test_warns_when_table_raises_runtime_error(self):
        output = self._run_save(_BenchmarkStub(table_exc=RuntimeError("unexpected")))
        self.assertIn("Warning: Could not save comparison table", output)

    def test_table_failure_does_not_block_plot_saving(self):
        """When table save fails, plot saving is still attempted when save_plots is enabled."""
        stub = _BenchmarkStub(table_exc=OSError("disk full"))
        output = self._run_save(stub, save_plots=True)
        self.assertIn("Warning: Could not save comparison table", output)
        self.assertIn("Warning: Could not save plots", output, "Plot saving must run despite table failure")


if __name__ == '__main__':
    unittest.main()
