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

    def test_save_comparison_outputs_warns_when_plot_raises_keyerror(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = Config()
            config.set('output.save_plots', True)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                save_comparison_outputs(
                    _BenchmarkStub(plot_exc=KeyError("model_a")),
                    output_dir,
                    config,
                )

        self.assertIn("Warning: Could not save plots", stdout.getvalue())

    def test_save_comparison_outputs_warns_when_table_generation_raises_keyerror(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = Config()
            config.set('output.save_plots', True)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                save_comparison_outputs(
                    _BenchmarkStub(table_exc=KeyError("test_metrics")),
                    output_dir,
                    config,
                )

        self.assertIn("Warning: Could not save comparison table", stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
