import unittest

from benchmarking.dissertation_scoreboard import (
    DISPLAY_ORDER,
    METRIC_COLUMNS,
    build_scoreboard,
    to_latex,
    to_markdown,
)


class _Bench:
    def __init__(self, results):
        self.results = results


def _row(p, r, fm, fw, pr, roc, mcc, bacc, tt):
    return {
        "test_metrics": {
            "precision": p,
            "recall": r,
            "f1_macro": fm,
            "f1_weighted": fw,
            "pr_auc": pr,
            "roc_auc": roc,
            "mcc": mcc,
            "balanced_accuracy": bacc,
        },
        "training_time": tt,
    }


class DissertationScoreboardTest(unittest.TestCase):
    def test_column_order_matches_plan(self):
        self.assertEqual(
            DISPLAY_ORDER,
            [
                "Rank",
                "Model",
                "Precision",
                "Recall",
                "F1-Macro",
                "F1-Weighted",
                "PR-AUC",
                "ROC-AUC",
                "MCC",
                "Balanced Accuracy",
                "Train Time (s)",
            ],
        )
        self.assertEqual(len(METRIC_COLUMNS), 8)

    def test_sort_primary_f1_macro_secondary_roc_auc(self):
        b = _Bench(
            {
                "low": _row(0.5, 0.5, 0.4, 0.4, 0.3, 0.7, 0.1, 0.5, 1.0),
                "mid": _row(0.6, 0.6, 0.5, 0.5, 0.4, 0.6, 0.2, 0.55, 2.0),
                "tie_first": _row(0.9, 0.9, 0.8, 0.8, 0.9, 0.95, 0.5, 0.9, 3.0),
                "tie_second": _row(0.8, 0.8, 0.8, 0.8, 0.85, 0.85, 0.5, 0.88, 3.5),
            }
        )
        df = build_scoreboard(b)
        models = df["Model"].tolist()
        self.assertEqual(models[0], "tie_first")
        self.assertEqual(models[1], "tie_second")
        self.assertEqual(models[2], "mid")
        self.assertEqual(models[3], "low")
        self.assertEqual(df.iloc[0]["Rank"], 1)

    def test_rounding_metrics_three_time_two(self):
        b = _Bench({"only": _row(0.123456, 0.9, 0.55, 0.56, 0.4444, 0.5555, 0.1, 0.2, 1.234567)})
        df = build_scoreboard(b)
        self.assertEqual(df["Precision"].iloc[0], 0.123)
        self.assertEqual(df["Train Time (s)"].iloc[0], 1.23)

    def test_markdown_bolds_ties(self):
        b = _Bench(
            {
                "a": _row(0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 1.0),
                "b": _row(0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.6, 2.0),
            }
        )
        md = to_markdown(build_scoreboard(b))
        self.assertIn("**0.900**", md)
        c_precision = md.splitlines()[3].count("|")
        self.assertGreater(c_precision, 0)

    def test_latex_escapes_underscores_in_model_name(self):
        b = _Bench(
            {"my_model": _row(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 1.0)}
        )
        tex = to_latex(build_scoreboard(b))
        self.assertIn(r"my\_model", tex)

    def test_latex_uses_valid_wrapped_tabular_spec(self):
        b = _Bench(
            {"only": _row(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 1.0)}
        )
        tex = to_latex(build_scoreboard(b))
        self.assertIn(r"\begin{tabular}{@{}rlrrrrrrrrr@{}}", tex)
        self.assertNotIn(r"\begin{tabular}{@rl", tex)

    def test_latex_deterministic(self):
        b = _Bench(
            {
                "a": _row(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 1.0),
                "b": _row(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.0, 0.4, 2.0),
            }
        )
        df = build_scoreboard(b)
        self.assertEqual(to_latex(df), to_latex(df))

    def test_empty_results_dataframe(self):
        df = build_scoreboard(_Bench({}))
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
