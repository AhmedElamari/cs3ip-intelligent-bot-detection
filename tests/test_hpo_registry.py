"""Unit tests for benchmarking.hpo registry."""
import unittest

from benchmarking.hpo.contracts import HPOEntry
from benchmarking.hpo import registry as reg
from benchmarking.hpo.register_entries import register_default_hpo_entries


def _dummy_suggest(trial):
    return {}


def _dummy_score(*args, **kwargs):
    return 0.0


class TestHPORegistry(unittest.TestCase):
    def setUp(self):
        reg.clear_for_tests()

    def tearDown(self):
        reg.clear_for_tests()
        register_default_hpo_entries()

    def test_register_and_get(self):
        entry = HPOEntry(
            name="logistic_regression",
            search_space_version="v1",
            suggest_fn=_dummy_suggest,
            score_fn=_dummy_score,
        )
        reg.register(entry)
        got = reg.get("logistic_regression")
        self.assertEqual(got.name, "logistic_regression")
        self.assertEqual(got.search_space_version, "v1")

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError) as ctx:
            reg.get("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_list_registered_sorted(self):
        reg.register(
            HPOEntry(
                name="z_model",
                search_space_version="v1",
                suggest_fn=_dummy_suggest,
                score_fn=_dummy_score,
            )
        )
        reg.register(
            HPOEntry(
                name="a_model",
                search_space_version="v1",
                suggest_fn=_dummy_suggest,
                score_fn=_dummy_score,
            )
        )
        self.assertEqual(reg.list_registered(), ["a_model", "z_model"])


if __name__ == "__main__":
    unittest.main()
