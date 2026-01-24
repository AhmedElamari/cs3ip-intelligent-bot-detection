import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class TimeStratifiedSplitTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")
        import pandas as pd
        import pipeline_utils
        self.pd = pd
        self.time_stratified_split = pipeline_utils.time_stratified_split

    def test_chronological_ordering(self):
        """Verify train gets oldest data, test gets newest."""
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.to_datetime([
                '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
                '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
                '2020-09-01', '2020-10-01'
            ]),
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'id': list(range(10))
        })
        
        train_df, val_df, test_df = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )
        
        # Get the max date in train and min date in test
        train_max_date = df.loc[df['id'].isin(train_df['id']), 'account_creation_date'].max()
        test_min_date = df.loc[df['id'].isin(test_df['id']), 'account_creation_date'].min()
        
        # Train max date should be before test min date (chronological)
        self.assertLess(train_max_date, test_min_date)

    def test_split_proportions(self):
        """Verify split sizes match requested proportions."""
        n = 100
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n),
            'label': [i % 2 for i in range(n)]
        })
        
        train_df, val_df, test_df = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )
        
        self.assertEqual(len(train_df), 70)  # 1 - 0.2 - 0.1 = 0.7
        self.assertEqual(len(val_df), 20)    # 0.2
        self.assertEqual(len(test_df), 10)   # 0.1

    def test_invalid_split_sizes_raise(self):
        """Verify invalid split sizes raise a ValueError."""
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=10),
            'label': [i % 2 for i in range(10)]
        })

        with self.assertRaises(ValueError):
            self.time_stratified_split(df, val_size=0.6, test_size=0.5, random_state=2112)

    def test_too_small_dataset_raises(self):
        """Verify too-small datasets raise when a split would be empty."""
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=3),
            'label': [0, 1, 0],
            'id': list(range(3))
        })

        with self.assertRaises(ValueError):
            self.time_stratified_split(df, val_size=0.2, test_size=0.1, random_state=2112)

    def test_small_dataset_splits(self):
        """Verify small datasets still split without errors."""
        n = 5
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n),
            'label': [i % 2 for i in range(n)],
            'id': list(range(n))
        })

        train_df, val_df, test_df = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )

        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), n)

    def test_no_overlap_between_splits(self):
        """Verify no samples appear in multiple splits."""
        n = 50
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n),
            'label': [i % 2 for i in range(n)],
            'id': list(range(n))
        })
        
        train_df, val_df, test_df = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )
        
        train_ids = set(train_df['id'])
        val_ids = set(val_df['id'])
        test_ids = set(test_df['id'])
        
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)
        self.assertEqual(len(train_ids | val_ids | test_ids), n)

    def test_fallback_without_time_column(self):
        """Verify graceful fallback when time column is missing."""
        df = self.pd.DataFrame({
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'id': list(range(10))
        })
        
        with self.assertLogs('pipeline_utils', level='WARNING') as logs:
            train_df, val_df, test_df = self.time_stratified_split(
                df, val_size=0.2, test_size=0.1, random_state=2112
            )
        
        # Should still produce valid splits
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 10)
        self.assertTrue(any('not found' in msg for msg in logs.output))

    def test_deterministic_with_random_state(self):
        """Verify same random_state produces identical results."""
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=50),
            'label': [i % 2 for i in range(50)]
        })
        
        train1, val1, test1 = self.time_stratified_split(df, random_state=2112)
        train2, val2, test2 = self.time_stratified_split(df, random_state=2112)
        
        self.pd.testing.assert_frame_equal(train1.reset_index(drop=True), 
                                            train2.reset_index(drop=True))
        self.pd.testing.assert_frame_equal(val1.reset_index(drop=True), 
                                            val2.reset_index(drop=True))
        self.pd.testing.assert_frame_equal(test1.reset_index(drop=True), 
                                            test2.reset_index(drop=True))

    def test_deterministic_with_duplicate_timestamps(self):
        """Verify deterministic results when timestamps are duplicated."""
        n = 40
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.to_datetime(
                ['2020-01-01'] * (n // 2) + ['2020-02-01'] * (n // 2)
            ),
            'label': [i % 2 for i in range(n)],
            'id': list(range(n))
        })

        train1, val1, test1 = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )
        train2, val2, test2 = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )

        self.pd.testing.assert_frame_equal(train1, train2)
        self.pd.testing.assert_frame_equal(val1, val2)
        self.pd.testing.assert_frame_equal(test1, test2)

    def test_shuffled_within_splits(self):
        """Verify data is shuffled within each split (not strictly sorted)."""
        n = 100
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n),
            'label': [i % 2 for i in range(n)],
            'original_order': list(range(n))
        })
        
        train_df, _, _ = self.time_stratified_split(df, random_state=2112)
        
        # After shuffle, the order should not be monotonically increasing
        original_orders = train_df['original_order'].tolist()
        is_sorted = all(original_orders[i] < original_orders[i+1] 
                       for i in range(len(original_orders)-1))
        self.assertFalse(is_sorted, "Data within split should be shuffled")

    def test_reference_date_from_all_data_prevents_negative_ages(self):
        """Verify that deriving reference date from all data prevents negative account ages.
        
        This tests the fix for the bug where deriving reference date from training data
        alone (oldest accounts) caused val/test accounts to have negative ages that got
        clipped to 1, corrupting derived features.
        """
        try:
            from FeatureEngineering import BotFeatureExtractor, derive_reference_date
        except Exception:
            self.skipTest("FeatureEngineering or its dependencies not installed")
        
        # Create data spanning 1 year (365 days)
        n = 100
        df = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n, freq='3D'),
            'label': [i % 2 for i in range(n)],
            'statuses_count': [1000] * n,  # Same statuses for all
            'followers_count': [500] * n,
            'favourites_count': [200] * n,
        })
        
        # Split chronologically: train=70%, val=20%, test=10%
        train_df, val_df, test_df = self.time_stratified_split(
            df, val_size=0.2, test_size=0.1, random_state=2112
        )
        
        # BUG: Reference date from train only (oldest accounts) -> early date
        train_only_ref = derive_reference_date(train_df)
        
        # FIX: Reference date from ALL data -> latest date
        all_data_ref = derive_reference_date(df)
        
        # With train-only reference, test accounts should have negative raw ages
        # (before clipping) because they were created after the reference date
        test_creation = self.pd.to_datetime(test_df['account_creation_date'])
        raw_ages_wrong = (train_only_ref - test_creation).dt.days
        
        # Verify the bug: test accounts created after train_only_ref have negative ages
        self.assertTrue(
            (raw_ages_wrong < 0).any(),
            "Test should demonstrate the bug: train-only reference causes negative ages"
        )
        
        # Extract features using the WRONG approach (train-only reference)
        extractor_wrong = BotFeatureExtractor(reference_date=train_only_ref)
        test_with_wrong_ref = extractor_wrong.extract_all_features(test_df.copy())
        
        # Extract features using the CORRECT approach (all-data reference)
        extractor_correct = BotFeatureExtractor(reference_date=all_data_ref)
        test_with_correct_ref = extractor_correct.extract_all_features(test_df.copy())
        
        # With all-data reference, no ages should be negative
        raw_ages_correct = (all_data_ref - test_creation).dt.days
        self.assertTrue(
            (raw_ages_correct >= 0).all(),
            "Fix should ensure no negative ages when using all-data reference"
        )
        
        # The key metric: account_age_days should be realistic with correct ref
        # With wrong ref, ages get clipped to 1 (artificially small)
        # With correct ref, ages should reflect actual days since creation
        wrong_ages = test_with_wrong_ref['account_age_days']
        correct_ages = test_with_correct_ref['account_age_days']
        
        # Verify that wrong reference causes ages to be clipped to minimum (1)
        # while correct reference preserves realistic ages
        self.assertTrue(
            (wrong_ages == 1).sum() > (correct_ages == 1).sum(),
            "Wrong reference should cause more ages to be clipped to 1"
        )
        
        # Verify derived features are more inflated with wrong reference
        # (because dividing by 1 instead of actual age)
        wrong_tpd_mean = test_with_wrong_ref['tweets_per_day'].mean()
        correct_tpd_mean = test_with_correct_ref['tweets_per_day'].mean()
        
        self.assertGreater(
            wrong_tpd_mean, correct_tpd_mean * 2,
            "Wrong reference should cause significantly inflated tweets_per_day"
        )

    def test_prepare_data_time_split_uses_combined_reference_date(self):
        """Integration test: prepare_data should avoid clipped ages in val/test."""
        from benchmarking.data_prep import prepare_data
        from config import Config

        n = 30
        df_all = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n, freq='D'),
            'label': [i % 2 for i in range(n)],
            'statuses_count': [100 + i for i in range(n)],
            'followers_count': [200 + i for i in range(n)],
            'favourites_count': [50 + i for i in range(n)],
        })

        splits = {
            'train': df_all.iloc[:10].copy(),
            'val': df_all.iloc[10:20].copy(),
            'test': df_all.iloc[20:].copy(),
        }

        config = Config({'time_split': True})
        X_train, X_val, X_test, *_ = prepare_data(splits, config)

        self.assertIn('account_age_days', X_val.columns)
        self.assertIn('account_age_days', X_test.columns)
        self.assertTrue(
            (X_val['account_age_days'] > 1).any(),
            "Validation ages should not all be clipped to 1 with time_split enabled"
        )
        self.assertTrue(
            (X_test['account_age_days'] > 1).any(),
            "Test ages should not all be clipped to 1 with time_split enabled"
        )

    def test_main_pipeline_time_split_uses_combined_reference_date(self):
        """Integration test: main pipeline should avoid clipped ages in val/test."""
        import main
        import training

        n = 30
        df_all = self.pd.DataFrame({
            'account_creation_date': self.pd.date_range('2020-01-01', periods=n, freq='D'),
            'label': [i % 2 for i in range(n)],
            'statuses_count': [100 + i for i in range(n)],
            'followers_count': [200 + i for i in range(n)],
            'favourites_count': [50 + i for i in range(n)],
        })

        splits = {
            'train': df_all.iloc[:10].copy(),
            'val': df_all.iloc[10:20].copy(),
            'test': df_all.iloc[20:].copy(),
        }

        captured = {}

        def capture_and_train(X_train, X_val, X_test, y_train, y_val, y_test, **kwargs):
            captured['X_val'] = X_val
            captured['X_test'] = X_test
            return training.train_and_evaluate(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                **kwargs
            )

        with patch.object(main, 'load_and_prepare_data', return_value=splits), \
             patch.object(main, 'train_and_evaluate', side_effect=capture_and_train):
            main.run_pipeline(
                model_type='random_forest',
                use_smote=False,
                use_scaling=False,
                num_features=None,
                use_time_split=True
            )

        self.assertIn('X_val', captured)
        self.assertIn('X_test', captured)
        self.assertIn('account_age_days', captured['X_val'].columns)
        self.assertIn('account_age_days', captured['X_test'].columns)
        self.assertTrue(
            (captured['X_val']['account_age_days'] > 1).any(),
            "Validation ages should not all be clipped to 1 in main pipeline"
        )
        self.assertTrue(
            (captured['X_test']['account_age_days'] > 1).any(),
            "Test ages should not all be clipped to 1 in main pipeline"
        )


if __name__ == "__main__":
    unittest.main()
