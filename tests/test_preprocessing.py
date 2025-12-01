###
# tests/test_preprocessing.py
#
# Unit tests for cluster_maker.preprocessing
###

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    # This test checks that select_features returns exactly the requested
    # columns in the right order. It would catch bugs where the function
    # silently drops columns, reorders them, or includes extra unwanted ones.
    def test_select_features_correct_columns_and_order(self):
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0],
                "x2": [10.0, 20.0, 30.0],
                "y": ["a", "b", "c"],  # non-numeric column
            }
        )
        feature_cols = ["x2", "x1"]

        selected = select_features(df, feature_cols)

        # We expect a DataFrame with only x2 and x1, in that order
        self.assertIsInstance(selected, pd.DataFrame)
        self.assertListEqual(list(selected.columns), feature_cols)
        # And the values should match the original data
        np.testing.assert_allclose(selected["x2"].values, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(selected["x1"].values, [1.0, 2.0, 3.0])

    # This test checks that select_features fails loudly when a requested
    # column name does not exist. It would catch bugs where missing columns
    # are silently ignored, leading to unexpected models using the wrong data.
    def test_select_features_raises_for_missing_column(self):
        df = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0],
                "b": [3.0, 4.0, 5.0],
            }
        )
        feature_cols = ["a", "c"]  # 'c' does not exist

        with self.assertRaises(KeyError):
            _ = select_features(df, feature_cols)

    # This test checks that standardise_features really scales each feature
    # to zero mean and unit variance. It would catch bugs where the scaler
    # is not fitted correctly, the wrong axis is used, or the function
    # returns unscaled data.
    def test_standardise_features_zero_mean_unit_variance(self):
        X = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )

        X_scaled = standardise_features(X)

        # Shape must be preserved
        self.assertEqual(X_scaled.shape, X.shape)

        # For each column: mean ~ 0, std ~ 1
        means = X_scaled.mean(axis=0)
        stds = X_scaled.std(axis=0)  # population std (ddof=0), like StandardScaler

        np.testing.assert_allclose(means, np.zeros(2), atol=1e-7)
        np.testing.assert_allclose(stds, np.ones(2), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
