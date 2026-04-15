"""
City Pulse — Test Suite
========================
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.data.loader import (
    load_raw, validate, clean, add_features, split, scale_features,
    NUMERIC_FEATURES, TARGET,
)
from src.models.clustering import run_kmeans, describe_clusters, elbow_analysis
from src.models.regression import (
    train_model, evaluate, cross_validate, get_coefficients,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw()

@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)

@pytest.fixture(scope="module")
def full_df(clean_df):
    return add_features(clean_df)

@pytest.fixture(scope="module")
def split_data(full_df):
    df_s, scaler = scale_features(full_df)
    return split(df_s)


# ─── Loader tests ─────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns(self, raw_df):
        for col in NUMERIC_FEATURES + [TARGET, "city", "country", "continent"]:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_40_cities(self, raw_df):
        assert len(raw_df) == 40

    def test_no_nulls_in_key_columns(self, raw_df):
        assert raw_df[["city"] + NUMERIC_FEATURES + [TARGET]].isnull().sum().sum() == 0

    def test_scores_in_range(self, raw_df):
        for col in NUMERIC_FEATURES + [TARGET]:
            assert raw_df[col].between(0, 100).all(), f"{col} has out-of-range values"

    def test_no_duplicate_cities(self, raw_df):
        assert raw_df["city"].is_unique

    def test_validate_passes(self, raw_df):
        result = validate(raw_df)
        assert result is not None

    def test_clean_returns_same_shape(self, raw_df, clean_df):
        assert len(clean_df) == len(raw_df)

    def test_engineered_features_added(self, full_df):
        for col in ["env_score", "social_score", "affordability", "quiet_score", "composite"]:
            assert col in full_df.columns

    def test_affordability_inverse_cost(self, full_df):
        expected = (100 - full_df["cost_of_living"]).round(1)
        pd.testing.assert_series_equal(
            full_df["affordability"], expected, check_names=False
        )

    def test_scale_features_zero_mean(self, full_df):
        df_s, _ = scale_features(full_df)
        means = df_s[NUMERIC_FEATURES].mean()
        assert (means.abs() < 1e-10).all(), "Scaled features should have zero mean"

    def test_scale_features_unit_variance(self, full_df):
        df_s, _ = scale_features(full_df)
        # sklearn uses ddof=0 (population std), pandas .std() uses ddof=1
        stds = df_s[NUMERIC_FEATURES].std(ddof=0)
        assert ((stds - 1).abs() < 1e-10).all(), "Scaled features should have unit variance"

    def test_split_sizes(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05  # ~20% test


# ─── Clustering tests ─────────────────────────────────────────────────────────

class TestClustering:
    def test_run_kmeans_adds_cluster_column(self, full_df):
        df_c, _, _, _ = run_kmeans(full_df, k=3)
        assert "cluster" in df_c.columns
        assert "cluster_label" in df_c.columns

    def test_kmeans_k_unique_clusters(self, full_df):
        for k in [2, 3, 4]:
            df_c, _, _, _ = run_kmeans(full_df, k=k)
            assert df_c["cluster"].nunique() == k

    def test_all_cities_assigned(self, full_df):
        df_c, _, _, _ = run_kmeans(full_df, k=3)
        assert df_c["cluster"].isnull().sum() == 0

    def test_describe_clusters_returns_dataframe(self, full_df):
        df_c, _, _, _ = run_kmeans(full_df, k=3)
        desc = describe_clusters(df_c)
        assert isinstance(desc, pd.DataFrame)
        assert len(desc) == 3

    def test_elbow_returns_correct_lengths(self, full_df):
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(full_df[NUMERIC_FEATURES])
        k_range = range(2, 7)
        inertias, sil = elbow_analysis(X, k_range)
        assert len(inertias) == len(list(k_range))
        assert len(sil) == len(list(k_range))

    def test_inertia_decreases_with_k(self, full_df):
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(full_df[NUMERIC_FEATURES])
        inertias, _ = elbow_analysis(X, range(2, 7))
        # Inertia should be non-increasing
        assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))

    def test_feature_groups(self, full_df):
        for group in ["all", "environmental", "social"]:
            df_c, _, _, _ = run_kmeans(full_df, k=3, feature_group=group)
            assert df_c["cluster"].nunique() == 3


# ─── Regression tests ─────────────────────────────────────────────────────────

class TestRegression:
    def test_train_model_returns_fitted(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        model = train_model(X_train, y_train, "Linear Regression")
        assert hasattr(model, "coef_")

    def test_evaluate_returns_metrics(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        model = train_model(X_train, y_train, "Linear Regression")
        metrics = evaluate(model, X_test, y_test)
        for key in ["R²", "RMSE", "MAE", "Max Error"]:
            assert key in metrics

    def test_r2_is_valid(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        model = train_model(X_train, y_train, "Linear Regression")
        metrics = evaluate(model, X_test, y_test)
        assert -1.0 <= metrics["R²"] <= 1.0

    def test_rmse_positive(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        model = train_model(X_train, y_train, "Linear Regression")
        metrics = evaluate(model, X_test, y_test)
        assert metrics["RMSE"] >= 0

    def test_coefficients_length(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        model = train_model(X_train, y_train, "Linear Regression")
        coef_df = get_coefficients(model)
        assert len(coef_df) == len(NUMERIC_FEATURES)

    def test_cross_validate_returns_dict(self, full_df):
        result = cross_validate(full_df, NUMERIC_FEATURES, "Linear Regression")
        assert "R² mean" in result
        assert "RMSE mean" in result

    def test_ridge_has_smaller_coefs_than_linear(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        lr = train_model(X_train, y_train, "Linear Regression")
        ridge = train_model(X_train, y_train, "Ridge (α=10)")
        lr_norm = np.linalg.norm(lr.coef_)
        ridge_norm = np.linalg.norm(ridge.coef_)
        assert ridge_norm <= lr_norm, "Ridge should shrink coefficients"

    def test_lasso_sparsity(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        lasso = train_model(X_train, y_train, "Lasso (α=1)")
        n_zero = np.sum(np.abs(lasso.coef_) < 1e-6)
        # Lasso should zero out at least one feature with alpha=1
        assert n_zero >= 1, "Lasso should produce sparse coefficients"

    def test_all_model_names_valid(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        from src.models.regression import MODELS
        for name in MODELS:
            model = train_model(X_train, y_train, name)
            assert hasattr(model, "predict")


# ─── Integration test ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: load → clean → scale → cluster → regress."""
        df = clean(load_raw())
        df = add_features(df)
        df_s, scaler = scale_features(df)
        X_tr, X_te, y_tr, y_te = split(df_s)

        # Cluster
        df_c, _, _, _ = run_kmeans(df, k=3)
        assert len(df_c) == len(df)

        # Regress
        model = train_model(X_tr, y_tr, "Linear Regression")
        metrics = evaluate(model, X_te, y_te)
        assert metrics["R²"] > 0.5, "Expected R² > 0.5 on this dataset"

    def test_pipeline_determinism(self):
        """Same random_state produces identical results."""
        df = add_features(clean(load_raw()))
        df_s, _ = scale_features(df)
        X_tr1, X_te1, y_tr1, _ = split(df_s, random_state=42)
        X_tr2, X_te2, y_tr2, _ = split(df_s, random_state=42)
        pd.testing.assert_frame_equal(X_tr1, X_tr2)
