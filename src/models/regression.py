"""
City Pulse — Regression Module
================================
Linear, Ridge, and Lasso regression for predicting liveability.
Includes model evaluation, coefficient plots, residual diagnostics,
cross-validation, and feature importance comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
from typing import Dict, List, Optional, Tuple

from src.data.loader import NUMERIC_FEATURES, TARGET, FEATURE_DISPLAY_NAMES

PALETTE = {
    "primary": "#7F77DD",
    "secondary": "#1D9E75",
    "accent": "#EF9F27",
    "danger": "#D85A30",
    "muted": "#888780",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "white",
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )


# ─── 1. Model Training ────────────────────────────────────────────────────────

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Ridge (α=10)": Ridge(alpha=10.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10_000),
    "Lasso (α=1)": Lasso(alpha=1.0, max_iter=10_000),
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10_000),
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Linear Regression",
) -> object:
    """Fit a named model (from MODELS dict) and return the fitted estimator."""
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    print(f"[train] Fitted '{model_name}' on {len(X_train)} samples.")
    return model


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """Fit all models in MODELS and return dict of fitted estimators."""
    fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


# ─── 2. Evaluation ────────────────────────────────────────────────────────────

def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Return a dict of evaluation metrics: R², RMSE, MAE, Max Error."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    max_err = np.max(np.abs(y_test - y_pred))
    metrics = {"R²": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "Max Error": round(max_err, 4)}
    print(f"[evaluate] {model_name} → R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return metrics


def evaluate_all(
    fitted_models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate every model in fitted_models and return a comparison DataFrame.
    """
    rows = []
    for name, model in fitted_models.items():
        m = evaluate(model, X_test, y_test, name)
        rows.append({"Model": name, **m})
    return pd.DataFrame(rows).set_index("Model").sort_values("R²", ascending=False)


def cross_validate(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    model_name: str = "Linear Regression",
    cv: int = 5,
) -> Dict[str, float]:
    """
    Run k-fold cross validation. Returns mean and std for R² and RMSE.
    Uses a pipeline so scaling is done inside each fold (no leakage).
    """
    features = features or NUMERIC_FEATURES
    X = df[features]
    y = df[TARGET]
    pipe = Pipeline([("scaler", StandardScaler()), ("model", MODELS[model_name])])

    r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    rmse_scores = cross_val_score(
        pipe, X, y, cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    return {
        "cv_folds": cv,
        "R² mean": round(r2_scores.mean(), 4),
        "R² std": round(r2_scores.std(), 4),
        "RMSE mean": round(-rmse_scores.mean(), 4),
        "RMSE std": round(rmse_scores.std(), 4),
    }


# ─── 3. Coefficient Analysis ──────────────────────────────────────────────────

def get_coefficients(
    model,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return feature coefficients as a sorted DataFrame.
    Works for LinearRegression, Ridge, Lasso.
    """
    features = features or NUMERIC_FEATURES
    coefs = model.coef_
    df = pd.DataFrame(
        {"Feature": [FEATURE_DISPLAY_NAMES.get(f, f) for f in features], "Coefficient": coefs}
    )
    df["Abs Coefficient"] = df["Coefficient"].abs()
    df = df.sort_values("Abs Coefficient", ascending=False).reset_index(drop=True)
    return df


def plot_coefficients(
    fitted_models: Dict[str, object],
    features: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing coefficients across models.
    """
    _style()
    features = features or NUMERIC_FEATURES
    feat_labels = [FEATURE_DISPLAY_NAMES[f] for f in features]
    n_models = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, fitted_models.items()):
        coefs = model.coef_
        colors = [PALETTE["primary"] if c >= 0 else PALETTE["danger"] for c in coefs]
        ax.barh(feat_labels, coefs, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient")
        for i, (v, c) in enumerate(zip(coefs, colors)):
            ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)

    fig.suptitle("Feature Coefficients by Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── 4. Diagnostic Plots ──────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    city_names: Optional[pd.Series] = None,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Scatter of predicted vs actual, with a perfect-fit diagonal
    and labelled outliers.
    """
    _style()
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        y_pred, y_test.values,
        c=np.abs(residuals), cmap="RdYlGn_r",
        s=80, alpha=0.85, edgecolors="white", lw=0.6,
    )
    plt.colorbar(sc, ax=ax, label="|Residual|")

    lo, hi = min(y_pred.min(), y_test.min()) - 2, max(y_pred.max(), y_test.max()) + 2
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["muted"], lw=1.5, label="Perfect fit")

    if city_names is not None:
        thresh = np.std(residuals) * 1.5
        for i, (pred, actual, name) in enumerate(zip(y_pred, y_test.values, city_names)):
            if abs(residuals.iloc[i] if hasattr(residuals, "iloc") else residuals[i]) > thresh:
                ax.annotate(name, (pred, actual), fontsize=8, color="#444",
                            xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Predicted Liveability Score", fontsize=12)
    ax.set_ylabel("Actual Liveability Score", fontsize=12)
    ax.set_title(
        f"{model_name} — Predicted vs Actual\n"
        f"R² = {r2:.3f}   RMSE = {rmse:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> plt.Figure:
    """
    2×2 residual diagnostic panel:
    1. Residuals vs Fitted
    2. QQ-plot (normality of residuals)
    3. Scale-location plot
    4. Histogram of residuals
    """
    _style()
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, color=PALETTE["primary"], alpha=0.75, s=60, edgecolors="white")
    ax.axhline(0, color=PALETTE["danger"], lw=1.5, ls="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. QQ Plot
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color=PALETTE["primary"], alpha=0.75, s=60, edgecolors="white")
    xline = np.array([min(osm), max(osm)])
    ax.plot(xline, slope * xline + intercept, color=PALETTE["accent"], lw=2)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered residuals")
    ax.set_title(f"QQ Plot (r={r:.3f})")

    # 3. Scale-Location
    ax = axes[1, 0]
    ax.scatter(y_pred, np.sqrt(np.abs(std_resid)), color=PALETTE["secondary"], alpha=0.75, s=60, edgecolors="white")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("√|Standardised residuals|")
    ax.set_title("Scale-Location")

    # 4. Histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=15, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
    xkde = np.linspace(residuals.min(), residuals.max(), 200)
    kde = stats.gaussian_kde(residuals)
    ax2 = ax.twinx()
    ax2.plot(xkde, kde(xkde), color=PALETTE["accent"], lw=2)
    ax2.set_yticks([])
    ax.axvline(0, color=PALETTE["danger"], lw=1.5, ls="--")
    ax.set_xlabel("Residuals")
    ax.set_title("Residual Distribution")

    fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── 5. Learning Curves ───────────────────────────────────────────────────────

def plot_learning_curves(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    model_name: str = "Linear Regression",
) -> plt.Figure:
    """
    Learning curve: training and validation score as a function
    of training set size. Reveals bias vs variance.
    """
    _style()
    features = features or NUMERIC_FEATURES
    X = df[features].values
    y = df[TARGET].values

    pipe = Pipeline([("scaler", StandardScaler()), ("model", MODELS[model_name])])
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 10),
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", color=PALETTE["primary"], lw=2, label="Training R²")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=PALETTE["primary"])
    ax.plot(train_sizes, val_mean, "s-", color=PALETTE["accent"], lw=2, label="Validation R²")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=PALETTE["accent"])
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R² score")
    ax.set_title(f"Learning Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


# ─── 6. Regularisation Path ───────────────────────────────────────────────────

def plot_regularisation_path(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Optional[List[str]] = None,
    model_type: str = "ridge",
) -> plt.Figure:
    """
    Show how coefficients shrink as regularisation strength (alpha) increases.
    Teaches the concept of L1/L2 regularisation.
    """
    _style()
    features = features or NUMERIC_FEATURES
    feat_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
    alphas = np.logspace(-3, 3, 60)
    coef_paths = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[features])

    for alpha in alphas:
        if model_type == "lasso":
            m = Lasso(alpha=alpha, max_iter=10_000).fit(X_scaled, y_train)
        else:
            m = Ridge(alpha=alpha).fit(X_scaled, y_train)
        coef_paths.append(m.coef_)

    coef_paths = np.array(coef_paths)
    colors = ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#378ADD", "#D4537E"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, color) in enumerate(zip(feat_labels, colors)):
        ax.semilogx(alphas, coef_paths[:, i], lw=2, color=color, label=label)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("Alpha (regularisation strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title(
        f"{'Lasso' if model_type=='lasso' else 'Ridge'} Regularisation Path",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig
