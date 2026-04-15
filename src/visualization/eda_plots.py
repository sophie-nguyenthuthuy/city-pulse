"""
City Pulse — Exploratory Data Analysis
=======================================
Statistical summaries, correlation analysis, distribution checks,
and outlier detection. All functions return figures or DataFrames
so they work in both notebooks and the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from typing import List, Optional, Tuple

from src.data.loader import NUMERIC_FEATURES, TARGET, FEATURE_DISPLAY_NAMES

# ─── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#7F77DD",
    "secondary": "#1D9E75",
    "accent": "#EF9F27",
    "danger": "#D85A30",
    "muted": "#888780",
}
CONTINENT_COLORS = {
    "Europe": "#7F77DD",
    "Asia": "#1D9E75",
    "North America": "#EF9F27",
    "South America": "#D85A30",
    "Oceania": "#378ADD",
    "Africa": "#D4537E",
}


def _style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F8F6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.2,
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )


# ─── 1. Overview ──────────────────────────────────────────────────────────────

def overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a rich summary table: mean, std, min, max, skewness,
    kurtosis, and missing count for every numeric feature.
    """
    rows = []
    for col in NUMERIC_FEATURES + [TARGET]:
        s = df[col]
        rows.append(
            {
                "Feature": FEATURE_DISPLAY_NAMES.get(col, col),
                "Mean": round(s.mean(), 2),
                "Std": round(s.std(), 2),
                "Min": round(s.min(), 2),
                "Max": round(s.max(), 2),
                "Skewness": round(s.skew(), 3),
                "Kurtosis": round(s.kurt(), 3),
                "Missing": int(s.isnull().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("Feature")


# ─── 2. Distributions ─────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, features: Optional[List[str]] = None) -> plt.Figure:
    """
    Grid of histograms + KDE curves for every numeric feature.
    Includes a vertical line for the mean.
    """
    _style()
    features = features or NUMERIC_FEATURES
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna()
        ax.hist(data, bins=15, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
        ax2 = ax.twinx()
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde = stats.gaussian_kde(data)
        ax2.plot(kde_x, kde(kde_x), color=PALETTE["accent"], lw=2)
        ax2.set_yticks([])
        ax.axvline(data.mean(), color=PALETTE["danger"], lw=1.5, ls="--", label=f"mean={data.mean():.1f}")
        ax.set_title(FEATURE_DISPLAY_NAMES.get(feat, feat), fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (0–100)")
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ─── 3. Correlation Matrix ────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Annotated heatmap of Pearson correlations among all numeric features
    including the target. Uses a purple-white-coral diverging palette.
    """
    _style()
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(260, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        square=True,
    )
    labels = [FEATURE_DISPLAY_NAMES.get(c, c) for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold", pad=16)
    fig.tight_layout()
    return fig


def top_correlations(df: pd.DataFrame, target: str = TARGET, n: int = 6) -> pd.Series:
    """Return top-n features most correlated with the target, sorted by |r|."""
    corr = df[NUMERIC_FEATURES].corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


# ─── 4. Scatter plots ─────────────────────────────────────────────────────────

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "continent",
    annotate: bool = True,
) -> plt.Figure:
    """
    Scatter plot of x vs y, colored by continent (or any hue column).
    Overlays a linear regression trend line and annotates outlier cities.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))

    continents = df[hue].unique() if hue in df.columns else ["all"]
    for cont in continents:
        sub = df[df[hue] == cont] if hue in df.columns else df
        color = CONTINENT_COLORS.get(cont, PALETTE["muted"])
        ax.scatter(sub[x], sub[y], c=color, label=cont, alpha=0.85, s=70, edgecolors="white", lw=0.5)

    # Trend line
    m, b, r, p, _ = stats.linregress(df[x], df[y])
    xr = np.linspace(df[x].min(), df[x].max(), 100)
    ax.plot(xr, m * xr + b, color=PALETTE["muted"], lw=1.5, ls="--", alpha=0.6)

    # Annotate outliers (residuals > 1.5 std)
    if annotate:
        predicted = m * df[x] + b
        residuals = df[y] - predicted
        thresh = residuals.std() * 1.5
        outliers = df[np.abs(residuals) > thresh]
        for _, row in outliers.iterrows():
            ax.annotate(
                row["city"],
                (row[x], row[y]),
                fontsize=8,
                color="#444",
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel(FEATURE_DISPLAY_NAMES.get(x, x), fontsize=12)
    ax.set_ylabel(FEATURE_DISPLAY_NAMES.get(y, y), fontsize=12)
    ax.set_title(
        f"{FEATURE_DISPLAY_NAMES.get(x,x)} vs {FEATURE_DISPLAY_NAMES.get(y,y)}  "
        f"(r = {r:.3f}, p = {p:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title=hue.capitalize(), fontsize=9, title_fontsize=9, loc="best")
    fig.tight_layout()
    return fig


# ─── 5. Pairplot ──────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, features: Optional[List[str]] = None) -> plt.Figure:
    """
    Seaborn pairplot for a subset of features, coloured by continent.
    Returns the underlying Figure object.
    """
    features = features or ["air_quality", "green_space", "transit_score", TARGET]
    palette = {k: v for k, v in CONTINENT_COLORS.items() if k in df["continent"].values}
    g = sns.pairplot(
        df[features + ["continent"]],
        hue="continent",
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "s": 50, "edgecolor": "white"},
    )
    g.figure.suptitle("Pairplot of Key Features", y=1.02, fontsize=14, fontweight="bold")
    return g.figure


# ─── 6. Outlier Detection ─────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Flag outliers using the IQR method (1.5× IQR rule).
    Returns a DataFrame with a boolean column per feature.
    """
    features = features or NUMERIC_FEATURES
    result = df[["city"]].copy()
    for feat in features:
        q1, q3 = df[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        result[feat] = (df[feat] < q1 - 1.5 * iqr) | (df[feat] > q3 + 1.5 * iqr)
    result["any_outlier"] = result[features].any(axis=1)
    return result


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag outliers using Z-score. |Z| > threshold => outlier.
    """
    result = df[["city"]].copy()
    for feat in NUMERIC_FEATURES:
        z = np.abs(stats.zscore(df[feat]))
        result[feat] = z > threshold
    result["any_outlier"] = result[NUMERIC_FEATURES].any(axis=1)
    return result


def plot_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Box-and-whisker plots for all features, sorted by median."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [df[f].dropna().values for f in NUMERIC_FEATURES]
    labels = [FEATURE_DISPLAY_NAMES[f] for f in NUMERIC_FEATURES]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        notch=True,
        medianprops=dict(color=PALETTE["accent"], lw=2),
    )
    colors = list(PALETTE.values())
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Feature Distributions — Boxplots", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0–100)")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    return fig


# ─── 7. Geographic summary ────────────────────────────────────────────────────

def continent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of all numeric features grouped by continent, sorted by liveability."""
    cols = NUMERIC_FEATURES + [TARGET]
    return (
        df.groupby("continent")[cols]
        .mean()
        .round(1)
        .sort_values(TARGET, ascending=False)
    )


def plot_continent_bars(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of mean liveability per continent."""
    _style()
    summary = continent_summary(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [CONTINENT_COLORS.get(c, PALETTE["muted"]) for c in summary.index]
    bars = ax.bar(summary.index, summary[TARGET], color=colors, edgecolor="white", width=0.6)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Mean Liveability Score by Continent", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Liveability Score")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig
