"""
City Pulse — Clustering Module
================================
K-Means clustering with elbow method, silhouette analysis,
PCA visualization, and cluster profiling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

from src.data.loader import NUMERIC_FEATURES, FEATURE_DISPLAY_NAMES

# ─── Palette ──────────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#7F77DD", "#1D9E75", "#EF9F27", "#D85A30", "#378ADD", "#D4537E"]

FEATURE_GROUPS = {
    "all": NUMERIC_FEATURES,
    "environmental": ["air_quality", "green_space", "noise_level"],
    "social": ["transit_score", "safety_index", "cost_of_living"],
    "quality_of_life": ["air_quality", "green_space", "transit_score", "safety_index"],
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


# ─── 1. Elbow Method ──────────────────────────────────────────────────────────

def elbow_analysis(
    X: np.ndarray,
    k_range: range = range(2, 10),
    random_state: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Compute inertia and silhouette score for each K.

    Returns
    -------
    inertias         : list of inertia values
    silhouette_scores: list of silhouette scores (k >= 2)
    """
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    return inertias, sil_scores


def plot_elbow(
    X: np.ndarray,
    k_range: range = range(2, 10),
    random_state: int = 42,
) -> plt.Figure:
    """
    Side-by-side elbow (inertia) and silhouette charts.
    The optimal K is automatically highlighted.
    """
    _style()
    ks = list(k_range)
    inertias, sil_scores = elbow_analysis(X, k_range, random_state)
    optimal_k = ks[np.argmax(sil_scores)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(ks, inertias, "o-", color="#7F77DD", lw=2, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax1.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Within-cluster sum of squares")
    for k, v in zip(ks, inertias):
        ax1.annotate(f"{v:.0f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    # Silhouette
    ax2.bar(ks, sil_scores, color=["#7F77DD" if k != optimal_k else "#EF9F27" for k in ks], edgecolor="white", width=0.6)
    ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Mean silhouette score")
    ax2.axvline(optimal_k, color="#EF9F27", lw=2, ls="--", label=f"Optimal K = {optimal_k}")
    ax2.legend()

    fig.suptitle("Choosing the Optimal Number of Clusters", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── 2. Run K-Means ───────────────────────────────────────────────────────────

def run_kmeans(
    df: pd.DataFrame,
    k: int = 3,
    feature_group: str = "all",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit K-Means and return the input DataFrame with a 'cluster' column added.

    Parameters
    ----------
    df            : cleaned city DataFrame (unscaled)
    k             : number of clusters
    feature_group : key in FEATURE_GROUPS dict
    random_state  : for reproducibility
    """
    features = FEATURE_GROUPS.get(feature_group, NUMERIC_FEATURES)
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = labels
    df["cluster_label"] = df["cluster"].apply(lambda x: f"Cluster {x + 1}")

    sil = silhouette_score(X_scaled, labels)
    print(f"[kmeans] K={k}, features={feature_group}, silhouette={sil:.3f}")
    return df, km, scaler, features


# ─── 3. Cluster Profiles ──────────────────────────────────────────────────────

def cluster_profiles(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return mean feature values per cluster + city count.
    Useful for interpreting what each cluster represents.
    """
    features = features or NUMERIC_FEATURES
    profile = (
        df.groupby("cluster_label")[features + ["liveability_score"]]
        .agg(["mean", "count"])
    )
    # Flatten column names
    profile.columns = [f"{col}_{stat}" for col, stat in profile.columns]
    return profile.round(2)


def describe_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Human-readable summary: for each cluster, list the cities,
    mean liveability, and the highest/lowest feature.
    """
    rows = []
    for label, group in df.groupby("cluster_label"):
        feat_means = group[NUMERIC_FEATURES].mean()
        rows.append(
            {
                "Cluster": label,
                "Cities": ", ".join(sorted(group["city"].tolist())),
                "Count": len(group),
                "Mean Liveability": round(group["liveability_score"].mean(), 1),
                "Strongest Feature": FEATURE_DISPLAY_NAMES[feat_means.idxmax()],
                "Weakest Feature": FEATURE_DISPLAY_NAMES[feat_means.idxmin()],
            }
        )
    return pd.DataFrame(rows).set_index("Cluster")


# ─── 4. PCA Scatter ───────────────────────────────────────────────────────────

def plot_pca_clusters(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Reduce to 2 PCA components and plot clusters.
    Annotates city names for notable outliers.
    """
    _style()
    features = features or NUMERIC_FEATURES
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (label, group) in enumerate(df.groupby("cluster_label")):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.scatter(
            group["pc1"], group["pc2"],
            c=color, label=label, s=90, alpha=0.85,
            edgecolors="white", lw=0.6,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["city"], (row["pc1"], row["pc2"]),
                fontsize=7.5, color="#444",
                xytext=(4, 3), textcoords="offset points",
            )

    var1, var2 = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12)
    ax.set_title(
        f"K-Means Clusters — PCA Projection  "
        f"(total variance explained: {var1+var2:.1f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="Cluster", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    return fig


# ─── 5. Radar Chart ───────────────────────────────────────────────────────────

def plot_radar(df: pd.DataFrame, features: Optional[List[str]] = None) -> plt.Figure:
    """
    Radar / spider chart showing the mean profile of each cluster
    across selected features.
    """
    features = features or NUMERIC_FEATURES
    labels = [FEATURE_DISPLAY_NAMES[f] for f in features]
    N = len(features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    cluster_means = df.groupby("cluster_label")[features].mean()

    for i, (cluster, means) in enumerate(cluster_means.iterrows()):
        values = list(means.values)
        values += values[:1]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.plot(angles, values, "o-", lw=2, color=color, label=cluster)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
    ax.set_title("Cluster Feature Profiles", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    return fig


# ─── 6. Silhouette Plot ───────────────────────────────────────────────────────

def plot_silhouette(X: np.ndarray, labels: np.ndarray, k: int) -> plt.Figure:
    """
    Per-sample silhouette coefficient plot, sorted within each cluster.
    Highlights clusters with negative silhouette values.
    """
    _style()
    sil_vals = silhouette_samples(X, labels)
    avg = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(k):
        cluster_vals = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(cluster_vals)
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, cluster_vals,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(-0.05, (y_lower + y_upper) / 2, f"C{i+1}", fontsize=9, ha="right")
        y_lower = y_upper + 10

    ax.axvline(avg, color="#EF9F27", lw=2, ls="--", label=f"Avg = {avg:.3f}")
    ax.set_title(f"Silhouette Plot — K={k}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    return fig
