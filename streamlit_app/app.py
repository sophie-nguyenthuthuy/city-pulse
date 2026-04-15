"""
City Pulse — Streamlit Dashboard
==================================
Run with:  streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_raw, clean, add_features, NUMERIC_FEATURES, TARGET, FEATURE_DISPLAY_NAMES
from src.visualization.eda_plots import (
    overview, plot_distributions, plot_correlation_matrix,
    plot_scatter, plot_boxplots, continent_summary, plot_continent_bars, top_correlations,
)
from src.models.clustering import (
    run_kmeans, plot_pca_clusters, plot_radar, plot_elbow, describe_clusters, FEATURE_GROUPS,
)
from src.models.regression import (
    train_model, evaluate, cross_validate, get_coefficients,
    plot_predicted_vs_actual, plot_residuals, plot_learning_curves, MODELS,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="City Pulse — Urban Data Explorer",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_raw()
    df = clean(df)
    df = add_features(df)
    return df

df = get_data()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌆 City Pulse")
    st.markdown("Urban liveability explorer for data science students.")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "EDA", "Clustering", "Regression", "Compare Models", "Learning Guide"],
        index=0,
    )
    st.divider()
    st.markdown("**Dataset**")
    st.markdown(f"- {len(df)} cities · {len(NUMERIC_FEATURES)} features")
    st.markdown("- Target: `liveability_score`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🌆 City Pulse — Urban Liveability Dataset")
    st.markdown(
        "Explore urban health indicators across **40 global cities**. "
        "Use the sidebar to navigate through EDA, clustering, and regression modules."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", len(df))
    c2.metric("Features", len(NUMERIC_FEATURES))
    c3.metric("Most liveable", df.loc[df[TARGET].idxmax(), "city"])
    c4.metric("Least liveable", df.loc[df[TARGET].idxmin(), "city"])

    st.subheader("Dataset preview")
    display_cols = ["city", "country", "continent"] + NUMERIC_FEATURES + [TARGET]
    st.dataframe(df[display_cols].sort_values(TARGET, ascending=False), use_container_width=True, height=380)

    st.subheader("Summary statistics")
    st.dataframe(overview(df), use_container_width=True)

    st.subheader("Liveability by continent")
    fig = plot_continent_bars(df)
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Scatter", "Correlation", "Outliers"])

    with tab1:
        st.subheader("Feature distributions")
        fig = plot_distributions(df)
        st.pyplot(fig, use_container_width=True)
        st.subheader("Boxplots")
        fig2 = plot_boxplots(df)
        st.pyplot(fig2, use_container_width=True)

    with tab2:
        st.subheader("Scatter explorer")
        c1, c2 = st.columns(2)
        x_feat = c1.selectbox("X axis", NUMERIC_FEATURES, index=0,
                               format_func=lambda x: FEATURE_DISPLAY_NAMES[x])
        y_feat = c2.selectbox("Y axis", NUMERIC_FEATURES + [TARGET], index=len(NUMERIC_FEATURES),
                               format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x))
        fig = plot_scatter(df, x_feat, y_feat)
        st.pyplot(fig, use_container_width=True)

        corr_top = top_correlations(df)
        st.markdown("**Top correlations with liveability score**")
        st.dataframe(corr_top.rename("Pearson r").to_frame(), use_container_width=True)

    with tab3:
        st.subheader("Correlation matrix")
        fig = plot_correlation_matrix(df)
        st.pyplot(fig, use_container_width=True)

    with tab4:
        st.subheader("Outlier detection (IQR method)")
        from src.visualization.eda_plots import detect_outliers_iqr
        outliers = detect_outliers_iqr(df)
        flag_cols = [c for c in outliers.columns if c not in ["city", "any_outlier"]]
        styled = outliers.set_index("city")[flag_cols + ["any_outlier"]]
        st.dataframe(
            styled.style.map(lambda v: "background-color:#FAECE7" if v else "", subset=flag_cols + ["any_outlier"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title("K-Means Clustering")
    st.markdown("Group cities by similarity. Use the controls to explore different K values and feature sets.")

    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (K)", 2, 7, 3)
        feat_group = st.selectbox(
            "Feature group",
            list(FEATURE_GROUPS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        run_btn = st.button("Run K-Means", type="primary")

    if run_btn or "cluster_df" not in st.session_state or st.session_state.get("cluster_k") != k:
        df_c, km, scaler, feats = run_kmeans(df, k=k, feature_group=feat_group)
        st.session_state["cluster_df"] = df_c
        st.session_state["cluster_k"] = k
        st.session_state["cluster_feats"] = feats
        st.session_state["cluster_km"] = km
    else:
        df_c = st.session_state["cluster_df"]
        feats = st.session_state["cluster_feats"]

    tab1, tab2, tab3, tab4 = st.tabs(["PCA Scatter", "Radar", "Elbow Analysis", "City Assignments"])

    with tab1:
        fig = plot_pca_clusters(df_c, feats)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig = plot_radar(df_c, feats)
        st.pyplot(fig, use_container_width=True)

    with tab3:
        X = df[feats].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        fig = plot_elbow(X_scaled)
        st.pyplot(fig, use_container_width=True)

    with tab4:
        st.dataframe(describe_clusters(df_c), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Regression":
    st.title("Liveability Regression")

    c1, c2 = st.columns([1, 2])
    with c1:
        model_name = st.selectbox("Model", list(MODELS.keys()))
        selected_feats = st.multiselect(
            "Features", NUMERIC_FEATURES,
            default=NUMERIC_FEATURES,
            format_func=lambda x: FEATURE_DISPLAY_NAMES[x],
        )
        if not selected_feats:
            st.warning("Select at least one feature.")
            st.stop()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = df[selected_feats]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_feats, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=selected_feats, index=X_test.index)

    model = train_model(X_train_s, y_train, model_name)
    metrics = evaluate(model, X_test_s, y_test, model_name)
    cv_res = cross_validate(df, selected_feats, model_name)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("MAE", f"{metrics['MAE']:.2f}")
    m4.metric("CV R² (5-fold)", f"{cv_res['R² mean']:.3f} ± {cv_res['R² std']:.3f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Predicted vs Actual", "Residuals", "Coefficients", "Learning Curve"])

    city_names_test = df.loc[X_test.index, "city"]

    with tab1:
        fig = plot_predicted_vs_actual(model, X_test_s, y_test, city_names_test, model_name)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig = plot_residuals(model, X_test_s, y_test, model_name)
        st.pyplot(fig, use_container_width=True)

    with tab3:
        coef_df = get_coefficients(model, selected_feats)
        st.dataframe(coef_df, use_container_width=True)

    with tab4:
        fig = plot_learning_curves(df, selected_feats, model_name)
        st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare Models":
    st.title("Model Comparison")
    st.markdown("Train all regression models and compare their test-set performance.")

    from sklearn.model_selection import train_test_split
    from src.models.regression import train_all, evaluate_all, plot_coefficients

    scaler = StandardScaler()
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=NUMERIC_FEATURES, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=NUMERIC_FEATURES, index=X_test.index)

    fitted = train_all(X_train_s, y_train)
    results = evaluate_all(fitted, X_test_s, y_test)

    st.subheader("Evaluation metrics")
    st.dataframe(
        results.style.background_gradient(subset=["R²"], cmap="Greens")
                     .background_gradient(subset=["RMSE", "MAE"], cmap="Reds_r"),
        use_container_width=True,
    )

    st.subheader("Coefficient comparison")
    fig = plot_coefficients(fitted)
    st.pyplot(fig, use_container_width=True)

    from src.models.regression import plot_regularisation_path
    st.subheader("Ridge regularisation path")
    fig = plot_regularisation_path(X_train_s, y_train, model_type="ridge")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Lasso regularisation path")
    fig = plot_regularisation_path(X_train_s, y_train, model_type="lasso")
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LEARNING GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Learning Guide":
    st.title("Learning Guide")
    st.markdown("A structured 7-step guide for completing this project as a university assignment.")

    steps = [
        ("Data Collection & Loading",
         "Load `cities.csv` with pandas. Call `df.info()` and `df.isnull().sum()`. "
         "Discuss: What does each feature represent? What potential biases exist in the data? "
         "Write a short paragraph documenting your observations."),
        ("Exploratory Data Analysis",
         "Plot histograms for every feature. Use the Scatter explorer to find correlated pairs. "
         "Compute the correlation matrix. Which features have the strongest linear relationship "
         "with liveability? Are any features redundant (multicollinear)?"),
        ("Data Preprocessing",
         "Standardize features using `StandardScaler`. Discuss why this matters for "
         "distance-based algorithms. Split data 80/20 using `train_test_split`. "
         "Consider: should you use stratified splitting? Why or why not?"),
        ("Clustering — K-Means",
         "Implement K-Means. Use the elbow method AND silhouette scores to choose K. "
         "Visualise clusters via PCA. Profile each cluster: what kind of city does each represent? "
         "Name the clusters (e.g. 'High-income, high-liveability'). Discuss limitations."),
        ("Regression — Predict Liveability",
         "Fit `LinearRegression`. Evaluate with R², RMSE, and MAE. Plot predicted vs actual. "
         "Run 5-fold cross-validation (no data leakage — use a Pipeline). "
         "Try Ridge and Lasso. Which regularisation helps? Why might Lasso be useful here?"),
        ("Interpretation & Storytelling",
         "Write a 500-word narrative: Which features matter most for liveability? "
         "Are 'liveable but expensive' cities a real cluster? Does air quality dominate? "
         "Present findings as if briefing a city mayor who has no data science background."),
        ("Extension Challenges",
         "- Try `RandomForestRegressor` and compare feature importances to linear coefficients.\n"
         "- Add SHAP values (`pip install shap`) for explainability.\n"
         "- Try DBSCAN clustering — how does it handle outlier cities?\n"
         "- Fetch real air quality data from OpenAQ API and replace simulated values.\n"
         "- Deploy your model as a Streamlit app (this file is the template!).\n"
         "- Write a pytest test suite for your preprocessing functions."),
    ]

    for i, (title, body) in enumerate(steps, 1):
        with st.expander(f"Step {i}: {title}", expanded=(i == 1)):
            st.markdown(body)

    st.divider()
    st.subheader("Grading rubric (suggested)")
    rubric = pd.DataFrame({
        "Component": ["Data loading & cleaning", "EDA (plots + analysis)", "Clustering",
                       "Regression", "Interpretation narrative", "Code quality & comments", "Extension (bonus)"],
        "Marks": [10, 20, 20, 25, 15, 10, "+10 bonus"],
        "Criteria": [
            "No errors, handles edge cases",
            "At least 4 informative plots with commentary",
            "Elbow plot, silhouette, PCA scatter, cluster profiles",
            "Train/test split, cross-validation, ≥2 models compared",
            "Clear, non-technical language, data-backed claims",
            "PEP8, docstrings, modular functions",
            "Any one extension challenge completed",
        ],
    })
    st.dataframe(rubric.set_index("Component"), use_container_width=True)
