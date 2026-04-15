# 🌆 City Pulse — Urban Liveability Data Science Project

A complete, structured data science project for university students. Explore what makes a city liveable through EDA, K-Means clustering, and regression — on a real-world-style dataset of 40 global cities.

---

## What you'll learn

| Skill | Where |
|-------|-------|
| Data loading & validation | `src/data/loader.py` |
| Exploratory data analysis | `src/visualization/eda_plots.py` |
| Feature scaling & train/test splits | `src/data/loader.py` |
| K-Means clustering + elbow method | `src/models/clustering.py` |
| Silhouette analysis | `src/models/clustering.py` |
| PCA dimensionality reduction | `src/models/clustering.py` |
| Linear, Ridge, and Lasso regression | `src/models/regression.py` |
| Residual diagnostics & learning curves | `src/models/regression.py` |
| Regularisation paths | `src/models/regression.py` |
| Writing pytest tests | `tests/test_pipeline.py` |
| Building a Streamlit dashboard | `streamlit_app/app.py` |

---

## Project structure

```
city-pulse/
├── data/
│   └── cities.csv              # 40 cities × 7 features
├── notebooks/
│   └── 01_city_pulse_walkthrough.ipynb   # Full guided notebook
├── src/
│   ├── data/
│   │   └── loader.py           # Loading, cleaning, feature engineering, scaling
│   ├── models/
│   │   ├── clustering.py       # K-Means, elbow, silhouette, PCA, radar
│   │   └── regression.py       # Linear/Ridge/Lasso, diagnostics, regularisation paths
│   └── visualization/
│       └── eda_plots.py        # Distributions, correlation matrix, scatter, outliers
├── streamlit_app/
│   └── app.py                  # Interactive web dashboard
├── tests/
│   └── test_pipeline.py        # 30+ pytest unit & integration tests
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
cd notebooks
jupyter notebook 01_city_pulse_walkthrough.ipynb
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

### 4. Run the tests

```bash
pytest tests/ -v
```

---

## Dataset

`data/cities.csv` — 40 global cities across 6 continents with the following features:

| Column | Description | Range |
|--------|-------------|-------|
| `air_quality` | Air quality index (higher = cleaner) | 0–100 |
| `green_space` | % of urban area that is parks/greenery | 0–100 |
| `transit_score` | Public transport connectivity & coverage | 0–100 |
| `safety_index` | Composite safety / low crime score | 0–100 |
| `cost_of_living` | Cost index (higher = more expensive) | 0–100 |
| `noise_level` | Urban noise pollution (higher = noisier) | 0–100 |
| `liveability_score` | **Target** — composite liveability rating | 0–100 |

**Note on cost and noise:** Higher values mean *worse* outcomes. When interpreting coefficients, a *negative* regression coefficient for these features is the expected and correct result.

---

## Step-by-step learning path

1. **Load & validate** — `loader.py` walks you through every quality check
2. **Explore** — run distributions, scatter plots, and the correlation matrix
3. **Preprocess** — scale inside a Pipeline to avoid data leakage
4. **Cluster** — use the elbow chart + silhouette score to pick K
5. **Regress** — compare Linear, Ridge, and Lasso; plot residuals
6. **Interpret** — write a data narrative for a non-technical audience
7. **Extend** — Random Forest, SHAP, DBSCAN, or a live API integration

---

## Extension challenges

- Replace simulated data with real OpenAQ air quality readings
- Add SHAP explainability (`pip install shap`)
- Try DBSCAN — which cities are "noise" (outliers)?
- Build a city recommender: given user preferences, find the best-fit city
- Deploy the Streamlit app to [Streamlit Cloud](https://streamlit.io/cloud) (free)

---

## Grading rubric (suggested)

| Component | Marks | Criteria |
|-----------|-------|----------|
| Data loading & cleaning | 10 | No errors, handles edge cases |
| EDA (plots + written analysis) | 20 | ≥4 informative plots with commentary |
| Clustering | 20 | Elbow, silhouette, PCA scatter, cluster profiles |
| Regression | 25 | Train/test split, cross-validation, ≥2 models |
| Interpretation narrative | 15 | Clear, non-technical, data-backed |
| Code quality & comments | 10 | PEP8, docstrings, modular functions |
| Extension (bonus) | +10 | Any one extension challenge completed |

---

## License

MIT — free to use, adapt, and redistribute for educational purposes.
