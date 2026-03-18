"""
ml_pipeline.py
──────────────
Two-stage ML pipeline for exoplanet habitability analysis.

Stage 1 — Unsupervised Clustering (KMeans)
    Groups exoplanets into natural clusters based on physical properties.

Stage 2 — Supervised Classification (Random Forest)
    Predicts habitability tier from physical features.
    Trains on scored data, reports metrics, saves the trained model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/habitable_scored_exoplanets.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "outputs/plots"
RANDOM_SEED = 42

FEATURES = ["pl_rade", "pl_bmasse", "pl_eqt"]
TARGET   = "habitability_tier"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(path)
    available = [f for f in FEATURES if f in df.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        print(f"  [warn] Missing feature columns: {missing}")
    return df, available


def elbow_plot(X_scaled: np.ndarray, max_k: int = 10):
    inertias = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), inertias, "o-", color="#4f86c6", linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot — KMeans Clustering")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/elbow_plot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {PLOT_DIR}/elbow_plot.png")


def cluster_scatter(df: pd.DataFrame, x: str = "pl_rade", y: str = "pl_eqt"):
    n_clusters = df["cluster"].nunique()
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, grp in df.groupby("cluster"):
        ax.scatter(grp[x], grp[y], label=f"Cluster {i}",
                   color=colors[i], alpha=0.75, edgecolors="k", linewidths=0.3, s=60)

    ax.set_xlabel("Planet Radius (R⊕)")
    ax.set_ylabel("Equilibrium Temperature (K)")
    ax.set_title("Exoplanet Clusters (KMeans)")
    ax.legend(title="Cluster")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/cluster_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {PLOT_DIR}/cluster_scatter.png")


def feature_importance_plot(model: Pipeline, feature_names: list):
    rf  = model.named_steps["clf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(imp)), imp[idx], color="#4f86c6", edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {PLOT_DIR}/feature_importance.png")


# ── Stage 1: Clustering ───────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, available_features: list, n_clusters: int = 4):
    print("\n── Stage 1: KMeans Clustering ──────────────────────────────────────")
    X = df[available_features].copy()

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X)
    X_sc    = scaler.fit_transform(X_imp)

    elbow_plot(X_sc)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    df["cluster"] = km.fit_predict(X_sc)

    print(f"  Cluster distribution:\n{df['cluster'].value_counts().sort_index().to_string()}")

    if "pl_rade" in df.columns and "pl_eqt" in df.columns:
        cluster_scatter(df)

    summary = df.groupby("cluster")[available_features + ["habitability_score"]].mean().round(3)
    print(f"\n  Cluster mean features:\n{summary.to_string()}")

    joblib.dump({"imputer": imputer, "scaler": scaler, "kmeans": km},
                f"{MODEL_DIR}/kmeans_pipeline.pkl")
    print(f"  Saved → {MODEL_DIR}/kmeans_pipeline.pkl")

    return df


# ── Stage 2: Classification ───────────────────────────────────────────────────

def run_classification(df: pd.DataFrame, available_features: list):
    print("\n── Stage 2: Random Forest Classification ───────────────────────────")

    # Use habitability_score to create tier labels if TARGET column missing
    if TARGET not in df.columns:
        print(f"  '{TARGET}' not found — generating from habitability_score")
        bins   = [0, 0.60, 0.75, 0.85, 1.01]
        labels = ["Low", "Moderate", "High", "Very High"]
        df[TARGET] = pd.cut(df["habitability_score"], bins=bins,
                            labels=labels, right=False).astype(str)

    df = df.dropna(subset=[TARGET])
    X = df[available_features]
    y = df[TARGET].astype(str)

    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    print(f"\n  5-Fold CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Habitability Tier")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {PLOT_DIR}/confusion_matrix.png")

    feature_importance_plot(pipeline, available_features)

    joblib.dump(pipeline, f"{MODEL_DIR}/rf_classifier.pkl")
    print(f"  Saved → {MODEL_DIR}/rf_classifier.pkl")

    return pipeline


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading scored dataset …")
    df, available = load_data(DATA_PATH)
    print(f"  {df.shape[0]:,} planets | features: {available}")

    df = run_clustering(df, available, n_clusters=4)
    pipeline = run_classification(df, available)

    print("\n✓ Pipeline complete.")
    print(f"  Models → {MODEL_DIR}/")
    print(f"  Plots  → {PLOT_DIR}/")