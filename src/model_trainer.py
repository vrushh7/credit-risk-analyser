"""
model_trainer.py
----------------
Trains and evaluates three ML models for credit risk prediction:
  1. Logistic Regression  — fast, interpretable, industry baseline
  2. Random Forest        — ensemble, handles non-linearity, robust
  3. Gradient Boosting    — highest accuracy, production favourite

Evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curve
  - Confusion matrix
  - Feature importance (RF & GB)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics             import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
TEXT_COL = "#eaeaea"
COLORS   = {"Logistic Regression": "#3498db",
            "Random Forest":       "#2ecc71",
            "Gradient Boosting":   "#e94560"}
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor":   "#444",  "axes.labelcolor": TEXT_COL,
    "text.color":       TEXT_COL,"xtick.color":     TEXT_COL,
    "ytick.color":      TEXT_COL,"grid.color":      "#333",
    "grid.linestyle":   "--",    "grid.alpha":      0.5,
    "font.family":      "monospace",
})


def _save(fig, filename):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[INFO] Saved: {path}")


# ── Model definitions ──────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dict of model_name → sklearn estimator.

    Why these models?
    -----------------
    Logistic Regression:
      - Linear decision boundary; works well when features scale linearly
      - Outputs calibrated probabilities (essential for credit scoring)
      - Highly interpretable coefficients — regulators can audit decisions
      - Fast to train even on large datasets

    Random Forest:
      - Ensemble of decorrelated trees → low variance, robust to outliers
      - Handles missing data, non-linear relationships, and interactions
      - Built-in feature importance; no need for feature scaling
      - Hard to overfit due to bagging + feature subsampling

    Gradient Boosting:
      - Sequential correction of residuals → highest accuracy in practice
      - GBDT family (XGBoost, LightGBM) dominates Kaggle credit risk tasks
      - Works well on imbalanced data with proper `scale_pos_weight`
      - Slower to train; hyperparameter-sensitive
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1,               # Regularisation strength (smaller = more regularised)
            solver="lbfgs",      # Efficient for small-to-medium datasets
            max_iter=1000,
            class_weight="balanced",  # Handles class imbalance automatically
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,        # Number of trees
            max_depth=10,            # Prevent overfitting
            min_samples_leaf=5,      # Smoothness
            class_weight="balanced",
            n_jobs=-1,               # Use all CPU cores
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,      # Shrinkage — smaller = better generalisation
            max_depth=4,
            subsample=0.8,           # Stochastic GB → faster, less overfit
            min_samples_leaf=5,
            random_state=42,
        ),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """Train all models and return fitted estimators."""
    models = get_models()
    fitted = {}
    print("\n[TRAINING] Fitting models …")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        fitted[name] = model
        print(f"  ✓ {name:<25} trained in {elapsed:.2f}s")

    # Persist models to disk
    for name, model in fitted.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        print(f"  [SAVED] models/{fname}")

    return fitted


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_models(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compute evaluation metrics for all models.

    Metric guide for banking:
      Accuracy  — overall correctness (can be misleading with imbalanced data)
      Precision — of all predicted defaults, how many actually defaulted?
                  (bank perspective: avoid false alarms)
      Recall    — of all actual defaults, how many did we catch?
                  (risk perspective: missing a default is very costly)
      F1-Score  — harmonic mean of precision & recall
      ROC-AUC   — probability that the model ranks a random defaulter
                  higher than a random non-defaulter (threshold-free)
    """
    results = []
    for name, model in fitted_models.items():
        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred),    4),
            "F1-Score":  round(f1_score(y_test, y_pred),        4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),   4),
        })

        print(f"\n{'─'*50}")
        print(f"  MODEL: {name}")
        print(f"{'─'*50}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Default", "Default"]))

    df_results = pd.DataFrame(results).set_index("Model")
    print("\n[RESULTS] Summary:\n")
    print(df_results.to_string())
    return df_results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_roc_curves(fitted_models: dict, X_test, y_test) -> plt.Figure:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, model in fitted_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2.5,
                color=COLORS[name], label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="white")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    _save(fig, "08_roc_curves.png")
    return fig


def plot_confusion_matrices(fitted_models: dict, X_test, y_test) -> plt.Figure:
    """Side-by-side confusion matrices for all models."""
    n = len(fitted_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

    for ax, (name, model) in zip(axes, fitted_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, ax=ax,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            linewidths=1, linecolor="#222",
            cbar=False, annot_kws={"size": 14},
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    _save(fig, "09_confusion_matrices.png")
    return fig


def plot_feature_importance(fitted_models: dict, feature_names: list) -> plt.Figure:
    """
    Feature importance for tree-based models.
    Logistic Regression uses |coefficient| as a proxy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Feature Importance — Top 15 Features", fontsize=15, fontweight="bold")
    top_n = 15

    for ax, (name, model) in zip(axes, fitted_models.items()):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            continue

        idx = np.argsort(importances)[-top_n:]
        feat_labels = [feature_names[i] for i in idx]
        feat_vals   = importances[idx]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        ax.barh(feat_labels, feat_vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    _save(fig, "10_feature_importance.png")
    return fig


def plot_metric_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing all metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model_name, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=model_name, color=list(COLORS.values())[i],
                      edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v:.2f}", ha="center", fontsize=7.5)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.legend()
    plt.tight_layout()
    _save(fig, "11_metric_comparison.png")
    return fig
