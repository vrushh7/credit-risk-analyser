"""
main.py
-------
Entry point for the Credit Risk Prediction pipeline.

Usage:
    python main.py                          # Use synthetic data
    python main.py --data data/german.data  # Use real dataset

Steps:
    1. Load data
    2. Clean & preprocess
    3. EDA (saves plots to reports/)
    4. Train models
    5. Evaluate & compare
    6. Business insights
"""

import argparse
import sys
import os

# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader      import load_german_credit, describe_dataset
from preprocessing    import CreditRiskPreprocessor
from eda              import run_full_eda
from model_trainer    import (
    train_models, evaluate_models,
    plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_metric_comparison,
)
from business_insights import (
    print_risk_summary, plot_threshold_analysis,
    print_business_insights, print_interview_notes,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Credit Risk Prediction Pipeline")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to german.data file (optional; uses synthetic data if absent)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA plots (faster run)"
    )
    parser.add_argument(
        "--interview", action="store_true",
        help="Print interview preparation notes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("   CREDIT RISK PREDICTION PIPELINE")
    print("   Banking-Grade Machine Learning System")
    print("=" * 60 + "\n")

    # ── STEP 1: Load data ──────────────────────────────────────────────────────
    print("STEP 1 ▶  Loading Dataset")
    df = load_german_credit(filepath=args.data)
    describe_dataset(df)

    # ── STEP 2: EDA ────────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\nSTEP 2 ▶  Exploratory Data Analysis")
        run_full_eda(df)

    # ── STEP 3: Preprocessing ─────────────────────────────────────────────────
    print("\nSTEP 3 ▶  Preprocessing & Feature Engineering")
    preprocessor = CreditRiskPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()

    # ── STEP 4: Train models ───────────────────────────────────────────────────
    print("\nSTEP 4 ▶  Training Models")
    fitted_models = train_models(X_train, y_train)

    # ── STEP 5: Evaluate ───────────────────────────────────────────────────────
    print("\nSTEP 5 ▶  Evaluating Models")
    results_df = evaluate_models(fitted_models, X_test, y_test)

    plot_roc_curves(fitted_models, X_test, y_test)
    plot_confusion_matrices(fitted_models, X_test, y_test)
    plot_feature_importance(fitted_models, feature_names)
    plot_metric_comparison(results_df)

    # ── STEP 6: Business Insights ──────────────────────────────────────────────
    print("\nSTEP 6 ▶  Business Insights")

    # Use best model (by ROC-AUC) for insights
    best_model_name = results_df["ROC-AUC"].idxmax()
    best_model      = fitted_models[best_model_name]
    print(f"\n[INFO] Best model: {best_model_name} (ROC-AUC = {results_df.loc[best_model_name, 'ROC-AUC']:.4f})")

    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    print_risk_summary(y_prob_best, y_test)
    plot_threshold_analysis(y_test, y_prob_best, model_name=best_model_name)

    import numpy as np
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    print_business_insights(feature_names, importances)

    if args.interview:
        print_interview_notes()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Reports saved to: reports/")
    print(f"  Models  saved to: models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
