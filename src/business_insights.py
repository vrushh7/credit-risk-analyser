"""
business_insights.py
--------------------
Translates ML model outputs into actionable banking insights.

Covers:
  - Risk segmentation (low / medium / high)
  - Key default drivers and their business interpretation
  - Threshold analysis for operational cut-offs
  - Policy recommendations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
TEXT_COL = "#eaeaea"
GREEN    = "#2ecc71"
YELLOW   = "#f39c12"
RED      = "#e74c3c"
BLUE     = "#3498db"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor":   "#444",  "axes.labelcolor": TEXT_COL,
    "text.color":       TEXT_COL,"xtick.color":     TEXT_COL,
    "ytick.color":      TEXT_COL,"grid.color":      "#333",
    "font.family":      "monospace",
})

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[INFO] Saved: {path}")


# ── Risk segmentation ──────────────────────────────────────────────────────────

def segment_risk(probabilities: np.ndarray) -> pd.Series:
    """
    Map predicted default probabilities to risk tiers.

    Banking standard:
      Low  (< 20%)  — Approve with standard rate
      Med  (20-50%) — Approve with higher rate / collateral
      High (> 50%)  — Reject or require guarantor
    """
    segments = pd.cut(
        probabilities,
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return segments


def print_risk_summary(probabilities: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Print default rates per risk segment."""
    segments = segment_risk(probabilities)
    df = pd.DataFrame({
        "risk_segment": segments,
        "actual_default": y_true,
        "default_prob": probabilities,
    })
    summary = df.groupby("risk_segment", observed=True).agg(
        count=("actual_default", "count"),
        actual_default_rate=("actual_default", "mean"),
        avg_predicted_prob=("default_prob", "mean"),
    )
    summary["actual_default_rate"] = summary["actual_default_rate"].map("{:.1%}".format)
    summary["avg_predicted_prob"]  = summary["avg_predicted_prob"].map("{:.1%}".format)

    print("\n[INSIGHTS] Risk Segment Summary:")
    print(summary.to_string())
    return df


# ── Threshold analysis ────────────────────────────────────────────────────────

def plot_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                            model_name: str = "Best Model") -> plt.Figure:
    """
    Show how precision, recall, and approval rate change with decision threshold.

    Why this matters:
      A bank cannot simply use 0.5 as its cut-off.
      A conservative bank (worried about losses) sets a lower threshold.
      A growth-oriented bank (worried about rejecting good customers) raises it.
    """
    from sklearn.metrics import precision_score, recall_score

    thresholds = np.linspace(0.05, 0.95, 50)
    precisions, recalls, approval_rates = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        approval_rates.append((y_pred == 0).mean())   # % of loans approved

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(thresholds, precisions,    color=GREEN,  linewidth=2, label="Precision (of flagged defaults, how many really default)")
    ax.plot(thresholds, recalls,       color=RED,    linewidth=2, label="Recall (of all actual defaults, how many we catch)")
    ax.plot(thresholds, approval_rates,color=BLUE,   linewidth=2, linestyle="--", label="Loan Approval Rate")

    # Annotate business operating points
    ax.axvline(0.3, color="white", linestyle=":", alpha=0.6)
    ax.text(0.31, 0.05, "Conservative\nBank (0.3)", fontsize=8, color="white")
    ax.axvline(0.5, color=YELLOW, linestyle=":", alpha=0.6)
    ax.text(0.51, 0.05, "Standard\n(0.5)", fontsize=8, color=YELLOW)

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score / Rate")
    ax.set_title(f"Threshold Analysis — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="center right", fontsize=9)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    _save(fig, "12_threshold_analysis.png")
    return fig


# ── Feature importance business translation ─────────────────────────────────

FEATURE_EXPLANATIONS = {
    "checking_account":        "💳 Checking account balance — direct liquidity signal",
    "credit_history":          "📋 Past repayment behaviour — strongest default predictor",
    "duration":                "⏳ Loan duration — longer loans carry more risk",
    "credit_amount":           "💰 Loan size — larger loans increase exposure",
    "savings_account":         "🏦 Savings — cushion against financial shocks",
    "employment":              "👔 Employment stability — income continuity matters",
    "age":                     "📅 Borrower age — younger borrowers default more",
    "installment_rate":        "📊 Instalment burden — high % of income is risky",
    "debt_to_income":          "⚖️  Debt-to-income ratio — key affordability metric",
    "credit_per_month":        "📆 Monthly obligation — cash flow pressure indicator",
    "purpose":                 "🎯 Loan purpose — business loans riskier than car loans",
    "housing":                 "🏠 Housing status — ownership signals stability",
    "other_installment_plans": "🔄 Other active loans — hidden liability risk",
}


def print_business_insights(feature_names: list, importances: np.ndarray) -> None:
    """Print a business-readable feature importance report."""
    print("\n" + "=" * 65)
    print("  BUSINESS INSIGHTS: KEY DRIVERS OF LOAN DEFAULT")
    print("=" * 65)

    idx = np.argsort(importances)[::-1][:10]
    rank = 1
    for i in idx:
        name  = feature_names[i]
        score = importances[i]
        expl  = FEATURE_EXPLANATIONS.get(name, f"Feature: {name}")
        print(f"\n  #{rank:02d}  {expl}")
        print(f"       Importance score: {score:.4f}  {'█' * int(score * 200)}")
        rank += 1

    print("\n" + "=" * 65)
    print("  POLICY RECOMMENDATIONS")
    print("=" * 65)
    recommendations = [
        "1. Require borrowers with no checking account to provide collateral.",
        "2. Loans > 36 months should trigger manual credit review.",
        "3. Borrowers under 25 should face a lower credit limit on first loan.",
        "4. Debt-to-income > 40% should automatically flag for review.",
        "5. Applicants with 'critical' credit history need guarantor.",
        "6. Regular model recalibration every 6 months for concept drift.",
    ]
    for rec in recommendations:
        print(f"  {rec}")
    print("=" * 65 + "\n")


# ── Interview Preparation Notes ───────────────────────────────────────────────

def print_interview_notes() -> None:
    """Print interview preparation talking points."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           INTERVIEW PREPARATION — CREDIT RISK ML                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Q: Why Logistic Regression for credit risk?                     ║
║  A: Banks operate under regulatory scrutiny (Basel III/IV,       ║
║     GDPR). Logistic Regression produces probability scores,      ║
║     has interpretable coefficients (regulators can audit why     ║
║     a loan was rejected), and scales well. It's the industry     ║
║     baseline — a scorecard is essentially LR with hand-crafted   ║
║     binning. If LR performs close to complex models, prefer it.  ║
║                                                                  ║
║  Q: Why Random Forest?                                           ║
║  A: RF handles non-linear interactions (e.g., young age +        ║
║     high loan amount is riskier than either alone). It's         ║
║     robust to outliers, requires minimal preprocessing, and      ║
║     provides stable feature importance. Bagging reduces          ║
║     variance which is crucial when training data is limited.     ║
║                                                                  ║
║  Q: How do banks use such models in production?                  ║
║  A: 1) Application scoring: real-time API called when a          ║
║        customer applies for a loan.                              ║
║     2) Behavioural scoring: monthly re-score of existing         ║
║        customers to detect early deterioration.                  ║
║     3) Regulatory reporting: PD (Probability of Default),        ║
║        LGD (Loss Given Default), EAD (Exposure at Default)       ║
║        feed into capital adequacy calculations.                  ║
║     4) Model validation: parallel runs, champion/challenger,     ║
║        Gini coefficient monitoring, population stability index.  ║
║                                                                  ║
║  Q: What is class imbalance and how did you handle it?           ║
║  A: Real credit data has ~5-20% default rate. We used            ║
║     class_weight='balanced' which re-weights the loss function   ║
║     so minority class errors cost more. Alternatives include     ║
║     SMOTE (synthetic oversampling) and threshold tuning.         ║
║                                                                  ║
║  Q: What metric matters most?                                    ║
║  A: ROC-AUC for ranking; Recall for risk management (we want     ║
║     to catch all defaults even if some false alarms arise);      ║
║     Precision for marketing (avoid rejecting good customers).    ║
║     In Basel models, Gini = 2×AUC−1 is the standard metric.     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
