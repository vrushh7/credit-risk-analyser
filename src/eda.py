"""
eda.py
------
Exploratory Data Analysis (EDA) module.

Generates all key visualisations for credit risk analysis:
  1. Loan default distribution (class balance)
  2. Numerical feature distributions split by default status
  3. Income (credit amount) vs default
  4. Credit history analysis
  5. Age distribution by default
  6. Feature correlation heatmap

All plots are saved to the /reports/ directory and also displayed.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style configuration ────────────────────────────────────────────────────────
PALETTE   = {"No Default": "#2ecc71", "Default": "#e74c3c"}
DARK_BG   = "#1a1a2e"
CARD_BG   = "#16213e"
TEXT_COL  = "#eaeaea"
ACCENT    = "#e94560"
GREEN     = "#2ecc71"
RED       = "#e74c3c"
BLUE      = "#3498db"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   TEXT_COL,
    "text.color":        TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "grid.color":        "#333",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[INFO] Saved: {path}")


# ── 1. Default distribution ────────────────────────────────────────────────────

def plot_default_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart showing class balance (default vs no-default)."""
    counts = df["target"].value_counts().rename({0: "No Default", 1: "Default"})
    pcts   = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Loan Default Distribution", fontsize=16, fontweight="bold", y=1.02)

    # ── Bar chart
    bars = axes[0].bar(
        counts.index, counts.values,
        color=[GREEN, RED], edgecolor="white", linewidth=0.8, width=0.5
    )
    for bar, pct in zip(bars, pcts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold"
        )
    axes[0].set_title("Count by Class", fontsize=13)
    axes[0].set_ylabel("Number of Borrowers")
    axes[0].set_xlabel("")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No Default", "Default"])

    # ── Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=[GREEN, RED],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(color=TEXT_COL, fontsize=12),
    )
    axes[1].set_title("Class Proportions", fontsize=13)

    plt.tight_layout()
    _save(fig, "01_default_distribution.png")
    return fig


# ── 2. Credit amount vs default ────────────────────────────────────────────────

def plot_credit_amount_vs_default(df: pd.DataFrame) -> plt.Figure:
    """Overlapping KDE + boxplot for credit amount by default status."""
    no_def = df[df["target"] == 0]["credit_amount"]
    default = df[df["target"] == 1]["credit_amount"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Credit Amount vs Default Status", fontsize=16, fontweight="bold")

    # KDE
    axes[0].set_title("Distribution of Credit Amount")
    no_def.plot.kde(ax=axes[0], color=GREEN, linewidth=2.5, label="No Default")
    default.plot.kde(ax=axes[0], color=RED,   linewidth=2.5, label="Default")
    line0 = axes[0].lines[0]
    axes[0].fill_between(
        line0.get_xdata(), 0, line0.get_ydata(),
        color=GREEN, alpha=0.15
    )
    axes[0].legend()
    axes[0].set_xlabel("Credit Amount (DM)")

    # Boxplot
    data_to_plot = [no_def.values, default.values]
    bp = axes[1].boxplot(
        data_to_plot,
        patch_artist=True,
        labels=["No Default", "Default"],
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], [GREEN, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Credit Amount Spread")
    axes[1].set_ylabel("Credit Amount (DM)")

    plt.tight_layout()
    _save(fig, "02_credit_amount_vs_default.png")
    return fig


# ── 3. Credit history analysis ─────────────────────────────────────────────────

def plot_credit_history(df: pd.DataFrame) -> plt.Figure:
    """Default rate per credit history category."""
    agg = (
        df.groupby("credit_history")["target"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "defaults", "count": "total"})
    )
    agg["default_rate"] = agg["defaults"] / agg["total"] * 100
    agg = agg.sort_values("default_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(agg)))
    bars = ax.barh(agg.index, agg["default_rate"], color=colors, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, agg["default_rate"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("Default Rate (%)")
    ax.set_title("Default Rate by Credit History Category", fontsize=14, fontweight="bold")
    ax.axvline(df["target"].mean() * 100, color="white", linestyle="--", alpha=0.6, label="Overall avg")
    ax.legend()
    plt.tight_layout()
    _save(fig, "03_credit_history_default_rate.png")
    return fig


# ── 4. Age distribution by default ────────────────────────────────────────────

def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of age coloured by default status."""
    fig, ax = plt.subplots(figsize=(12, 5))

    bins = range(18, 76, 3)
    ax.hist(df[df["target"] == 0]["age"], bins=bins,
            alpha=0.7, color=GREEN, label="No Default", edgecolor="white", linewidth=0.4)
    ax.hist(df[df["target"] == 1]["age"], bins=bins,
            alpha=0.7, color=RED, label="Default", edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Borrowers")
    ax.set_title("Age Distribution by Default Status", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "04_age_distribution.png")
    return fig


# ── 5. Loan duration analysis ──────────────────────────────────────────────────

def plot_duration_vs_default(df: pd.DataFrame) -> plt.Figure:
    """Default rate across loan duration buckets."""
    df = df.copy()
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[0, 12, 24, 36, 48, 72],
        labels=["0-12m", "12-24m", "24-36m", "36-48m", "48-72m"]
    )
    rates = df.groupby("duration_bucket", observed=True)["target"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [GREEN if r < 30 else ACCENT if r < 45 else RED for r in rates]
    bars = ax.bar(rates.index.astype(str), rates.values, color=colors, edgecolor="white", linewidth=0.7)

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Loan Duration")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Loan Duration", fontsize=14, fontweight="bold")
    ax.axhline(df["target"].mean() * 100, color="white", linestyle="--", alpha=0.6, label="Avg")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_duration_vs_default.png")
    return fig


# ── 6. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of correlations between numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # Upper triangle mask

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#222",
        square=True,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, "06_correlation_heatmap.png")
    return fig


# ── 7. Savings account vs default ─────────────────────────────────────────────

def plot_savings_vs_default(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: default rate per savings account tier."""
    agg = df.groupby("savings_account")["target"].mean() * 100
    agg = agg.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(agg)))
    bars = ax.bar(range(len(agg)), agg.values, color=colors, edgecolor="white", linewidth=0.6, width=0.6)

    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10)

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=15, ha="right")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Savings Account Tier", fontsize=14, fontweight="bold")
    ax.axhline(df["target"].mean() * 100, color="white", linestyle="--", alpha=0.6, label="Avg")
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_savings_vs_default.png")
    return fig


# ── Run all EDA ────────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> None:
    """Execute the complete EDA suite."""
    print("\n[EDA] Running full exploratory data analysis …\n")
    plot_default_distribution(df)
    plot_credit_amount_vs_default(df)
    plot_credit_history(df)
    plot_age_distribution(df)
    plot_duration_vs_default(df)
    plot_correlation_heatmap(df)
    plot_savings_vs_default(df)
    print("\n[EDA] All plots saved to /reports/\n")
