"""
data_loader.py
--------------
Handles dataset loading and initial inspection.

Dataset: German Credit Dataset (UCI Machine Learning Repository)
Download: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Alternatively via Kaggle:
https://www.kaggle.com/datasets/uciml/german-credit

This module provides functions to load the dataset from a local path or
download it programmatically.
"""

import pandas as pd
import numpy as np
import os


# ─── Column names for the German Credit Dataset (raw format has no headers) ───
COLUMN_NAMES = [
    "checking_account",        # Status of existing checking account
    "duration",                # Duration in months
    "credit_history",          # Credit history
    "purpose",                 # Purpose of loan
    "credit_amount",           # Credit amount
    "savings_account",         # Savings account / bonds
    "employment",              # Present employment since
    "installment_rate",        # Installment rate as % of disposable income
    "personal_status",         # Personal status and sex
    "other_debtors",           # Other debtors / guarantors
    "residence_since",         # Present residence since
    "property",                # Property
    "age",                     # Age in years
    "other_installment_plans", # Other installment plans
    "housing",                 # Housing
    "existing_credits",        # Number of existing credits at this bank
    "job",                     # Job
    "num_dependents",          # Number of people liable to provide maintenance
    "telephone",               # Telephone
    "foreign_worker",          # Foreign worker
    "target",                  # 1 = Good (no default), 2 = Bad (default)
]


def load_german_credit(filepath: str = None) -> pd.DataFrame:
    """
    Load the German Credit dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the 'german.data' file. If None, generates a synthetic
        dataset with the same structure for demonstration purposes.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all 21 columns and proper column names.
    """
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath, sep=" ", header=None, names=COLUMN_NAMES)
    else:
        print("[INFO] No local file found — generating synthetic German Credit data.")
        print("[INFO] To use the real dataset, download from:")
        print("       https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data")
        df = _generate_synthetic_data(n_samples=1000)

    # Convert target: 1 (good) → 0 (no default), 2 (bad) → 1 (default)
    df["target"] = df["target"].map({1: 0, 2: 1})

    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Default rate: {df['target'].mean():.1%}")
    return df


def _generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset mirroring the German Credit schema.
    Used when the real dataset is unavailable.
    """
    rng = np.random.default_rng(random_state)

    checking_account    = rng.choice(["A11", "A12", "A13", "A14"], n_samples, p=[0.27, 0.27, 0.06, 0.40])
    duration            = rng.integers(4, 73, n_samples)
    credit_history      = rng.choice(["A30","A31","A32","A33","A34"], n_samples, p=[0.04,0.05,0.53,0.09,0.29])
    purpose             = rng.choice(["A40","A41","A42","A43","A45","A46","A48","A49"], n_samples)
    credit_amount       = rng.integers(250, 18500, n_samples)
    savings_account     = rng.choice(["A61","A62","A63","A64","A65"], n_samples, p=[0.60,0.10,0.06,0.06,0.18])
    employment          = rng.choice(["A71","A72","A73","A74","A75"], n_samples, p=[0.06,0.17,0.34,0.18,0.25])
    installment_rate    = rng.integers(1, 5, n_samples)
    personal_status     = rng.choice(["A91","A92","A93","A94"], n_samples, p=[0.05,0.31,0.55,0.09])
    other_debtors       = rng.choice(["A101","A102","A103"], n_samples, p=[0.91,0.04,0.05])
    residence_since     = rng.integers(1, 5, n_samples)
    property_           = rng.choice(["A121","A122","A123","A124"], n_samples, p=[0.28,0.23,0.24,0.25])
    age                 = rng.integers(19, 75, n_samples)
    other_installments  = rng.choice(["A141","A142","A143"], n_samples, p=[0.14,0.10,0.76])
    housing             = rng.choice(["A151","A152","A153"], n_samples, p=[0.18,0.71,0.11])
    existing_credits    = rng.integers(1, 5, n_samples)
    job                 = rng.choice(["A171","A172","A173","A174"], n_samples, p=[0.02,0.20,0.63,0.15])
    num_dependents      = rng.integers(1, 3, n_samples)
    telephone           = rng.choice(["A191","A192"], n_samples, p=[0.60,0.40])
    foreign_worker      = rng.choice(["A201","A202"], n_samples, p=[0.96,0.04])

    # Construct default probability with strong, realistic correlations
    base = np.full(n_samples, 0.05)

    checking_effect = np.select(
        [checking_account == "A11", checking_account == "A12",
         checking_account == "A13", checking_account == "A14"],
        [+0.35, +0.10, -0.10, +0.15], default=0.0
    )
    history_effect = np.select(
        [credit_history == "A30", credit_history == "A31",
         credit_history == "A32", credit_history == "A33", credit_history == "A34"],
        [-0.05, -0.15, -0.10, +0.25, +0.30], default=0.0
    )
    duration_effect = np.where(duration > 48, +0.25,
                      np.where(duration > 24, +0.10,
                      np.where(duration <= 12, -0.10, 0.0)))
    savings_effect = np.select(
        [savings_account == "A61", savings_account == "A62",
         savings_account == "A63", savings_account == "A64", savings_account == "A65"],
        [+0.20, +0.05, -0.05, -0.15, +0.10], default=0.0
    )
    amount_effect = np.where(credit_amount > 12000, +0.20,
                    np.where(credit_amount > 6000, +0.08,
                    np.where(credit_amount < 2000, -0.08, 0.0)))
    age_effect = np.where(age < 25, +0.20, np.where(age < 30, +0.10,
                 np.where(age > 50, -0.10, 0.0)))
    employ_effect = np.select(
        [employment == "A71", employment == "A72", employment == "A73",
         employment == "A74", employment == "A75"],
        [+0.25, +0.10, 0.0, -0.08, -0.12], default=0.0
    )
    install_effect = np.where(installment_rate >= 4, +0.15,
                     np.where(installment_rate == 3, +0.05, -0.05))

    prob = np.clip(
        base + checking_effect + history_effect + duration_effect
             + savings_effect  + amount_effect  + age_effect
             + employ_effect   + install_effect,
        0.04, 0.96
    )
    target = rng.binomial(1, prob, n_samples) + 1   # 1 = good, 2 = bad (raw format)

    df = pd.DataFrame({
        "checking_account": checking_account,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status": personal_status,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_,
        "age": age,
        "other_installment_plans": other_installments,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
        "target": target,
    })
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a structured summary of the DataFrame."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicate rows : {df.duplicated().sum()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True).rename({0:'Good (No Default)', 1:'Bad (Default)'})}")
    print("\nData Types:")
    print(df.dtypes.to_string())
    print("=" * 60)
