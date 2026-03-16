"""
preprocessing.py
----------------
Handles all data cleaning, encoding, and feature engineering.

Steps:
  1. Handle missing values
  2. Remove duplicates
  3. Decode categorical codes → readable labels
  4. Encode categoricals → numeric (for ML models)
  5. Scale numerical features
  6. Provide train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─── Human-readable mappings for German Credit categorical codes ──────────────

CATEGORY_MAPS = {
    "checking_account": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": "> 200 DM",
        "A14": "No account",
    },
    "credit_history": {
        "A30": "No credits taken",
        "A31": "All paid duly",
        "A32": "Existing paid duly",
        "A33": "Delay in past",
        "A34": "Critical account",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture",
        "A43": "Radio/TV",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": "> 1000 DM",
        "A65": "Unknown/None",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": "> 7 years",
    },
    "personal_status": {
        "A91": "Male: divorced/separated",
        "A92": "Female: divorced/separated/married",
        "A93": "Male: single",
        "A94": "Male: married/widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings",
        "A123": "Car or other",
        "A124": "Unknown/None",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free",
    },
    "job": {
        "A171": "Unemployed/unskilled (non-resident)",
        "A172": "Unskilled (resident)",
        "A173": "Skilled employee",
        "A174": "Management/self-employed",
    },
    "telephone": {
        "A191": "None",
        "A192": "Yes, registered",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

# Numerical columns in the raw dataset
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents"
]


class CreditRiskPreprocessor:
    """
    Full preprocessing pipeline for the German Credit dataset.

    Usage
    -----
    preprocessor = CreditRiskPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    # Later on new data:
    X_new = preprocessor.transform(new_df)
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []   # Saved after fit for transform consistency

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """
        Full pipeline: clean → decode → encode → split → scale.
        Fits the scaler on training data only (no data leakage).

        Returns
        -------
        X_train, X_test, y_train, y_test : numpy arrays
        """
        df = self._clean(df)
        df = self._decode_categoricals(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=True)

        X = df.drop(columns=["target"])
        y = df["target"]
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        # Scale only numerical columns; fit on train, apply to both
        X_train[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]] = \
            self.scaler.fit_transform(
                X_train[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]]
            )
        X_test[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]] = \
            self.scaler.transform(
                X_test[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]]
            )

        print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
        print(f"[INFO] Features   : {len(self.feature_columns)}")
        return (
            X_train.values, X_test.values,
            y_train.values, y_test.values
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to new data (no fitting)."""
        df = self._clean(df)
        df = self._decode_categoricals(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)

        # Align columns with training set
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        df[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]] = \
            self.scaler.transform(
                df[NUMERICAL_COLS + ["debt_to_income", "credit_per_month"]]
            )
        return df.values

    def get_feature_names(self) -> list:
        return self.feature_columns

    # ── Private helpers ────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and handle missing values."""
        df = df.copy()

        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            print(f"[INFO] Removed {before - len(df)} duplicate rows.")

        # Fill numerical missing values with median
        for col in NUMERICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical missing values with mode
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def _decode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace coded categories (A11, A12 …) with readable strings."""
        df = df.copy()
        for col, mapping in CATEGORY_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-informed features for banking context."""
        df = df.copy()

        # Debt-to-income proxy: credit amount relative to duration installments
        df["debt_to_income"] = df["credit_amount"] / (df["duration"] * df["installment_rate"] + 1)

        # Monthly payment proxy
        df["credit_per_month"] = df["credit_amount"] / df["duration"].clip(lower=1)

        # Age bucket (risk differs across life stages)
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["Young", "Early-career", "Mid-career", "Senior"]
        ).astype(str)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode all object columns."""
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels gracefully
                    df[col] = df[col].astype(str).map(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0   # Column not seen during training
        return df
