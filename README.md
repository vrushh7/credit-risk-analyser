# 🏦 Credit Risk Prediction

> A production-quality machine learning system for predicting loan default probability — built for banking data science roles.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Project Overview

This project implements a complete end-to-end credit risk prediction pipeline — from raw data ingestion through EDA, feature engineering, model training, and business insight generation — culminating in an interactive Streamlit dashboard.

**Use case**: Predict the probability that a loan applicant will default, enabling banks to make data-driven lending decisions.

---

## 📁 Project Structure

```
credit-risk-prediction/
├── data/                          # Raw dataset (download separately)
│   └── german.data                # UCI German Credit dataset
├── notebooks/
│   └── credit_risk_analysis.ipynb # Full interactive walkthrough
├── src/
│   ├── data_loader.py             # Dataset loading & inspection
│   ├── preprocessing.py           # Cleaning, encoding, feature engineering
│   ├── eda.py                     # Visualisation suite (7 charts)
│   ├── model_trainer.py           # Training + evaluation (3 models)
│   └── business_insights.py      # Risk segmentation & policy recommendations
├── models/                        # Saved trained models (.pkl)
├── reports/                       # Auto-generated plots (PNG)
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── main.py                        # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🗃 Dataset

**German Credit Dataset** — UCI Machine Learning Repository  
1,000 loan applicants with 20 features and binary target (default / no default).

| Source | Link |
|--------|------|
| UCI Repository | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Kaggle mirror  | https://www.kaggle.com/datasets/uciml/german-credit |

> **Note**: If no dataset file is provided, the pipeline auto-generates realistic synthetic data with the same schema.

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-prediction.git
cd credit-risk-prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Download real dataset

```bash
# Place the downloaded german.data file in data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data \
     -O data/german.data
```

### 3. Run the full pipeline

```bash
# With synthetic data (no download needed)
python main.py

# With real dataset
python main.py --data data/german.data

# Include interview prep notes
python main.py --interview

# Skip EDA for faster run
python main.py --skip-eda
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Open the Jupyter notebook

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

---

## 🤖 Models

| Model | Why it's used in banking |
|-------|--------------------------|
| **Logistic Regression** | Interpretable coefficients; regulatory compliance; calibrated probabilities |
| **Random Forest** | Handles non-linear interactions; robust to outliers; stable feature importance |
| **Gradient Boosting** | Highest accuracy; industry standard for credit scoring (GBDT family) |

---

## 📊 Evaluation Metrics

| Metric | Banking interpretation |
|--------|------------------------|
| Accuracy | Overall correctness (misleading with imbalanced data) |
| Precision | Of all flagged defaults, how many actually defaulted? |
| Recall | Of all real defaults, how many did we catch? (most important for risk) |
| F1-Score | Harmonic balance of precision and recall |
| ROC-AUC | Ranking ability; equivalent to Gini / 2 + 0.5 in Basel models |

---

## 📈 Generated Reports

All plots are saved automatically to `reports/`:

| # | Chart |
|---|-------|
| 01 | Default distribution (bar + pie) |
| 02 | Credit amount vs default (KDE + boxplot) |
| 03 | Default rate by credit history |
| 04 | Age distribution by default status |
| 05 | Default rate by loan duration |
| 06 | Feature correlation heatmap |
| 07 | Default rate by savings account tier |
| 08 | ROC curves (all models) |
| 09 | Confusion matrices (all models) |
| 10 | Feature importance (all models) |
| 11 | Metric comparison dashboard |
| 12 | Decision threshold analysis |

---

## 🎯 Key Findings

1. **Checking account status** is the strongest default predictor — borrowers with no account default at 2× the rate.
2. **Credit history** is the second most important feature — critical account history → ~45% default rate.
3. **Loan duration** strongly correlates with default: loans > 48 months default at 40%+ vs < 15% for short-term loans.
4. **Young borrowers** (under 25) default significantly more than those aged 35–50.
5. **Debt-to-income ratio** is a reliable engineered feature matching real-world bank policy thresholds.

---

## 💼 Interview Talking Points

**Why Logistic Regression for credit risk?**  
Banks operate under Basel III/IV and GDPR. Logistic Regression produces calibrated probabilities, has auditable coefficients (regulators can verify why a loan was rejected), and scales to millions of customers. It's the industry baseline — a traditional scorecard is essentially LR with binned variables.

**Why Random Forest?**  
RF handles non-linear interactions (young age + high loan amount is riskier than either alone), is robust to outliers without preprocessing, and provides stable feature importance rankings used in regulatory model documentation.

**How do banks use these models in production?**  
1. **Application scoring** — real-time API call on loan application
2. **Behavioural scoring** — monthly re-score of existing portfolio
3. **Regulatory capital** — PD feeds into Basel capital calculations
4. **Model monitoring** — Population Stability Index, Gini drift alerts

---

## 🛠 Technology Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit
- Jupyter Notebook
- Joblib (model persistence)

---

## 📄 License

MIT — free to use for portfolio, research, and commercial projects.
