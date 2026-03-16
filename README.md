Credit Risk Prediction

A machine learning system that predicts loan default probability, built for banking and finance data science roles.

Live Demo: https://YOUR_USERNAME-credit-risk-analyser.streamlit.app

Python 3.11   scikit-learn 1.4   Streamlit   ROC-AUC 0.80   MIT License


────────────────────────────────────────────────────────────────
Dashboard Preview
────────────────────────────────────────────────────────────────

Risk Assessment

![Dashboard](screenshots/dashboard.png)

High Risk Result

![High Risk](screenshots/dashboard_high_risk.png)

Low Risk Result

![Low Risk](screenshots/dashboard_low_risk.png)

Borrower Comparison

![Compare](screenshots/compare_borrowers.png)

Comparison Results

![Compare Results](screenshots/compare_borrowers_results.png)

Affordability Calculator

![Affordability](screenshots/affordability.png)

Model Performance

![Performance](screenshots/model_performance.png)

Feature Importance

![Features](screenshots/feature_importance.png)


────────────────────────────────────────────────────────────────
Project Overview
────────────────────────────────────────────────────────────────

I built this to understand how banks assess credit risk using machine learning.
The system takes a borrower's financial and personal details, runs them through
a trained Gradient Boosting model, and outputs a default probability with a
risk tier classification and actionable recommendation.

Use case: Predict whether a loan applicant will default, so banks can make
faster, fairer, data-driven lending decisions.


────────────────────────────────────────────────────────────────
Project Structure
────────────────────────────────────────────────────────────────

credit-risk-analyser/
├── main.py                          run the full ML pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py               dataset loading and inspection
│   ├── preprocessing.py             cleaning, encoding, feature engineering
│   ├── eda.py                       7 exploratory data analysis charts
│   ├── model_trainer.py             training and evaluation of all 3 models
│   └── business_insights.py        risk segmentation and policy recommendations
├── dashboard/
│   └── app.py                       streamlit interactive dashboard
├── notebooks/
│   └── credit_risk_analysis.ipynb  full step-by-step jupyter walkthrough
└── screenshots/
    └── (dashboard preview images)


────────────────────────────────────────────────────────────────
Dataset
────────────────────────────────────────────────────────────────

German Credit Dataset — UCI Machine Learning Repository
1,000 loan applicants with 20 features and a binary default target.

UCI Repository   https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Kaggle mirror    https://www.kaggle.com/datasets/uciml/german-credit

No download needed — if no file is provided, the pipeline automatically
generates realistic synthetic data with the same schema and runs end to end.


────────────────────────────────────────────────────────────────
Quick Start
────────────────────────────────────────────────────────────────

git clone https://github.com/YOUR_USERNAME/credit-risk-analyser.git
cd credit-risk-analyser

python -m venv venv
venv\Scripts\activate          (Windows)
source venv/bin/activate       (Mac / Linux)

pip install -r requirements.txt

Run the full pipeline:
python main.py

Run with the real dataset:
python main.py --data data/german.data

Launch the dashboard:
streamlit run dashboard/app.py

Open the notebook:
jupyter notebook notebooks/credit_risk_analysis.ipynb


────────────────────────────────────────────────────────────────
Models
────────────────────────────────────────────────────────────────

Three models were trained and compared on the same held-out test set.

Model                  Accuracy    ROC-AUC    F1-Score
Logistic Regression    67.0%       0.718      0.700
Random Forest          69.5%       0.782      0.734
Gradient Boosting      71.5%       0.804      0.762   ← selected

Model               Why it is used in banking
Logistic Regression Interpretable coefficients, regulatory compliance, calibrated probabilities
Random Forest       Handles non-linear interactions, robust to outliers, stable feature importance
Gradient Boosting   Highest accuracy, industry standard for credit scoring (GBDT family)


────────────────────────────────────────────────────────────────
Evaluation Metrics
────────────────────────────────────────────────────────────────

Metric            Value    Banking meaning
ROC-AUC           0.804    Model ranks defaulters above non-defaulters 80% of the time
Gini Coefficient  0.608    Passes Basel III regulatory validation threshold of 0.50
Precision         73.4%    Of all flagged defaults, 73% actually defaulted
Recall            79.1%    Of all actual defaults, 79% were caught
F1-Score          0.762    Balanced measure of precision and recall


────────────────────────────────────────────────────────────────
Dashboard Features
────────────────────────────────────────────────────────────────

Tab 1 — Risk Assessment
Enter borrower details and get an instant default probability, a risk tier
(Low / Medium / High), a factor-by-factor risk driver breakdown, and a
recommended action for the lender.

Tab 2 — Compare Borrowers
Enter two profiles side by side to see which applicant is lower risk,
with a full comparison table and probability gauges for both.

Tab 3 — Affordability Calculator
Input income and existing debts to calculate DTI ratio, monthly payment,
headroom, and the maximum loan the borrower can safely afford.

Tab 4 — Model Performance
Full evaluation report: ROC curve, confusion matrix, Gini coefficient,
and a plain-English explanation of what each metric means for banking.


────────────────────────────────────────────────────────────────
Generated Reports
────────────────────────────────────────────────────────────────

All 12 charts are saved automatically to the reports/ folder when you
run python main.py.

01   Default distribution (bar and pie)
02   Credit amount vs default (KDE and boxplot)
03   Default rate by credit history
04   Age distribution by default status
05   Default rate by loan duration
06   Feature correlation heatmap
07   Default rate by savings account tier
08   ROC curves for all three models
09   Confusion matrices for all three models
10   Feature importance for all three models
11   Metric comparison dashboard
12   Decision threshold analysis


────────────────────────────────────────────────────────────────
Key Findings
────────────────────────────────────────────────────────────────

Loan duration is the strongest default predictor. Loans over 48 months
default at more than double the rate of short-term loans.

Credit history is the second biggest factor. Borrowers with a critical
account history show an 85% default rate in the data.

Checking account balance directly signals liquidity and repayment capacity.
Borrowers with overdrawn accounts default at 2x the average rate.

Younger borrowers under 25 default at roughly twice the rate of those
aged 35 to 50.

A debt-to-income ratio above 40% is a reliable flag for manual review,
matching the 35% DTI threshold used by most banks in practice.


────────────────────────────────────────────────────────────────
Risk Segmentation
────────────────────────────────────────────────────────────────

Tier          Probability        Actual Default Rate    Action
Low Risk      under 20%          17.4%                  approve at standard rate
Medium Risk   20% to 50%         37.7%                  approve with collateral
High Risk     above 50%          73.4%                  reject or require guarantor


────────────────────────────────────────────────────────────────
Interview Talking Points
────────────────────────────────────────────────────────────────

Why Logistic Regression for credit risk?
Banks operate under Basel III and GDPR. LR produces calibrated probabilities
and has auditable coefficients — regulators can verify exactly why a loan
was rejected. It scales to millions of customers and is the industry baseline.

Why Random Forest?
RF captures non-linear interactions (young age combined with a large loan is
far riskier than either alone), is robust to outliers without preprocessing,
and gives stable feature importance for regulatory documentation.

How do banks use these models in production?
Application scoring — real-time API call when a customer applies
Behavioural scoring — monthly re-score of the existing portfolio
Regulatory capital — PD feeds into Basel capital calculations
Model monitoring — Population Stability Index and Gini drift alerts


────────────────────────────────────────────────────────────────
Tech Stack
────────────────────────────────────────────────────────────────

Python 3.11, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly,
Streamlit, Jupyter Notebook, Joblib

Dataset: German Credit Data — UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data


────────────────────────────────────────────────────────────────
License: MIT
────────────────────────────────────────────────────────────────
