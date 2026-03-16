🏦 Credit Risk Prediction
A production-quality machine learning system for predicting loan default probability — built for banking data science roles.

Python   scikit-learn   Streamlit   ROC-AUC 0.80   MIT License

🚀 Live Demo: https://YOUR_USERNAME-credit-risk-analyser.streamlit.app

---

🖥️ Dashboard Preview

Risk Assessment

![Dashboard](screenshots/dashboard.png)

High Risk Borrower

![High Risk](screenshots/dashboard_high_risk.png)

Low Risk Borrower

![Low Risk](screenshots/dashboard_low_risk.png)

Side-by-Side Borrower Comparison

![Compare](screenshots/compare_borrowers.png)

Comparison Results

![Compare Results](screenshots/compare_borrowers_results.png)

Loan Affordability Calculator

![Affordability](screenshots/affordability.png)

Model Performance Report

![Performance](screenshots/model_performance.png)

Feature Importance

![Features](screenshots/feature_importance.png)

---

📋 Project Overview
This project implements a complete end-to-end credit risk prediction pipeline — from raw data ingestion through EDA, feature engineering, model training, and business insight generation — culminating in an interactive Streamlit dashboard.

Use case: Predict the probability that a loan applicant will default, enabling banks to make data-driven lending decisions.

---

📁 Project Structure
credit-risk-analyser/
├── main.py                          # Run the full pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py               # Dataset loading & inspection
│   ├── preprocessing.py             # Cleaning, encoding, feature engineering
│   ├── eda.py                       # Visualisation suite (7 charts)
│   ├── model_trainer.py             # Training + evaluation (3 models)
│   └── business_insights.py        # Risk segmentation & policy recommendations
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
├── notebooks/
│   └── credit_risk_analysis.ipynb  # Full interactive walkthrough
└── screenshots/                     # Dashboard preview images

---

🗃 Dataset
German Credit Dataset — UCI Machine Learning Repository
1,000 loan applicants with 20 features and binary target (default / no default).

Source          Link
UCI Repository  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Kaggle mirror   https://www.kaggle.com/datasets/uciml/german-credit

Note: If no dataset file is provided, the pipeline auto-generates realistic synthetic data with the same schema.

---

⚡ Quick Start

1. Clone & set up environment
git clone https://github.com/YOUR_USERNAME/credit-risk-analyser.git
cd credit-risk-analyser

python -m venv venv
venv\Scripts\activate             # Windows
source venv/bin/activate          # Mac / Linux

pip install -r requirements.txt

2. Run the full pipeline
# With synthetic data (no download needed)
python main.py

# With real dataset
python main.py --data data/german.data

# Include interview prep notes
python main.py --interview

# Skip EDA for faster run
python main.py --skip-eda

3. Launch the Streamlit dashboard
streamlit run dashboard/app.py

4. Open the Jupyter notebook
jupyter notebook notebooks/credit_risk_analysis.ipynb

---

🤖 Models

Model                  Accuracy   ROC-AUC   F1-Score
Logistic Regression    67.0%      0.718     0.700
Random Forest          69.5%      0.782     0.734
Gradient Boosting      71.5%      0.804     0.762   ← selected

Model                Why it's used in banking
Logistic Regression  Interpretable coefficients; regulatory compliance; calibrated probabilities
Random Forest        Handles non-linear interactions; robust to outliers; stable feature importance
Gradient Boosting    Highest accuracy; industry standard for credit scoring (GBDT family)

---

📊 Evaluation Metrics

Metric            Value    Banking interpretation
ROC-AUC           0.804    Model ranks defaulters above non-defaulters 80% of the time
Gini Coefficient  0.608    Passes Basel III regulatory validation threshold (>0.50)
Precision         73.4%    Of all flagged defaults, how many actually defaulted?
Recall            79.1%    Of all real defaults, how many did we catch? (most important for risk)
F1-Score          0.762    Harmonic balance of precision and recall

---

🖥️ Dashboard Features

Tab                      Features
🔍 Risk Assessment       Input form → probability gauge → risk driver bars → feature importance
⚖️ Compare Borrowers     Two profiles side-by-side → comparison table → winner highlighted
🧮 Affordability Calc    DTI ratio checker → max loan calculator → monthly headroom
📊 Model Performance     ROC curve → confusion matrix → Gini coefficient → metric guide

---

📈 Generated Reports
All 12 charts are saved automatically to reports/ when you run python main.py.

#    Chart
01   Default distribution (bar + pie)
02   Credit amount vs default (KDE + boxplot)
03   Default rate by credit history
04   Age distribution by default status
05   Default rate by loan duration
06   Feature correlation heatmap
07   Default rate by savings account tier
08   ROC curves (all models)
09   Confusion matrices (all models)
10   Feature importance (all models)
11   Metric comparison dashboard
12   Decision threshold analysis

---

🎯 Key Findings
- Loan duration is the strongest default predictor — loans over 48 months default at 40%+ vs under 15% for short-term loans.
- Credit history is the second most important feature — critical account history leads to an 85% default rate.
- Checking account balance directly signals liquidity — borrowers with overdrawn accounts default at 2× the average rate.
- Young borrowers under 25 default significantly more than those aged 35–50.
- Debt-to-income ratio above 40% is a reliable flag for manual review, matching real-world bank policy thresholds.

---

💼 Risk Segmentation

Tier          Probability     Actual Default Rate    Bank Action
Low Risk      under 20%       17.4%                  Approve at standard rate
Medium Risk   20% to 50%      37.7%                  Approve with collateral or co-signer
High Risk     above 50%       73.4%                  Reject or require guarantor

---

💼 Interview Talking Points

Why Logistic Regression for credit risk?
Banks operate under Basel III/IV and GDPR. Logistic Regression produces calibrated probabilities, has auditable coefficients (regulators can verify why a loan was rejected), and scales to millions of customers. It's the industry baseline — a traditional scorecard is essentially LR with binned variables.

Why Random Forest?
RF handles non-linear interactions (young age + high loan amount is riskier than either alone), is robust to outliers without preprocessing, and provides stable feature importance rankings used in regulatory model documentation.

How do banks use these models in production?
- Application scoring — real-time API call on loan application
- Behavioural scoring — monthly re-score of existing portfolio
- Regulatory capital — PD feeds into Basel capital calculations
- Model monitoring — Population Stability Index, Gini drift alerts

---

🛠 Technology Stack
- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit
- Jupyter Notebook
- Joblib (model persistence)

---

📄 License
MIT — free to use for portfolio, research, and commercial projects.
