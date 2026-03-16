Credit Risk Prediction System

A machine learning system that predicts loan default probability, built to demonstrate
production-level data science skills for banking and finance roles.

Live Demo: https://YOUR_USERNAME-credit-risk-analyser.streamlit.app

────────────────────────────────────────────────────────────────

Dashboard Preview

Risk Assessment – High Risk

![High Risk](screenshots/dashboard_high_risk.png)

Risk Assessment – Low Risk

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

About the Project

I built this to understand how banks actually assess credit risk using machine learning.
The system takes a borrower's financial and personal details, runs them through a trained
Gradient Boosting model, and outputs a default probability with a risk tier classification.

The dataset used is the German Credit Dataset from the UCI Machine Learning Repository —
1,000 real loan applicants with 20 features each.

────────────────────────────────────────────────────────────────

Project Structure

credit-risk-analyser/
    main.py                         run the full pipeline from terminal
    requirements.txt
    README.md
    src/
        data_loader.py              loads and inspects the dataset
        preprocessing.py            cleaning, encoding, feature engineering
        eda.py                      exploratory data analysis charts
        model_trainer.py            trains and evaluates all three models
        business_insights.py        risk segmentation and policy recommendations
    dashboard/
        app.py                      streamlit dashboard
    notebooks/
        credit_risk_analysis.ipynb  step-by-step jupyter walkthrough
    screenshots/
        dashboard previews

────────────────────────────────────────────────────────────────

How to Run It Locally

    git clone https://github.com/YOUR_USERNAME/credit-risk-analyser.git
    cd credit-risk-analyser

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

    python main.py
    streamlit run dashboard/app.py

────────────────────────────────────────────────────────────────

Model Results

Three models were trained and compared on the same test set of 200 borrowers.

    Logistic Regression     Accuracy 67.0%    ROC-AUC 0.718
    Random Forest           Accuracy 69.5%    ROC-AUC 0.782
    Gradient Boosting       Accuracy 71.5%    ROC-AUC 0.804  (selected)

Gini Coefficient: 0.608 — passes the Basel III regulatory validation threshold of 0.50,
meaning this model would be considered acceptable for real banking use.

────────────────────────────────────────────────────────────────

What the Dashboard Does

Tab 1 — Risk Assessment
Enter a borrower's details and get an instant default probability score,
a risk tier (Low / Medium / High), and a breakdown of which factors are
driving the risk up or down.

Tab 2 — Compare Borrowers
Enter two borrower profiles side by side and see which one is lower risk,
with a full comparison table and recommended action for each.

Tab 3 — Affordability Calculator
Input a borrower's income and existing debts to calculate their DTI ratio,
monthly payment, and the maximum loan amount they can safely afford.

Tab 4 — Model Performance
Full evaluation report including ROC curve, confusion matrix, and an
explanation of what each metric means in a banking context.

────────────────────────────────────────────────────────────────

Key Findings from the Data

Loan duration is the strongest predictor of default. Loans over 48 months
default at more than double the rate of short-term loans.

Credit history is the second biggest factor. Borrowers with critical account
history have an 85% default rate in the data.

Younger borrowers under 25 default at roughly twice the rate of borrowers
aged 35 to 50.

A debt-to-income ratio above 40% is a reliable signal for manual review,
which aligns with the 35% DTI threshold used by most banks.

────────────────────────────────────────────────────────────────

Risk Tiers

    Low Risk       under 20% probability    approve at standard rate
    Medium Risk    20 to 50% probability    approve with collateral or co-signer
    High Risk      above 50% probability    reject or require guarantor

────────────────────────────────────────────────────────────────

Tech Stack

Python 3.11, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib

Dataset: German Credit Data — UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

────────────────────────────────────────────────────────────────

License: MIT
