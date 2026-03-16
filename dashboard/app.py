"""
dashboard/app.py  —  CreditIQ  (Redesigned v2)
Clean light theme, 4 tabs, advanced features.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                              confusion_matrix, precision_score, recall_score, f1_score)
from preprocessing import CreditRiskPreprocessor, CATEGORY_MAPS
from data_loader   import load_german_credit

st.set_page_config(page_title="CreditIQ — Loan Risk Analyser",
                   page_icon="🏦", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{
  background:#F7F8FC!important;color:#1C2B4A!important;
  font-family:'DM Sans',sans-serif!important}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #E5E9F2}
#MainMenu,footer,header,[data-testid="stDecoration"]{visibility:hidden;display:none}

.topbar{background:#fff;border-bottom:1px solid #E5E9F2;padding:14px 40px;
  display:flex;align-items:center;justify-content:space-between;
  margin:-1rem -1rem 2rem -1rem;box-shadow:0 1px 8px rgba(28,43,74,.06)}
.topbar-brand{font-family:'DM Serif Display',serif;font-size:22px;
  color:#1C2B4A;letter-spacing:-.3px}
.topbar-brand span{color:#2563EB}
.topbar-badge{background:#EFF6FF;color:#2563EB;font-size:11px;font-weight:600;
  padding:4px 10px;border-radius:20px;letter-spacing:.5px}

.card{background:#fff;border-radius:16px;padding:24px 28px;
  border:1px solid #E5E9F2;box-shadow:0 2px 12px rgba(28,43,74,.05);margin-bottom:20px}
.card-title{font-family:'DM Serif Display',serif;font-size:17px;
  color:#1C2B4A;margin-bottom:4px}
.card-sub{font-size:13px;color:#6B7A99;margin-bottom:16px;line-height:1.5}

.result-hero{border-radius:20px;padding:28px;text-align:center;margin-bottom:16px}
.result-hero.low   {background:linear-gradient(135deg,#F0FDF4,#DCFCE7);border:2px solid #86EFAC}
.result-hero.medium{background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:2px solid #FCD34D}
.result-hero.high  {background:linear-gradient(135deg,#FFF1F2,#FFE4E6);border:2px solid #FECDD3}
.result-pct{font-family:'DM Serif Display',serif;font-size:56px;line-height:1;margin-bottom:6px}
.result-pct.low   {color:#16A34A}.result-pct.medium{color:#D97706}.result-pct.high{color:#DC2626}
.result-tier{font-size:15px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.result-action{font-size:13px;margin-top:8px;color:#4B5563}

.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.metric-chip{flex:1;min-width:110px;background:#F7F8FC;border:1px solid #E5E9F2;
  border-radius:12px;padding:12px 14px;text-align:center}
.metric-chip .val{font-size:20px;font-weight:700;color:#1C2B4A}
.metric-chip .lbl{font-size:10px;color:#6B7A99;margin-top:2px;
  text-transform:uppercase;letter-spacing:.5px}

.factor-row{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.factor-name{font-size:13px;color:#374151;width:145px;flex-shrink:0;font-weight:500}
.factor-bar-bg{flex:1;height:7px;background:#F1F5F9;border-radius:4px;overflow:hidden}
.factor-bar-fill{height:100%;border-radius:4px}
.factor-val{font-size:11px;color:#6B7A99;width:38px;text-align:right}

[data-testid="stTabs"] [role="tablist"]{gap:4px;background:#F7F8FC;padding:4px;
  border-radius:12px;border:1px solid #E5E9F2}
[data-testid="stTabs"] [role="tab"]{border-radius:9px!important;font-size:14px!important;
  font-weight:500!important;padding:8px 18px!important;color:#6B7A99!important;
  border:none!important;background:transparent!important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:#fff!important;
  color:#1C2B4A!important;box-shadow:0 1px 6px rgba(28,43,74,.10)!important}

.stButton>button{background:#2563EB!important;color:#fff!important;border:none!important;
  border-radius:12px!important;padding:12px 28px!important;font-size:15px!important;
  font-weight:600!important;font-family:'DM Sans',sans-serif!important;
  box-shadow:0 4px 14px rgba(37,99,235,.30)!important;width:100%!important}
.stButton>button:hover{background:#1D4ED8!important;transform:translateY(-1px)!important}

.callout{background:#EFF6FF;border-left:3px solid #2563EB;border-radius:0 10px 10px 0;
  padding:12px 16px;font-size:13px;color:#1E40AF;margin:12px 0;line-height:1.6}
.callout.warn{background:#FFFBEB;border-color:#F59E0B;color:#92400E}
.callout.danger{background:#FFF1F2;border-color:#EF4444;color:#991B1B}
.callout.success{background:#F0FDF4;border-color:#22C55E;color:#166534}

.compare-table{width:100%;border-collapse:collapse;font-size:13px}
.compare-table th{background:#F7F8FC;padding:10px 14px;text-align:left;
  font-weight:600;color:#6B7A99;font-size:11px;text-transform:uppercase;
  letter-spacing:.5px;border-bottom:1px solid #E5E9F2}
.compare-table td{padding:10px 14px;border-bottom:1px solid #F1F5F9;color:#374151}
.compare-table tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
.badge.low{background:#DCFCE7;color:#16A34A}
.badge.medium{background:#FEF3C7;color:#D97706}
.badge.high{background:#FFE4E6;color:#DC2626}
hr.soft{border:none;border-top:1px solid #E5E9F2;margin:18px 0}
.tooltip-label{font-size:13px;font-weight:500;color:#374151;margin-bottom:4px;
  display:flex;align-items:center;gap:4px}
.tooltip-label .tip{font-size:11px;color:#9CA3AF;background:#F3F4F6;
  border-radius:4px;padding:1px 5px}
</style>
""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_pipeline():
    df = load_german_credit()
    preprocessor = CreditRiskPreprocessor(random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, subsample=0.8,
                                   min_samples_leaf=5, random_state=42)
    m.fit(X_train, y_train)
    return m, preprocessor, X_test, y_test

model, preprocessor, X_test, y_test = load_pipeline()
feat_names  = preprocessor.get_feature_names()
importances = model.feature_importances_

def predict(inp):
    return float(model.predict_proba(preprocessor.transform(pd.DataFrame([inp])))[0][1])

def tier(p):
    if p < .20: return "low",    "✅ LOW RISK",    "#16A34A", "Approve at standard rate"
    if p < .50: return "medium", "⚠️ MEDIUM RISK", "#D97706", "Approve with collateral"
    return              "high",  "❌ HIGH RISK",   "#DC2626", "Reject or require guarantor"

def rev(m, v): return {vv: k for k, vv in m.items()}.get(v, v)

def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p*100,
        number={"suffix":"%","font":{"size":34,"color":color,"family":"DM Serif Display"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"size":10}},
               "bar":{"color":color,"thickness":.22},
               "bgcolor":"#F7F8FC","borderwidth":0,
               "steps":[{"range":[0,20],"color":"#DCFCE7"},
                        {"range":[20,50],"color":"#FEF3C7"},
                        {"range":[50,100],"color":"#FFE4E6"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":.8,"value":p*100}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#1C2B4A",height=200,margin=dict(l=20,r=20,t=10,b=10))
    return fig

def feat_fig(n=10):
    idx   = np.argsort(importances)[-n:][::-1]
    names = [feat_names[i].replace("_"," ").title() for i in idx]
    vals  = importances[idx]
    cols  = ["#2563EB" if v>.09 else "#60A5FA" if v>.05 else "#BAD1FF" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=cols,
                           text=[f"{v:.3f}" for v in vals], textposition="outside",
                           textfont=dict(size=11,color="#6B7A99")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans",color="#1C2B4A"),height=340,
                      xaxis=dict(showgrid=True,gridcolor="#F1F5F9",title="Importance"),
                      yaxis=dict(autorange="reversed",tickfont=dict(size=12)),
                      margin=dict(l=10,r=60,t=10,b=30),showlegend=False)
    return fig

def roc_fig():
    yp = model.predict_proba(X_test)[:,1]
    fpr,tpr,_ = roc_curve(y_test,yp)
    auc = roc_auc_score(y_test,yp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,fill="tozeroy",
        fillcolor="rgba(37,99,235,.08)",
        line=dict(color="#2563EB",width=2.5),name=f"Model (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],
        line=dict(color="#D1D5DB",dash="dash",width=1.5),name="Random"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans",color="#1C2B4A"),height=300,
                      xaxis=dict(title="FPR",showgrid=True,gridcolor="#F1F5F9"),
                      yaxis=dict(title="TPR",showgrid=True,gridcolor="#F1F5F9"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10,r=10,t=10,b=40))
    return fig

def default_inputs():
    return dict(checking_account="A11",duration=24,credit_history="A32",
                purpose="A40",credit_amount=5000,savings_account="A61",
                employment="A73",installment_rate=2,personal_status="A93",
                other_debtors="A101",residence_since=2,property="A121",
                age=35,other_installment_plans="A143",housing="A152",
                existing_credits=1,job="A173",num_dependents=1,
                telephone="A191",foreign_worker="A201")

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">Credit<span>IQ</span> &nbsp;·&nbsp; Loan Risk Analyser</div>
  <div class="topbar-badge">🏦 Gradient Boosting · German Credit Dataset</div>
</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs([
    "🔍  Risk Assessment",
    "⚖️  Compare Borrowers",
    "🧮  Affordability Calculator",
    "📊  Model Performance"])

# ════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="card"><div class="card-title">📋 Borrower Profile</div>'
                '<div class="card-sub">Enter borrower details to get an instant risk score. '
                'The most impactful fields are highlighted with tips.</div></div>',
                unsafe_allow_html=True)

    st.markdown("#### 💰 Loan Details")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<div class="tooltip-label">Loan Amount (DM)</div>',unsafe_allow_html=True)
        ca = st.number_input("ca",500,20000,5000,500,key="t1_ca",label_visibility="collapsed")
    with c2:
        st.markdown('<div class="tooltip-label">Duration (months) <span class="tip">longer=riskier</span></div>',unsafe_allow_html=True)
        dur = st.slider("dur",4,72,24,key="t1_dur",label_visibility="collapsed")
    with c3:
        st.markdown('<div class="tooltip-label">Instalment Rate <span class="tip">% of income</span></div>',unsafe_allow_html=True)
        ir = st.select_slider("ir",[1,2,3,4],2,key="t1_ir",label_visibility="collapsed")
    with c4:
        st.markdown('<div class="tooltip-label">Loan Purpose</div>',unsafe_allow_html=True)
        pur = st.selectbox("pur",list(CATEGORY_MAPS["purpose"].values()),key="t1_pur",label_visibility="collapsed")

    st.markdown('<hr class="soft">',unsafe_allow_html=True)
    st.markdown("#### 🏦 Financial Profile")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="tooltip-label">Checking Account <span class="tip">★ most important</span></div>',unsafe_allow_html=True)
        chk = st.selectbox("chk",list(CATEGORY_MAPS["checking_account"].values()),key="t1_chk",label_visibility="collapsed")
    with c2:
        st.markdown('<div class="tooltip-label">Savings Account</div>',unsafe_allow_html=True)
        sav = st.selectbox("sav",list(CATEGORY_MAPS["savings_account"].values()),key="t1_sav",label_visibility="collapsed")
    with c3:
        st.markdown('<div class="tooltip-label">Credit History <span class="tip">★ 2nd most important</span></div>',unsafe_allow_html=True)
        ch = st.selectbox("ch",list(CATEGORY_MAPS["credit_history"].values()),key="t1_ch",label_visibility="collapsed")

    st.markdown('<hr class="soft">',unsafe_allow_html=True)
    st.markdown("#### 👤 Personal Profile")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<div class="tooltip-label">Age <span class="tip">under 25 = higher risk</span></div>',unsafe_allow_html=True)
        age = st.slider("age",18,75,35,key="t1_age",label_visibility="collapsed")
    with c2:
        st.markdown('<div class="tooltip-label">Employment Since</div>',unsafe_allow_html=True)
        emp = st.selectbox("emp",list(CATEGORY_MAPS["employment"].values()),key="t1_emp",label_visibility="collapsed")
    with c3:
        st.markdown('<div class="tooltip-label">Housing</div>',unsafe_allow_html=True)
        hse = st.selectbox("hse",list(CATEGORY_MAPS["housing"].values()),key="t1_hse",label_visibility="collapsed")
    with c4:
        st.markdown('<div class="tooltip-label">Other Instalment Plans</div>',unsafe_allow_html=True)
        oi = st.selectbox("oi",list(CATEGORY_MAPS["other_installment_plans"].values()),key="t1_oi",label_visibility="collapsed")

    st.markdown("<br>",unsafe_allow_html=True)
    col_btn,col_hint = st.columns([1,3])
    with col_btn:
        go_btn = st.button("🔍  Assess Risk Now",use_container_width=True)
    with col_hint:
        st.markdown('<div class="callout">💡 <b>Tip:</b> The three biggest risk drivers are '
                    '<b>Checking Account status</b>, <b>Credit History</b>, and <b>Loan Duration</b>. '
                    'Try changing those to see the biggest effect on the score.</div>',
                    unsafe_allow_html=True)

    if go_btn:
        inp = default_inputs()
        inp.update(dict(
            checking_account=rev(CATEGORY_MAPS["checking_account"],chk),
            duration=dur, credit_history=rev(CATEGORY_MAPS["credit_history"],ch),
            purpose=rev(CATEGORY_MAPS["purpose"],pur), credit_amount=ca,
            savings_account=rev(CATEGORY_MAPS["savings_account"],sav),
            employment=rev(CATEGORY_MAPS["employment"],emp),
            installment_rate=ir, housing=rev(CATEGORY_MAPS["housing"],hse),
            other_installment_plans=rev(CATEGORY_MAPS["other_installment_plans"],oi),
            age=age,
        ))
        prob = predict(inp)
        t,label,color,action = tier(prob)

        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Result")
        left,right = st.columns([1,1])

        with left:
            st.plotly_chart(gauge(prob,color),use_container_width=True)
            st.markdown(f"""
            <div class="result-hero {t}">
              <div class="result-pct {t}">{prob:.1%}</div>
              <div class="result-tier" style="color:{color}">{label}</div>
              <div class="result-action">{action}</div>
            </div>""",unsafe_allow_html=True)

        with right:
            st.markdown("#### 🔑 Risk Driver Breakdown")
            st.markdown('<div class="card-sub">How each factor influences the risk score</div>',unsafe_allow_html=True)
            drivers = [
                ("Loan Duration",    dur/72),
                ("Credit Amount",    ca/20000),
                ("Checking Account", 0.9 if "<0" in chk else 0.55 if "No account" in chk else 0.15),
                ("Credit History",   0.88 if "Critical" in ch else 0.65 if "Delay" in ch else 0.18),
                ("Age Risk",         max(0,(30-age)/30) if age<30 else 0.1),
                ("Employment",       0.82 if "Unemployed" in emp else 0.5 if "<1" in emp else 0.2),
                ("Savings",          0.72 if "<100" in sav else 0.3),
                ("Instalment Rate",  ir/4),
            ]
            for name,val in drivers:
                val = min(max(val,0.02),1.0)
                bc  = "#EF4444" if val>.6 else "#F59E0B" if val>.35 else "#22C55E"
                st.markdown(f"""
                <div class="factor-row">
                  <div class="factor-name">{name}</div>
                  <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{val*100:.0f}%;background:{bc}"></div>
                  </div>
                  <div class="factor-val">{val*100:.0f}%</div>
                </div>""",unsafe_allow_html=True)

            callout_cls = "success" if t=="low" else "warn" if t=="medium" else "danger"
            msgs = {
                "low":    "✅ <b>Low risk.</b> Standard approval recommended.",
                "medium": "⚠️ <b>Medium risk.</b> Consider requesting collateral or a co-signer.",
                "high":   "❌ <b>High risk.</b> Recommend rejection or mandatory guarantor.",
            }
            st.markdown(f'<div class="callout {callout_cls}">{msgs[t]}</div>',unsafe_allow_html=True)

            monthly = ca/dur
            dti_est = (monthly*ir)/3000*100
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-chip"><div class="val">DM {monthly:,.0f}</div><div class="lbl">Monthly Payment</div></div>
              <div class="metric-chip"><div class="val">{dti_est:.0f}%</div><div class="lbl">Est. DTI</div></div>
              <div class="metric-chip"><div class="val">{dur}m</div><div class="lbl">Loan Term</div></div>
              <div class="metric-chip"><div class="val">{age}y</div><div class="lbl">Age</div></div>
            </div>""",unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📈 Global Feature Importance")
        st.markdown('<div class="card-sub">Top features the model uses across all borrowers</div>',unsafe_allow_html=True)
        st.plotly_chart(feat_fig(),use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — COMPARE BORROWERS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><div class="card-title">⚖️ Side-by-Side Comparison</div>'
                '<div class="card-sub">Compare two borrower profiles to evaluate co-applicants '
                'or alternative loan structures.</div></div>',unsafe_allow_html=True)

    def bform(col, lbl, kp, defs):
        with col:
            st.markdown(f"#### 👤 {lbl}")
            ca2   = st.number_input("Loan Amount (DM)",500,20000,defs["ca"],500,key=f"{kp}_ca")
            dur2  = st.slider("Duration (months)",4,72,defs["dur"],key=f"{kp}_dur")
            chk2  = st.selectbox("Checking Account",list(CATEGORY_MAPS["checking_account"].values()),index=defs["chk"],key=f"{kp}_chk")
            sav2  = st.selectbox("Savings Account",list(CATEGORY_MAPS["savings_account"].values()),index=defs["sav"],key=f"{kp}_sav")
            ch2   = st.selectbox("Credit History",list(CATEGORY_MAPS["credit_history"].values()),index=defs["ch"],key=f"{kp}_ch")
            age2  = st.slider("Age",18,75,defs["age"],key=f"{kp}_age")
            emp2  = st.selectbox("Employment",list(CATEGORY_MAPS["employment"].values()),index=defs["emp"],key=f"{kp}_emp")
            ir2   = st.select_slider("Instalment Rate",[1,2,3,4],defs["ir"],key=f"{kp}_ir")
            inp2  = default_inputs()
            inp2.update(dict(
                checking_account=rev(CATEGORY_MAPS["checking_account"],chk2),
                duration=dur2,credit_history=rev(CATEGORY_MAPS["credit_history"],ch2),
                credit_amount=ca2,savings_account=rev(CATEGORY_MAPS["savings_account"],sav2),
                employment=rev(CATEGORY_MAPS["employment"],emp2),
                installment_rate=ir2,age=age2))
            return inp2

    ca,cb,_ = st.columns([5,5,0.1])
    ia = bform(ca,"Borrower A","a",{"ca":3000,"dur":12,"chk":2,"sav":3,"ch":2,"age":42,"emp":4,"ir":1})
    ib = bform(cb,"Borrower B","b",{"ca":9000,"dur":48,"chk":0,"sav":0,"ch":4,"age":23,"emp":1,"ir":3})

    if st.button("⚖️  Compare Both Borrowers"):
        pa,pb = predict(ia),predict(ib)
        ta,la,ca_c,aa = tier(pa)
        tb,lb,cb_c,ab = tier(pb)
        st.markdown("---"); st.markdown("### Results")
        r1,r2 = st.columns(2)
        for col,p,t2,lbl,col_c,act,nm in [(r1,pa,ta,la,ca_c,aa,"Borrower A"),(r2,pb,tb,lb,cb_c,ab,"Borrower B")]:
            with col:
                st.plotly_chart(gauge(p,col_c),use_container_width=True)
                st.markdown(f"""<div class="result-hero {t2}">
                  <div style="font-size:12px;font-weight:600;color:#6B7A99">{nm}</div>
                  <div class="result-pct {t2}">{p:.1%}</div>
                  <div class="result-tier" style="color:{col_c}">{lbl}</div>
                  <div class="result-action">{act}</div></div>""",unsafe_allow_html=True)
        winner = "Borrower A" if pa<pb else "Borrower B"
        diff   = abs(pa-pb)
        st.markdown(f"""
        <table class="compare-table">
          <tr><th>Metric</th><th>Borrower A</th><th>Borrower B</th></tr>
          <tr><td>Default Probability</td><td><b>{pa:.1%}</b></td><td><b>{pb:.1%}</b></td></tr>
          <tr><td>Risk Tier</td>
              <td><span class="badge {ta}">{la}</span></td>
              <td><span class="badge {tb}">{lb}</span></td></tr>
          <tr><td>Recommended Action</td><td>{aa}</td><td>{ab}</td></tr>
          <tr><td>Est. Monthly Payment</td><td>DM {ia['credit_amount']/ia['duration']:,.0f}</td>
              <td>DM {ib['credit_amount']/ib['duration']:,.0f}</td></tr>
        </table>
        <br><div class="callout">📌 <b>{winner}</b> is lower risk by <b>{diff:.1%}</b>.</div>
        """,unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 3 — AFFORDABILITY CALCULATOR
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card"><div class="card-title">🧮 Affordability Calculator</div>'
                '<div class="card-sub">Check if a borrower can comfortably afford a loan '
                'using the 35% DTI threshold — the same standard used by major banks.</div></div>',
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### 📥 Inputs")
        inc  = st.number_input("Monthly Net Income (DM)",500,50000,3500,100)
        la   = st.number_input("Requested Loan Amount (DM)",500,50000,8000,500)
        ld   = st.slider("Loan Duration (months)",6,84,36)
        lir  = st.select_slider("Instalment Rate",[1,2,3,4],2)
        ex   = st.number_input("Existing Monthly Debt (DM)",0,10000,500,100)
        if st.button("🧮  Calculate Affordability",use_container_width=True):
            st.session_state["aff_done"] = True
            mp   = la/ld
            tot  = mp*lir+ex
            dti  = tot/inc*100
            maxl = max((inc*.35-ex)/lir*ld,0)
            head = inc*.35-tot
            st.session_state["aff"] = dict(mp=mp,tot=tot,dti=dti,maxl=maxl,head=head)
    with c2:
        st.markdown("#### 📊 Results")
        if st.session_state.get("aff_done"):
            a = st.session_state["aff"]
            tc = "low" if a["dti"]<25 else "medium" if a["dti"]<35 else "high"
            dc = "#16A34A" if a["dti"]<25 else "#D97706" if a["dti"]<35 else "#DC2626"
            ok = a["dti"]<=35
            st.markdown(f"""<div class="result-hero {tc}">
              <div class="result-pct {tc}">{a['dti']:.1f}%</div>
              <div class="result-tier" style="color:{dc}">DEBT-TO-INCOME RATIO</div>
              <div class="result-action">{"✅ Within safe threshold (≤35%)" if ok else "❌ Exceeds safe threshold"}</div>
            </div>""",unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-row">
              <div class="metric-chip"><div class="val">DM {a['mp']:,.0f}</div><div class="lbl">Monthly Payment</div></div>
              <div class="metric-chip"><div class="val">DM {a['tot']:,.0f}</div><div class="lbl">Total Monthly Debt</div></div>
              <div class="metric-chip"><div class="val">DM {max(a['head'],0):,.0f}</div><div class="lbl">Headroom</div></div>
              <div class="metric-chip"><div class="val">DM {a['maxl']:,.0f}</div><div class="lbl">Max Affordable</div></div>
            </div><br>""",unsafe_allow_html=True)
            cls = "success" if ok and a["dti"]<25 else "warn" if ok else "danger"
            msg = (f"✅ DTI of <b>{a['dti']:.1f}%</b> is healthy." if ok and a["dti"]<25
                   else f"⚠️ DTI of <b>{a['dti']:.1f}%</b> is within limits but elevated."
                   if ok else f"❌ DTI exceeds threshold. Max affordable loan: <b>DM {a['maxl']:,.0f}</b>.")
            st.markdown(f'<div class="callout {cls}">{msg}</div>',unsafe_allow_html=True)
            st.plotly_chart(gauge(a["dti"]/100,dc),use_container_width=True)
        else:
            st.markdown('<div class="callout">👆 Fill in the inputs and click <b>Calculate Affordability</b>.</div>',unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="card"><div class="card-title">📊 Model Performance Report</div>'
                '<div class="card-sub">Gradient Boosting evaluated on 200 held-out test borrowers.</div></div>',
                unsafe_allow_html=True)
    yp = model.predict_proba(X_test)[:,1]
    ypr = model.predict(X_test)
    auc  = roc_auc_score(y_test,yp)
    acc  = accuracy_score(y_test,ypr)
    prec = precision_score(y_test,ypr)
    rec  = recall_score(y_test,ypr)
    f1   = f1_score(y_test,ypr)
    gini = 2*auc-1
    st.markdown(f"""<div class="metric-row">
      <div class="metric-chip"><div class="val">{auc:.3f}</div><div class="lbl">ROC-AUC</div></div>
      <div class="metric-chip"><div class="val">{gini:.3f}</div><div class="lbl">Gini Coeff.</div></div>
      <div class="metric-chip"><div class="val">{acc:.1%}</div><div class="lbl">Accuracy</div></div>
      <div class="metric-chip"><div class="val">{prec:.1%}</div><div class="lbl">Precision</div></div>
      <div class="metric-chip"><div class="val">{rec:.1%}</div><div class="lbl">Recall</div></div>
      <div class="metric-chip"><div class="val">{f1:.3f}</div><div class="lbl">F1-Score</div></div>
    </div><br>""",unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC Curve")
        st.plotly_chart(roc_fig(),use_container_width=True)
    with col2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test,ypr)
        fig_cm = px.imshow(cm,labels=dict(x="Predicted",y="Actual"),
                           x=["No Default","Default"],y=["No Default","Default"],
                           color_continuous_scale=[[0,"#EFF6FF"],[1,"#2563EB"]],
                           text_auto=True)
        fig_cm.update_traces(textfont_size=20)
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(family="DM Sans",color="#1C2B4A"),
                             height=300,margin=dict(l=10,r=10,t=10,b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_cm,use_container_width=True)

    st.markdown("#### Feature Importance")
    st.plotly_chart(feat_fig(12),use_container_width=True)

    st.markdown("""#### 📖 Metric Guide
<table class="compare-table">
  <tr><th>Metric</th><th>Banking Meaning</th></tr>
  <tr><td><b>ROC-AUC</b></td><td>Probability the model ranks a defaulter above a non-defaulter. Industry benchmark: >0.70 = good.</td></tr>
  <tr><td><b>Gini Coefficient</b></td><td>Basel III standard = 2×AUC−1. Values >0.50 pass regulatory validation.</td></tr>
  <tr><td><b>Precision</b></td><td>Of all flagged defaults, how many actually defaulted? Higher = fewer false rejections.</td></tr>
  <tr><td><b>Recall</b></td><td>Of all actual defaults, how many did we catch? Higher = fewer missed defaults (most critical for risk).</td></tr>
</table>""",unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:28px 0 12px;font-size:12px;color:#9CA3AF;">'
            'CreditIQ · Python · Scikit-learn · Streamlit · German Credit Dataset (UCI)</div>',
            unsafe_allow_html=True)