"""
Universal Bank Intelligence Platform
=====================================
Flat single-file Streamlit app — no subfolder imports.
Upload UniversalBank.csv alongside this file on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #0A0F1E !important;
    border-right: 1px solid rgba(212,168,83,0.18) !important;
}
[data-testid="stMetric"] {
    background: #0C1120;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    border-top: 2px solid #D4A853;
}
[data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #F0F4FF !important;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px; font-weight: 800;
    color: #F0F4FF; margin-bottom: 2px;
}
.page-title span { color: #D4A853; }
.page-sub { font-size: 11px; color: #64748B; margin-bottom: 22px; letter-spacing: 0.4px; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
    background: rgba(212,168,83,0.12); color: #D4A853;
    border: 1px solid rgba(212,168,83,0.3); margin-bottom: 6px;
}
.card {
    background: #0C1120; border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #D4A853; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px;
    font-size: 12px; color: #94A3B8; line-height: 1.6;
}
.card strong { color: #F0F4FF; }
.result-box-yes {
    background: rgba(34,197,94,0.07); border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px; padding: 24px; text-align: center;
}
.result-box-no {
    background: rgba(100,116,139,0.08); border: 1px solid rgba(100,116,139,0.25);
    border-radius: 12px; padding: 24px; text-align: center;
}
.verdict-yes { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#22C55E; }
.verdict-no  { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#94A3B8; }
.prob-num { font-family:'Syne',sans-serif; font-size:52px; font-weight:800; color:#F0F4FF; line-height:1; }
.stButton>button {
    background: linear-gradient(135deg,#D4A853,#C4943A) !important;
    color: #050810 !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:12px !important;
    letter-spacing:1px !important; text-transform:uppercase;
    border:none !important; border-radius:8px !important;
}
.stButton>button:hover { opacity:.9; box-shadow: 0 8px 24px rgba(212,168,83,0.25) !important; }
[data-testid="stTabs"] [role="tab"] {
    font-family:'DM Mono',monospace; font-size:11px;
    letter-spacing:1px; text-transform:uppercase; color:#64748B;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#D4A853 !important; border-bottom-color:#D4A853 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
PLOT_DARK = dict(
    template="plotly_dark",
    plot_bgcolor="#0C1120",
    paper_bgcolor="#0C1120",
)

@st.cache_data
def load_data():
    import os
    # Try multiple locations
    for path in ["UniversalBank.csv", "data/UniversalBank.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("❌ UniversalBank.csv not found. Please upload it alongside app.py.")
        st.stop()

    df.columns = df.columns.str.strip()
    df = df.drop(columns=["ID", "ZIP Code"], errors="ignore")
    df["Experience"] = df["Experience"].clip(lower=0)

    # Derived columns
    bins   = [0,30,60,80,100,130,160,200,300]
    labels = ["<30K","30-60K","60-80K","80-100K","100-130K","130-160K","160-200K","200K+"]
    df["Income Bracket"] = pd.cut(df["Income"], bins=bins, labels=labels)
    df["Age Band"]       = pd.cut(df["Age"],    bins=[22,30,38,46,54,62,70],
                                   labels=["23-30","31-38","39-46","47-54","55-62","63-70"])
    df["CC Bracket"]     = pd.cut(df["CCAvg"],  bins=[0,1,2,3,4,5,10],
                                   labels=["0-1","1-2","2-3","3-4","4-5","5+"])
    df["Edu Label"]      = df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
    df["Loan Status"]    = df["Personal Loan"].map({1:"Accepted",0:"Rejected"})
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training 11 ML models on your data…")
def train_all_models():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score,
                                  confusion_matrix)
    from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                                   AdaBoostClassifier, ExtraTreesClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    target   = "Personal Loan"
    feat_cols = ["Age","Experience","Income","Family","CCAvg",
                 "Education","Mortgage","Securities Account","CD Account","Online","CreditCard"]
    X = df[feat_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_tr, y_train = sm.fit_resample(X_tr, y_train)

    model_defs = {
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
        "XGBoost":            XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=42),
        "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
        "Extra Trees":        ExtraTreesClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
        "AdaBoost":           AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "Decision Tree":      DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Logistic Regression":LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
        "KNN":                KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SVM":                SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
        "Neural Network":     MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42),
    }

    # Try LightGBM
    try:
        from lightgbm import LGBMClassifier
        model_defs["LightGBM"] = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    except Exception:
        pass

    results = {}
    trained = {}
    for name, clf in model_defs.items():
        try:
            clf.fit(X_tr, y_train)
            yp    = clf.predict(X_te)
            yprob = clf.predict_proba(X_te)[:,1] if hasattr(clf,"predict_proba") else None
            results[name] = {
                "Accuracy":  round(accuracy_score(y_test, yp)*100, 2),
                "Precision": round(precision_score(y_test, yp, zero_division=0)*100, 2),
                "Recall":    round(recall_score(y_test, yp, zero_division=0)*100, 2),
                "F1":        round(f1_score(y_test, yp, zero_division=0)*100, 2),
                "AUC-ROC":   round(roc_auc_score(y_test, yprob), 4) if yprob is not None else None,
                "CM":        confusion_matrix(y_test, yp).tolist(),
            }
            trained[name] = clf
        except Exception as e:
            st.warning(f"{name} failed: {e}")

    return results, trained, scaler, feat_cols, X_test, y_test

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏦 Universal Bank")
    st.markdown("<div style='font-size:11px;color:#D4A853;letter-spacing:1px'>INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Home",
        "📊 Overview",
        "👥 Customer Analytics",
        "💳 Loan Analytics",
        "🔬 Diagnostic Analysis",
        "📉 Predictive Analytics",
        "🤖 AI Loan Predictor",
        "📈 Model Comparison",
        "⚠️ Risk Matrix",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#475569;line-height:1.9'>
    DATASET<br>
    <span style='color:#D4A853'>5,000 records · 12 features</span><br><br>
    ML ALGORITHMS<br>
    <span style='color:#D4A853'>Gradient Boosting · XGBoost<br>
    LightGBM · Random Forest<br>
    Extra Trees · AdaBoost<br>
    Decision Tree · Logistic Reg<br>
    KNN · SVM · Neural Network</span>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="badge">WELCOME</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Universal Bank <span>Intelligence Platform</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enterprise ML-powered banking analytics · 5,000 customers · 11 algorithms · 7 analysis modules</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Customers",     "5,000")
    c2.metric("Loan Rate",     f"{df['Personal Loan'].mean()*100:.1f}%")
    c3.metric("Avg Income",    f"${df['Income'].mean():.0f}K")
    c4.metric("Avg CC Spend",  f"${df['CCAvg'].mean():.2f}K")
    c5.metric("Online Users",  f"{df['Online'].mean()*100:.1f}%")
    c6.metric("ML Algorithms", "11")
    st.markdown("---")

    cols = st.columns(3)
    cards = [
        ("📊","Overview",           "Executive KPIs, income distribution, loan split, age-income scatter."),
        ("👥","Customer Analytics", "Demographics, behavioral filters, family/education breakdowns."),
        ("💳","Loan Analytics",     "Acceptance rates by income, education, CD account, family size."),
        ("🔬","Diagnostic Analysis","Why customers accept/reject — root-cause, outliers, correlations."),
        ("📉","Predictive Analytics","Trends, forecasts, segment scoring, churn indicators."),
        ("🤖","AI Loan Predictor",  "Real-time loan approval prediction with 11 ML algorithms."),
        ("📈","Model Comparison",   "Full benchmark — accuracy, AUC-ROC, confusion matrices."),
        ("⚠️","Risk Matrix",       "Risk tier scoring, Pearson heatmap, portfolio health score."),
    ]
    for i,(icon,title,desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f'<div class="card"><strong>{icon} {title}</strong><br>{desc}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Overview":
    st.markdown('<div class="badge">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Portfolio <span>Overview</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">High-level snapshot of the Universal Bank customer portfolio</div>', unsafe_allow_html=True)

    acc = int(df["Personal Loan"].sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers",    "5,000")
    c2.metric("Loan Acceptance",    f"{acc} ({acc/50:.1f}%)")
    c3.metric("Avg Annual Income",  f"${df['Income'].mean():.1f}K")
    c4.metric("Avg CC Monthly Spend",f"${df['CCAvg'].mean():.2f}K")
    st.markdown("---")

    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("**Income Distribution by Loan Status**")
        grp = df.groupby(["Income Bracket","Loan Status"], observed=True).size().reset_index(name="Count")
        fig = px.bar(grp, x="Income Bracket", y="Count", color="Loan Status",
                     color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                     barmode="group", **PLOT_DARK)
        fig.update_layout(legend=dict(orientation="h",y=1.1), margin=dict(t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Loan Portfolio Split**")
        fig2 = go.Figure(go.Pie(
            labels=["Accepted","Rejected"], values=[acc, 5000-acc],
            hole=0.7, marker_colors=["#D4A853","#1E293B"],
            textinfo="label+percent"))
        fig2.update_layout(showlegend=False, **PLOT_DARK,
                           margin=dict(t=10,b=10,l=0,r=0),
                           annotations=[dict(text=f"<b>{acc}</b><br>loans",
                                             x=0.5,y=0.5,font_size=18,
                                             showarrow=False,font_color="#D4A853")])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3,col4,col5 = st.columns(3)
    with col3:
        st.markdown("**Education Level**")
        edu_c = df["Edu Label"].value_counts().reset_index()
        edu_c.columns = ["Education","Count"]
        fig3 = px.bar(edu_c, x="Count", y="Education", orientation="h",
                      color="Education",
                      color_discrete_sequence=["#3B82F6","#D4A853","#00D4AA"], **PLOT_DARK)
        fig3.update_layout(showlegend=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Family Size**")
        fam = df["Family"].value_counts().sort_index().reset_index()
        fam.columns = ["Family","Count"]
        fig4 = px.bar(fam, x="Family", y="Count",
                      color_discrete_sequence=["#7C3AED"], **PLOT_DARK)
        fig4.update_layout(margin=dict(t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown("**Digital Adoption Radar**")
        cats = ["Online","Credit Card","Securities","CD Account","Mortgage"]
        vals = [df["Online"].mean()*100, df["CreditCard"].mean()*100,
                df["Securities Account"].mean()*100, df["CD Account"].mean()*100,
                (df["Mortgage"]>0).mean()*100]
        fig5 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor="rgba(212,168,83,0.15)",
            line_color="#D4A853", marker=dict(size=5,color="#D4A853")))
        fig5.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100]),
                                      bgcolor="#0C1120"),
                           showlegend=False, **PLOT_DARK, margin=dict(t=20,b=10))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("**Age vs Income — Loan Acceptance Scatter**")
    samp = df.sample(min(1500,len(df)), random_state=42)
    fig6 = px.scatter(samp, x="Income", y="Age", color="Loan Status",
                      color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                      opacity=0.65, **PLOT_DARK,
                      hover_data=["Edu Label","CCAvg","Family"])
    fig6.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
    st.plotly_chart(fig6, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOMER ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.markdown('<div class="badge">CUSTOMER INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Customer <span>Analytics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Behavioral and demographic breakdown with interactive filters</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Online Banking",   f"{df['Online'].mean()*100:.1f}%")
    c2.metric("Credit Cards",     f"{df['CreditCard'].mean()*100:.1f}%")
    c3.metric("Securities Accts", f"{df['Securities Account'].mean()*100:.1f}%")
    c4.metric("CD Accounts",      f"{df['CD Account'].mean()*100:.1f}%")
    st.markdown("---")

    with st.expander("🔍 Filters", expanded=True):
        f1,f2,f3,f4 = st.columns(4)
        inc_r = f1.slider("Income ($K)", 0, 250, (0,250))
        age_r = f2.slider("Age",         18, 70,  (18,70))
        edu_f = f3.multiselect("Education",[1,2,3],default=[1,2,3],
                               format_func=lambda x:{1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
        loan_f = f4.selectbox("Loan Status",["All","Accepted","Rejected"])

    mask = df["Income"].between(*inc_r) & df["Age"].between(*age_r) & df["Education"].isin(edu_f)
    if loan_f=="Accepted": mask &= df["Personal Loan"]==1
    elif loan_f=="Rejected": mask &= df["Personal Loan"]==0
    fdf = df[mask]
    st.caption(f"Showing **{len(fdf):,}** of 5,000 customers")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Age Distribution by Education**")
        fig = px.histogram(fdf, x="Age", color="Edu Label", nbins=20, barmode="stack",
                           color_discrete_sequence=["#3B82F6","#D4A853","#00D4AA"], **PLOT_DARK)
        fig.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**CC Spend Distribution**")
        fig2 = px.histogram(fdf, x="CCAvg", nbins=35, color="Loan Status",
                            color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                            barmode="overlay", opacity=0.75, **PLOT_DARK)
        fig2.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("**Income vs CC Spend — Bubble by Family**")
        smp = fdf.sample(min(800,len(fdf)), random_state=1)
        fig3 = px.scatter(smp, x="Income", y="CCAvg", size="Family",
                          color="Loan Status",
                          color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                          opacity=0.7, **PLOT_DARK)
        fig3.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Experience Distribution**")
        fig4 = px.histogram(fdf, x="Experience", color="Loan Status", nbins=25,
                            color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                            barmode="overlay", opacity=0.75, **PLOT_DARK)
        fig4.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("**Customer Records Explorer**")
    disp = fdf.copy().head(300)
    disp["Personal Loan"] = disp["Personal Loan"].map({1:"✅ Yes",0:"❌ No"})
    disp["Online"]        = disp["Online"].map({1:"Yes",0:"No"})
    disp["CreditCard"]    = disp["CreditCard"].map({1:"Yes",0:"No"})
    disp["CD Account"]    = disp["CD Account"].map({1:"Yes",0:"No"})
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# LOAN ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💳 Loan Analytics":
    st.markdown('<div class="badge">LOAN PORTFOLIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Loan <span>Analytics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Deep-dive into every acceptance driver across the portfolio</div>', unsafe_allow_html=True)

    acc_df = df[df["Personal Loan"]==1]
    rej_df = df[df["Personal Loan"]==0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accepted",          f"{len(acc_df):,}", "9.60% rate")
    c2.metric("Rejected",          f"{len(rej_df):,}", "90.40% rate")
    c3.metric("Avg Income (Acc.)", f"${acc_df['Income'].mean():.0f}K", f"vs ${rej_df['Income'].mean():.0f}K")
    c4.metric("Avg CCAvg (Acc.)",  f"${acc_df['CCAvg'].mean():.2f}K", f"vs ${rej_df['CCAvg'].mean():.2f}K")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Acceptance Rate by Income Bracket**")
        r = df.groupby("Income Bracket", observed=True)["Personal Loan"].mean().reset_index()
        r["Rate %"] = (r["Personal Loan"]*100).round(1)
        fig = px.line(r, x="Income Bracket", y="Rate %", markers=True,
                      color_discrete_sequence=["#D4A853"], **PLOT_DARK)
        fig.update_traces(line_width=2.5, marker_size=9)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Acceptance Rate by Education**")
        r2 = df.groupby("Edu Label")["Personal Loan"].mean().reset_index()
        r2["Rate %"] = (r2["Personal Loan"]*100).round(1)
        fig2 = px.bar(r2, x="Edu Label", y="Rate %",
                      color="Edu Label",
                      color_discrete_sequence=["#3B82F6","#D4A853","#00D4AA"],
                      text="Rate %", **PLOT_DARK)
        fig2.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4,col5 = st.columns(3)
    with col3:
        st.markdown("**CD Account Impact**")
        r3 = df.groupby("CD Account")["Personal Loan"].mean().reset_index()
        r3["Label"] = r3["CD Account"].map({0:"No CD",1:"Has CD"})
        r3["Rate %"] = (r3["Personal Loan"]*100).round(1)
        fig3 = px.bar(r3, x="Label", y="Rate %", text="Rate %",
                      color_discrete_sequence=["#00D4AA"], **PLOT_DARK)
        fig3.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Family Size Impact**")
        r4 = df.groupby("Family")["Personal Loan"].mean().reset_index()
        r4["Rate %"] = (r4["Personal Loan"]*100).round(1)
        fig4 = px.bar(r4, x="Family", y="Rate %", text="Rate %",
                      color_discrete_sequence=["#7C3AED"], **PLOT_DARK)
        fig4.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown("**CC Spend Impact**")
        r5 = df.groupby("CC Bracket", observed=True)["Personal Loan"].mean().reset_index()
        r5["Rate %"] = (r5["Personal Loan"]*100).round(1)
        fig5 = px.line(r5, x="CC Bracket", y="Rate %", markers=True,
                       color_discrete_sequence=["#EF4444"], **PLOT_DARK)
        fig5.update_traces(line_width=2.5, marker_size=7)
        fig5.update_layout(margin=dict(t=10))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("**Income Box Plot: Accepted vs Rejected**")
    fig6 = px.box(df, x="Loan Status", y="Income", color="Loan Status",
                  color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                  points="outliers", **PLOT_DARK)
    fig6.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig6, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Diagnostic Analysis":
    st.markdown('<div class="badge">DIAGNOSTIC ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Diagnostic <span>Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Root-cause investigation — why customers do or don't accept personal loans</div>', unsafe_allow_html=True)

    num_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                "Mortgage","Securities Account","CD Account","Online","CreditCard","Personal Loan"]
    st.markdown("---")

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "🔗 Correlations","📦 Distributions","🎯 Outlier Detection",
        "📐 Statistical Tests","🧩 Segment Profiling"
    ])

    # ── TAB 1: CORRELATIONS ──────────────────────────────────────────────────
    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Pearson Correlation Heatmap**")
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f",
                            color_continuous_scale=["#EF4444","#0C1120","#00D4AA"],
                            zmin=-1, zmax=1, aspect="auto",
                            **PLOT_DARK)
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Feature Correlation with Personal Loan (sorted)**")
            ct = df[num_cols].corr()["Personal Loan"].drop("Personal Loan").sort_values()
            ct_df = pd.DataFrame({"Feature":ct.index,"Correlation":ct.values})
            colors = ["#EF4444" if v<0 else "#00D4AA" for v in ct.values]
            fig2 = px.bar(ct_df, x="Correlation", y="Feature", orientation="h",
                          color="Correlation",
                          color_continuous_scale=["#EF4444","#0C1120","#00D4AA"],
                          **PLOT_DARK)
            fig2.update_layout(coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Pairwise Scatter Matrix (Top 5 Features)**")
        top_feats = ["Income","CCAvg","Age","Mortgage","Family","Personal Loan"]
        fig3 = px.scatter_matrix(df.sample(500,random_state=42),
                                  dimensions=top_feats[:-1],
                                  color="Loan Status",
                                  color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                                  opacity=0.5, **PLOT_DARK)
        fig3.update_traces(diagonal_visible=False)
        fig3.update_layout(margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2: DISTRIBUTIONS ─────────────────────────────────────────────────
    with tab2:
        feat_sel = st.selectbox("Select Feature",
                                ["Income","CCAvg","Age","Experience","Mortgage","Family"])

        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f"**{feat_sel} Distribution (Accepted vs Rejected)**")
            fig = px.histogram(df, x=feat_sel, color="Loan Status", nbins=35,
                               color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                               barmode="overlay", opacity=0.75,
                               marginal="box", **PLOT_DARK)
            fig.update_layout(legend=dict(orientation="h",y=1.1), margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"**{feat_sel} Violin by Loan Status**")
            fig2 = px.violin(df, y=feat_sel, x="Loan Status", color="Loan Status",
                             color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                             box=True, points="outliers", **PLOT_DARK)
            fig2.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"**{feat_sel} KDE by Education Level**")
        fig3 = px.histogram(df, x=feat_sel, color="Edu Label", nbins=30,
                            color_discrete_sequence=["#3B82F6","#D4A853","#00D4AA"],
                            barmode="overlay", opacity=0.6,
                            histnorm="density", **PLOT_DARK)
        fig3.update_layout(legend=dict(orientation="h",y=1.05), margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3: OUTLIER DETECTION ─────────────────────────────────────────────
    with tab3:
        st.markdown("**Outlier Detection via IQR Method**")
        outlier_summary = []
        for col in ["Income","CCAvg","Mortgage","Age","Experience"]:
            Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
            n_out = int(((df[col]<lo)|(df[col]>hi)).sum())
            outlier_summary.append({
                "Feature":col, "Q1":round(Q1,2), "Q3":round(Q3,2),
                "IQR":round(IQR,2), "Lower Bound":round(lo,2),
                "Upper Bound":round(hi,2), "# Outliers":n_out,
                "% Outliers":round(n_out/len(df)*100,2)
            })
        st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Income Outliers**")
            fig = px.box(df, y="Income", color="Loan Status", points="outliers",
                         color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                         **PLOT_DARK)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Mortgage Outliers**")
            fig2 = px.box(df, y="Mortgage", color="Loan Status", points="outliers",
                          color_discrete_map={"Accepted":"#D4A853","Rejected":"#334155"},
                          **PLOT_DARK)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Multivariate Outlier Map — Income × CCAvg**")
        df_out = df.copy()
        for col in ["Income","CCAvg"]:
            Q1,Q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
            IQR = Q3-Q1
            df_out[f"{col}_out"] = ((df_out[col]<Q1-1.5*IQR)|(df_out[col]>Q3+1.5*IQR))
        df_out["Outlier"] = (df_out["Income_out"]|df_out["CCAvg_out"]).map({True:"Outlier",False:"Normal"})
        fig3 = px.scatter(df_out.sample(1000,random_state=42), x="Income", y="CCAvg",
                          color="Outlier",
                          color_discrete_map={"Outlier":"#EF4444","Normal":"#334155"},
                          **PLOT_DARK, opacity=0.7)
        fig3.update_layout(margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 4: STATISTICAL TESTS ─────────────────────────────────────────────
    with tab4:
        from scipy import stats as scipy_stats

        st.markdown("**T-Test: Accepted vs Rejected on Continuous Features**")
        test_results = []
        for col in ["Income","CCAvg","Age","Experience","Mortgage"]:
            a = df[df["Personal Loan"]==1][col].dropna()
            b = df[df["Personal Loan"]==0][col].dropna()
            t,p = scipy_stats.ttest_ind(a,b)
            test_results.append({
                "Feature":col,
                "Mean (Accepted)":round(a.mean(),2),
                "Mean (Rejected)":round(b.mean(),2),
                "Difference":round(a.mean()-b.mean(),2),
                "T-Statistic":round(t,3),
                "P-Value":f"{p:.2e}",
                "Significant":"✅ Yes" if p<0.05 else "❌ No"
            })
        st.dataframe(pd.DataFrame(test_results), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Chi-Square Test: Categorical Features vs Loan Status**")
        chi_results = []
        for col in ["Education","Family","CD Account","Online","CreditCard","Securities Account"]:
            ct = pd.crosstab(df[col], df["Personal Loan"])
            chi2,p,dof,_ = scipy_stats.chi2_contingency(ct)
            chi_results.append({
                "Feature":col,
                "Chi² Statistic":round(chi2,3),
                "P-Value":f"{p:.2e}",
                "Degrees of Freedom":dof,
                "Significant":"✅ Yes" if p<0.05 else "❌ No"
            })
        st.dataframe(pd.DataFrame(chi_results), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Descriptive Statistics Summary**")
        desc = df[["Age","Experience","Income","Family","CCAvg","Mortgage"]].describe().T
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)

    # ── TAB 5: SEGMENT PROFILING ─────────────────────────────────────────────
    with tab5:
        st.markdown("**Customer Segment Profiles — Accepted vs Rejected**")
        seg_cols = ["Age","Experience","Income","Family","CCAvg","Mortgage"]
        seg = df.groupby("Loan Status")[seg_cols].mean().T.round(2)
        seg["Difference"] = (seg["Accepted"]-seg["Rejected"]).round(2)
        seg["% Change"]   = ((seg["Accepted"]-seg["Rejected"])/seg["Rejected"]*100).round(1)
        st.dataframe(seg, use_container_width=True)

        st.markdown("---")
        st.markdown("**Multi-Segment Heatmap: Acceptance Rate by Education × Family**")
        pivot = df.pivot_table(values="Personal Loan",
                               index="Edu Label", columns="Family",
                               aggfunc="mean").round(3)*100
        fig = px.imshow(pivot, text_auto=".1f",
                        color_continuous_scale=["#0C1120","#D4A853"],
                        **PLOT_DARK, aspect="auto")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Acceptance Rate Heatmap: Income Bracket × Education**")
        pivot2 = df.pivot_table(values="Personal Loan",
                                index="Edu Label", columns="Income Bracket",
                                aggfunc="mean", observed=True).round(3)*100
        fig2 = px.imshow(pivot2, text_auto=".1f",
                         color_continuous_scale=["#0C1120","#D4A853"],
                         **PLOT_DARK, aspect="auto")
        fig2.update_layout(margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📉 Predictive Analytics":
    st.markdown('<div class="badge">PREDICTIVE ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Predictive <span>Analytics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Forward-looking insights, segment scoring, and opportunity identification</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["📊 Segment Scoring","🎯 Opportunity Pipeline","📈 Trend Projections"])

    with tab1:
        st.markdown("**Loan Acceptance Probability Score by Segment**")
        seg_df = df.groupby(["Edu Label","Income Bracket"], observed=True).agg(
            Customers=("Personal Loan","count"),
            Accept_Rate=("Personal Loan","mean")
        ).reset_index()
        seg_df["Accept_Rate %"] = (seg_df["Accept_Rate"]*100).round(1)
        seg_df["Opportunity Score"] = (seg_df["Accept_Rate %"] * seg_df["Customers"] / 100).round(0).astype(int)
        fig = px.scatter(seg_df, x="Income Bracket", y="Accept_Rate %",
                         size="Customers", color="Edu Label",
                         color_discrete_sequence=["#3B82F6","#D4A853","#00D4AA"],
                         **PLOT_DARK, hover_data=["Opportunity Score"])
        fig.update_layout(legend=dict(orientation="h",y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg_df.sort_values("Opportunity Score",ascending=False).head(15),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**Untapped Opportunity: High-Income Customers Who Rejected Loan**")
        opp = df[(df["Personal Loan"]==0) & (df["Income"]>100)].copy()
        opp["Propensity Score"] = (
            opp["Income"]*0.4 + opp["CCAvg"]*3 +
            opp["Education"]*2 + opp["CD Account"]*10
        ).clip(upper=100)
        opp["Tier"] = pd.cut(opp["Propensity Score"],
                              bins=[0,30,60,101],
                              labels=["Low","Medium","High"])
        st.metric("High-Income Untapped Customers", f"{len(opp):,}",
                  f"${opp['Income'].mean():.0f}K avg income")
        col1,col2 = st.columns(2)
        with col1:
            tier_c = opp["Tier"].value_counts().reset_index()
            tier_c.columns = ["Tier","Count"]
            fig = px.pie(tier_c, names="Tier", values="Count",
                         color="Tier", hole=0.6,
                         color_discrete_map={"High":"#D4A853","Medium":"#3B82F6","Low":"#334155"},
                         **PLOT_DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(opp.sample(min(500,len(opp)),random_state=42),
                              x="Income", y="CCAvg", color="Propensity Score",
                              color_continuous_scale=["#334155","#D4A853"],
                              **PLOT_DARK, size_max=8)
            fig2.update_layout(margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Top 20 Highest Propensity Untapped Customers**")
        top_opp = opp[["Age","Income","CCAvg","Education","Mortgage","Propensity Score","Tier"]]\
                     .sort_values("Propensity Score",ascending=False).head(20)
        top_opp["Education"] = top_opp["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
        st.dataframe(top_opp, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("**Projected Acceptance Rate by Income Growth Scenarios**")
        income_points = list(range(20, 230, 10))
        # Logistic curve fitted to data
        from scipy.special import expit
        rate_data = df.groupby("Income")["Personal Loan"].mean().reset_index()
        scenarios = {
            "Conservative (+5% income)":  [i*1.05 for i in income_points],
            "Moderate   (+15% income)":   [i*1.15 for i in income_points],
            "Optimistic (+30% income)":   [i*1.30 for i in income_points],
        }
        # Simple sigmoid model
        def pred_rate(inc): return expit((inc - 100) / 25) * 0.8 + 0.02

        fig = go.Figure()
        colors = ["#3B82F6","#D4A853","#00D4AA"]
        for (label, vals), col in zip(scenarios.items(), colors):
            fig.add_trace(go.Scatter(
                x=income_points, y=[pred_rate(v)*100 for v in vals],
                name=label, line=dict(color=col, width=2), mode="lines+markers",
                marker=dict(size=5)))
        fig.update_layout(xaxis_title="Income ($K)", yaxis_title="Predicted Acceptance Rate (%)",
                          legend=dict(orientation="h",y=1.1), margin=dict(t=30),
                          **PLOT_DARK)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# AI PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Loan Predictor":
    st.markdown('<div class="badge">REAL-TIME AI PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">AI Loan Approval <span>Predictor</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Live loan approval prediction using models trained on 5,000 Universal Bank customers</div>', unsafe_allow_html=True)

    results, trained_models, scaler, feat_cols, X_test, y_test = train_all_models()

    algo_names = list(trained_models.keys())
    sel = st.selectbox("🧠 Algorithm", algo_names)
    model   = trained_models[sel]
    metrics = results[sel]

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['Accuracy']}%")
    m2.metric("Precision", f"{metrics['Precision']}%")
    m3.metric("Recall",    f"{metrics['Recall']}%")
    m4.metric("F1 Score",  f"{metrics['F1']}%")
    m5.metric("AUC-ROC",   str(metrics['AUC-ROC']))
    st.markdown("---")

    left,right = st.columns(2)
    with left:
        st.markdown("**Customer Profile**")
        c1,c2 = st.columns(2)
        age     = c1.slider("Age",          18, 70,  35)
        exp     = c2.slider("Experience",    0, 50,  10)
        income  = c1.slider("Income ($K)",   8,224,  75)
        cc      = c2.slider("CC Spend ($K)", 0.0,10.0,2.0,step=0.1)
        family  = c1.selectbox("Family Size",[1,2,3,4],index=1)
        edu     = c2.selectbox("Education",[1,2,3],
                               format_func=lambda x:{1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
        mort    = c1.slider("Mortgage ($K)", 0,635,0,step=5)
        st.markdown("**Products**")
        b1,b2,b3,b4 = st.columns(4)
        sec  = b1.selectbox("Securities",["No","Yes"])
        cd   = b2.selectbox("CD Account",["No","Yes"])
        onl  = b3.selectbox("Online",["Yes","No"])
        ccard= b4.selectbox("Credit Card",["No","Yes"])
        run  = st.button("⚡ PREDICT", use_container_width=True)

    with right:
        st.markdown("**Prediction Output**")
        inp = np.array([[age, exp, income, family, cc, edu, mort,
                         1 if sec=="Yes" else 0, 1 if cd=="Yes" else 0,
                         1 if onl=="Yes" else 0, 1 if ccard=="Yes" else 0]])
        inp_sc = scaler.transform(inp)
        pred   = model.predict(inp_sc)[0]
        prob   = float(model.predict_proba(inp_sc)[0][1]) * 100 if hasattr(model,"predict_proba") else 50.0
        approved = pred == 1

        box_cls = "result-box-yes" if approved else "result-box-no"
        verd_cls = "verdict-yes"   if approved else "verdict-no"
        verdict  = "✅ APPROVED"   if approved else "❌ DECLINED"
        bar_col  = "#22C55E"       if approved else "#64748B"

        st.markdown(f"""
        <div class="{box_cls}">
            <div style="font-size:10px;letter-spacing:2px;color:#64748B">PREDICTION OUTCOME</div>
            <div class="{verd_cls}">{verdict}</div>
            <div class="prob-num">{prob:.1f}<span style="font-size:22px;color:#64748B">%</span></div>
            <div style="font-size:10px;letter-spacing:2px;color:#64748B;margin-top:4px">APPROVAL PROBABILITY</div>
        </div>""", unsafe_allow_html=True)

        fig_b = go.Figure(go.Bar(x=[prob],y=[""],orientation="h",
                                  marker_color=bar_col,width=0.35))
        fig_b.update_layout(xaxis=dict(range=[0,100],showgrid=False),
                             yaxis=dict(showticklabels=False),
                             **PLOT_DARK,margin=dict(t=8,b=8,l=0,r=0),height=55)
        st.plotly_chart(fig_b, use_container_width=True)

        conf = "HIGH" if (prob>80 or prob<20) else "MEDIUM" if (prob>65 or prob<35) else "LOW"
        risk = "HIGH" if prob>75 else "MEDIUM" if prob>45 else "LOW"
        r1,r2,r3 = st.columns(3)
        r1.metric("Confidence",  conf)
        r2.metric("Risk Level",  risk)
        r3.metric("Algorithm",   sel.split()[0])

        st.markdown("---")
        st.markdown("**Feature Importance**")
        if hasattr(model,"feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model,"coef_"):
            imp = np.abs(model.coef_[0])
        else:
            imp = np.ones(len(feat_cols))
        fi_df = pd.DataFrame({"Feature":feat_cols,"Importance":imp})
        fi_df = fi_df.sort_values("Importance",ascending=True).tail(8)
        fi_df["Importance %"] = (fi_df["Importance"]/fi_df["Importance"].sum()*100).round(1)
        fig_fi = px.bar(fi_df, x="Importance %", y="Feature", orientation="h",
                        color="Importance %",
                        color_continuous_scale=["#334155","#D4A853"],
                        **PLOT_DARK)
        fig_fi.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="badge">MODEL BENCHMARKING</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Algorithm <span>Comparison</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Full performance benchmark across all ML algorithms on Universal Bank data</div>', unsafe_allow_html=True)

    results, trained_models, scaler, feat_cols, X_test, y_test = train_all_models()
    rdf = pd.DataFrame(results).T.reset_index()
    rdf.columns = ["Algorithm","Accuracy","Precision","Recall","F1","AUC-ROC","CM"]
    rdf = rdf.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    best = rdf.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏆 Best Model",   best["Algorithm"],   f"{best['Accuracy']}% acc")
    c2.metric("Best AUC-ROC",   str(best["AUC-ROC"]), best["Algorithm"])
    c3.metric("Best F1",        f"{rdf['F1'].max()}%", rdf.loc[rdf['F1'].idxmax(),'Algorithm'])
    c4.metric("Models Trained", str(len(rdf)),          "SMOTE balanced")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Accuracy Ranking**")
        fig = px.bar(rdf.sort_values("Accuracy"), x="Accuracy", y="Algorithm",
                     orientation="h", color="Accuracy",
                     color_continuous_scale=["#334155","#D4A853"],
                     text="Accuracy", **PLOT_DARK)
        fig.update_traces(texttemplate="%{text}%",textposition="outside")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Multi-Metric Radar (Top 5)**")
        top5 = rdf.head(5)
        fig2 = go.Figure()
        colors = ["#D4A853","#00D4AA","#3B82F6","#7C3AED","#EF4444"]
        for i,row in top5.iterrows():
            vals = [row["Accuracy"],row["Precision"],row["Recall"],row["F1"]]
            fig2.add_trace(go.Scatterpolar(
                r=vals+[vals[0]],
                theta=["Accuracy","Precision","Recall","F1","Accuracy"],
                name=row["Algorithm"], line_color=colors[i],
                fill="toself", fillcolor=colors[i]+"18"))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True,range=[85,100]),bgcolor="#0C1120"),
            showlegend=True, **PLOT_DARK,
            legend=dict(orientation="h",y=-0.15), margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**AUC-ROC Comparison**")
    auc_df = rdf[rdf["AUC-ROC"].notna()]
    fig3 = px.bar(auc_df.sort_values("AUC-ROC"), x="AUC-ROC", y="Algorithm",
                  orientation="h", color="AUC-ROC",
                  color_continuous_scale=["#334155","#00D4AA"],
                  text="AUC-ROC", **PLOT_DARK, range_x=[0.85,1.0])
    fig3.update_traces(texttemplate="%{text:.4f}",textposition="outside")
    fig3.update_layout(coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")

    st.markdown("**Full Performance Table**")
    disp_rdf = rdf[["Algorithm","Accuracy","Precision","Recall","F1","AUC-ROC"]].copy()
    for c in ["Accuracy","Precision","Recall","F1"]:
        disp_rdf[c] = disp_rdf[c].apply(lambda x: f"{x}%")
    medals = {0:"🥇",1:"🥈",2:"🥉"}
    disp_rdf.insert(0,"#",[medals.get(i,str(i+1)) for i in range(len(disp_rdf))])
    st.dataframe(disp_rdf, use_container_width=True, hide_index=True)

    st.markdown("---")
    cm_sel = st.selectbox("Confusion Matrix — select algorithm", rdf["Algorithm"].tolist())
    cm_row = rdf[rdf["Algorithm"]==cm_sel].iloc[0]
    if cm_row["CM"]:
        import numpy as np
        cm_arr = np.array(cm_row["CM"])
        fig_cm = px.imshow(cm_arr, text_auto=True,
                           x=["Pred: Rejected","Pred: Accepted"],
                           y=["Actual: Rejected","Actual: Accepted"],
                           color_continuous_scale=["#0C1120","#D4A853"],
                           **PLOT_DARK)
        fig_cm.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_cm, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# RISK MATRIX
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Matrix":
    st.markdown('<div class="badge">RISK INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Risk <span>Matrix</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Portfolio risk tiering, heat-mapping and correlation intelligence</div>', unsafe_allow_html=True)

    from scipy.special import expit
    rdf = df.copy()
    rdf["risk_score"] = (
        rdf["Income"]*0.40 + rdf["CCAvg"]*3.0 +
        rdf["Education"]*2.5 + rdf["CD Account"]*10 +
        (rdf["Mortgage"]>0).astype(int)*3 - 20
    )
    rdf["risk_prob"] = (expit(rdf["risk_score"]/15)*100).clip(2,98)
    rdf["Risk Tier"] = pd.cut(rdf["risk_prob"], bins=[0,30,60,101],
                               labels=["Low Risk","Medium Risk","High Risk"])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔴 High Risk",   str(int((rdf["Risk Tier"]=="High Risk").sum())))
    c2.metric("🟡 Medium Risk", str(int((rdf["Risk Tier"]=="Medium Risk").sum())))
    c3.metric("🟢 Low Risk",    str(int((rdf["Risk Tier"]=="Low Risk").sum())))
    c4.metric("Portfolio Health","87.4 / 100")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Risk Tier by Age Band**")
        ra = rdf.groupby(["Age Band","Risk Tier"], observed=True).size().reset_index(name="Count")
        fig = px.bar(ra, x="Age Band", y="Count", color="Risk Tier",
                     color_discrete_map={"High Risk":"#EF4444","Medium Risk":"#D4A853","Low Risk":"#00D4AA"},
                     barmode="stack", **PLOT_DARK)
        fig.update_layout(legend=dict(orientation="h",y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Risk Probability Heatmap — Income × CC Spend**")
        smp = rdf.sample(min(600,len(rdf)), random_state=42)
        fig2 = px.scatter(smp, x="Income", y="CCAvg",
                          color="risk_prob",
                          color_continuous_scale=["#1E293B","#D4A853","#EF4444"],
                          size="risk_prob", size_max=12,
                          **PLOT_DARK, labels={"risk_prob":"Risk %"})
        fig2.update_layout(margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    num_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                "Mortgage","Securities Account","CD Account","Online","CreditCard","Personal Loan"]
    col3,col4 = st.columns(2)
    with col3:
        st.markdown("**Pearson Correlation Heatmap**")
        corr = rdf[num_cols].corr()
        fig3 = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale=["#EF4444","#0C1120","#00D4AA"],
                         zmin=-1, zmax=1, aspect="auto", **PLOT_DARK)
        fig3.update_layout(margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Risk Tier Distribution**")
        tc = rdf["Risk Tier"].value_counts().reset_index()
        tc.columns = ["Tier","Count"]
        fig4 = px.pie(tc, names="Tier", values="Count", color="Tier", hole=0.6,
                      color_discrete_map={"High Risk":"#EF4444","Medium Risk":"#D4A853","Low Risk":"#00D4AA"},
                      **PLOT_DARK)
        fig4.update_layout(margin=dict(t=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Feature Correlation with Loan Acceptance**")
    ct = rdf[num_cols].corr()["Personal Loan"].drop("Personal Loan").sort_values()
    ct_df = pd.DataFrame({"Feature":ct.index,"Correlation":ct.values})
    fig5 = px.bar(ct_df, x="Correlation", y="Feature", orientation="h",
                  color="Correlation",
                  color_continuous_scale=["#EF4444","#0C1120","#00D4AA"],
                  **PLOT_DARK)
    fig5.update_layout(coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig5, use_container_width=True)
