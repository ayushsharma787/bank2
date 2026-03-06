# 🏦 Universal Bank Intelligence Platform

> Enterprise ML-powered banking analytics dashboard — **flat file structure, Streamlit Cloud ready**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📁 Files to Upload (FLAT — no subfolders needed)

```
app.py                    ← Main Streamlit app (everything in one file)
UniversalBank.csv         ← Dataset (MUST be in same folder as app.py)
requirements.txt          ← Python dependencies
.streamlit/config.toml    ← Dark gold theme
README.md
```

> ⚠️ **Important**: `UniversalBank.csv` must be in the **same folder** as `app.py`

---

## 🚀 Deploy to Streamlit Cloud (Step-by-Step)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Universal Bank Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/universalbank-dashboard.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Set:
   - **Repository**: `YOUR_USERNAME/universalbank-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy** ✅

---

## 🏠 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 8 Dashboard Sections

| # | Section | Description |
|---|---|---|
| 1 | 🏠 Home | Overview cards, navigation guide |
| 2 | 📊 Overview | KPIs, income charts, digital adoption radar |
| 3 | 👥 Customer Analytics | Filtered demographics, scatter plots, data explorer |
| 4 | 💳 Loan Analytics | Acceptance rates by income, education, CD, family |
| 5 | 🔬 Diagnostic Analysis | Correlations, outliers, t-tests, chi-square, segment profiling |
| 6 | 📉 Predictive Analytics | Segment scoring, opportunity pipeline, projections |
| 7 | 🤖 AI Loan Predictor | Real-time prediction with 11 algorithms + feature importance |
| 8 | 📈 Model Comparison | Full benchmark table, radar chart, confusion matrices |
| 9 | ⚠️ Risk Matrix | Risk tiers, heatmaps, Pearson correlations |

---

## 🧠 11 ML Algorithms

Gradient Boosting · XGBoost · LightGBM · Random Forest · Extra Trees · AdaBoost · Decision Tree · Logistic Regression · KNN · SVM · Neural Network (MLP)

> All trained with **SMOTE** to handle 9.6% class imbalance.
