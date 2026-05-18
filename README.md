# 🎯 Customer Lifetime Value (CLV) Prediction
### Auto Insurance · XGBoost Regressor · R² 0.681 · MAPE 14.93% · Gap R² 0.029

> **Predict the total revenue a customer will generate — from day one — to optimize retention strategy, marketing budget allocation, and long-term profitability.**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Key Findings (AHA Moments)](#key-findings)
- [Methodology](#methodology)
- [Model Journey & Results](#model-journey)
- [Feature Importance](#feature-importance)
- [Business Impact](#business-impact)
- [Customer Segmentation](#customer-segmentation)
- [Model Limitations](#model-limitations)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## 🔍 Project Overview

This capstone project builds a **regression machine learning model** that predicts Customer Lifetime Value (CLV) for auto insurance policyholders. The final model — **XGBoost (Tuned)** — was selected after a systematic 3-algorithm comparison based on both predictive accuracy and overfitting control.

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| R² Score | ≥ 0.60 | **0.681** | ✅ Achieved |
| MAPE | ≤ 20% | **14.93%** | ✅ Achieved |
| Gap R² | < 0.10 | **0.029** | ✅ No overfitting |

---

## 💼 Business Problem

An auto insurance company has **no systematic way to predict a customer's CLV at onboarding**. Without this, retention budgets are allocated uniformly — spending the same on a $2,000 customer as on an $83,000 customer.

**Stakeholders:** Head of Marketing · CRM Team · CFO

**Why it matters:**
- Top 20% of customers contribute **46.4%** of total company CLV
- Mean CLV = **$8,030** but range spans **$1,898 – $83,325** (44× gap)
- Acquiring a new customer costs **~5× more** than retaining one
- Without targeting, **80% of retention effort** protects only **54% of value**

**Goal:** Predict CLV at onboarding → segment into 3 tiers → allocate resources with precision.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | US Auto Insurance Company (CRM) |
| Rows | 5,669 customers |
| Columns | 11 features + 1 target |
| Missing Values | 0 |
| Duplicates | 0 |
| Target | `Customer Lifetime Value` (continuous, USD) |

**Feature Overview:**

| Feature | Type | Business Relevance |
|---------|------|--------------------|
| `Vehicle Class` | str | Luxury vehicle = CLV 2.67× average |
| `Coverage` | str | Premium coverage = CLV 1.56× Basic |
| `Renew Offer Type` | str | Influences renewal decisions |
| `EmploymentStatus` | str | Income stability signal |
| `Marital Status` | str | Insurance consumption pattern |
| `Education` | str | Coverage awareness |
| `Number of Policies` | float | **Predictor #1** — 2 policies = CLV 2× |
| `Monthly Premium Auto` | float | **Predictor #2** — direct revenue source |
| `Total Claim Amount` | float | Profitability proxy |
| `Income` | float | Financial capacity for upgrades |

**3 Engineered Features:**

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| `premium_to_income` | premium / income | Low ratio = upgrade candidate |
| `claim_ratio` | total_claim / (premium × 12) | High ratio = less profitable |
| `high_value_vehicle` | binary flag (Luxury/Sports) | From AHA insight: 2.67× CLV |

---

## 💡 Key Findings

### AHA Moment #1 — Pareto Effect
> Top 20% of customers → **46.4% of total CLV**. Uniform retention strategy means 80% of spend protects only 54% of value.

### AHA Moment #2 — Luxury Vehicle Premium
> Luxury SUV customers: mean CLV **$18,990** vs **$7,120** for standard cars.  
> **2.67× gap** — vehicle class is the most visible high-value segment signal.

### AHA Moment #3 — The 2-Policy Sweet Spot
> Cross-selling from 1 → 2 policies nearly **doubles CLV**.  
> Beyond 2 policies the curve plateaus — making **1→2 the highest-ROI action** in the entire dataset.

---

## 🔬 Methodology

```
Raw Data (5,669 rows)
        ↓
Data Cleaning (0 missing, 0 duplicates, outliers retained)
        ↓
Feature Engineering (3 new features: premium_to_income, claim_ratio, high_value_vehicle)
        ↓
Feature Selection (correlation analysis, all 13 features retained with business justification)
        ↓
Train/Test Split (80/20, random_state=42)
        ↓
Baseline: Linear Regression R²=0.17 ❌
        ↓
RF Baseline: Gap R²=0.227 ❌ Overfit
        ↓
RF Re-Tuned (RandomizedSearchCV): Gap R²=0.132 ⚠️
        ↓
XGBoost Tuned (RandomizedSearchCV 60 iter): Gap R²=0.029 ✅ FINAL MODEL
        ↓
Customer Segmentation → 3 Tiers → Business Recommendations
```

---

## 🏆 Model Journey & Results

| Step | Model | Train R² | Test R² | Gap R² | Status |
|------|-------|----------|---------|--------|--------|
| 1 | Linear Regression | 0.18 | 0.17 | 0.01 | ❌ Too simple |
| 2 | RF Baseline | 0.888 | 0.661 | 0.227 | ❌ Overfit |
| 3 | RF Re-Tuned | 0.834 | 0.702 | 0.132 | ⚠️ Still overfit |
| 4 | **XGBoost Tuned** | **0.710** | **0.681** | **0.029** | ✅ **WINNER** |
| 5 | LightGBM Tuned | 0.725 | 0.680 | 0.045 | ✅ Runner-up |

**Why XGBoost wins over RF Re-Tuned (higher raw Test R²)?**  
RF Re-Tuned's Test R² = 0.702 vs XGBoost's 0.681 — a 2.1pp difference. But the overfitting gap tells a different story: RF 0.132 vs XGBoost **0.029**. In production deployment, a reliable model (XGBoost) is worth more than a slightly more accurate but unstable one. Composite score: XGBoost **0.797** vs RF 0.769.

> **Composite Score** = Test R² × 0.6 + (1 − Gap R²) × 0.4  
> *Primary objective (accuracy) weighted 60%, deployment constraint (reliability) weighted 40%.*

**XGBoost Best Parameters (RandomizedSearchCV, 60 iterations, 5-Fold CV):**
```python
{
  'n_estimators': 300,
  'max_depth': 5,
  'learning_rate': 0.05,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'min_child_weight': 5,
  'reg_alpha': 0.1,
  'reg_lambda': 2
}
```

---

## 📈 Feature Importance

```
Num. Policies        ████████████████████████████████████  47.0% ← #1
Monthly Premium      ████████████████████                  27.0% ← #2
Coverage             █████                                  7.0%
Vehicle Class        ████                                   5.5%
premium_to_income    ████ [ENG]                             5.6%
Total Claim Amount   ███                                    4.8%
claim_ratio          ███ [ENG]                              4.4%
Income               ██                                     3.5%
high_value_vehicle   ██ [ENG]                               3.1%
Renew Offer Type     █                                      2.2%
EmploymentStatus     █                                      1.9%
Marital Status       █                                      1.8%
Education            ▌                                      1.0%
```

**Top 2 features = 74% of all predictions.**  
All 3 engineered features appear in top 9 — feature engineering paid off.

**Business Actions from Feature Importance:**
1. **Cross-sell 1→2 policies** (47%) — highest ROI action for Medium CLV customers
2. **Push coverage upgrades** (27%) — every premium increase directly lifts CLV
3. **Target low premium_to_income** (5.6%) — can afford more, not paying it
4. **Flag high claim_ratio** (4.4%) — reprice at renewal

---

## 💰 Business Impact

**Simulation: Same $11,400 budget, 228 customers targeted (top 20%)**

| Strategy | Mean CLV Targeted | Total CLV Captured |
|----------|------------------|--------------------|
| Without Model (random 20%) | $8,023 | $1,829,000 |
| **With Model (top 20% predicted)** | **$17,368** | **$3,960,000** |
| **Lift** | **2.18×** | **+$2,131,000** |

> *Budget assumption: $50/customer outreach cost (US insurance industry avg: $30–80/customer, McKinsey Insurance Report 2022)*

---

## 👥 Customer Segmentation — 3 Tiers

| Tier | CLV Range | % Customers | % Total CLV | Strategy | Priority |
|------|-----------|-------------|-------------|----------|----------|
| **HIGH** | >$7,950 | ~34% | 55% | VIP Retention | 🔴 Highest |
| **MEDIUM** | $4,715–$7,950 | ~33% | 29% | Upsell & Cross-sell | 🟡 Medium |
| **LOW** | <$4,715 | ~33% | 16% | Automation | 🟢 Minimal |

**HIGH CLV:** Dedicated account manager · Early renewal 60 days out · Exclusive loyalty program  
**MEDIUM CLV:** Cross-sell 1→2 policies (10–15% discount) · Coverage upgrade campaign  
**LOW CLV:** Self-service digital · Automated renewal reminder · Chatbot for claims

---

## ⚠️ Model Limitations

**Model CAN be trusted for:**
- Personal auto insurance customers (not corporate/fleet)
- Customer profiles within training data distribution
- Vehicle classes and coverage types present in training data
- Stable macroeconomic & regulatory conditions

**Use with caution when:**
- **Income = $0** (~20% of dataset) — high uncertainty for this segment
- **Ultra-high CLV > $30,000** — model under-predicts; manual validation recommended
- **Corporate / fleet customers** — very different profile from training distribution
- **Regulatory changes** that fundamentally alter premium or claim structures
- **No tenure data** — relationship duration is an unmodeled CLV driver

**Future Improvements:**
1. Add `tenure` feature — most critical missing CLV driver
2. Build separate model for ultra-high CLV (>$30K) segment
3. Retrain every 6 months or after significant market changes
4. Add SHAP values for individual prediction explainability
5. Explore ensemble: XGBoost + Neural Network for VIP segment

---

## 🚀 How to Run

```bash
# 1. Clone repository
git clone https://github.com/riskyadipratama/clv-prediction.git
cd clv-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook Capstone_Module3_CLV_FIXED.ipynb
```

**requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

**Quick Predict:**
```python
import pickle
import pandas as pd

# Load model
with open('model/model_xgboost_clv.pkl', 'rb') as f:
    payload = pickle.load(f)

model = payload['model']
feature_cols = payload['feature_cols']

# Predict CLV for new customer
new_customer = pd.DataFrame([{
    'Vehicle Class': 2,        # encoded: Luxury SUV
    'Coverage': 2,             # encoded: Premium
    'Renew Offer Type': 0,     # encoded: Offer1
    'EmploymentStatus': 1,     # encoded: Employed
    'Marital Status': 2,       # encoded: Single
    'Education': 2,            # encoded: Bachelor
    'Number of Policies': 2,
    'Monthly Premium Auto': 95,
    'Total Claim Amount': 500,
    'Income': 65000,
    'premium_to_income': 95/65000,
    'claim_ratio': 500/(95*12),
    'high_value_vehicle': 1
}])

predicted_clv = model.predict(new_customer)[0]
tier = 'High' if predicted_clv > 7950 else 'Medium' if predicted_clv > 4715 else 'Low'
print(f"Predicted CLV: ${predicted_clv:,.0f} → {tier} CLV Tier")
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9 |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model Persistence | pickle |
| Notebook | Jupyter |
| Presentation | PowerPoint (pptxgenjs) |

---

## 👤 Author

**Risky Adipratama**  
Capstone Project — Module 3  
JCDSOHAM · Data Science Bootcamp

---

*If you find this project useful, feel free to ⭐ the repository!*
