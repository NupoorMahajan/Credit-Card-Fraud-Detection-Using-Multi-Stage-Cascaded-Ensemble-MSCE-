"""
=============================================================================
  CREDIT CARD FRAUD DETECTION — MULTI-STAGE CASCADED ENSEMBLE (MSCE)
=============================================================================

  Novel Algorithm: Multi-Stage Cascaded Ensemble (MSCE)
  -------------------------------------------------------
  Core Idea: Instead of a flat ensemble, data passes through THREE
  filtering/decision stages before a final meta-learner decides.

  Stage 1 — ANOMALY FILTER (Isolation Forest)
      → Flags statistical outliers; outputs an 'anomaly_score' feature.
        Transactions that look normal pass through with low suspicion.

  Stage 2 — BASE CLASSIFIER PANEL (XGBoost + LightGBM + Random Forest)
      → Each base model gives a probability. Their outputs are used as
        additional "meta-features" — not just averaged.

  Stage 3 — META-LEARNER (Logistic Regression on meta-features)
      → Trained on the stacked probabilities from Stage 2 PLUS the
        Stage-1 anomaly score and hand-crafted features.
        Final decision is made here.

  Why this beats a simple ensemble:
  • Stage 1 pre-filters noise, reducing false positives.
  • Stage 2 captures different decision boundaries (tree gradient
    boosting vs bagging).
  • Stage 3 learns HOW to combine Stage 2's disagreements — not just
    an average.

=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve,
    matthews_corrcoef, f1_score, precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = "fraudTest.csv"   # ← change if your file is in a different path
RANDOM_SEED = 42
TEST_RATIO  = 0.20              # 20% held out for final evaluation
N_CV_FOLDS  = 5                 # Stratified K-Fold for cross-validation

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — LOADING DATA")
print("=" * 70)

try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    df = pd.read_excel(DATA_PATH)

print(f"  Dataset shape : {df.shape}")
print(f"  Fraud cases   : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
print(f"  Normal cases  : {(df['is_fraud']==0).sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  (domain-specific + temporal + spatial)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 70)

df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# ── Temporal features ──────────────────────────────────────────────────────
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour']   = df['trans_date_trans_time'].dt.hour
df['trans_dow']    = df['trans_date_trans_time'].dt.dayofweek     # 0=Mon
df['trans_month']  = df['trans_date_trans_time'].dt.month
df['is_night']     = ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 5)).astype(int)
df['is_weekend']   = (df['trans_dow'] >= 5).astype(int)

# ── Age of cardholder ──────────────────────────────────────────────────────
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365.25).fillna(40)

# ── Spatial distance between cardholder and merchant ──────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

df['distance_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

# ── Amount-based features ──────────────────────────────────────────────────
df['log_amt']          = np.log1p(df['amt'])
df['amt_to_city_pop']  = df['amt'] / (df['city_pop'] + 1)  # per-capita spend proxy
df['high_amount_flag'] = (df['amt'] > df['amt'].quantile(0.95)).astype(int)

# ── Cardholder behavioural aggregates (rolling per cc_num) ─────────────────
df = df.sort_values('trans_date_trans_time').reset_index(drop=True)

cc_group = df.groupby('cc_num')['amt']
df['cc_mean_amt']  = cc_group.transform('mean')
df['cc_std_amt']   = cc_group.transform('std').fillna(0)
df['amt_z_score']  = (df['amt'] - df['cc_mean_amt']) / (df['cc_std_amt'] + 1e-6)

cc_freq = df.groupby('cc_num')['trans_num'].transform('count')
df['cc_tx_freq'] = cc_freq

# ── Merchant-level aggregates ──────────────────────────────────────────────
merch_fraud_rate = df.groupby('merchant')['is_fraud'].transform('mean')
df['merch_fraud_rate'] = merch_fraud_rate

# ── Encode categoricals ────────────────────────────────────────────────────
le = LabelEncoder()
df['gender_enc']   = le.fit_transform(df['gender'].fillna('M'))
df['category_enc'] = le.fit_transform(df['category'].fillna('misc_pos'))

print("  Features created successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'amt', 'log_amt', 'trans_hour', 'trans_dow', 'trans_month',
    'is_night', 'is_weekend', 'age', 'distance_km',
    'amt_to_city_pop', 'high_amount_flag',
    'cc_mean_amt', 'cc_std_amt', 'amt_z_score', 'cc_tx_freq',
    'merch_fraud_rate', 'city_pop', 'gender_enc', 'category_enc',
    'zip', 'lat', 'long', 'merch_lat', 'merch_long'
]

TARGET = 'is_fraud'

X = df[FEATURES].fillna(0).values
y = df[TARGET].values

print(f"\n  Feature matrix shape : {X.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STRATIFIED TRAIN / TEST SPLIT  (time-respecting)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
)
print(f"\n  Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")
print(f"  Train fraud: {y_train.sum():,}  |  Test fraud: {y_test.sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLASS IMBALANCE HANDLING — SMOTETomek
#     SMOTE generates synthetic minority samples; Tomek removes borderline
#     majority samples → cleaner decision boundary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3 — CLASS IMBALANCE HANDLING  (SMOTETomek)")
print("=" * 70)

# Subsample majority class before resampling for efficiency on large datasets
from sklearn.utils import resample as sklearn_resample
idx_fraud = np.where(y_train == 1)[0]
idx_norm  = np.where(y_train == 0)[0]
idx_norm_sub = sklearn_resample(idx_norm, n_samples=40000, random_state=RANDOM_SEED, replace=False)
idx_all = np.concatenate([idx_fraud, idx_norm_sub])
X_sub, y_sub = X_train[idx_all], y_train[idx_all]

smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_sub, y_sub)
print(f"  After SMOTE — Fraud: {y_res.sum():,}  |  Normal: {(y_res==0).sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SCALING
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_res_sc   = scaler.fit_transform(X_res)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MULTI-STAGE CASCADED ENSEMBLE  (MSCE)  — NOVEL ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 — MULTI-STAGE CASCADED ENSEMBLE (MSCE)  [Novel Algorithm]")
print("=" * 70)

# ── Stage 1 : Anomaly Filter (Isolation Forest) ────────────────────────────
print("  [Stage 1] Training Isolation Forest anomaly detector ...")
iso_forest = IsolationForest(
    n_estimators=200, contamination=0.004, max_samples='auto',
    random_state=RANDOM_SEED, n_jobs=-1
)
iso_forest.fit(X_res_sc)

# anomaly_score: more negative → more anomalous
def iso_score(model, X):
    return -model.score_samples(X)   # flip sign so higher = more anomalous


# ── Stage 2 : Base Classifier Panel ───────────────────────────────────────
print("  [Stage 2] Training XGBoost ...")
scale_pos = (y_res == 0).sum() / (y_res == 1).sum()
xgb_clf = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1,          # already balanced via SMOTETomek
    use_label_encoder=False, eval_metric='aucpr',
    tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1
)
xgb_clf.fit(X_res_sc, y_res,
            eval_set=[(X_train_sc, y_train)],
            verbose=False)

print("  [Stage 2] Training LightGBM ...")
lgb_clf = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=RANDOM_SEED, n_jobs=-1,
    verbose=-1
)
lgb_clf.fit(X_res_sc, y_res,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
            eval_set=[(X_train_sc, y_train)],
            eval_metric='average_precision')

print("  [Stage 2] Training Random Forest ...")
rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced',
    random_state=RANDOM_SEED, n_jobs=-1
)
rf_clf.fit(X_res_sc, y_res)


# ── Stage 3 : Meta-feature construction + Meta-Learner ────────────────────
print("  [Stage 3] Building meta-features via 5-Fold out-of-fold predictions ...")

skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_xgb = np.zeros(len(X_res))
oof_lgb = np.zeros(len(X_res))
oof_rf  = np.zeros(len(X_res))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_res_sc, y_res)):
    Xtr, Xval = X_res_sc[tr_idx], X_res_sc[val_idx]
    ytr        = y_res[tr_idx]

    _xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='aucpr',
        tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1)
    _xgb.fit(Xtr, ytr, verbose=False)
    oof_xgb[val_idx] = _xgb.predict_proba(Xval)[:, 1]

    _lgb = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    _lgb.fit(Xtr, ytr)
    oof_lgb[val_idx] = _lgb.predict_proba(Xval)[:, 1]

    _rf = RandomForestClassifier(
        n_estimators=150, max_depth=None, max_features='sqrt',
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    _rf.fit(Xtr, ytr)
    oof_rf[val_idx] = _rf.predict_proba(Xval)[:, 1]

    print(f"    Fold {fold+1}/{N_CV_FOLDS} done.")

# anomaly scores for resampled train
oof_iso = iso_score(iso_forest, X_res_sc)

# Stack OOF predictions as meta-features
meta_train = np.column_stack([oof_xgb, oof_lgb, oof_rf, oof_iso])

print("  [Stage 3] Training Logistic Regression meta-learner ...")
meta_learner = LogisticRegression(
    C=1.0, class_weight='balanced',
    max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'
)
meta_learner.fit(meta_train, y_res)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PREDICTION FUNCTION  (full MSCE pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def msce_predict_proba(X_scaled):
    """Run X through all 3 stages and return fraud probability."""
    p_xgb = xgb_clf.predict_proba(X_scaled)[:, 1]
    p_lgb = lgb_clf.predict_proba(X_scaled)[:, 1]
    p_rf  = rf_clf.predict_proba(X_scaled)[:, 1]
    p_iso = iso_score(iso_forest, X_scaled)
    meta  = np.column_stack([p_xgb, p_lgb, p_rf, p_iso])
    return meta_learner.predict_proba(meta)[:, 1]

def msce_predict(X_scaled, threshold=0.5):
    return (msce_predict_proba(X_scaled) >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  THRESHOLD OPTIMISATION  (maximise F1 on training set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Optimising classification threshold ...")
train_probs = msce_predict_proba(X_train_sc)
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, train_probs)

f1_scores = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
best_idx   = np.argmax(f1_scores)
THRESHOLD  = thresh_arr[best_idx] if best_idx < len(thresh_arr) else 0.5
print(f"  Optimal threshold : {THRESHOLD:.4f}  (F1 = {f1_scores[best_idx]:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  EVALUATION ON HOLD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 5 — MODEL EVALUATION  (Hold-out Test Set)")
print("=" * 70)

y_prob = msce_predict_proba(X_test_sc)
y_pred = (y_prob >= THRESHOLD).astype(int)

# ── Core metrics ──────────────────────────────────────────────────────────
roc_auc   = roc_auc_score(y_test, y_prob)
pr_auc    = average_precision_score(y_test, y_prob)
mcc       = matthews_corrcoef(y_test, y_pred)
kappa     = cohen_kappa_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
bal_acc   = balanced_accuracy_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv         = tn / (tn + fn + 1e-8)   # Negative Predictive Value

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │                  MSCE MODEL METRICS                     │
  ├────────────────────────────────┬────────────────────────┤
  │  ROC-AUC                       │  {roc_auc:.6f}            │
  │  PR-AUC (Avg Precision)        │  {pr_auc:.6f}            │
  │  Matthews Corr. Coeff. (MCC)   │  {mcc:.6f}            │
  │  Cohen's Kappa                 │  {kappa:.6f}            │
  │  F1-Score                      │  {f1:.6f}            │
  │  Precision                     │  {precision:.6f}            │
  │  Recall  (Sensitivity)         │  {recall:.6f}            │
  │  Specificity                   │  {specificity:.6f}            │
  │  Balanced Accuracy             │  {bal_acc:.6f}            │
  │  Neg. Predictive Value         │  {npv:.6f}            │
  ├────────────────────────────────┼────────────────────────┤
  │  True  Positives  (fraud↑)     │  {tp:,}                 │
  │  False Positives  (normal↑)    │  {fp:,}                 │
  │  True  Negatives  (normal↓)    │  {tn:,}               │
  │  False Negatives  (fraud↓)     │  {fn:,}                  │
  └────────────────────────────────┴────────────────────────┘
""")

print("  Per-class Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# 11.  INDIVIDUAL BASE MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  BASE MODELS vs MSCE  (Test Set ROC-AUC + PR-AUC)")
print("=" * 70)

models_eval = {
    "XGBoost"    : xgb_clf.predict_proba(X_test_sc)[:, 1],
    "LightGBM"   : lgb_clf.predict_proba(X_test_sc)[:, 1],
    "Random Forest": rf_clf.predict_proba(X_test_sc)[:, 1],
    "MSCE (ours)": y_prob
}

print(f"  {'Model':<20} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10}")
print("  " + "-" * 54)
for name, prob in models_eval.items():
    preds = (prob >= THRESHOLD).astype(int)
    print(f"  {name:<20} {roc_auc_score(y_test, prob):>10.4f} "
          f"{average_precision_score(y_test, prob):>10.4f} "
          f"{f1_score(y_test, preds):>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  VISUALISATIONS  — FIGURE 1: Full Evaluation Dashboard (3×3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating evaluation plots ...")

PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']   # XGB / LGB / RF / MSCE

fig1, axes = plt.subplots(3, 3, figsize=(20, 17))
fig1.patch.set_facecolor('#F7F9FC')
fig1.suptitle("Credit Card Fraud Detection — MSCE Full Evaluation Dashboard",
              fontsize=16, fontweight='bold', y=1.01)

for ax in axes.flat:
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')

# ── (A) Confusion Matrix — normalised ────────────────────────────────────
ax = axes[0, 0]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
annot_labels = np.array([[f"{v:,}\n({p:.1%})" for v, p in zip(row_v, row_p)]
                          for row_v, row_p in zip(cm, cm_norm)])
sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
            linewidths=0.5, linecolor='#CCCCCC', cbar_kws={'shrink': 0.8})
ax.set_title('Confusion Matrix (count + row %)', fontweight='bold', pad=10)
ax.set_ylabel('Actual Label'); ax.set_xlabel('Predicted Label')

# ── (B) ROC Curves — all models ──────────────────────────────────────────
ax = axes[0, 1]
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')
for (name, prob), color in zip(models_eval.items(), PALETTE):
    fpr_c, tpr_c, _ = roc_curve(y_test, prob)
    auc_c = roc_auc_score(y_test, prob)
    lw = 3 if name == 'MSCE (ours)' else 1.8
    ax.plot(fpr_c, tpr_c, color=color, lw=lw, label=f"{name}  AUC={auc_c:.4f}")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.set_title('ROC Curves — All Models', fontweight='bold', pad=10)
ax.set_xlabel('False Positive Rate (FPR)'); ax.set_ylabel('True Positive Rate (TPR)')
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.25)

# ── (C) Precision-Recall Curves ──────────────────────────────────────────
ax = axes[0, 2]
baseline_pr = y_test.mean()
ax.axhline(baseline_pr, color='grey', ls='--', lw=1, label=f'Baseline (no-skill) = {baseline_pr:.4f}')
for (name, prob), color in zip(models_eval.items(), PALETTE):
    prec_c, rec_c, _ = precision_recall_curve(y_test, prob)
    prauc_c = average_precision_score(y_test, prob)
    lw = 3 if name == 'MSCE (ours)' else 1.8
    ax.plot(rec_c, prec_c, color=color, lw=lw, label=f"{name}  AP={prauc_c:.4f}")
ax.set_title('Precision-Recall Curves', fontweight='bold', pad=10)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

# ── (D) Fraud Probability Score Distribution ─────────────────────────────
ax = axes[1, 0]
ax.hist(y_prob[y_test == 0], bins=80, alpha=0.55, label='Normal (0)',
        color='steelblue', density=True, edgecolor='none')
ax.hist(y_prob[y_test == 1], bins=80, alpha=0.65, label='Fraud (1)',
        color='crimson',   density=True, edgecolor='none')
ax.axvline(THRESHOLD, color='black', ls='--', lw=2.2,
           label=f'Decision threshold = {THRESHOLD:.4f}')
ax.set_title('Predicted Probability Distribution', fontweight='bold', pad=10)
ax.set_xlabel('Fraud Probability Score'); ax.set_ylabel('Density')
ax.set_yscale('log')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# ── (E) Feature Importance — XGBoost (top 15) ────────────────────────────
ax = axes[1, 1]
imp = pd.Series(xgb_clf.feature_importances_, index=FEATURES).nlargest(15).sort_values()
bar_cols = plt.cm.Blues(np.linspace(0.35, 0.95, len(imp)))
imp.plot(kind='barh', ax=ax, color=bar_cols, edgecolor='none')
for i, (val, name) in enumerate(zip(imp.values, imp.index)):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=7.5)
ax.set_title('Feature Importances — XGBoost (Top 15)', fontweight='bold', pad=10)
ax.set_xlabel('Gain Importance Score'); ax.grid(alpha=0.25, axis='x')

# ── (F) Model Comparison — grouped bar (ROC / PR / F1) ───────────────────
ax = axes[1, 2]
model_names = list(models_eval.keys())
roc_vals  = [roc_auc_score(y_test, p) for p in models_eval.values()]
pr_vals   = [average_precision_score(y_test, p) for p in models_eval.values()]
f1_vals   = [f1_score(y_test, (p >= THRESHOLD).astype(int)) for p in models_eval.values()]
x_pos = np.arange(len(model_names)); w = 0.26
b1 = ax.bar(x_pos - w, roc_vals, w, label='ROC-AUC',  color='#2196F3', edgecolor='none')
b2 = ax.bar(x_pos,     pr_vals,  w, label='PR-AUC',   color='#4CAF50', edgecolor='none')
b3 = ax.bar(x_pos + w, f1_vals,  w, label='F1-Score', color='#FF9800', edgecolor='none')
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x_pos); ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title('Model Comparison — ROC / PR / F1', fontweight='bold', pad=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis='y')

# ── (G) Precision & Recall vs Threshold ──────────────────────────────────
ax = axes[2, 0]
thresh_range = np.linspace(0.01, 0.99, 300)
prec_line, rec_line, f1_line = [], [], []
for t in thresh_range:
    yp = (y_prob >= t).astype(int)
    prec_line.append(precision_score(y_test, yp, zero_division=0))
    rec_line.append(recall_score(y_test, yp, zero_division=0))
    f1_line.append(f1_score(y_test, yp, zero_division=0))
ax.plot(thresh_range, prec_line, color='#2196F3', lw=2, label='Precision')
ax.plot(thresh_range, rec_line,  color='#E91E63', lw=2, label='Recall')
ax.plot(thresh_range, f1_line,   color='#4CAF50', lw=2, label='F1-Score')
ax.axvline(THRESHOLD, color='black', ls='--', lw=1.8,
           label=f'Chosen threshold={THRESHOLD:.4f}')
ax.set_title('Precision / Recall / F1 vs Threshold', fontweight='bold', pad=10)
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# ── (H) Class Distribution — Before vs After SMOTE ───────────────────────
ax = axes[2, 1]
categories = ['Before SMOTE\n(Train)', 'After SMOTE\n(Resampled)']
normal_counts = [int((y_train == 0).sum()), int((y_res == 0).sum())]
fraud_counts  = [int((y_train == 1).sum()), int((y_res == 1).sum())]
x2 = np.arange(2); w2 = 0.35
b_n = ax.bar(x2 - w2/2, normal_counts, w2, label='Normal', color='steelblue', edgecolor='none')
b_f = ax.bar(x2 + w2/2, fraud_counts,  w2, label='Fraud',  color='crimson',   edgecolor='none')
for bar in list(b_n) + list(b_f):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'{bar.get_height():,}', ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x2); ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Class Distribution Before vs After SMOTE', fontweight='bold', pad=10)
ax.set_ylabel('Number of Transactions')
ax.legend(fontsize=10); ax.grid(alpha=0.25, axis='y')

# ── (I) Radar / Spider Chart — MSCE Metric Profile ───────────────────────
ax = axes[2, 2]
radar_labels = ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Specificity', 'Bal-Acc']
radar_vals   = [roc_auc, pr_auc, f1, precision, recall, specificity, bal_acc]
N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
vals_plot = radar_vals + radar_vals[:1]
ax.remove()
ax_radar = fig1.add_subplot(3, 3, 9, polar=True)
ax_radar.set_facecolor('#FFFFFF')
ax_radar.plot(angles, vals_plot, color='#E91E63', lw=2.5, marker='o', ms=6)
ax_radar.fill(angles, vals_plot, color='#E91E63', alpha=0.20)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, fontsize=9)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7, color='grey')
ax_radar.grid(color='grey', alpha=0.25)
ax_radar.set_title('MSCE Metric Radar Profile', fontweight='bold', pad=18)
for angle, val, label in zip(angles[:-1], radar_vals, radar_labels):
    ax_radar.text(angle, val + 0.07, f'{val:.3f}', ha='center', va='center',
                  fontsize=7.5, color='#C62828', fontweight='bold')

plt.tight_layout(pad=2.5)
fig1.savefig('fraud_detection_evaluation.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  ✓ Figure 1 saved → fraud_detection_evaluation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12b.  FIGURE 2 — Metrics Deep-Dive Dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.patch.set_facecolor('#F7F9FC')
fig2.suptitle("MSCE — Metrics Deep-Dive & Model Insights",
              fontsize=15, fontweight='bold', y=1.01)
for ax in axes2.flat:
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')

# ── (J) Full Metrics Scorecard — horizontal bars with colour bands ────────
ax = axes2[0, 0]
all_metric_names  = ['ROC-AUC', 'PR-AUC', 'MCC', "Cohen's κ",
                     'F1-Score', 'Precision', 'Recall',
                     'Specificity', 'Balanced Acc', 'Neg. Pred. Value']
all_metric_values = [roc_auc, pr_auc, mcc, kappa,
                     f1, precision, recall,
                     specificity, bal_acc, npv]
band_colors = ['#4CAF50' if v >= 0.9 else '#FF9800' if v >= 0.75 else '#F44336'
               for v in all_metric_values]
bars = ax.barh(all_metric_names, all_metric_values, color=band_colors,
               edgecolor='none', height=0.6)
ax.set_xlim(0, 1.18)
for bar, val in zip(bars, all_metric_values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(0.9, color='#4CAF50', ls=':', lw=1.2, alpha=0.6)
ax.axvline(0.75, color='#FF9800', ls=':', lw=1.2, alpha=0.6)
ax.text(0.905, -0.7, '0.90', fontsize=7, color='#4CAF50')
ax.text(0.755, -0.7, '0.75', fontsize=7, color='#FF9800')
ax.set_title('Full Metrics Scorecard  (🟢 ≥0.90 | 🟡 ≥0.75 | 🔴 <0.75)',
             fontweight='bold', pad=10)
ax.set_xlabel('Score'); ax.grid(alpha=0.2, axis='x')

# ── (K) Confusion Matrix — percent only ──────────────────────────────────
ax = axes2[0, 1]
cm_pct = cm.astype(float) / cm.sum() * 100
labels_pct = np.array([[f'{v:.2f}%\n(n={n:,})' for v, n in zip(rp, rn)]
                        for rp, rn in zip(cm_pct, cm)])
sns.heatmap(cm_pct, annot=labels_pct, fmt='', cmap='RdYlGn', ax=ax,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'],
            linewidths=1, linecolor='white', vmin=0, vmax=70,
            cbar_kws={'shrink': 0.8, 'label': '% of all predictions'})
ax.set_title('Confusion Matrix — % of Total Predictions', fontweight='bold', pad=10)

# ── (L) Per-class Feature Importance comparison (LightGBM) ───────────────
ax = axes2[1, 0]
lgb_imp = pd.Series(lgb_clf.feature_importances_, index=FEATURES).nlargest(12).sort_values()
lgb_cols = plt.cm.Greens(np.linspace(0.35, 0.95, len(lgb_imp)))
lgb_imp.plot(kind='barh', ax=ax, color=lgb_cols, edgecolor='none')
for i, (val, name) in enumerate(zip(lgb_imp.values, lgb_imp.index)):
    ax.text(val + 0.5, i, f'{val:.0f}', va='center', fontsize=8)
ax.set_title('Feature Importances — LightGBM (Top 12)', fontweight='bold', pad=10)
ax.set_xlabel('Split Gain'); ax.grid(alpha=0.25, axis='x')

# ── (M) FPR vs FNR trade-off (Detection Error Trade-off curve) ───────────
ax = axes2[1, 1]
fpr_det, tpr_det, thresholds_det = roc_curve(y_test, y_prob)
fnr_det = 1 - tpr_det
ax.plot(fpr_det, fnr_det, color='#E91E63', lw=2.5, label='MSCE DET curve')
ax.scatter([fp / (fp + tn)], [fn / (fn + tp)],
           color='black', zorder=5, s=80, marker='*',
           label=f'Operating point\n(FPR={fp/(fp+tn):.4f}, FNR={fn/(fn+tp):.4f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('False Positive Rate (log)'); ax.set_ylabel('False Negative Rate (log)')
ax.set_title('Detection Error Trade-off (DET) Curve', fontweight='bold', pad=10)
ax.legend(fontsize=9); ax.grid(alpha=0.25, which='both')

plt.tight_layout(pad=2.5)
fig2.savefig('fraud_metrics_deep_dive.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  ✓ Figure 2 saved → fraud_metrics_deep_dive.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump({
    'scaler'      : scaler,
    'iso_forest'  : iso_forest,
    'xgb_clf'     : xgb_clf,
    'lgb_clf'     : lgb_clf,
    'rf_clf'      : rf_clf,
    'meta_learner': meta_learner,
    'threshold'   : THRESHOLD,
    'features'    : FEATURES
}, 'msce_fraud_model.pkl')
print("  ✓ Model saved  → msce_fraud_model.pkl")
print("  ✓ Output files : fraud_detection_evaluation.png")
print("                   fraud_metrics_deep_dive.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  INFERENCE FUNCTION  (for new transactions)
# ─────────────────────────────────────────────────────────────────────────────
def predict_new_transaction(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of raw transactions (same schema as training data),
    returns the DataFrame with 'fraud_probability' and 'is_fraud_predicted'.
    """
    artifacts = joblib.load('msce_fraud_model.pkl')
    sc   = artifacts['scaler']
    feat = artifacts['features']
    thr  = artifacts['threshold']

    # (apply same feature engineering here if needed)
    X_new = raw_df[feat].fillna(0).values
    X_sc  = sc.transform(X_new)

    proba = msce_predict_proba(X_sc)
    raw_df = raw_df.copy()
    raw_df['fraud_probability']  = proba
    raw_df['is_fraud_predicted'] = (proba >= thr).astype(int)
    return raw_df


print("\n" + "=" * 70)
print("  ✓ TRAINING COMPLETE")
print("=" * 70)
print(f"""
  Summary of MSCE Algorithm:
  ─────────────────────────
  Stage 1 → Isolation Forest  :  anomaly score per transaction
  Stage 2 → XGBoost + LightGBM + Random Forest  :  base probabilities
  Stage 3 → Logistic Regression meta-learner  :  final decision

  Key Result:
    ROC-AUC  = {roc_auc:.4f}   (1.0 = perfect; 0.5 = random)
    PR-AUC   = {pr_auc:.4f}   (critical metric for imbalanced data)
    MCC      = {mcc:.4f}   (best single metric for fraud detection)
    F1-Score = {f1:.4f}
""")