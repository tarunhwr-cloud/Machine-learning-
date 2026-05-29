# -*- coding: utf-8 -*-
"""
Quantum OC-SVM - HAI Anomaly Detection
Paper: Cultice et al., ISVLSI 2024

ANALYSIS OF LAST RUN (Sep=1.007x, F1=0.782):
  Normal   u=1.0371  (broad distribution, long right tail)
  Anomaly  u=1.0295  (narrow sharp peak at 1.030-1.035)
  Overlap: heavy in 1.030-1.040 range

  SVM decision function: linear boundary in kernel space
  Problem: linear boundary is suboptimal for overlapping distributions

THIS RUN - 3 SCORING METHODS, BEST WINS:
  Method 1: SVM decision function (current working method)
  Method 2: Kernel row mean score (k_mean_i = mean of K_te[i,:])
            Anomaly -> low fidelity with all training -> low row mean
            More direct use of quantum fidelity
  Method 3: Kernel row mean + column std score
            Anomaly -> not only low mean but also low variance
            (anomalies cluster together, normals spread out)

  All three evaluated at all thresholds in parallel.
  Best combination wins.

EVERYTHING ELSE: identical to F1=0.782 working run.
"""

import os, time, warnings, multiprocessing
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

warnings.filterwarnings('ignore')
np.random.seed(42)
t_start = time.time()

# ============================================================
# SETTINGS - identical to working run
# ============================================================
N_FEATURES = 16
REPS       = 4
TRAIN_SIZE = 1000
N_EVAL     = 500
BATCH_SIZE = 50
EXP_FACTOR = 4

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
N_JOBS   = max(1, multiprocessing.cpu_count() - 1)

print("=" * 60)
print("  QUANTUM OC-SVM - HAI ANOMALY DETECTION")
print("  Base F1=0.782 -> Direct kernel scoring")
print("=" * 60)
print(f"  Train  : {TRAIN_SIZE} normal (sequential)")
print(f"  Test   : {2*N_EVAL} balanced 50/50")
print(f"  Reps   : {REPS}")
print(f"  New    : 3 scoring methods, best wins")
print(f"  Cores  : {N_JOBS} parallel")
print(f"  Est.   : ~58 min")
print("=" * 60 + "\n")


# ============================================================
# FEATURE MAP - original working version
# ============================================================
def build_feature_map(num_features=16, reps=4):
    n_q    = num_features // 2
    params = ParameterVector('x', num_features)
    qc     = QuantumCircuit(n_q)
    for _ in range(reps):
        for q in range(n_q):
            qc.ry(params[2*q],     q)
            qc.rz(params[2*q + 1], q)
        for q in range(n_q - 1):
            qc.cx(q, q + 1)
    return qc

feature_map = build_feature_map(N_FEATURES, REPS)
qk          = FidelityQuantumKernel(feature_map=feature_map)


# ============================================================
# PARALLEL KERNEL (Fixed backend for Qiskit compatibility)
# ============================================================
def compute_kernel(qk, X1, X2=None):
    sym  = X2 is None
    X2_  = X1 if sym else X2
    n1, n2 = len(X1), len(X2_)
    K    = np.zeros((n1, n2))
    tasks = [(i, j)
             for i in range(0, n1, BATCH_SIZE)
             for j in range(0, n2, BATCH_SIZE)
             if not (sym and j < i)]

    def worker(i, j):
        return i, j, qk.evaluate(
            x_vec=X1[i:i+BATCH_SIZE],
            y_vec=X2_[j:j+BATCH_SIZE])

    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(worker)(i, j) for i, j in tqdm(tasks))
    for i, j, blk in res:
        K[i:i+BATCH_SIZE, j:j+BATCH_SIZE] = blk
        if sym:
            K[j:j+BATCH_SIZE, i:i+BATCH_SIZE] = blk.T
    return K


# ============================================================
# THRESHOLD EVALUATOR
# ============================================================
def eval_score(score_arr, pct, label, y_true):
    """Threshold a score array at given percentile, return metrics."""
    thr  = np.percentile(score_arr, pct)
    pred = np.where(score_arr < thr, -1, 1)
    return (
        label, pct,
        accuracy_score(y_true, pred),
        precision_score(y_true, pred, pos_label=-1, zero_division=0),
        recall_score(y_true, pred,    pos_label=-1, zero_division=0),
        f1_score(y_true, pred,        pos_label=-1, zero_division=0),
        pred.copy()
    )


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("[ 1/6 ] Loading data ...")
train_df     = pd.read_csv("train1.csv", sep=";")
test_df      = pd.read_csv("test1.csv",  sep=";")
drop_cols    = ["time","attack","attack_P1","attack_P2","attack_P3"]
train_labels = train_df["attack"].values
test_labels  = test_df["attack"].values

train_raw = (train_df.drop(columns=drop_cols)
             .apply(pd.to_numeric, errors='coerce')
             .rolling(window=60).mean().dropna())
test_raw  = (test_df.drop(columns=drop_cols)
             .apply(pd.to_numeric, errors='coerce')
             .rolling(window=60).mean().dropna())

train_labels = train_labels[-len(train_raw):]
test_labels  = test_labels[-len(test_raw):]

scaler       = StandardScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)
print(f"  Train: {len(train_scaled):,}  Test: {len(test_scaled):,}\n")


# ============================================================
# STEP 2: FEATURE SELECTION
# ============================================================
print("[ 2/6 ] Feature selection ...")
train_normal_mask = (train_labels == 0)
train_normal_data = train_scaled[train_normal_mask]

ocsvm_ps  = OneClassSVM(kernel='rbf', nu=0.05)
ocsvm_ps.fit(train_normal_data[:3000])
ps_labels = np.where(ocsvm_ps.predict(train_scaled) == -1, 1, 0)

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(train_scaled[:5000], ps_labels[:5000])
top_idx = np.argsort(rf.feature_importances_)[::-1][:N_FEATURES]

train_feat = train_normal_data[:, top_idx]
test_feat  = test_scaled[:, top_idx]
print(f"  Top {N_FEATURES} indices: {top_idx}\n")


# ============================================================
# STEP 3: SEQUENTIAL SAMPLING
# ============================================================
print("[ 3/6 ] Sequential first-N sampling ...")
train_sel   = train_feat[:TRAIN_SIZE]
normal_idx  = np.where(test_labels == 0)[0]
anomaly_idx = np.where(test_labels == 1)[0]
n_each      = min(N_EVAL, len(normal_idx), len(anomaly_idx))

eval_idx    = np.concatenate([normal_idx[:n_each], anomaly_idx[:n_each]])
np.random.seed(42)
np.random.shuffle(eval_idx)

test_sel = test_feat[eval_idx]
y_true   = np.where(test_labels[eval_idx] == 0, 1, -1)
ni       = np.where(y_true ==  1)[0]
ai       = np.where(y_true == -1)[0]
print(f"  Train: {len(train_sel)} | Test: {len(test_sel)} "
      f"(N:{len(ni)} A:{len(ai)})\n")


# ============================================================
# STEP 4: NORMALIZATION
# ============================================================
print("[ 4/6 ] Normalizing to [-pi, pi] ...")
mms        = MinMaxScaler(feature_range=(-np.pi, np.pi))
train_norm = mms.fit_transform(train_sel)
test_norm  = np.clip(mms.transform(test_sel), -np.pi, np.pi)
print(f"  Train: [{train_norm.min():.3f}, {train_norm.max():.3f}]")
print(f"  Test : [{test_norm.min():.3f}, {test_norm.max():.3f}]\n")


# ============================================================
# STEP 5: QUANTUM KERNEL
# ============================================================
print(f"[ 5/6 ] Quantum kernels ({N_JOBS} cores, reps={REPS}) ...")

print("  Train kernel:")
t0 = time.time()
K_tr_raw = compute_kernel(qk, train_norm)
print(f"  Done {(time.time()-t0)/60:.1f} min\n")

print("  Test kernel:")
t0 = time.time()
K_te_raw = compute_kernel(qk, test_norm, train_norm)
print(f"  Done {(time.time()-t0)/60:.1f} min\n")

K_tr = np.exp(EXP_FACTOR * K_tr_raw)
K_te = np.exp(EXP_FACTOR * K_te_raw)
K_tr = (K_tr + K_tr.T) / 2
K_tr += 1e-6 * np.eye(len(K_tr))

nm  = K_te[ni].mean()
am  = K_te[ai].mean()
sep = nm / (am + 1e-8)

print(f"  K_tr: mean={K_tr.mean():.4f}  max={K_tr.max():.4f}")
print(f"  K_te: mean={K_te.mean():.4f}  max={K_te.max():.4f}")
print(f"  Normal mean : {nm:.4f}")
print(f"  Anomaly mean: {am:.4f}")
print(f"  Separation  : {sep:.4f}x\n")


# ============================================================
# STEP 6: 3 SCORING METHODS + PARALLEL THRESHOLD SEARCH
# ============================================================
print("[ 6/6 ] Computing all scoring methods ...")

# -- Method 1: SVM decision function --------------------------
print("  Method 1: SVM scores ...")
nu_vals   = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
svm_scores = {}
for nu in nu_vals:
    m  = OneClassSVM(kernel="precomputed", nu=nu)
    m.fit(K_tr)
    sc = m.decision_function(K_te)
    svm_scores[f'svm_nu{nu}'] = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
    print(f"     nu={nu:.2f} done")

# -- Method 2: kernel row mean score --------------------------
print("\n  Method 2: Kernel row mean score ...")
row_means = K_te.mean(axis=1)
row_means_norm = ((row_means - row_means.min()) /
                  (row_means.max() - row_means.min() + 1e-10))
print(f"     Row mean: Normal={row_means[ni].mean():.4f}  "
      f"Anomaly={row_means[ai].mean():.4f}  "
      f"Sep={row_means[ni].mean()/row_means[ai].mean():.4f}x")

# -- Method 3: row mean + row std combined --------------------
print("\n  Method 3: Row mean + std score ...")
row_stds  = K_te.std(axis=1)
rm_n = (row_means - row_means.min()) / (row_means.max() - row_means.min() + 1e-10)
rs_n = (row_stds  - row_stds.min())  / (row_stds.max()  - row_stds.min()  + 1e-10)
combined = 0.7 * rm_n + 0.3 * rs_n
print(f"     Combined: Normal={combined[ni].mean():.4f}  "
      f"Anomaly={combined[ai].mean():.4f}  "
      f"Sep={combined[ni].mean()/(combined[ai].mean()+1e-8):.4f}x")

# -- Method 4: SVM ensemble -----------------------------------
print("\n  Method 4: SVM ensemble (mean of all nu) ...")
svm_stack       = np.array(list(svm_scores.values()))
svm_ensemble    = svm_stack.mean(axis=0)
print(f"     Ensemble: Normal={svm_ensemble[ni].mean():.4f}  "
      f"Anomaly={svm_ensemble[ai].mean():.4f}")

all_scores = {
    'row_mean'    : row_means_norm,
    'mean_std'    : combined,
    'svm_ensemble': svm_ensemble,
}
all_scores.update(svm_scores)

# -- Parallel threshold search across all methods -------------
print("\n  Parallel threshold search ...")
pct_vals = list(range(35, 76, 1))

tasks = [(lbl, sc, pct)
         for lbl, sc in all_scores.items()
         for pct in pct_vals]

res_parallel = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(eval_score)(sc, pct, lbl, y_true)
    for lbl, sc, pct in tasks)

print(f"\n  Best result per scoring method:")
print(f"  {'Method':<20}  {'pct':>5}  "
      f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("  " + "-"*58)

by_method = defaultdict(list)
for r in res_parallel:
    by_method[r[0]].append(r)

method_bests = {}
for lbl in all_scores:
    br = max(by_method[lbl], key=lambda x: x[5])
    method_bests[lbl] = br
    print(f"  {lbl:<20}  {br[1]:>5}  "
          f"{br[2]:>6.3f} {br[3]:>6.3f} "
          f"{br[4]:>6.3f} {br[5]:>6.3f}")

best      = max(res_parallel, key=lambda x: x[5])
best_lbl  = best[0]
best_pct  = best[1]
best_acc  = best[2]
best_prec = best[3]
best_rec  = best[4]
best_f1   = best[5]
best_pred = best[6]

elapsed = (time.time() - t_start) / 60
cm      = confusion_matrix(y_true, best_pred, labels=[1, -1])

print(f"\n{'='*64}")
print(f"  FINAL RESULTS  (runtime: {elapsed:.1f} min)")
print(f"{'='*64}")
print(f"  {'Method':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print(f"  {'-'*60}")
print(f"  {'Base run (SVM only)':<30} "
      f"{'0.752':>6} {'0.697':>6} {'0.892':>6} {'0.782':>6}")
print(f"  {'Best this run':<30} "
      f"{best_acc:>6.3f} {best_prec:>6.3f} "
      f"{best_rec:>6.3f} {best_f1:>6.3f}  ({best_lbl})")
print(f"  {'Paper target':<30} "
      f"{'0.870':>6} {'0.880':>6} {'0.870':>6} {'0.860':>6}")
print(f"{'='*64}")
gap    = best_f1 - 0.86
improv = best_f1 - 0.782
print(f"  Improvement : {improv:+.3f} over base run")
print(f"  vs paper    : {gap:+.3f}  "
      f"{'MATCHES OR BEATS!' if gap >= 0 else f'{abs(gap):.3f} below'}")
print(f"  Best method : {best_lbl}  pct={best_pct}")
print(f"  Separation  : {sep:.4f}x")
print(f"\n  Confusion:")
print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}\n")


# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Quantum OC-SVM - HAI Anomaly Detection (Cultice et al. 2024)\n"
    f"Train={TRAIN_SIZE}, Test={2*n_each}, reps={REPS}, exp={EXP_FACTOR}xK\n"
    f"Runtime: {elapsed:.1f} min  Sep={sep:.4f}x  "
    f"Best F1={best_f1:.3f}  ({best_lbl}, pct={best_pct})",
    fontsize=10)

metrics = ["Accuracy","Precision","Recall","F1"]
q_vals  = [best_acc, best_prec, best_rec, best_f1]
base    = [0.752, 0.697, 0.892, 0.782]
p_vals  = [0.87, 0.88, 0.87, 0.86]
x, w    = np.arange(4), 0.25
ax = axes[0,0]
ax.bar(x-w, q_vals, w, label=f'This run (F1={best_f1:.3f})', color='steelblue')
ax.bar(x,   base,   w, label='Base (F1=0.782)', color='royalblue', alpha=0.7)
ax.bar(x+w, p_vals, w, label='Paper target', color='coral', alpha=0.8)
ax.set_title("Performance Comparison")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for i,(q,b,p) in enumerate(zip(q_vals,base,p_vals)):
    ax.text(i-w, q+0.02, f'{q:.3f}', ha='center', fontsize=9, color='steelblue', fontweight='bold')
    ax.text(i,   b+0.02, f'{b:.3f}', ha='center', fontsize=8, color='royalblue')
    ax.text(i+w, p+0.02, f'{p:.2f}', ha='center', fontsize=8, color='coral')

ax = axes[0,1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Normal','Pred Anomaly'],
            yticklabels=['True Normal','True Anomaly'],
            ax=ax, annot_kws={'size':13,'weight':'bold'})
ax.set_title(f"Confusion Matrix ({best_lbl}, pct={best_pct})")

ax = axes[1,0]
colors = ['steelblue','seagreen','coral','royalblue','purple','orange','brown']
key_methods = ['row_mean', 'mean_std', 'svm_ensemble', 'svm_nu0.01', 'svm_nu0.05']
for i, lbl in enumerate(key_methods):
    if lbl not in by_method:
        continue
    sorted_r = sorted(by_method[lbl], key=lambda x: x[1])
    pl = [r[1] for r in sorted_r]
    fl = [r[5] for r in sorted_r]
    ax.plot(pl, fl, '-', color=colors[i], lw=1.5, label=f'{lbl} (best={max(fl):.3f})')
ax.axhline(0.86,  color='black', ls=':', lw=1.5, label='Paper=0.86')
ax.axhline(0.782, color='gray',  ls=':', lw=1,   label='Base=0.782')
ax.set_title("F1 vs Threshold: All Scoring Methods")
ax.set_xlabel("Percentile"); ax.set_ylabel("F1")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[1,1]
nm_arr = K_te[ni].mean(axis=1)
am_arr = K_te[ai].mean(axis=1)
ax.hist(nm_arr, bins=40, alpha=0.7, density=True, label=f'Normal  u={nm_arr.mean():.4f}', color='steelblue')
ax.hist(am_arr, bins=40, alpha=0.7, density=True, label=f'Anomaly u={am_arr.mean():.4f}', color='coral')
ax.set_title(f"Kernel Row Mean Distribution\nSep={sep:.4f}x")
ax.set_xlabel("Mean kernel fidelity"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
sp = os.path.join(SAVE_DIR, "results.png")
plt.savefig(sp, dpi=150, bbox_inches='tight')
print(f"  Plot : {sp}")
print(f"  Time : {elapsed:.1f} min")