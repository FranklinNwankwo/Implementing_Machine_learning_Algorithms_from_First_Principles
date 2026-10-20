"""
Rebuild the dashboard's model artifacts from the raw dataset.

This script reproduces the training pipeline in
`notebook/Logistic_Regression_Credit_Card_Dataset.ipynb` exactly:
same preprocessing, same stratified 70/15/15 split (seed 42), same
train-only z-score normalisation, same from-scratch SMOTE, same
from-scratch gradient-descent logistic regression, same F2 threshold
search on the validation set. It then saves only what the dashboard needs:

    artifacts/model.npz       weights, normalisation stats, thresholds,
                              training histories
    artifacts/eval_data.npz   validation + test features and labels
                              (no training rows are shipped)

The run asserts that the test-set operating points match the numbers
printed in the notebook, so the app can never silently drift from it.

Usage
-----
    python build_artifacts.py                       # downloads the CSV
    python build_artifacts.py --csv path/to/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
ART_DIR = Path(__file__).resolve().parent / "artifacts"
N_DUPLICATES = 0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — copied from the notebook (Full Model Implementation Pipeline cell)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(source: str):
    global N_DUPLICATES
    df = pd.read_csv(source)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    N_DUPLICATES = before - len(df)
    print(f"Dropped {before - len(df):,} duplicate rows. Remaining: {len(df):,}")
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount", "Time"])
    feature_names = [c for c in df.columns if c != "Class"]
    X = df.drop(columns=["Class"]).values.astype(np.float64)
    y = df["Class"].values.astype(np.float64)
    return X, y, feature_names


def stratified_split(X, y, test_size=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])
    n_pos_test = int(len(idx_pos) * test_size)
    n_neg_test = int(len(idx_neg) * test_size)
    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.concatenate([idx_pos[n_pos_test:], idx_neg[n_neg_test:]])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def z_score_normalise(X_train, X_val, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, sample_weights=None):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if sample_weights is not None:
        return np.mean(sample_weights * per_sample)
    return np.mean(per_sample)


def compute_gradient(X, y_true, y_pred, sample_weights=None):
    error = y_pred - y_true
    if sample_weights is not None:
        error = sample_weights * error
    return (1 / X.shape[0]) * (X.T @ error)


def gradient_check(X, y, w, sample_weights=None, eps=1e-5):
    analytic = compute_gradient(X, y, sigmoid(X @ w), sample_weights)
    numerical = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        numerical[i] = (binary_cross_entropy(y, sigmoid(X @ wp), sample_weights)
                        - binary_cross_entropy(y, sigmoid(X @ wm), sample_weights)) / (2 * eps)
    return np.linalg.norm(analytic - numerical) / (
        np.linalg.norm(analytic) + np.linalg.norm(numerical) + 1e-15)


def smote(X, y, target_ratio=0.1, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    X_min, X_maj = X[y == 1], X[y == 0]
    n_maj, n_min = len(X_maj), len(X_min)
    n_needed = int(n_maj * target_ratio / (1 - target_ratio)) - n_min
    if n_needed <= 0:
        return X, y
    sq = np.sum(X_min ** 2, axis=1, keepdims=True)
    dists = np.maximum(sq + sq.T - 2 * (X_min @ X_min.T), 0)
    nn_idx = np.argsort(dists, axis=1)[:, 1:k + 1]
    synthetics = []
    for _ in range(n_needed):
        i = rng.integers(0, n_min)
        nn = nn_idx[i, rng.integers(0, k)]
        lam = rng.uniform(0, 1)
        synthetics.append(X_min[i] + lam * (X_min[nn] - X_min[i]))
    X_bal = np.vstack([X, np.vstack(synthetics)])
    y_bal = np.concatenate([y, np.ones(n_needed)])
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


class LogisticRegression:
    """Gradient-descent logistic regression, identical to the notebook's class."""

    def __init__(self, learning_rate=0.1, n_iterations=2000, class_weight=None, patience=30):
        self.lr, self.n_iter = learning_rate, n_iterations
        self.class_weight, self.patience = class_weight, patience
        self.weights = None
        self.loss_history, self.val_loss_history, self.grad_norm_history = [], [], []

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _sample_weights(self, y):
        if self.class_weight == "balanced":
            m, m0, m1 = len(y), np.sum(y == 0), np.sum(y == 1)
            w0, w1 = m / (2 * m0), m / (2 * m1)
            return np.where(y == 1, w1, w0), w0, w1
        return np.ones(len(y)), 1.0, 1.0

    def fit(self, X, y, X_val, y_val):
        X = self._add_bias(X)
        self.weights = np.zeros(X.shape[1])
        c, self.w0, self.w1 = self._sample_weights(y)
        best_val, best_w, no_improve = np.inf, self.weights.copy(), 0
        X_val_b = self._add_bias(X_val)
        self.stopped_epoch = self.n_iter
        for epoch in range(self.n_iter):
            y_hat = sigmoid(X @ self.weights)
            train_loss = binary_cross_entropy(y, y_hat, sample_weights=c)
            grad = compute_gradient(X, y, y_hat, sample_weights=c)
            self.weights -= self.lr * grad
            self.loss_history.append(train_loss)
            self.grad_norm_history.append(np.linalg.norm(grad))
            val_loss = binary_cross_entropy(y_val, sigmoid(X_val_b @ self.weights))
            self.val_loss_history.append(val_loss)
            if val_loss < best_val:
                best_val, best_w, no_improve = val_loss, self.weights.copy(), 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                self.stopped_epoch = epoch + 1
                break
        self.weights = best_w
        self.best_val_loss = best_val
        return self

    def predict_proba(self, X):
        return sigmoid(self._add_bias(X) @ self.weights)


def precision_recall_f1(y_true, y_pred, beta=1.0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp + 1e-15)
    rec = tp / (tp + fn + 1e-15)
    return prec, rec, (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-15)


def find_optimal_threshold(y_true, y_scores, beta=2.0, n_thresholds=300):
    best_t, best_fb = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n_thresholds):
        _, _, fb = precision_recall_f1(y_true, (y_scores >= t).astype(int), beta=beta)
        if fb > best_fb:
            best_fb, best_t = fb, t
    return best_t, best_fb


def notebook_grid_auc_pr(y_true, y_scores, n_thresholds=300):
    """The notebook's 300-point linear-grid AUC-PR, kept for reference only."""
    recs, precs = [], []
    for t in np.linspace(0, 1, n_thresholds):
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        precs.append(1.0 if tp + fp == 0 else tp / (tp + fp))
        recs.append(precision_recall_f1(y_true, y_pred)[1])
    recs, precs = np.array(recs), np.array(precs)
    order = np.argsort(recs)
    return float(np.trapezoid(precs[order], recs[order]))


# ─────────────────────────────────────────────────────────────────────────────
# Expected notebook results (test set, operating point chosen on validation)
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED = {
    "A": dict(tau=0.987, tp=58, fn=12, fp=29),
    "B": dict(tau=0.767, tp=58, fn=12, fp=29),
}


def main(source: str) -> None:
    X, y, feature_names = load_and_preprocess(source)
    X_temp, X_test, y_temp, y_test = stratified_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = stratified_split(X_temp, y_temp, test_size=round(0.15 / 0.85, 6))
    print(f"Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
    X_train_n, X_val_n, X_test_n, mu, sigma = z_score_normalise(X_train, X_val, X_test)

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(X_train_n), 200, replace=False)
    Xc = np.hstack([np.ones((200, 1)), X_train_n[sub]])
    grad_err = gradient_check(Xc, y_train[sub], np.zeros(Xc.shape[1]))
    print(f"Gradient check relative error: {grad_err:.2e}")

    print("Training Model A (weighted BCE)...")
    model_A = LogisticRegression(class_weight="balanced").fit(X_train_n, y_train, X_val_n, y_val)
    print("Generating SMOTE samples and training Model B...")
    X_smote, y_smote = smote(X_train_n, y_train, target_ratio=0.1, k=5)
    model_B = LogisticRegression(class_weight=None).fit(X_smote, y_smote, X_val_n, y_val)

    out = {}
    for name, model in [("A", model_A), ("B", model_B)]:
        tau, f2_val = find_optimal_threshold(y_val, model.predict_proba(X_val_n), beta=2.0)
        p_test = model.predict_proba(X_test_n)
        pred = (p_test >= tau).astype(int)
        tp = int(np.sum((y_test == 1) & (pred == 1)))
        fn = int(np.sum((y_test == 1) & (pred == 0)))
        fp = int(np.sum((y_test == 0) & (pred == 1)))
        exp = EXPECTED[name]
        assert round(tau, 3) == exp["tau"] and (tp, fn, fp) == (exp["tp"], exp["fn"], exp["fp"]), (
            f"Model {name} does not reproduce the notebook: tau={tau:.4f} TP={tp} FN={fn} FP={fp}")
        out[name] = dict(tau=tau, f2_val=f2_val, grid_auc_pr=notebook_grid_auc_pr(y_test, p_test))
        print(f"Model {name}: tau={tau:.4f}  TP={tp} FN={fn} FP={fp}  (matches notebook)")

    ART_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        ART_DIR / "model.npz",
        weights_A=model_A.weights, weights_B=model_B.weights,
        mu=mu, sigma=sigma,
        tau_A=out["A"]["tau"], tau_B=out["B"]["tau"],
        loss_A=np.array(model_A.loss_history), val_loss_A=np.array(model_A.val_loss_history),
        grad_norm_A=np.array(model_A.grad_norm_history),
        loss_B=np.array(model_B.loss_history), val_loss_B=np.array(model_B.val_loss_history),
        grad_norm_B=np.array(model_B.grad_norm_history),
        feature_names=np.array(feature_names),
    )
    np.savez_compressed(
        ART_DIR / "eval_data.npz",
        X_val=X_val.astype(np.float32), y_val=y_val.astype(np.int8),
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.int8),
    )
    meta = {
        "seed": SEED,
        "n_duplicates_dropped": int(N_DUPLICATES),
        "n_rows_after_dedup": int(len(y)),
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "train_fraud": int(y_train.sum()), "val_fraud": int(y_val.sum()), "test_fraud": int(y_test.sum()),
        "smote_rows": int(len(y_smote)), "smote_fraud": int(y_smote.sum()),
        "class_weight_fraud": float(model_A.w1), "class_weight_legit": float(model_A.w0),
        "gradient_check_rel_error": float(grad_err),
        "train_fraud_amount_mean": float(np.expm1(X_train[y_train == 1, -1]).mean()),
        "train_fraud_amount_median": float(np.median(np.expm1(X_train[y_train == 1, -1]))),
        "epochs_A": len(model_A.loss_history), "epochs_B": len(model_B.loss_history),
        "f2_val_A": float(out["A"]["f2_val"]), "f2_val_B": float(out["B"]["f2_val"]),
        "notebook_grid_auc_pr_A": out["A"]["grid_auc_pr"], "notebook_grid_auc_pr_B": out["B"]["grid_auc_pr"],
    }
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Artifacts written to {ART_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DATA_URL, help="Path or URL to creditcard.csv")
    main(ap.parse_args().csv)
