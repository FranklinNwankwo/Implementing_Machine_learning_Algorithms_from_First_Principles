"""
Inference for the two from-scratch logistic regression models.

Scores are recomputed live from the saved weights, so the dashboard runs the
actual model rather than replaying stored predictions:

    z = (x − μ) / σ                 (training-set statistics only)
    logit = w₀ + Σⱼ wⱼ zⱼ
    P(fraud | x) = σ(logit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ART_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_LABELS = {"A": "Model A · Weighted BCE", "B": "Model B · SMOTE"}


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class FraudModel:
    key: str
    weights: np.ndarray           # [bias, w_1 … w_29]
    mu: np.ndarray
    sigma: np.ndarray
    tau: float                    # F2-optimal threshold chosen on validation
    feature_names: list[str]
    loss: np.ndarray
    val_loss: np.ndarray
    grad_norm: np.ndarray

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.key]

    @property
    def bias(self) -> float:
        return float(self.weights[0])

    @property
    def coef(self) -> np.ndarray:
        return self.weights[1:]

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma

    def logit(self, X: np.ndarray) -> np.ndarray:
        return self.bias + self.standardise(X) @ self.coef

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(X))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Exact additive decomposition of one transaction's logit:
        logit = bias + Σ contributions."""
        return self.standardise(x.reshape(1, -1))[0] * self.coef


@dataclass
class EvalData:
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (self.X_val, self.y_val) if name == "validation" else (self.X_test, self.y_test)


def load_models(art_dir: Path = ART_DIR) -> dict[str, FraudModel]:
    m = np.load(art_dir / "model.npz", allow_pickle=False)
    names = [str(s) for s in m["feature_names"]]
    return {
        k: FraudModel(
            key=k, weights=m[f"weights_{k}"], mu=m["mu"], sigma=m["sigma"],
            tau=float(m[f"tau_{k}"]), feature_names=names,
            loss=m[f"loss_{k}"], val_loss=m[f"val_loss_{k}"], grad_norm=m[f"grad_norm_{k}"],
        )
        for k in ("A", "B")
    }


def load_eval_data(art_dir: Path = ART_DIR) -> EvalData:
    d = np.load(art_dir / "eval_data.npz", allow_pickle=False)
    return EvalData(
        X_val=d["X_val"].astype(np.float64), y_val=d["y_val"].astype(np.int64),
        X_test=d["X_test"].astype(np.float64), y_test=d["y_test"].astype(np.int64),
    )


def load_meta(art_dir: Path = ART_DIR) -> dict:
    return json.loads((art_dir / "meta.json").read_text())


def amounts_from_features(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Recover the original transaction amount (€) from the Amount_log column."""
    return np.expm1(X[:, feature_names.index("Amount_log")])
