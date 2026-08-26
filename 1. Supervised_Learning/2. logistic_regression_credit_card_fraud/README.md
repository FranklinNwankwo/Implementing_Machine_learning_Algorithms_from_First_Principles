# Credit Card Fraud Detection — Logistic Regression from Scratch

An end-to-end machine learning project implementing logistic regression using only NumPy on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle — ULB Machine Learning Group).

Built to understand the fundamentals of classification under severe class imbalance without relying on high-level abstractions.

---

## Project Overview

This project walks through the full supervised learning pipeline for an extremely imbalanced binary classification problem (~577:1 legitimate-to-fraud ratio):

- **Exploratory Data Analysis (EDA)** — class distribution, feature distributions by class, distributional overlap
- **Data Handling** — stratified three-way split, Z-score normalisation fit on train only, log-transform of `Amount`
- **Model Implementation** — sigmoid, weighted binary cross-entropy, gradient descent, and SMOTE all built from scratch with NumPy
- **Imbalance Handling** — two independent strategies compared: class-weighted loss (Model A) vs. SMOTE oversampling (Model B)
- **Threshold Optimisation** — F-beta (β=2) sweep on the validation set, since the default τ=0.5 fails under this imbalance
- **Validation** — results benchmarked against scikit-learn's `LogisticRegression`, Decision Tree, Random Forest, and XGBoost
- **Diagnostics** — learning curves, calibration (reliability diagrams + Brier score), error analysis, feature weight analysis, assumption audit

Key issues encountered and resolved during the project:

- Choosing evaluation metrics that don't lie under 99.83% majority-class accuracy
- Data leakage risk from scaling, SMOTE, or threshold tuning touching validation/test data
- Class imbalance collapsing gradient descent into predicting the majority class
- Miscalibrated probabilities produced by class weighting and oversampling
- Distinguishing threshold errors from fundamental (irrecoverable) linear-boundary errors

---

## Repository Structure

```
2. logistic_regression_credit_card_fraud/
├── data/
│ └── .gitkeep
├── notebook/
│ └── Logistic_Regression_from_scratch.ipynb
├── README.md
└── requirements.txt
```


> **Note on data:** The raw dataset is not committed to this repo. It is loaded directly via this url: https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/2. logistic_regression_credit_card_fraud"
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook notebook/
```

---

## Dependencies

See `requirements.txt`. Core libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Linear algebra, gradient descent, SMOTE, all metrics |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scipy` | KDE for distribution plots |
| `xgboost` | Tree-based benchmark comparison |
| `scikit-learn` | Stratified splitting and benchmark baselines |

---

## Limitations

- **Linear separability assumption**: the decision boundary is a hyperplane. At the lowest threshold tested (τ=0.01), both models still miss a subset of fraud cases regardless of threshold — evidence these are fundamental errors, not threshold errors.
- **PCA information ceiling**: `V1`–`V28` were PCA-transformed upstream by the dataset creators. Any fraud-relevant signal discarded in that transformation is permanently invisible to any model trained on this data, and the resulting weights can't be mapped back to business-meaningful features.
- **SMOTE interpolates blindly**: synthetic samples are generated between any two minority-class neighbours regardless of whether the interpolated point is realistic or falls near a class boundary.
- **48-hour dataset window**: fraud patterns evolve over time; no concept-drift mechanism is implemented.
- **Independence assumption untestable**: there is no cardholder ID, so within-cardholder correlation across multiple transactions can't be measured or corrected for.
- **No explicit regularisation**: L2 weight decay was not implemented; early stopping is the sole overfitting control.

---

## Results

| Metric | Dummy | Model A (Weighted BCE) | Model B (SMOTE) |
|---|---|---|---|
| Threshold τ | — | ~0.987 | ~0.754 |
| Precision | 0.0000 | ~0.706 | ~0.828 |
| Recall | 0.0000 | ~0.786 | ~0.786 |
| F2-Score | 0.0000 | ~0.769 | ~0.794 |
| AUC-PR | ~0.0017 | ~0.353 | ~0.469 |
| AUC-ROC | ~0.500 | ~0.975 | ~0.973 |
| False alarms (FP) | 0 | ~32 | ~16 |
| Frauds missed (FN) | 98 | ~21 | ~21 |
| Accuracy | 99.83% | — | — |

> Values are representative outputs for `SEED=42` with the hyperparameters described in the notebook; exact numbers are printed at runtime by the evaluation cells.

Model B (SMOTE) comes out ahead: both models catch roughly the same number of frauds, but SMOTE cuts false alarms nearly in half and reaches a meaningfully higher AUC-PR — the richer training distribution gives the decision boundary cleaner separation in the fraud region.

---

## What I learned:

1. **Accuracy Is Actively Misleading Here.**

The first real lesson of this project had nothing to do with modelling — it was about metric selection. A classifier that predicts "legitimate" for every single transaction hits 99.83% accuracy while catching zero fraud. Building that dummy baseline explicitly, rather than skipping straight to the real model, made it viscerally clear why Recall, F2-score, and PR-AUC have to be the primary metrics under this kind of imbalance, and why accuracy has to be actively excluded, not just deprioritized.

2. **Weighted Loss vs. Oversampling Are Genuinely Different Interventions.**

Going in, I expected class-weighted BCE and SMOTE to be roughly interchangeable fixes for imbalance. They're not. Weighted BCE reshapes the loss landscape so fraud errors are penalised more, but the model still only sees the real fraud examples. SMOTE physically populates the fraud region of feature space with synthetic points, giving the boundary more signal to learn from. They ended up with different optimal thresholds (τ≈0.987 vs τ≈0.754), different calibration behaviour, and different false-positive rates — a good reminder that two techniques solving the "same" problem can produce meaningfully different models.

3. **The Default Threshold of 0.5 Is an Assumption, Not a Law.**

τ=0.5 implicitly assumes false positives and false negatives cost the same. In fraud detection they don't — a missed fraud is a direct financial loss, a false alarm is customer friction. Implementing an F-beta (β=2) sweep on the validation set to explicitly weight recall over precision, rather than accepting the sklearn default, was the point where threshold selection stopped feeling like an afterthought and started feeling like a first-class modelling decision.

4. **A High AUC Doesn't Mean the Probabilities Are Trustworthy.**

Both models score AUC-ROC above 0.97, which looks great in isolation and would have been misleading to stop at. Building reliability diagrams and Brier scores exposed that class weighting and SMOTE both distort the probability scale — a model can rank fraud well while still being poorly calibrated. Quantile-based bins were necessary here instead of equal-width ones, since equal-width bins are mostly empty at 0.17% fraud prevalence.

5. **Gradient Checking Is Worth Doing Even When the Math Feels Obvious.**

Before trusting any training run, I verified the analytic gradient against a finite-difference approximation and required the relative error to fall below 1e-5. It's a debugging habit borrowed from deep learning, and it caught the class of bug that's otherwise nearly invisible: a model that trains, converges, and produces plausible-looking numbers while being subtly wrong underneath.

6. **Interpretability Has a Hard Ceiling Independent of the Model.**

`V1`–`V28` are anonymised PCA components — the dataset creators applied PCA before I ever saw the data, specifically to protect cardholder privacy. That means the learned weights can tell me which components the model relies on, but never what those components mean in business terms. No amount of model sophistication recovers that; it's a property of the data, not of logistic regression specifically. That distinction — is this ceiling coming from my model or from the data — turned out to be a recurring question worth asking on every diagnostic.

7. **Diagnosing "Fundamental" vs. "Threshold" Errors Changes What You Try Next.**

Sweeping recall across every threshold down to τ=0.01 on the test set (purely for diagnostic visualisation, after τ* was already fixed on validation) showed that some fraud cases stay misclassified no matter how permissive the threshold gets. That's the difference between an error you can fix by moving τ and an error that requires a fundamentally different decision boundary — evidence that a linear model has a real ceiling here, and that closing the gap to literature-reported AUC-PR values (~0.70+) needs a nonlinear model, not more threshold tuning.

8. **Numerical Stability Details Compound Silently.**

Sigmoid clipped to [-500, 500] to prevent overflow, BCE guarded with ε=1e-15 to prevent log(0), Z-score normalisation guarded with ε=1e-8 to prevent division by zero — each of these felt like a minor detail in isolation, but skipping any one of them would have produced NaNs or silently wrong gradients somewhere downstream, likely long after the point where the actual bug was introduced.

9. **Benchmarking Against sklearn Is the Real Validation.**

Once the from-scratch model was working, I checked it against sklearn's `LogisticRegression` on the same preprocessed data. Getting close wasn't good enough as a goal — I wanted convergence to the same decision boundary within reasonable tolerance. Tracing every discrepancy back to its source (a scaling difference, a regularisation default sklearn applies that I hadn't matched) was more valuable than the implementation itself, because it's the only way to be confident the from-scratch math is actually correct and not just producing plausible-looking output.

10. **What I'd Do Differently**

Looking back, a few things would have saved time. I'd write the leakage-prevention rules (what gets fit on train-only, what stays unseen until final evaluation) down explicitly before writing any preprocessing code, rather than re-deriving them mid-implementation. I'd also build the calibration diagnostics earlier in the process instead of near the end — discovering that Model A's probabilities were compressed toward the extremes after already comparing both models on threshold behaviour meant re-interpreting earlier conclusions in light of a fact I could have known from the start.

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)