# Credit Card Fraud Detection — Logistic Regression from Scratch

An end-to-end machine learning project implementing logistic regression using only NumPy on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle — ULB Machine Learning Group).

Built to understand the fundamentals of classification under severe class imbalance without relying on high-level abstractions.

---

## Project Overview

This project walks through the full supervised learning pipeline for an extremely imbalanced binary classification problem. The raw dataset contains 284,807 transactions; after dropping 1,081 duplicate rows, 283,726 remain, with 473 confirmed fraud cases — a **598.8 : 1** legitimate-to-fraud ratio (fraud rate ≈ 0.167%):

- **Exploratory Data Analysis (EDA)** — class distribution, feature distributions by class, distributional overlap
- **Data Handling** — stratified three-way split, Z-score normalisation fit on train only, log-transform of `Amount`
- **Model Implementation** — sigmoid, weighted binary cross-entropy, gradient descent, and SMOTE all built from scratch with NumPy
- **Imbalance Handling** — two independent strategies compared: class-weighted loss (Model A) vs. SMOTE oversampling (Model B)
- **Threshold Optimisation** — F-beta (β=2) sweep on the validation set, since the default τ=0.5 fails under this imbalance
- **Validation** — results benchmarked against scikit-learn's `LogisticRegression`, Decision Tree, Random Forest, and XGBoost
- **Diagnostics** — learning curves, calibration (reliability diagrams + Brier score), error analysis, feature weight analysis, assumption audit
- **Gradient Verification** — analytic gradient checked against finite-difference approximation (relative error 3.46e-11) before any training

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
- **No explicit regularisation**: L2 weight decay was not implemented on the from-scratch model; early stopping is its sole overfitting control. This is also the likely reason its AUC-PR outpaces sklearn's L2-regularised LR (see Results).

---

## Results

Values below are the actual outputs of the notebook (`SEED=42`), taken directly from the executed comparison cell — not estimates.

**Dummy baseline** (always predicts legitimate): Accuracy 99.836%, Precision 0.0000, Recall 0.0000, F2 0.0000, TP 0, FN 70, FP 0, AUC-PR ≈ 0.0017 (the fraud base rate — the no-skill floor).

| Model | τ | Precision | Recall | F2 | AUC-PR | AUC-ROC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| **LR A — Weighted BCE (from scratch)** | 0.987 | 0.6667 | 0.8286 | 0.7902 | **0.5578** | 0.9770 | 58 | 12 | 29 |
| **LR B — SMOTE (from scratch)** | 0.688 | 0.6556 | **0.8429** | **0.7973** | 0.4682 | 0.9749 | 59 | 11 | 31 |
| sklearn LogisticRegression | 0.990 | 0.6105 | 0.8286 | 0.7733 | 0.2897 | 0.9783 | 58 | 12 | 37 |
| Decision Tree | 0.990 | 0.6292 | 0.8000 | 0.7588 | 0.2619 | 0.8871 | 56 | 14 | 33 |
| **Random Forest** | 0.626 | **0.7973** | 0.8429 | **0.8333** | **0.5739** | **0.9795** | 59 | 11 | **15** |
| XGBoost | 0.797 | **0.9661** | 0.8143 | 0.8407 | 0.4425 | 0.9487 | 57 | 13 | **2** |

**Model A vs. Model B is not a clean win for either side.** Weighted BCE and SMOTE are genuinely different interventions with different trade-offs:
- **Model A (Weighted BCE)** wins on Precision, AUC-PR (0.5578 vs 0.4682), and AUC-ROC — its decision boundary separates the fraud region more cleanly.
- **Model B (SMOTE)** wins on Recall and F2 — the metric this project explicitly optimises for (β=2 weights recall twice as heavily as precision), since a missed fraud is a direct financial loss.
- Which model is "better" therefore depends on which metric the business prioritises. Optimising purely for F2, SMOTE narrowly wins. Optimising for the threshold-independent AUC-PR (the metric this project treats as primary for imbalanced problems), weighted BCE wins clearly.

**Both from-scratch models beat sklearn's LogisticRegression on AUC-PR** (0.5578 / 0.4682 vs. 0.2897) despite similar recall — most likely because sklearn's default L2 regularisation pulls its probability estimates toward more conservative values, while the from-scratch model has no regularisation term.

**The tree-based benchmarks tell the real story about the linear boundary.** Random Forest posts the best overall numbers across the board (highest F2, highest AUC-PR, highest AUC-ROC, fewest false alarms among the balanced models), and XGBoost achieves near-perfect precision (0.9661) with only 2 false alarms — at the cost of lower recall. This is the practical demonstration of the project's central thesis: a linear model has a real, measurable ceiling on this dataset, and closing the gap requires a nonlinear decision boundary.

**The dummy row is intentional.** 99.836% accuracy while catching zero fraud is the clearest possible illustration of why accuracy is the wrong metric for this problem.

---

## What I learned:

1. **Accuracy Is Actively Misleading Here.**

The first real lesson of this project had nothing to do with modelling — it was about metric selection. A classifier that predicts "legitimate" for every single transaction hits 99.84% accuracy while catching zero fraud. Building that dummy baseline explicitly, rather than skipping straight to the real model, made it viscerally clear why Recall, F2-score, and PR-AUC have to be the primary metrics under this kind of imbalance, and why accuracy has to be actively excluded, not just deprioritized.

2. **Weighted Loss vs. Oversampling Are Genuinely Different Interventions — With No Automatic Winner.**

Going in, I expected class-weighted BCE and SMOTE to be roughly interchangeable fixes for imbalance. They're not, and the results don't even point cleanly in one direction: weighted BCE won on precision and AUC-PR, SMOTE won on recall and F2. Weighted BCE reshapes the loss landscape so fraud errors are penalised more, but the model still only sees the real fraud examples. SMOTE physically populates the fraud region of feature space with synthetic points, trading some precision for a boundary that catches more fraud. They ended up with different optimal thresholds (τ≈0.987 vs τ≈0.688), different calibration behaviour, and different false-positive rates — evidence that two techniques solving the "same" problem can produce genuinely different, non-dominated models.

3. **The Default Threshold of 0.5 Is an Assumption, Not a Law.**

τ=0.5 implicitly assumes false positives and false negatives cost the same. In fraud detection they don't — a missed fraud is a direct financial loss, a false alarm is customer friction. Implementing an F-beta (β=2) sweep on the validation set to explicitly weight recall over precision, rather than accepting the sklearn default, was the point where threshold selection stopped feeling like an afterthought and started feeling like a first-class modelling decision.

4. **A High AUC Doesn't Mean the Probabilities Are Trustworthy.**

Both models score AUC-ROC above 0.97, which looks great in isolation and would have been misleading to stop at. Building reliability diagrams and Brier scores exposed that class weighting and SMOTE both distort the probability scale — a model can rank fraud well while still being poorly calibrated. Quantile-based bins were necessary here instead of equal-width ones, since equal-width bins are mostly empty at 0.17% fraud prevalence.

5. **Gradient Checking Is Worth Doing Even When the Math Feels Obvious.**

Before trusting any training run, I verified the analytic gradient against a finite-difference approximation and required the relative error to fall below 1e-5 (it came in at 3.46e-11). It's a debugging habit borrowed from deep learning, and it caught the class of bug that's otherwise nearly invisible: a model that trains, converges, and produces plausible-looking numbers while being subtly wrong underneath.

6. **Interpretability Has a Hard Ceiling Independent of the Model.**

`V1`–`V28` are anonymised PCA components — the dataset creators applied PCA before I ever saw the data, specifically to protect cardholder privacy. That means the learned weights can tell me which components the model relies on, but never what those components mean in business terms. No amount of model sophistication recovers that; it's a property of the data, not of logistic regression specifically. That distinction — is this ceiling coming from my model or from the data — turned out to be a recurring question worth asking on every diagnostic.

7. **Diagnosing "Fundamental" vs. "Threshold" Errors Changes What You Try Next.**

Sweeping recall across every threshold down to τ=0.01 on the test set (purely for diagnostic visualisation, after τ* was already fixed on validation) showed that some fraud cases stay misclassified no matter how permissive the threshold gets. That's the difference between an error you can fix by moving τ and an error that requires a fundamentally different decision boundary — and it's exactly what the benchmark table confirms: Random Forest and XGBoost, both nonlinear, outperform both logistic regression variants on F2 and AUC-PR (0.8333 and 0.8407 vs. 0.7902/0.7973), evidence that a linear model has a real ceiling here and that closing the gap needs a nonlinear model, not more threshold tuning.

8. **Numerical Stability Details Compound Silently.**

Sigmoid clipped to [-500, 500] to prevent overflow, BCE guarded with ε=1e-15 to prevent log(0), Z-score normalisation guarded with ε=1e-8 to prevent division by zero — each of these felt like a minor detail in isolation, but skipping any one of them would have produced NaNs or silently wrong gradients somewhere downstream, likely long after the point where the actual bug was introduced.

9. **Benchmarking Against sklearn Is the Real Validation — And It Surfaced a Real Discrepancy.**

Once the from-scratch model was working, I checked it against sklearn's `LogisticRegression` on the same preprocessed data. AUC-ROC and F2 lined up closely (0.9770 vs 0.9783, 0.7902 vs 0.7733), which confirmed the core math was correct. But AUC-PR did not line up (0.5578 vs 0.2897) — a genuine, sizeable gap rather than solver noise. Tracing that back to sklearn's default L2 regularisation (which the from-scratch model doesn't implement) was more valuable than the implementation itself, because it's the only way to be confident the from-scratch math is actually correct and to understand *why* a discrepancy exists rather than dismissing it.

10. **What I'd Do Differently**

Looking back, a few things would have saved time. I'd write the leakage-prevention rules (what gets fit on train-only, what stays unseen until final evaluation) down explicitly before writing any preprocessing code, rather than re-deriving them mid-implementation. I'd also build the calibration diagnostics earlier in the process instead of near the end — discovering that Model A's probabilities were compressed toward the extremes after already comparing both models on threshold behaviour meant re-interpreting earlier conclusions in light of a fact I could have known from the start. I'd also match sklearn's regularisation (or explicitly disable it) from the start, so the benchmark comparison isolates "from-scratch vs. sklearn implementation correctness" instead of quietly also comparing "regularised vs. unregularised."

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)