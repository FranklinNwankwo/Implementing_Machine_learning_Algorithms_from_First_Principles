# Support Vector Machine from Scratch — Breast Cancer Wisconsin Diagnosis

An end-to-end machine learning project implementing a Support Vector Machine (soft-margin, linear kernel) classifier using only NumPy and Pandas, applied to the [Breast Cancer Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

Built to understand the fundamentals of margin-based, convex-optimization-driven learning, from the maximum-margin derivation through subgradient descent to support vector identification, with no reliance on high-level abstractions.

---

## Project Overview

Unlike a lazy learner like KNN, an SVM **learns a parametric decision boundary**: a weight vector `w` and bias `b` defining the hyperplane `w·x + b = 0` that maximizes the margin between classes. This project implements that idea from first principles:

1. **Mathematical Derivation** — the hard-margin maximum-margin classifier, the soft-margin primal with slack variables, the equivalent hinge-loss formulation `L(w,b) = (λ/2)‖w‖² + (1/n)Σ max(0, 1 − yᵢ(w·xᵢ + b))`, and the subgradient of the hinge loss at its non-differentiable kink
2. **Optimization** — mini-batch subgradient descent (batch size 32), since the hinge loss is not differentiable everywhere and a standard closed-form or gradient solution doesn't apply
3. **Support Vector Identification** — points within a tolerance `ε = 0.05` of the margin boundary (`yᵢ(w·xᵢ + b) ≤ 1 + ε`) are flagged as support vectors; everything else could be removed from training without changing the learned hyperplane
4. **Clinical Framing** — malignant tumors are the positive-risk class; a false negative (malignant called benign) is far costlier than a false positive, so malignant recall and balanced accuracy are treated as primary metrics alongside raw accuracy

This project walks through the full supervised learning pipeline:

- **Structural Auditing** — 569 observations, 30 features, zero missing values, zero duplicates, all `float64`, mild class imbalance (37.3% malignant / 62.7% benign)
- **Exploratory Data Analysis (EDA)** — target distribution, mean/worst feature distributions by class, correlation heatmap, Cohen's-d feature-separation boxplots, and a PCA 2D projection to visually assess linear separability before committing to a linear kernel
- **Preprocessing** — stratified 80/20 train/test split, `StandardScaler` fit on the training split only (SVMs are not scale-invariant — the `½‖w‖²` objective is dominated by large-range features otherwise), and signed `{-1, +1}` target encoding required by the hinge loss
- **Baseline** — a most-frequent-class dummy classifier, to establish the accuracy floor and expose why accuracy alone is insufficient in a medical context
- **Model Implementation** — a `SupportVectorMachineFromScratch` class (mini-batch subgradient descent, configurable `C`/learning rate/epochs, loss-history tracking, support-vector extraction, a sigmoid-based `predict_proba` for ROC/PR curves) built from scratch with NumPy
- **Hyperparameter Experimentation** — a learning-rate sweep, a regularization (`C`) sweep evaluated on an internal validation split carved out of the training set only (test set never touched during model selection), and an epochs sweep
- **Validation** — a scikit-learn-compatible wrapper (`SKLearnCompatibleSVM`) enabling native `cross_validate` 5-fold stratified CV, plus a head-to-head comparison against `sklearn.svm.SVC` with both linear and RBF kernels
- **Comparative Modeling** — benchmarked against sklearn's SVC (linear and RBF), Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost, Naive Bayes, and KNN, all under 5-fold stratified CV
- **Diagnostics** — confusion matrices, ROC and precision-recall curves, a feature-weight bar chart (with an explicit caution against reading weight magnitude as feature importance under multicollinearity), and a side-by-side custom-vs-sklearn evaluation
- **Hypothesis Testing** — four pre-registered hypotheses checked explicitly against final results

Key issues encountered and resolved during the project:

- **The regularization (`C`) sweep was deliberately evaluated on an internal validation split carved out of the training set** — never on `X_test`/`y_test` — so the test set stays fully unseen during model selection, the same test-set discipline enforced in the rest of the portfolio series
- **`ClassifierMixin`/`BaseEstimator` inheritance order mattered**: the sklearn-compatibility wrapper (`SKLearnCompatibleSVM`) requires `ClassifierMixin` to come *before* `BaseEstimator` in the class definition — with the reverse order, scikit-learn's current tag-resolution system breaks `cross_validate` outright, not just silently misbehaves
- **Hypothesis H4 ("ensembles slightly outperform linear SVM") was only conditionally confirmed, not confirmed outright** — on the 5-fold CV benchmark, Logistic Regression actually ranked #1 by ROC-AUC ahead of every ensemble method, and the Custom SVM itself outranked Random Forest, Gradient Boosting, and XGBoost; the honest read is "some models beat linear SVM, but not consistently the ensembles," not the cleaner story the hypothesis predicted

---

## Repository Structure

```
08. SVM_breast_cancer
├── data/
│   └── .gitk
├── notebook/
│   └── SVM_From_Scratch_Breast_Cancer.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via scikit-learn's built-in Breast Cancer Wisconsin dataset inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/08. SVM_breast_cancer"
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
| `numpy` | Subgradient descent, hinge-loss/margin computation, support-vector identification |
| `pandas` | Data manipulation, feature/target framing, descriptive statistics |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots, correlation heatmap |
| `scikit-learn` | Dataset loader, `StandardScaler`, PCA, validation-phase SVC (linear/RBF), benchmark models, cross-validation utilities |
| `xgboost` | Comparative benchmark model |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `C` (regularization) | 10 |
| `λ = 1/C` | 0.1 |
| Learning rate `η` | 0.01 |
| Epochs `T` | 1000 |
| Batch size | 32 |
| Support-vector tolerance `ε` | 0.05 |

Selected via a regularization sweep over `C ∈ {0.01, 0.1, 1, 10, 100}` on an internal validation split, balancing accuracy and malignant recall equally (`0.5·accuracy + 0.5·recall`) rather than maximizing raw accuracy alone, given the clinical cost of a missed malignant case.

---

## Limitations

- **Linear kernel only**: the custom implementation supports the primal linear SVM. Non-linear boundaries (RBF, polynomial) require switching to the dual form and the kernel trick, which this implementation doesn't attempt.
- **Solver quality, not correctness, is the real gap to sklearn**: the custom SVM matched sklearn's Linear SVC exactly on every test-set metric (0.00 pp gap), but its 5-fold CV wall time (6.70s) was roughly 15–17× slower than sklearn's SVC implementations (0.35–0.43s), since LibSVM solves the dual QP via SMO while this implementation uses primal mini-batch subgradient descent.
- **Probability calibration is a rough approximation**: `predict_proba` uses a sigmoid transform of the decision score, not true Platt scaling (a cross-validated logistic fit on decision scores, as sklearn does internally).
- **Multicollinearity limits weight interpretability**: radius/perimeter/area and mean/worst versions of the same measurement are highly correlated, so individual feature-weight magnitudes can't be read as clean feature importances — the optimizer may arbitrarily split weight among correlated features.
- **Small, clean, single-domain dataset (569 samples)**: no missing values, no noisy features, and PCA already shows strong linear separability — an ideal dataset for illustrating the algorithm's mechanics, not a stress test of its robustness on noisy or highly non-linear data.
- **H4 was only conditionally confirmed**: the assumption that ensemble methods outperform a linear SVM didn't hold cleanly on this dataset — Logistic Regression and the Custom SVM itself outranked all three ensemble methods on 5-fold CV ROC-AUC.

---

## Results

**Initial model (`C=1`, default sweep starting point) — test set:**

| Metric | Value |
|---|---|
| Accuracy | 0.9561 |
| Balanced Accuracy | 0.9405 |
| Precision (malignant) | 1.0000 |
| Recall (malignant) | 0.8810 |
| F1 (malignant) | 0.9367 |
| ROC-AUC | 0.9940 |
| Support Vectors | 177 (38.9% of training set) |

**Final model (`C=10`, selected via sweep) — test set:**

| Metric | Value |
|---|---|
| Accuracy | 0.9825 |
| Balanced Accuracy | 0.9812 |
| Precision (malignant) | 0.9762 |
| Recall (malignant) | 0.9762 |
| F1 (malignant) | 0.9762 |
| ROC-AUC | 0.9954 |
| Support Vectors | 86 |

**Baseline — most-frequent-class dummy classifier:** 63.2% accuracy, 0.0% malignant recall (misses every cancer case), confirming accuracy alone is clinically insufficient.

**Custom SVM vs. sklearn, matched hyperparameters (`C=10`, linear kernel), test set:**

| Metric | Custom SVM | sklearn Linear SVC | sklearn RBF SVC |
|---|---|---|---|
| Accuracy | 0.9825 | 0.9825 | 0.9737 |
| Balanced Accuracy | 0.9812 | 0.9812 | 0.9742 |
| Precision (malignant) | 0.9762 | 0.9762 | 0.9535 |
| Recall (malignant) | 0.9762 | 0.9762 | 0.9762 |
| F1 (malignant) | 0.9762 | 0.9762 | 0.9647 |
| ROC-AUC | 0.9954 | 0.9954 | 0.9957 |
| Support Vectors | 86 | 26 | 84 |

**Performance gap, Custom vs. sklearn Linear SVC: 0.0000 (0.00 pp) on every metric,** the custom implementation matches sklearn's production solver exactly on the held-out test set.

**5-fold stratified CV of the custom SVM** (metrics computed for the malignant class, `pos_label=0`):

| Metric | Test (mean ± std) | Train (mean) |
|---|---|---|
| Accuracy | 0.9684 ± 0.0189 | 0.9789 |
| Balanced Accuracy | 0.9596 ± 0.0262 | 0.9727 |
| Precision | 0.9905 ± 0.0117 | 0.9951 |
| Recall | 0.9248 ± 0.0560 | 0.9481 |
| F1 | 0.9554 ± 0.0278 | 0.9710 |
| ROC-AUC | 0.9951 ± 0.0061 | 0.9961 |

**5-fold CV benchmark — full model comparison (sorted by ROC-AUC):**

| Rank | Model | Accuracy | Bal. Acc. | ROC-AUC | CV Time (s) |
|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.9737 ± 0.0166 | 0.9676 | 0.9953 | 0.35 |
| 2 | **Custom SVM** | **0.9684 ± 0.0189** | **0.9596** | **0.9951** | **6.70** |
| 3 | SVC (RBF) | 0.9789 ± 0.0119 | 0.9754 | 0.9945 | 0.41 |
| 4 | AdaBoost | 0.9666 ± 0.0151 | 0.9620 | 0.9939 | 7.54 |
| 5 | XGBoost | 0.9649 ± 0.0096 | 0.9615 | 0.9931 | 5.63 |
| 6 | Gradient Boosting | 0.9561 ± 0.0241 | 0.9479 | 0.9929 | 9.14 |
| 7 | SVC (Linear) | 0.9631 ± 0.0244 | 0.9592 | 0.9914 | 0.43 |
| 8 | Random Forest | 0.9543 ± 0.0102 | 0.9503 | 0.9896 | 5.53 |
| 9 | Naive Bayes | 0.9385 ± 0.0235 | 0.9280 | 0.9887 | 0.16 |
| 10 | K-Nearest Neighbors | 0.9649 ± 0.0184 | 0.9559 | 0.9863 | 5.58 |
| 11 | Decision Tree | 0.9104 ± 0.0279 | 0.9000 | 0.9021 | 0.26 |

The custom implementation ranks **2nd of 11 models on ROC-AUC**, outranking every ensemble method (Random Forest, Gradient Boosting, XGBoost, AdaBoost), while taking ~15–20× longer to cross-validate than sklearn's optimized SVC implementations.

**Hyperparameter sweeps (learning rate, `C`, epochs):**

- **Learning rate:** all three tested rates (η = 0.1, 0.01, 0.001) converged to essentially the same solution (accuracy 0.9474–0.9561, malignant recall 0.8810 across the board) at 1000 epochs, indicating the objective's convexity makes the final solution insensitive to learning rate at this scale.
- **Regularization (`C`), on the internal validation split:** accuracy and malignant recall both rose sharply from `C=0.01` (0.6264 acc / 0.0 recall / 355 SVs) through `C=10` (0.9780 acc / 0.9412 recall / 66 SVs), then accuracy dipped slightly at `C=100` (0.9670 acc / 36 SVs) even as ROC-AUC kept climbing (0.9985) — consistent with the classic narrow-margin overfitting risk at very high `C`. `C=10` was selected as the best accuracy/recall balance.
- **Epochs:** accuracy, recall, and final loss were already stable at 100 epochs (0.9561 / 0.8810 / loss 0.2651) and stayed essentially unchanged through 5000 epochs, while training time scaled linearly from 0.18s to 7.2s — confirming 1000 epochs is sufficient and additional epochs buy nothing but wall time.

**Hypothesis Evaluation:**

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: Linear SVM will achieve strong performance | ✓ Confirmed | PCA showed strong linear separability (PC1 44.3% + PC2 19.0% = 63.2% variance); linear SVM achieved 0.9954 test ROC-AUC |
| H2: Feature standardization improves optimization | ✓ Confirmed | High coefficient-of-variation across features made scaling necessary for the `½‖w‖²` objective to converge stably |
| H3: Custom SVM approaches sklearn SVC within acceptable margin | ✓ Confirmed | Performance gap vs. sklearn Linear SVC was 0.00 pp on every test-set metric |
| H4: Ensemble models slightly outperform linear SVM | Conditionally confirmed | SVC (RBF) edged out the Custom SVM on accuracy, but Logistic Regression and the Custom SVM itself outranked all three ensemble methods on CV ROC-AUC |

---

## What I learned:

1. **Matching Sklearn Exactly Validates the Objective, Not Just the Code.**

Getting a 0.00 pp gap against sklearn's Linear SVC on every single test-set metric (accuracy, precision, recall, F1, ROC-AUC) was a stronger signal than "close enough" — since both implementations solve the identical convex soft-margin objective, an exact match confirms there's no subtle bug (a wrong sign in the subgradient, a mishandled kink, an off-by-one in the margin normalization) hiding anywhere in the derivation-to-code translation.

2. **Solver Quality and Conceptual Correctness Are Different Axes.**

The custom SVM matched sklearn's accuracy exactly, but took roughly 15–20x longer per cross-validation fold (6.70s vs. 0.35–0.43s). Primal mini-batch subgradient descent and LibSVM's dual SMO solver reach the same optimum, but SMO gets there far more efficiently. Correctness and speed are separate things to verify, and matching on one says nothing about the other.

3. **`C` Is the Lever, Learning Rate Is Mostly Noise (Given Enough Epochs).**

The learning-rate sweep barely moved final accuracy across two orders of magnitude of η, while the `C` sweep swung malignant recall from 0.0 to 0.94 and accuracy from 0.63 to 0.98. For a convex objective at 1000 epochs, learning rate mostly changes how fast you get to the optimum; `C` changes what the optimum actually is.

4. **An Honest Hypothesis Result Beats a Tidy One.**

The pre-registered hypothesis that ensembles would edge out the linear SVM only held loosely: Logistic Regression, not an ensemble, ranked #1 by ROC-AUC, and the Custom SVM itself beat Random Forest, Gradient Boosting, and XGBoost. Marking H4 "conditionally confirmed" rather than forcing a clean confirm/deny kept the actual CV table as the source of truth instead of the hypothesis's prior expectation.

5. **A Silent MRO Bug Is Worse Than a Loud One.**

Getting the base-class order backwards in the sklearn-compatibility wrapper (`BaseEstimator` before `ClassifierMixin` instead of after) didn't produce subtly wrong cross-validation scores — it broke `cross_validate` outright via scikit-learn's tag-resolution system. A hard failure at the wrong inheritance order was actually the better outcome; a quieter failure mode there would have been much easier to miss.

6. **A Wide Margin Isn't Automatically the "Safe" Choice in a Clinical Setting.**

Small `C` (heavy regularization, wide margin) sounds like the conservative choice, but on the validation sweep it produced 0.0 malignant recall at `C=0.01` — the wide margin came from a degenerate, nearly unweighted decision boundary, not from genuinely confident separation. The margin's width only means something once the classifier is actually discriminating between classes.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)