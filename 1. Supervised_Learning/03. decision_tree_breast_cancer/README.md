# Decision Tree from Scratch — Breast Cancer Wisconsin Classification

An end-to-end machine learning project implementing a CART-style Decision Tree classifier using only NumPy, applied to the [Breast Cancer Wisconsin Diagnostic dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset).

Built to understand the fundamentals of Gini-impurity-based recursive splitting without relying on high-level abstractions.

---

## Project Overview

This is a binary classification problem — **Malignant** (positive class, cancer present) vs. **Benign** (negative class). Because a missed cancer diagnosis carries extremely high clinical risk, **Recall on the Malignant class is the primary optimisation target throughout, not overall accuracy.**

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — class distribution, correlation heatmap, feature distributions by class, outlier inspection, pairplots on the most separable features.
- **Mathematical Foundations** — Gini impurity, weighted Gini for a split, information gain, and a worked numerical example, plus a Gini-vs-Entropy comparison.
- **Model Implementation** — a full CART-style tree (`Node` class, recursive Gini-impurity splitting, threshold-midpoint search, `predict_proba`, and Gini-reduction-based feature importances) built from scratch with NumPy.
- **Hyperparameter Sensitivity Analysis** — `max_depth`, `min_samples_split`, and `min_samples_leaf` sweeps, plus 5-fold stratified cross-validation for stability.
- **Comparison Models** — benchmarked against `sklearn`'s `DecisionTreeClassifier`, Random Forest, XGBoost, Logistic Regression, and SVM (RBF).
- **Final Evaluation** — ROC and Precision-Recall curves, confusion matrices, cost-complexity (post-)pruning, and decision-threshold optimisation for recall.

Key issues encountered and resolved during the project:

- sklearn's `load_breast_cancer` encodes labels as `0 = Malignant, 1 = Benign`, flipped to the clinical convention (`1 = Malignant`) so Recall/Precision/F1 are computed with Malignant as the positive class
- Choosing Recall over accuracy as the primary metric, since a naive "always Benign" classifier already scores ~62% accuracy
- Validating that the scratch tree's minor performance gap versus sklearn comes from threshold-enumeration and tie-breaking differences, not implementation bugs.
- Balancing interpretability against ensemble performance for a clinical-deployment recommendation

---

## Repository Structure

```
03. decision_tree_breast_cancer/
├── data/
│   └── .gitkeep
├── notebook/
│   └── Decision_tree_(Breast_cancer_dataset).ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via `sklearn.datasets.load_breast_cancer()` inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/03. decision_tree_breast_cancer"
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
| `numpy` | Linear algebra, Gini impurity, recursive tree splitting |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scipy` | Spearman correlation for feature-importance validation |
| `xgboost` | Gradient-boosted ensemble benchmark |
| `scikit-learn` | Dataset loading, splitting, metrics, and benchmark models |

---

## Limitations

- **Exhaustive threshold search is O(n·d) per split**: the scratch tree evaluates every midpoint between sorted unique feature values at every node, which doesn't scale to very large datasets the way sklearn's optimised Cython implementation does.
- **No native multi-class support**: the implementation is built and validated for binary classification only.
- **Small test set (15%, ~85 samples)**: several models including Logistic Regression reach 100% test accuracy, which likely reflects the dataset's strong class separability and limited test-set size rather than guaranteed generalisation to unseen clinical data.
- **Post-pruning is exploratory only**: cost-complexity pruning is demonstrated via sklearn's `ccp_alpha` path for theoretical illustration, not implemented from scratch or applied to the final scratch tree.
- **No external validation cohort**: all evaluation is on splits of the same single-institution dataset; no true out-of-distribution or multi-site validation is performed.

---

## Results

Test-set performance across all six models:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Scratch DT (Tuned) | 0.9535 | 0.9118 | 0.9688 | 0.9394 | 0.9754 | 0.9687 | 1 | 3 |
| sklearn DT (Tuned) | 0.9535 | 0.9118 | 0.9688 | 0.9394 | 0.9754 | 0.9687 | 1 | 3 |
| Random Forest | 0.9884 | 1.0000 | 0.9688 | 0.9841 | 1.0000 | 1.0000 | 1 | 0 |
| XGBoost | 0.9767 | 1.0000 | 0.9375 | 0.9677 | 0.9994 | 0.9991 | 2 | 0 |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** | **0** |
| SVM (RBF) | 0.9767 | 1.0000 | 0.9375 | 0.9677 | 1.0000 | 1.0000 | 2 | 0 |

The scratch Decision Tree matches sklearn's `DecisionTreeClassifier` exactly on every metric, confirming the from-scratch CART implementation is correct. Logistic Regression achieves a perfect test score with zero false negatives, but for **clinical deployment** the recommendation is the tuned Decision Tree with a lowered decision threshold (0.3–0.4). It trades a small amount of raw performance for fully auditable, clinician-readable if-then decision rules, which matters for regulatory and clinical-trust reasons that a linear model's coefficients or an ensemble's aggregate votes don't provide.

---

## What I learned:

1. **Matching sklearn Exactly Is a Stronger Validation Than "Close Enough."**

The scratch tree didn't just come close to sklearn's `DecisionTreeClassifier`, it matched it on every single metric (0.9535 accuracy, 0.9688 recall, identical FN/FP counts). That level of agreement is a much stronger signal than a small performance gap would have been, because it rules out subtle bugs in the Gini computation, the split-selection logic, or the recursive tree-building itself, rather than just showing the two models land in a similar performance neighbourhood by coincidence.

2. **The Positive-Class Encoding Is Not a Cosmetic Choice.**

sklearn's `load_breast_cancer` ships with `0 = Malignant, 1 = Benign`, which is backwards from clinical convention and, more importantly, backwards from how Recall, Precision, and the confusion matrix get interpreted. Flipping the encoding before any modelling started meant every downstream metric, especially which class "Recall" refers to was unambiguous for the rest of the notebook, instead of requiring a mental relabelling every time a result was read.

3. **Optimising for Accuracy Would Have Been the Wrong Default.**

With a ~62:38 Benign-to-Malignant split, a classifier that always predicts "Benign" already scores 62% accuracy while catching zero cancers. Naming Recall on the Malignant class as the primary metric from the start before writing a single line of tree code meant every later decision (threshold selection, hyperparameter tuning, model comparison) had a consistent yardstick, rather than accuracy quietly creeping back in as the tie-breaker.

4. **`np.bincount` Over `Counter` Is a Real Performance Decision, Not Premature Optimisation.**

Gini impurity gets computed at every candidate split during tree building, which for exhaustive threshold search across 30 features means millions of calls. Choosing `np.bincount` (O(n), no hashing) instead of Python's `Counter` for the class-count step wasn't a micro-optimisation for its own sake. It's the difference between a tree that builds in a reasonable time and one that doesn't, once the candidate-threshold count scales up.

5. **Threshold Midpoints Are Mathematically Equivalent to Raw Values, but Practically Better.**

Rather than testing every raw feature value as a candidate split point, the scratch tree computes midpoints between consecutive sorted unique values. This halves the number of candidates to evaluate and removes an entire class of boundary-ambiguity bugs (does `x <= v` or `x < v` put a tied value on the left or right?), a good example of a mathematically neutral choice that meaningfully simplifies the implementation.

6. **Ensemble Gains Come at a Real Interpretability Cost.**

Random Forest and XGBoost both reach 1.0000 ROC-AUC, and Random Forest gets zero false positives, genuinely better raw numbers than the single tree. But neither produces a human-readable decision path the way a single tree's if-then rules do. Seeing the ensemble's numerical advantage sit right next to its interpretability cost, rather than treating "better metrics" as an automatic win, is what shaped the eventual recommendation for a tuned single tree in the clinical-deployment scenario.

7. **A Perfect Score Is a Prompt to Check the Test Set, Not a Finish Line.**

Logistic Regression hitting 1.0000 on every single metric is a good result, but with only ~85 test samples and strong class separability in this dataset, a perfect score is at least as likely to reflect the small evaluation set as it is to reflect flawless generalisation. Treating a perfect number as something to interrogate rather than something to simply report was a useful instinct to build here, especially with a small clinical dataset.

8. **Threshold Tuning Applies to Trees Just as Much as to Logistic Regression.**

`predict_proba()` on the scratch tree returns the leaf's training class distribution, which meant the same threshold-sweep-for-recall approach used in earlier projects (in the fraud-detection logistic regression notebook) applied directly here too. Lowering the operating threshold below 0.5 trades some precision for higher recall, the right trade in a screening context where a missed malignancy is far costlier than a false alarm that gets ruled out on follow-up.

9. **Feature Importance Cross-Validated Against a Second Model Is More Trustworthy Than Feature Importance Alone.**

Comparing the scratch tree's Gini-reduction-based feature importances against Random Forest's importances (and against Spearman correlation with the target) meant the top-ranked features weren't just an artefact of one particular tree's split order, they showed up as informative across multiple independent methods, which is a stronger basis for any downstream claim about which measurements matter clinically.

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)