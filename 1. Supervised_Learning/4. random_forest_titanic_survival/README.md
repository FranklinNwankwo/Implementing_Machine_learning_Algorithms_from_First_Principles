# Random Forest from Scratch — Titanic Survival Prediction

An end-to-end machine learning project implementing a Random Forest classifier using only NumPy and Pandas, applied to the [Titanic survival dataset](https://www.kaggle.com/c/titanic).

Built to understand the fundamentals of ensemble learning — bagging, random feature subsampling, and Out-of-Bag estimation, without relying on high-level abstractions.

---

## Project Overview

A single Decision Tree trained to low bias tends to overfit: small perturbations in the training data produce substantially different trees — a **high-variance** model. Random Forest addresses this through three mechanisms, all implemented from scratch here:

1. **Bootstrap Aggregation (Bagging)** — each tree trains on a bootstrap sample drawn with replacement, giving trees overlapping but non-identical data
2. **Random Feature Subsampling** — only a random subset of features is considered at each split, decorrelating individual trees
3. **Majority / Probability Voting** — final predictions aggregate all trees' outputs, smoothing out individual errors

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — target distribution, survival by categorical features, Sex×Pclass interaction, numerical feature distributions, correlation analysis
- **Feature Engineering** — title extraction from `Name` via regex, family-size features, cabin-missingness and deck features, ticket-group size and fare-per-person, domain-informed age/fare binning
- **Preprocessing** — stratified three-way split, training-set-only imputation, log1p(Fare) transform, and a manual one-hot encoder class (no `sklearn.preprocessing`, no `pd.get_dummies`)
- **Model Implementation** — a `DecisionTreeClassifier` (Gini impurity, random feature subsampling) and a `RandomForestClassifier` (bagging, majority/probability voting, OOB scoring) both built from scratch with NumPy
- **Hyperparameter Sensitivity Analysis** — `n_estimators`, `max_depth`, `max_features`, and `min_samples_leaf` sweeps, each checked against theoretical bias-variance predictions
- **Validation** — 5-fold stratified cross-validation, decision-threshold optimisation (F1-optimal τ*), and a custom-vs-sklearn parity check
- **Comparative Modeling** — benchmarked against sklearn's Random Forest, Gradient Boosting, AdaBoost, SVM (RBF), and KNN
- **Diagnostics** — feature importance (Mean Decrease in Gini), an ablation study on each feature-engineering step, learning curves, and subgroup error analysis (by Sex, Pclass, Title, FamilyGroup)

Key issues encountered and resolved during the project:

- High-cardinality categorical encoding (`Cabin`, `Ticket`) risking sparse, low-frequency split candidates — resolved by encoding missingness and extracting compact derived features (deck letter, group size) instead of raw values
- Data leakage risk from imputation and one-hot vocabulary being fit anywhere other than the training set
- Distinguishing genuine ensemble gains from variance an individual tree would show under a different random seed — addressed with an explicit single-tree-vs-forest variance experiment
- Known Mean-Decrease-in-Gini bias toward high-cardinality features, flagged explicitly when interpreting feature importances

---

## Repository Structure

```
4. random_forest_titanic_survival/
├── data/
│   └── .gitkeep
├── notebook/
│   └── random_forest (Titanic_Survival).ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via this url: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/4. random_forest_titanic_survival"
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
| `numpy` | Linear algebra, Gini impurity, bootstrap sampling, tree/forest construction |
| `pandas` | Data manipulation, feature engineering, EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | Validation-phase benchmark models (Random Forest, GBM, AdaBoost, SVM, KNN) and cross-validation utilities |

---

## Limitations

- **`Cabin` is ~77% missing**: rather than imputing, missingness itself is encoded as a signal (`HasCabin`) and a `Deck` feature is extracted only where known — the majority of cabin information for this dataset is simply unrecoverable.
- **Mean-Decrease-in-Gini importance bias**: MDI is known to be biased toward high-cardinality features, so the feature-importance ranking should be read as directionally informative, not as a precise ranking.
- **Small dataset (~891 passengers)**: after a three-way stratified split, the validation and test sets are small enough that subgroup error analysis (e.g. by `Title` or `FamilyGroup`) can be noisy for the smallest subgroups.
- **No native handling of unseen categories beyond all-zero rows**: the manual one-hot encoder maps unseen validation/test categories to an all-zero indicator row rather than a dedicated "unknown" category, which slightly under-represents genuinely novel categories.
- **Historical, single-event dataset**: the model captures evacuation-priority patterns specific to this ship and disaster; it has no claim to generalising beyond Titanic-like scenarios.

---

## Results

Validation-set performance across the custom Random Forest, sklearn's Random Forest, and the comparison models (Gradient Boosting, AdaBoost, SVM, KNN), sorted by F1 score:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 0.8496 | 0.7818 | 0.8431 | 0.8113 | 0.8611 |
| Gradient Boosting | 0.8571 | 0.8810 | 0.7255 | 0.7957 | 0.9376 |
| sklearn Random Forest | 0.8421 | 0.8000 | 0.7843 | 0.7921 | 0.9065 |
| **Custom Random Forest** | **0.8421** | **0.8000** | **0.7843** | **0.7921** | **0.9075** |
| AdaBoost | 0.8271 | 0.7692 | 0.7843 | 0.7767 | 0.8980 |
| KNN (k=5) | 0.8195 | 0.7755 | 0.7451 | 0.7600 | 0.8615 |

**Custom-vs-sklearn parity check** (matched hyperparameters, validation set): `|Δ Accuracy| = 0.00%`, `|Δ F1| = 0.0000`, `|Δ ROC-AUC| = 0.0010`. OOB accuracy is likewise close — 82.66% (custom) vs. 83.15% (sklearn) — a particularly strong correctness signal since OOB estimation depends on the bootstrap-sample tracking logic being right.

**Final model summary** (custom Random Forest, tuned hyperparameters):

| Metric | Value |
|---|---|
| OOB Accuracy | 82.66% |
| 5-Fold CV Accuracy | 83.47% ± 1.98% |
| Validation Accuracy (τ=0.5) | 84.21% |
| Validation F1 | 0.7921 |
| Validation ROC-AUC | 0.9075 |
| Test Accuracy (τ=0.5) | 80.00% |
| Test F1 | 0.7379 |
| Test ROC-AUC | 0.8255 |
| Optimal threshold τ* | 0.48 |
| Test Accuracy (τ*=0.48) | 80.00% |

Gradient Boosting and SVM edge out the Random Forest on raw metrics (highest F1 and ROC-AUC respectively), and the custom implementation tracks sklearn's Random Forest almost exactly, the small residual ROC-AUC gap (0.0010) is consistent with split-threshold sampling and floating-point aggregation differences rather than an implementation bug. The gap between validation accuracy (84.21%) and test accuracy (80.00%) is within the expected noise for a ~135-sample test set and is not evidence of overfitting on its own, given the close OOB/CV/validation agreement above.

---

## What I learned:

1. **OOB Scoring Is Cross-Validation You Get for Free.**

Roughly 36.8% of training samples are excluded from any given tree's bootstrap sample — and aggregated across the whole forest, virtually every sample ends up out-of-bag for a meaningful fraction of trees. Implementing OOB scoring meant the forest carries its own internal generalisation estimate at zero extra computational cost, which turned out to be more useful during hyperparameter search than repeatedly re-checking against a held-out validation set.

2. **Variance Reduction Isn't Just a Claim — It's Directly Measurable.**

Rather than taking bagging's variance-reduction benefit on faith, running the same single Decision Tree across many random seeds and comparing its accuracy spread against the forest's accuracy spread made the effect concrete: the single tree's accuracy swings noticeably with the seed, while the forest's does not. Seeing $\text{Var}(\bar{T}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ show up as an actual, visible flattening of a distribution — not just a formula — was the most convincing part of the whole implementation.

3. **The `n_estimators` Plateau Is the Bias-Variance Formula Showing Its Face.**

Accuracy climbs quickly as trees are added, then flattens. That's not a coincidence — it's the $1/B$ term in the variance formula shrinking toward zero while the correlated $\rho\sigma^2$ term stays put, since it doesn't depend on $B$ at all. The OOB curve tracking the validation curve closely throughout the sweep was a good secondary confirmation that OOB estimation could be trusted as a proxy during the sweep itself, not just after the fact.

4. **`max_depth` Sensitivity Reveals the Bias-Variance Trade-off More Clearly Than Any Single Metric.**

Watching the train-minus-OOB gap widen as `max_depth` increases — while validation accuracy peaks in the middle rather than at the extremes — made the abstract bias-variance trade-off into something directly readable off a chart, rather than something to reason about only in the abstract.

5. **`max_features='sqrt'` Isn't an Arbitrary Default.**

Sweeping `max_features` across `log2`, `sqrt`, and using all features showed the diversity-accuracy trade-off directly: too few candidate features per split risks weak splits when the randomly-chosen subset happens to be uninformative; using all features collapses back toward a plain bagged-tree ensemble as trees become more correlated. Seeing `sqrt` land near the empirical peak for this dataset — matching what's typically cited as a sensible classification default — was a useful check against just trusting the convention.

6. **Feature Engineering From `Name` and `Ticket` Mattered More Than Expected.**

The ablation study quantifying each feature-engineering step's incremental contribution showed `Title` (extracted via regex from `Name`) as one of the largest single gains — it cleanly separates social role, gender, and age-proxy signal that raw `Sex` and `Age` only partially encode on their own. That the single biggest lever wasn't a modelling choice but a feature-engineering one was the most transferable lesson of the whole project.

7. **A Manual One-Hot Encoder Forces You to Think About the Train/Val/Test Contract Explicitly.**

Writing the encoder's vocabulary-fitting logic by hand — rather than calling `pd.get_dummies` on the full dataset — made the leakage risk unavoidable to confront directly: the vocabulary has to come from the training set only, and validation/test categories not seen in training have to degrade gracefully (here, to an all-zero row) rather than silently creating new columns.

8. **Threshold Tuning and Subgroup Analysis Together Catch What Aggregate Metrics Hide.**

A single F1-optimal threshold and a single aggregate accuracy number can both look healthy while masking uneven performance across passenger subgroups. Running subgroup error analysis by Sex, Pclass, Title, and FamilyGroup — rather than stopping at the aggregate confusion matrix — surfaced where the model's confidence was and wasn't well-calibrated, which is a habit worth carrying into every future classification project, not just this one.

9. **Matching sklearn Isn't Just a Sanity Check — It's a Debugging Tool.**

Comparing the custom Random Forest against sklearn's implementation with matched hyperparameters, and explicitly tracing the residual gap to split-threshold sampling and floating-point aggregation differences rather than treating "close enough" as good enough, meant any larger discrepancy would have been a signal to go back and re-check the bootstrap or split-selection logic rather than something to explain away.

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)