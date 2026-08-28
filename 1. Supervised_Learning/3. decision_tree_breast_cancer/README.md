# Random Forest from Scratch — Titanic Survival Prediction

An end-to-end machine learning project implementing a Random Forest classifier using only NumPy and Pandas, applied to the [Titanic survival dataset](https://www.kaggle.com/c/titanic).

Built to understand the fundamentals of ensemble learning — bagging, random feature subsampling, and Out-of-Bag estimation without relying on high-level abstractions.

---

## Project Overview

A single Decision Tree trained to low bias tends to overfit: small perturbations in the training data produce substantially different trees, a **high-variance** model. Random Forest addresses this through three mechanisms, all implemented from scratch here:

1. **Bootstrap Aggregation (Bagging):** Each tree trains on a bootstrap sample drawn with replacement, giving trees overlapping but non-identical data
2. **Random Feature Subsampling:** Only a random subset of features is considered at each split, decorrelating individual trees
3. **Majority / Probability Voting:** Final predictions aggregate all trees' outputs, smoothing out individual errors

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — target distribution, survival by categorical features, Sex × Pclass interaction, numerical feature distributions, correlation analysis
- **Feature Engineering** — title extraction from `Name` via regex, family-size features, cabin-missingness and deck features, ticket-group size and fare-per-person, domain-informed age/fare binning
- **Preprocessing** — stratified three-way split, training-set-only imputation, log1p(Fare) transform, and a manual one-hot encoder class (no `sklearn.preprocessing`, no `pd.get_dummies`)
- **Model Implementation** — a `DecisionTreeClassifier` (Gini impurity, random feature subsampling) and a `RandomForestClassifier` (bagging, majority/probability voting, OOB scoring) both built from scratch with NumPy
- **Hyperparameter Sensitivity Analysis** — `n_estimators`, `max_depth`, `max_features`, and `min_samples_leaf` sweeps, each checked against theoretical bias-variance predictions
- **Validation** — 5-fold stratified cross-validation, decision-threshold optimisation (F1-optimal τ*), and a custom-vs-sklearn parity check
- **Comparative Modeling** — benchmarked against sklearn's Random Forest, Gradient Boosting, AdaBoost, SVM (RBF), and KNN
- **Diagnostics** — feature importance (Mean Decrease in Gini), an ablation study on each feature-engineering step, learning curves, and subgroup error analysis (by Sex, Pclass, Title, FamilyGroup)

Key issues encountered and resolved during the project:

- High-cardinality categorical encoding (`Cabin`, `Ticket`) risking sparse, low-frequency split candidates, resolved by encoding missingness and extracting compact derived features (deck letter, group size) instead of raw values.
- **Data leakage from features computed before the train/val/test split**: `FareBin`'s quartile edges and `TicketGroupSize`/`FarePerPerson` were originally derived from the full pre-split dataset. Both were rebuilt to fit exclusively on the training partition, with unseen validation/test values (fares above the training max, tickets not seen in training) handled explicitly via clipping and sensible defaults rather than silently leaking split-boundary information.
- **A hard-vote / soft-average inconsistency between `predict()` and `predict_proba()`**: the original `RandomForestClassifier.predict()` majority-voted each tree's discrete label, which isn't guaranteed to agree with thresholding the averaged probability from `predict_proba()`. `predict()` now thresholds the averaged probability directly, matching sklearn's own convention and tightening custom-vs-sklearn parity to `|Δ Accuracy| = 0.00%`.
- Distinguishing genuine ensemble gains from variance an individual tree would show under a different random seed
- Known Mean-Decrease-in-Gini bias toward high-cardinality features, flagged explicitly when interpreting feature importances

---

## Repository Structure

```
4. random_forest/
├── data/
│   └── .gitkeep
├── notebook/
│   └── random_forest.ipynb
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

- **`Cabin` is ~77% missing**: rather than imputing, missingness itself is encoded as a signal (`HasCabin`) and a `Deck` feature is extracted only where known, the majority of cabin information for this dataset is simply unrecoverable.
- **Mean-Decrease-in-Gini importance bias**: MDI is known to be biased toward high-cardinality features, so the feature-importance ranking should be read as directionally informative, not as a precise ranking.
- **Small dataset (~891 passengers)**: the validation and test sets are only 133 and 135 rows respectively. This shows up directly in the results below, a single-point ablation or hyperparameter sweep on a set this size can move by a couple of accuracy points from ordinary sampling noise alone, not just from the change being tested.
- **No native handling of unseen categories beyond all-zero rows**: the manual one-hot encoder maps unseen validation/test categories to an all-zero indicator row rather than a dedicated "unknown" category, which slightly under-represents genuinely novel categories.
- **Historical, single-event dataset**: the model captures evacuation-priority patterns specific to this ship and disaster; it has no claim to generalising beyond Titanic-like scenarios.

---

## Results

All figures below are the notebook's own printed output (`SEED=42`), taken after fixing the leakage and `predict()` issues described above, not estimates.

### Final model

| Metric | Value | Note |
|---|---|---|
| OOB Accuracy | 82.66% | Training-set internal estimate |
| 5-Fold CV Accuracy | 83.47% ± 2.21% | Generalisation stability |
| Val Accuracy (τ=0.5) | 84.21% | Held-out validation |
| Val F1 Score | 0.7921 | |
| Val ROC-AUC | 0.9075 | |
| Val PR-AUC | 0.8952 | Swept over every unique validation probability |
| **Test Accuracy (τ=0.5)** | **80.00%** | Final unbiased estimate |
| Test F1 Score | 0.7379 | |
| Test ROC-AUC | 0.8255 | |
| Optimal Threshold τ* | 0.48 | Max F1 on validation set — coincides with τ=0.5 on this test run |

Final hyperparameters: `n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features='sqrt'`, chosen from the sensitivity sweeps below rather than by default.

### Custom vs. sklearn parity

| Metric | Custom | sklearn | \|Δ\| |
|---|---|---|---|
| Accuracy | 0.8421 | 0.8421 | 0.0000 |
| Precision | 0.8000 | 0.8000 | 0.0000 |
| Recall | 0.7843 | 0.7843 | 0.0000 |
| F1 | 0.7921 | 0.7921 | 0.0000 |
| ROC-AUC | 0.9075 | 0.9065 | 0.0010 |

OOB accuracy also agrees closely (custom 82.66% vs. sklearn 83.15%), which is a stronger correctness signal than accuracy parity alone, since OOB estimation depends on the bootstrap-tracking logic being right, not just the split-finding logic.

### Model comparison (validation set, sorted by F1)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 84.96% | 0.7818 | 0.8431 | **0.8113** | 0.8611 |
| Gradient Boosting | **85.71%** | 0.8810 | 0.7255 | 0.7957 | **0.9376** |
| sklearn Random Forest | 84.21% | 0.8000 | 0.7843 | 0.7921 | 0.9065 |
| Custom Random Forest | 84.21% | 0.8000 | 0.7843 | 0.7921 | 0.9075 |
| AdaBoost | 82.71% | 0.7692 | 0.7843 | 0.7767 | 0.8980 |
| KNN (k=5) | 81.95% | 0.7755 | 0.7451 | 0.7600 | 0.8615 |

5-fold CV accuracy tells a similar story: SVM (RBF) 83.87% ± 2.24%, Gradient Boosting 83.60% ± 2.09%, Custom RF 83.47% ± 1.98%, sklearn RF 82.94% ± 2.32%, AdaBoost 81.75% ± 2.23%, KNN 80.43% ± 2.43%.

**Random Forest is competitive but not dominant here.** Gradient Boosting edges it out on accuracy and ROC-AUC, and SVM (RBF) — with no hyperparameter tuning beyond defaults — edges it out on F1 and recall. That's a fair result for a ~890-row tabular dataset: it doesn't undercut the value of the from-scratch implementation, which is validated separately by how closely it tracks sklearn's own Random Forest, not by whether Random Forest happens to top this particular leaderboard.

### Ablation study — feature engineering contribution

| Config | Features Added | # Features | Val Accuracy | Val F1 |
|---|---|---|---|---|
| C0 — Raw baseline | `Pclass`, `Age`, `Fare`, `SibSp`, `Parch`, `Sex`, `Embarked` | 10 | 83.46% | 0.7660 |
| C1 — + Family | `FamilySize`, `IsAlone`, `FamilyGroup` | 15 | 82.71% | 0.7527 |
| C2 — + Title | `Title` | 20 | **84.96%** | **0.8000** |
| C3 — + Discretisation | `AgeBin`, `FareBin`, `IsChild` | 30 | 83.46% | 0.7708 |
| C4 — Full feature set | `HasCabin`, `Deck`, `TicketGroupSize`, `FarePerPerson` | 42 | 84.21% | 0.7921 |

`Title` (C1→C2) is the one unambiguous win, the largest single jump in both accuracy and F1, consistent with it cleanly separating social role, gender, and age-proxy signal that raw `Sex`/`Age` only partially encode. But Family features (C0→C1) and discretisation (C2→C3) both *decrease* validation performance relative to the step before them, and the final feature block (C3→C4) only partially recovers. On a 133-row validation set, single-point deltas of a percentage point or two are within the range of ordinary sampling noise, so the honest read is "Title clearly helps; the rest of the feature engineering is a wash or mildly negative on this particular split," not "each step adds incremental value." A cross-validated ablation (averaging the delta across folds rather than one split) would be the natural next step to get a more decisive answer on the smaller effects.

### Sensitivity sweeps

- **`n_estimators`**: rapid improvement from 5→50 trees (OOB 79.68%→82.66%), then a plateau through 200 trees (OOB settling around 82.7–83.0%), consistent with the $1/B$ term in the ensemble variance formula shrinking toward zero while the correlated component stays put.
- **`max_depth`**: the train−val gap widens steadily from 1.03% at depth 3 to 2.63% at depth 12/unconstrained, while validation accuracy peaks in the 5–8 range — the textbook bias-variance signature, motivating `max_depth=8` for the final model.
- **`max_features`**: `sqrt` (6 features) and `0.75` (31 features) both land near the top of the validation-accuracy range tested (`0.3`→12 features actually edges out `sqrt` slightly at 84.96% vs. 84.21%, well within this dataset's noise floor), while `log2` (5 features, too few for a strong split) and `1.0` (all 42 features, more correlated trees) are the weakest of the six configurations tried — supporting `sqrt` as a reasonable default without claiming it's uniquely optimal on this dataset.
- **`min_samples_leaf`**: acts as a soft regulariser; validation accuracy is essentially flat across leaf sizes 1–20 on this dataset, so the final choice of 5 was made for its interpretability (fewer near-single-sample leaves) rather than a decisive accuracy gain.

### Subgroup error analysis

Overall validation error rate: 15.79% (21 of 133 misclassified). Errors are broken down by Sex, Pclass, Title, and FamilyGroup in the notebook's subgroup plots, checking for systematic failure on specific passenger groups rather than relying on the aggregate rate alone, the standard concern being that a model can look healthy in aggregate while performing poorly on a specific subgroup that a single accuracy number would hide.

---

## What I learned:

1. **OOB Scoring Is Cross-Validation You Get for Free.**

Roughly 36.8% of training samples are excluded from any given tree's bootstrap sample, and aggregated across the whole forest, virtually every sample ends up out-of-bag for a meaningful fraction of trees. Implementing OOB scoring meant the forest carries its own internal generalisation estimate at zero extra computational cost, which turned out to be more useful during hyperparameter search than repeatedly re-checking against a held-out validation set, confirmed by how closely the OOB curve tracked validation accuracy throughout the `n_estimators` sweep (both settling in the low-to-mid 80s from 50 trees onward).

2. **A Variance-Reduction Experiment Is Only as Good as the Randomness It Actually Exercises.**

The initial single-tree-vs-forest comparison was meant to make bagging's variance-reduction benefit directly measurable rather than taken on faith, but training the "single tree" baseline on the full training set with no bootstrap resampling meant changing its random seed changed nothing: the same deterministic tree came out every time (Std Dev: 0.00%). The fix was to bootstrap-sample the single tree too, so the comparison isolates *ensembling* rather than accidentally comparing "no randomness" to "some randomness." With that corrected, across 20 seeds the single tree's accuracy has a real spread (79.92% ± 3.68%) against the forest's much tighter one (84.74% ± 1.07%), an **11.9× variance reduction**, and the forest also scores higher on average. That's the result the experiment was supposed to produce from the start; a variance experiment's baseline needs to actually vary before its output can be trusted.

3. **The `n_estimators` Plateau Is the Bias-Variance Formula Showing Its Face.**

Accuracy climbs quickly as trees are added, then flattens. That's not a coincidence, it's the $1/B$ term in the variance formula shrinking toward zero while the correlated $\rho\sigma^2$ term stays put, since it doesn't depend on $B$ at all.

4. **`max_depth` Sensitivity Reveals the Bias-Variance Trade-off More Clearly Than Any Single Metric.**

Watching the train-minus-OOB gap widen as `max_depth` increases, while validation accuracy peaks in the middle rather than at the extremes made the abstract bias-variance trade-off into something directly readable off a chart, rather than something to reason about only in the abstract.

5. **Feature Engineering From `Name` Mattered More Than the Rest of the Feature Engineering Combined.**

The ablation study's one unambiguous result was `Title`: the single largest gain in the whole sweep, and the only step whose improvement clearly exceeded what this dataset's sampling noise could plausibly produce on its own. That the family-size and discretisation features didn't show a similarly clean win, on a single validation split, at least, was a useful reminder not to write a smooth "every step helped a little" narrative just because that's the expected shape; the data doesn't always cooperate with the story, and a single 133-row validation split isn't enough to fully resolve the smaller effects.

6. **A Manual One-Hot Encoder Forces You to Think About the Train/Val/Test Contract Explicitly — And It's Easy to Get Half of It Right.**

Writing the encoder's vocabulary-fitting logic by hand made the leakage risk unavoidable to confront directly: the vocabulary has to come from the training set only, and validation/test categories not seen in training have to degrade gracefully (here, to an all-zero row) rather than silently creating new columns. But the encoder being correct didn't automatically mean every *feature feeding into it* was, `FareBin`'s bin edges and `TicketGroupSize`'s counts were both originally computed on the full pre-split dataset upstream of the encoder, which the encoder itself had no way to catch. The lesson generalises: a correct encoding step doesn't guarantee a leakage-free pipeline if something earlier in the chain already crossed the split boundary.

7. **Internal Consistency Within a Class Is Its Own Category of Bug.**

`RandomForestClassifier.predict()` (hard-voting each tree's label) and `predict_proba()` (averaging soft probabilities) can disagree at a near-even split across trees, even though both "look" correct in isolation and neither throws an error. It's the kind of bug that never surfaces unless you go looking for it, parity checks and threshold-optimisation logic downstream both implicitly assumed `predict()` was just a thresholded `predict_proba()`, and it quietly wasn't. Fixing it tightened custom-vs-sklearn parity to `|Δ Accuracy| = 0.00%`, which is a good sign, but the real lesson is that "each method works" isn't the same claim as "the methods agree with each other."

8. **Threshold Tuning and Subgroup Analysis Together Catch What Aggregate Metrics Hide.**

A single F1-optimal threshold and a single aggregate accuracy number can both look healthy while masking uneven performance across passenger subgroups. Running subgroup error analysis by Sex, Pclass, Title, and FamilyGroup, rather than stopping at the aggregate confusion matrix surfaced where the model's confidence was and wasn't well-calibrated, a habit worth carrying into every future classification project.

9. **Matching sklearn Isn't Just a Sanity Check — It's a Debugging Tool.**

Comparing the custom Random Forest against sklearn's implementation with matched hyperparameters, and explicitly tracing the residual gap to split-threshold sampling and floating-point aggregation differences rather than treating "close enough" as good enough, meant any larger discrepancy would have been a signal to go back and re-check the bootstrap or split-selection logic rather than something to explain away. In practice this caught two real problems: the `predict()` inconsistency showed up first as a nonzero parity delta before it was traced to its actual cause, and the leakage fixes were confirmed correct partly by parity *tightening* afterward rather than loosening.

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)