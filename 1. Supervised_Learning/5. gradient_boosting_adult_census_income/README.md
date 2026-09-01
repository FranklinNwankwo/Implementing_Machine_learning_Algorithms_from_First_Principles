# Gradient Boosting from Scratch — Adult Census Income Prediction

An end-to-end machine learning project implementing a Gradient Boosting classifier using only NumPy and Pandas, applied to the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) (UCI Machine Learning Repository).

Built to understand the fundamentals of boosting; sequential error-correction via functional gradient descent, without relying on high-level abstractions.

---

## Project Overview

Where bagging (Random Forest) reduces variance by averaging many independent, high-variance trees, boosting takes the opposite approach: it builds trees **sequentially**, each one trained not on the raw target but on the *pseudo-residuals* of the ensemble so far; the negative gradient of the loss function with respect to the current predictions. Each new tree is a small step of gradient descent, taken in function space rather than parameter space, and shrunk by a learning rate before being added to the running prediction. This project implements that idea from first principles for binary classification under log-loss:

1. **Functional Gradient Descent:** At each iteration, the ensemble's current log-odds predictions are converted to pseudo-residuals `y - p`, and a new regression tree is fit to those residuals
2. **Shrinkage:** Each tree's contribution is scaled by a learning rate before being added, trading more iterations for better generalisation
3. **Stochastic Subsampling:** Each tree trains on a random subsample of the training data (subsample < 1.0), reducing correlation between successive trees
4. **Early Stopping:** Training halts once validation loss stops improving for a set number of rounds, rather than running a fixed iteration count blindly.

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — target class imbalance, numerical/categorical feature distributions, correlation analysis, capital gain/loss sparsity, fairness-relevant subgroup patterns (sex, race)
- **Preprocessing** — stratified three-way split, training-set-only mode imputation for missing categorical values, and ordinal encoding fit only on the training split (unseen validation/test categories mapped to a dedicated "unknown" code rather than crashing or leaking)
- **Model Implementation** — a `RegressionTree` (variance-reduction splitting, vectorised sorted-cumulative-sum split search) and a `GradientBoostingClassifierScratch` (log-odds initialisation, pseudo-residual fitting, shrinkage, subsampling, early stopping) both built from scratch with NumPy
- **Validation Against Sklearn** — matched-hyperparameter comparison against `sklearn.ensemble.GradientBoostingClassifier`, including a feature-importance rank correlation check and an explicit accounting of *why* the two diverge where they do
- **Hyperparameter Sensitivity Analysis** — `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, and `subsample` sweeps against train/validation curves only (the test set is never touched during tuning)
- **Validation** — 5-fold stratified cross-validation, calibration analysis (Brier score, reliability curve)
- **Comparative Modeling** — benchmarked against sklearn's Gradient Boosting, Random Forest, AdaBoost, a single Decision Tree, Logistic Regression, XGBoost, LightGBM, and a majority-class baseline
- **Diagnostics** — confusion matrix, ROC curve, feature importance (scratch vs. sklearn consensus), and an explicit hypothesis-validation summary checked against the project's own pre-registered predictions

Key issues encountered and resolved during the project:

- **Data leakage in missing-value imputation** — the mode used to fill missing `workclass`/`occupation`/`native_country` values was originally computed from the full dataset before the train/val/test split; fixed to fit the mode on the training split only and apply it to validation/test
- **Test set touched during hyperparameter tuning** — the sweep visualisation originally plotted test-set AUC alongside train/validation for every hyperparameter value tried; removed so the test set is evaluated exactly once, on the final chosen configuration
- **O(n²) split search** — the original `_best_split` recomputed variance from scratch for every candidate threshold via boolean masking; rewritten to sort each feature once and sweep thresholds with running cumulative sums, cutting full training time from ~58 minutes to under 90 seconds with numerically equivalent results
- **A silent pandas `na_values`/`skipinitialspace` interaction bug** — `skipinitialspace=True` strips leading whitespace *before* `na_values=" ?"` gets a chance to match, so the dataset's missing-value marker was never actually being converted to `NaN` — it was training on the literal string `"?"` as if it were a real category. Fixed by matching on `"?"` (without the leading space) instead

---

## Repository Structure

```
5. gradient_boosting_adult_census_income/
├── data/
│   └── .gitkeep
├── notebook/
│   └── gradient_boosting (Adult_Census_Income).ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly from the UCI Machine Learning Repository (`adult.data` and `adult.test`) inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/5. gradient_boosting_adult_census_income"
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
| `numpy` | Linear algebra, variance-reduction splitting, pseudo-residual computation, tree/ensemble construction |
| `pandas` | Data manipulation, feature engineering, EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | Validation-phase benchmark model, cross-validation utilities, evaluation metrics |
| `xgboost` | Comparative benchmark model |
| `lightgbm` | Comparative benchmark model |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `max_depth` | 3 |
| `min_samples_split` | 10 |
| `subsample` | 0.8 |
| `early_stopping_rounds` | 20 |

Training ran the full 200 iterations without early stopping triggering (validation loss was still improving at the final iteration) and completed in **79.58 seconds** after vectorising the split search.

---

## Limitations

- **Leaf values use the mean pseudo-residual, not a Newton-Raphson update**: the textbook Friedman TreeBoost algorithm for log-loss sets each leaf's value via a one-step Newton update (weighted by `p·(1-p)`), not the plain mean of the residuals landing in that leaf. This implementation uses the simpler mean-residual leaf value, which is mathematically valid but under-confident relative to sklearn's classifier; it's the primary reason recall and F1 trail sklearn's `GradientBoostingClassifier` by a wider margin than accuracy or ROC-AUC do (see Results).
- **Pure Python/NumPy is still slower than compiled libraries**: even after vectorising the split search (~44x speedup, from ~58 minutes to under 90 seconds for 200 trees), this remains far slower than XGBoost or LightGBM's compiled implementations, which is an expected and acceptable trade-off for implementation transparency.
- **Historical, socioeconomic dataset**: this is 1994 U.S. Census data. It includes protected attributes (`sex`, `race`) as raw features and reflects the labor-market patterns of its era, it has no claim to generalising to current income distributions or to populations outside the U.S., and any subgroup performance differences reflect patterns in 30-year-old data, not a normative claim about any group.
- **No native handling of unseen categories beyond a dedicated unknown code**: the ordinal encoder maps unseen validation/test categories to a reserved "unknown" integer code rather than a richer strategy (e.g. frequency-based smoothing); in practice this affected only 1 value across the entire validation set and 0 in test.

---

## Results

**Test set — full model comparison (sorted by ROC-AUC):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LightGBM | 0.8762 | 0.7925 | 0.6541 | 0.7167 | 0.9286 |
| XGBoost | 0.8734 | 0.7830 | 0.6520 | 0.7115 | 0.9285 |
| Sklearn GBM | 0.8745 | 0.7865 | 0.6528 | 0.7135 | 0.9269 |
| Random Forest | 0.8685 | 0.7685 | 0.6451 | 0.7014 | 0.9179 |
| **GBM Scratch** | **0.8584** | **0.8105** | **0.5330** | **0.6431** | **0.9123** |
| Decision Tree (depth=6) | 0.7933 | 0.5428 | 0.8656 | 0.6672 | 0.8973 |
| AdaBoost | 0.8328 | 0.8644 | 0.3574 | 0.5058 | 0.8931 |
| Logistic Regression | 0.7501 | 0.4852 | 0.7235 | 0.5809 | 0.8285 |
| Majority Class (baseline) | 0.2394 | 0.2394 | 1.0000 | 0.3863 | 0.5000 |

The custom implementation ranks **#5 of 9 models** by ROC-AUC — squarely mid-pack, ahead of a single decision tree, AdaBoost, logistic regression, and the baseline, and within 0.015 AUC of sklearn's own `GradientBoostingClassifier` (0.9123 vs. 0.9269).

**Custom vs. sklearn GBM, matched hyperparameters:**

| Metric | GBM Scratch | Sklearn GBM | Δ (abs.) |
|---|---|---|---|
| Accuracy | 0.8584 | 0.8745 | 0.0161 |
| Precision | 0.8105 | 0.7865 | 0.0241 |
| Recall | 0.5330 | 0.6528 | 0.1199 |
| F1 | 0.6431 | 0.7135 | 0.0704 |
| ROC-AUC | 0.9123 | 0.9269 | 0.0146 |

Feature-importance rank correlation between the two implementations: **Spearman ρ = 0.9648** (p = 2.5 × 10⁻⁸) — near-perfect agreement on which features matter, even though the two models disagree on decision thresholds. Consensus top features: `relationship`, `education_num`, `capital_gain`, `age`, `hours_per_week`.

**5-fold stratified cross-validation** (sklearn GBM used as a computationally efficient surrogate, same hyperparameters):

| Metric | Mean ± Std | Range |
|---|---|---|
| ROC-AUC | 0.9239 ± 0.0041 | 0.9163 – 0.9279 |
| F1 | 0.6983 ± 0.0107 | 0.6797 – 0.7100 |
| Accuracy | 0.8699 ± 0.0039 | 0.8637 – 0.8747 |

**Calibration:** Brier score of **0.1003** (0.25 = random on a balanced problem; 0 = perfect).

---

## What I learned:

1. **Boosting and Bagging Solve Different Halves of the Bias-Variance Trade-off.**

Having built a Random Forest first, implementing boosting right after made the contrast concrete rather than theoretical: bagging averages independent high-variance learners to cancel out their noise, while boosting chains together high-bias, low-variance weak learners and lets each one correct what the ensemble got wrong so far. Watching the pseudo-residuals shrink iteration by iteration was a much more direct way to internalise "boosting reduces bias" than reading the definition ever was.

2. **A Model Being "Correct" Doesn't Mean It Matches a Reference Implementation Line for Line.**

The custom GBM's recall and F1 trail sklearn's implementation by a real margin, and rather than treating that as a bug to chase down, tracing it to its root cause; mean-of-residuals leaf values instead of a Newton-Raphson update, turned a worrying-looking gap into a documented, understood design trade-off. The near-perfect Spearman correlation on feature importance (0.9648) confirmed the model had learned the right *structure* even where it diverged on decision thresholds.

3. **"Leakage-Safe Order" Is a Claim That Has to Be Checked, Not Assumed.**

A cell literally commented "leakage-safe order" was still computing the missing-value imputation mode from the full dataset before the split, the comment described the intent, not what the code actually did. The habit that caught it was checking every markdown/comment claim against what the code and its output actually showed, not just reading the code once and trusting its own narration of itself.

4. **The Test Set Is a Resource That Depletes the Moment You Look at It.**

The hyperparameter sweep originally plotted test-set AUC across every value tried, which doesn't corrupt the final reported numbers if the config was already fixed beforehand, but it's exactly the kind of habit that turns into real leakage the next time under time pressure. Removing it and confirming the test set is touched exactly once, at the end, was worth doing even though it didn't change a single reported metric.

5. **Vectorising a Split Search Is About Restructuring the Computation, Not Just Adding NumPy Calls.**

The original split search recomputed variance from a boolean mask for every candidate threshold — O(n²) per feature. Sorting each feature once and sweeping thresholds with running cumulative sums turned that into an O(n log n) computation and cut training time by roughly 44x (58 minutes → 90 seconds), with results matching the unoptimised version to 5–6 decimal places. The lesson wasn't "use NumPy" — it was recognising that `Σy² - (Σy)²/n` lets variance be tracked incrementally instead of recomputed from scratch.

6. **A Correct `na_values` Argument and a Working One Aren't the Same Thing.**

`na_values=" ?"` combined with `skipinitialspace=True` silently failed to convert the dataset's missing-value marker to `NaN`, because the whitespace got stripped before the match could happen — so an entire category of "missing" data was training as a literal `"?"` string for most of this project's development. The dataset's own structural audit cell (printing zero missing values, when the well-documented rate for this dataset is ~5–6%) was the tell, a sanity check against known facts about the data caught something a purely internal code review wouldn't have.

7. **Matching Sklearn Isn't Just a Sanity Check — It's a Debugging Tool.**

Comparing the custom implementation against sklearn's with matched hyperparameters, and explicitly tracing the residual gap to a specific, named algorithmic difference (leaf-value computation) rather than treating "close enough" as good enough, meant a larger or unexplained discrepancy would have been a signal to go back and re-check the pseudo-residual or split-selection logic, not something to wave away.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)