# K-Nearest Neighbors from Scratch — Iris Flower Classification

An end-to-end machine learning project implementing a K-Nearest Neighbors classifier using only NumPy and Pandas, applied to the classic [Iris flower dataset](https://archive.ics.uci.edu/dataset/53/iris).

Built to understand the fundamentals of instance-based, distance-driven learning, with no training phase, no parametric assumptions, and no reliance on high-level abstractions.

---

## Project Overview

Unlike every other model in this repository, KNN doesn't learn a set of parameters at all, it's a **lazy learner**. `fit()` does nothing but store the training data; all the actual work happens at prediction time, when a query point's distance to every training sample is computed and its class is decided by majority vote among the K closest neighbors. This project implements that idea from first principles:

1. **Distance Computation** — Euclidean, Manhattan, and general Minkowski distance, all hand-coded with NumPy broadcasting rather than any distance utility
2. **Neighbor Search** — brute-force `O(N·d)` distance computation followed by an `argsort` to find the K nearest training points
3. **Majority Voting** — the predicted class is whichever label is most common among the K nearest neighbors, with ties broken by proximity order
4. **Feature Scaling** — since KNN is purely distance-based, unscaled features with larger numeric ranges can silently dominate the distance calculation; a from-scratch standardizer addresses this

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — class balance, per-class feature distributions (histograms, boxplots, violin plots, KDE overlap), pairplot, correlation heatmap
- **Preprocessing** — stratified train/test split, and a manual standardization class (`z = (x - μ) / σ`, statistics fit on the training split only, no `sklearn.preprocessing`)
- **Model Implementation** — a `KNNClassifier` (Euclidean/Manhattan/Minkowski distance, brute-force neighbor search, majority-vote prediction, a `get_neighbors()` diagnostic utility) built from scratch with NumPy
- **Hyperparameter Sensitivity Analysis** — a `K` sweep (1–21) evaluated via 5-fold cross-validation, checked against the classic bias-variance narrative (K=1 overfits, large K underfits)
- **Validation** — matched-hyperparameter comparison against `sklearn.neighbors.KNeighborsClassifier` with `algorithm='brute'`, including a full prediction-by-prediction agreement check
- **Comparative Modeling** — benchmarked against sklearn's KNN, Logistic Regression, Decision Tree, Random Forest, SVM (RBF), Gradient Boosting, and XGBoost
- **Diagnostics** — misclassified-sample inspection with per-error vote breakdowns, a decision-boundary visualization in petal space, and an empirical prediction-time-vs-N timing study confirming `O(N)` complexity
- **Hypothesis Testing** — five pre-registered hypotheses about the data and model, checked explicitly against final results rather than assumed

Key issues encountered and resolved during the project:

- **A hypothesis marked "Confirmed" that the notebook's own numbers contradicted** — the scaling-impact hypothesis (H4) was originally reported as confirmed, but the actual comparison showed unscaled KNN (100.0%) outperforming scaled KNN (93.3%) on the test set; corrected to "Not confirmed," with the likely explanation (a 30-sample test set, and Iris's features already sharing comparable units) stated explicitly rather than glossed over
- **Test-set accuracy visualized during hyperparameter search** — the K-sweep plot originally charted test accuracy alongside train and cross-validation accuracy across all 11 K values; removed so the test set isn't visually inspected during model selection, even though the actual K choice was always driven by cross-validation, not test performance
- **A hardcoded audit number that didn't match its own computed output** — a structural-audit table claimed 3 duplicate rows while the cell computing that count directly above it reported 1; corrected to match the actual `df.duplicated().sum()` result

---

## Repository Structure

```
06. KNN_iris_flower/
├── data/
│   └── .gitkeep
├── notebook/
│   └── knn_iris_flower_dataset.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via scikit-learn's built-in Iris dataset inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/06. KNN_iris_flower"
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
| `numpy` | Distance computation, neighbor search, majority voting, standardization |
| `pandas` | Data manipulation, feature/target framing, EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | Iris dataset loader, validation-phase benchmark models, cross-validation utilities |
| `xgboost` | Comparative benchmark model (used if installed; benchmark suite degrades gracefully without it) |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `k` | 3 |
| `metric` | Euclidean |

Selected via 5-fold cross-validation over `k ∈ {1, 3, 5, ..., 21}` (CV accuracy = 0.9667), not via test-set performance.

---

## Limitations

- **Feature scaling did not help on this dataset**: the project's own scaling-impact experiment found unscaled KNN slightly *outperforming* scaled KNN on the test set (100.0% vs. 93.3%). This is best read as noise from a 30-sample test set combined with Iris's four features already being measured in comparable units (cm) — not evidence that scaling is unimportant for KNN in general. On datasets with features spanning different orders of magnitude, scaling remains essential.
- **Brute-force search is `O(N·d)` per prediction**: fine at Iris's scale (microseconds per prediction, confirmed empirically), but this implementation has no spatial indexing (KD-tree, ball tree) and would become impractical well before `N` reaches production scale.
- **One duplicate row is present and deliberately retained**: two Virginica samples share identical measurements. This was a deliberate choice (plausible in real botanical data) rather than an oversight, but it does mean the duplicate could in principle land on both sides of the train/test split.
- **Small, well-studied, single-domain dataset (150 samples)**: Iris is close to linearly separable and has no missing values or noisy features — it's an ideal dataset for illustrating an algorithm's mechanics, not a stress test of its robustness.
- **Majority-vote ties are broken by proximity order, not by a documented explicit rule**: `Counter.most_common()` resolves ties by first-encountered order among the K nearest neighbors (i.e., favors the closer of two tied classes), which is a reasonable default but isn't a configurable option.

---

## Results

**Test set — full model comparison (sorted by test accuracy):**

| Rank | Model | Test Acc | F1 | CV (5-fold) |
|---|---|---|---|---|
| 1 | Gradient Boosting | 0.9667 | 0.9666 | 0.9417 ± 0.0204 |
| 2 | SVM (RBF) | 0.9667 | 0.9666 | 0.9667 ± 0.0167 |
| 3 | **Custom KNN** | **0.9333** | **0.9327** | **0.9667 ± 0.0167** |
| 4 | Sklearn KNN | 0.9333 | 0.9327 | 0.9667 ± 0.0167 |
| 5 | Decision Tree | 0.9333 | 0.9333 | 0.9500 ± 0.0167 |
| 6 | Logistic Regression | 0.9333 | 0.9333 | 0.9583 ± 0.0264 |
| 7 | XGBoost | 0.9333 | 0.9333 | 0.9500 ± 0.0312 |
| 8 | Random Forest | 0.9000 | 0.8997 | 0.9500 ± 0.0312 |

The custom implementation ties for 3rd of 8 models on test accuracy, and **matches sklearn's own KNN implementation exactly,** same test accuracy, same F1, same cross-validation score, to four decimal places.

**Custom vs. sklearn KNN, matched hyperparameters (k=3, Euclidean, brute-force):**

| Metric | Custom KNN | Sklearn KNN |
|---|---|---|
| Test Accuracy | 0.9333 | 0.9333 |
| Macro Precision | 0.9444 | 0.9444 |
| Macro Recall | 0.9333 | 0.9333 |
| Macro F1 | 0.9327 | 0.9327 |

**Prediction agreement: 30 / 30 test samples (100%)** — the custom implementation produces an identical prediction to sklearn on every single test point, not just matching aggregate metrics.

**Error analysis:** 2 of 30 test samples misclassified, both true Virginica predicted as Versicolor, consistent with the two species' known overlap at the decision boundary; Setosa was classified perfectly (precision, recall, and F1 all 1.00).

**Complexity:** empirical timing confirms `O(N)` prediction cost — 0.09 ms/prediction at N=120 training samples, scaling to 1.18 ms/prediction at N=10,000.

---

## What I learned:

1. **A Model With No Training Phase Still Has a Model Selection Problem.**

KNN has no loss function to minimize during `fit()` — but choosing `k` is exactly as consequential as choosing any other model's hyperparameters, and it has the same bias-variance shape as everything else in this repository: `k=1` memorizes the training set (100% train accuracy, visibly worse test accuracy), while large `k` smooths the decision boundary until it collapses toward the majority class. Watching that curve appear from a plain cross-validation sweep — with no actual training happening anywhere — was a clean illustration that "hyperparameter tuning" and "training" are separable ideas.

2. **Matching Sklearn Exactly Is a Stronger Signal for a Lazy Learner Than for a Trained One.**

For the tree- and boosting-based projects in this repository, "close to sklearn" was the realistic bar, because randomness in split-threshold sampling and floating-point aggregation prevents exact reproduction. KNN has no such randomness — same distance formula and same K should mean *identical* predictions, deterministically. Getting 100% prediction agreement (not just matching metrics) was confirmation that there was nowhere left for a subtle bug to hide.

3. **An Unconfirmed Hypothesis Is a More Useful Result Than a Falsely Confirmed One.**

The scaling experiment came back the "wrong" way — unscaled KNN beat scaled KNN — and the honest move was reporting that plainly rather than reflexively marking the hypothesis confirmed because scaling "should" help. Iris's four features are already all measured in centimeters, so there was never much distortion for standardization to fix; that's a more interesting and more transferable finding than a comforting but inaccurate "confirmed."

4. **A Comment Describing What Code *Should* Do Isn't Evidence That It Does.**

A duplicate-row count sitting in a hardcoded markdown table (3) directly contradicted the number the same notebook's own `df.duplicated().sum()` call produced two cells earlier (1). Nothing downstream depended on the wrong number, but it's a reminder that narration and computation can drift apart silently, and the only real check is comparing a written claim against the specific cell that's supposed to back it up.

5. **The Test Set Doesn't Need to Change an Outcome to Be Worth Protecting.**

Plotting test accuracy across every value in the K sweep never actually changed which `k` got selected — cross-validation accuracy did that on its own. Removing it from the chart anyway was about the discipline of never visually referencing the test set during model selection, not about fixing a result that was already correct.

6. **`O(N)` Complexity Is Easy to State and More Convincing to Measure.**

Timing the same prediction across training set sizes from 120 to 10,000 and watching the per-prediction cost scale linearly turned "KNN is O(N) per query" from a textbook fact into something with an actual number attached — and made the case for spatial indexing (KD-trees, ball trees) at production scale concrete rather than theoretical.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)