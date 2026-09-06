# Linear Discriminant Analysis from Scratch — IMDb Sentiment Analysis

An end-to-end machine learning project implementing Linear Discriminant Analysis using only NumPy, applied to binary sentiment classification on the [IMDb Movie Review dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (50,000 reviews).

Built to understand LDA's full mathematical machinery, class means, scatter matrices, Fisher's criterion, pooled covariance, regularised inversion, with no reliance on `sklearn.discriminant_analysis` until the custom implementation was fully derived and validated.

---

## Project Overview

Unlike a lazy learner like KNN, LDA estimates explicit parameters at fit time, per-class means, a pooled within-class covariance, and the discriminant weights derived from it, then classifies new points with a single matrix-vector multiplication. This project implements that idea from first principles:

1. **Mathematical Derivation** — class means, within-class scatter $\mathbf{S}_W$, between-class scatter $\mathbf{S}_B$, and Fisher's criterion, worked out in full before any code is written
2. **From-Scratch Implementation** — a `CustomLDA` class (NumPy only) estimating class priors, means, and pooled covariance; Cholesky-based inversion with a `pinv` fallback; precomputed discriminant coefficients $w_k = \Sigma^{-1}\mu_k$; a numerically stable log-sum-exp softmax for `predict_proba()`
3. **Ridge Regularisation** — a tunable $\alpha$ added to $\mathbf{S}_W$ before inversion, required because high-dimensional TF-IDF features make the scatter matrix rank-deficient
4. **Text Feature Engineering** — a cleaning pipeline (HTML/punctuation stripping, stopword removal) feeding both TF-IDF (unigrams + bigrams, sublinear TF) and Bag-of-Words vectorisers, fit on the training split only

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — class balance, review-length distributions by class, distinctive-word frequency analysis, vocabulary growth (Heaps' Law), and measured TF-IDF sparsity across candidate vocabulary sizes
- **Preprocessing** — HTML/punctuation/stopword-stripping text cleaner, a 70/15/15 stratified train/validation/test split, and a TF-IDF vectoriser (`max_features=2000`, `min_df=3`, `sublinear_tf=True`, unigrams+bigrams) fit on the training split only
- **Model Implementation** — a `CustomLDA` class (ridge-regularised pooled covariance, Cholesky solve, softmax posterior) built from scratch with NumPy
- **Hyperparameter Sensitivity Analysis** — a regularisation ($\alpha$) sweep from $10^{-6}$ to $1.0$ evaluated on the validation set
- **Validation** — matched comparison against `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (`solver='lsqr'`, `shrinkage='auto'`), including a full prediction-agreement check
- **Dimensionality Reduction Experiment** — PCA (50–200 components) as a preprocessing step to produce a full-rank scatter matrix without regularisation
- **Comparative Modeling** — benchmarked against Sklearn LDA, Logistic Regression, Linear SVM, Bernoulli Naive Bayes, Random Forest, and Gradient Boosting
- **Assumption Analysis** — an explicit, honest check of LDA's three core assumptions (Gaussian class-conditionals, shared covariance, feature independence) against TF-IDF's actual distribution
- **Diagnostics** — 5-fold stratified cross-validation (with the vectoriser refit inside each fold), false-positive/false-negative inspection with confidence scores, a prediction-confidence and calibration (reliability diagram) analysis, and LDA's discriminant weights as a feature-importance/interpretability tool

Key issues encountered and resolved during the project:

- **Cross-validation would have leaked information through the vectoriser** — pooling the already-vectorised TF-IDF matrices across folds would have reused a vocabulary/IDF fit on the original 70% training split for every fold; instead, Section 13 refits the `TfidfVectorizer` from raw cleaned text inside each fold independently
- **A second, distinct duplication source after text cleaning** — the raw `review`/`sentiment` pair was deduplicated first (418 rows dropped), but cleaning (HTML/punctuation/stopword removal) can make two originally different reviews collapse into the same `clean_review` string; the notebook re-checks duplicates against `clean_review` and drops those separately (49,582 → 49,574 rows) rather than assuming the first dedup pass caught everything
- **PCA preprocessing was tested, not assumed to help** — Section 10 ran regularised direct LDA against PCA+LDA at `n_components ∈ {50,100,150,200}`; even at 200 components (26.5% variance explained), PCA+LDA's test accuracy (0.8602) did not surpass regularised direct LDA (0.8697), so the direct, regularised approach was kept as the primary model rather than defaulting to PCA on the theoretical argument that it produces a "cleaner" full-rank solution
- **The Gaussian assumption is violated and the notebook says so directly** — TF-IDF features are bounded, sparse, and right-skewed, a clear violation of LDA's normality assumption; rather than glossing over this, Section 8 states the consequence explicitly (a suboptimal discriminant relative to non-Gaussian-assuming models) and treats LDA's competitive results as an example of robustness to assumption violations at large sample size, not evidence the assumption doesn't matter

---

## Repository Structure

```
9. LDA_imdb_movie/
├── data/
│   └── .gitkeep
├── notebook/
│   └── LDA_IMDb_Sentiment_Analysis.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly from Hugging Face's `datasets` hub (`stanfordnlp/imdb`) inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/9. LDA_imdb_sentiment_analysis"
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
| `numpy` | Class means, scatter matrices, Cholesky inversion, discriminant scoring, softmax |
| `pandas` | Data manipulation, structural audit, EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Confusion matrix heatmap |
| `datasets` (Hugging Face) | IMDb dataset loader |
| `nltk` | English stopword list |
| `scikit-learn` | TF-IDF/BoW vectorisation, `train_test_split`, `StratifiedKFold`, PCA, validation-phase benchmark models, evaluation metrics |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `alpha` (ridge regularisation) | 1e-4 |
| Feature representation | TF-IDF (unigrams + bigrams, `max_features=2000`, `min_df=3`, `sublinear_tf=True`) |

Selected via a validation-set sweep over $\alpha \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 0.1, 1.0\}$ (best validation accuracy = 0.8744), not via test-set performance.

---

## Limitations

- **LDA's Gaussian assumption is violated by construction**: TF-IDF features are bounded in [0, 1], sparse (a spike at zero), and right-skewed for non-zero values. LDA remains competitive in practice because it is robust to this violation at large sample sizes with genuinely discriminative features, not because the assumption holds.
- **Ridge regularisation, not covariance shrinkage, is used**: a fixed $\alpha$ tuned by validation-set sweep is simpler than Scikit-Learn's data-adaptive Ledoit-Wolf shrinkage (`shrinkage='auto'`), and accounts for most of the accuracy gap to Sklearn's implementation (0.8697 vs. 0.8728).
- **Dense arrays are required**: the from-scratch NumPy implementation needs dense matrix arithmetic, which is feasible at a capped vocabulary of 2,000 features (96.7% sparse) but would not scale to the full unconstrained vocabulary without sparse-aware linear algebra.
- **PCA preprocessing was tested and did not outperform direct regularised LDA**: at the largest tested `n_components` (200, 26.5% variance explained), PCA+LDA reached 0.8602 test accuracy versus 0.8697 for direct LDA — useful for illustrating the rank-deficiency/full-rank trade-off, but not adopted as the primary model.
- **Predicted probabilities are not well calibrated**: LDA's softmax-derived probabilities amplify discriminant score differences non-linearly; the reliability diagram in Section 14 should be consulted before using raw `predict_proba()` output as a calibrated confidence score in production (Platt scaling or isotonic regression would be needed).
- **Negation and mixed-sentiment language remain hard**: error analysis shows false negatives concentrated in reviews with negated positive phrasing (e.g. "not bad at all") and false positives concentrated in mixed or sarcastic reviews — LDA's linear boundary cannot model this non-linear interaction, and bigrams only partially mitigate it.

---

## Results

**Test set — full model comparison (sorted by test accuracy):**

| Rank | Model | Accuracy | Precision | Recall | F1 | AUC | Train Time |
|---|---|---|---|---|---|---|---|
| 1 | Sklearn LDA | 0.8728 | 0.8546 | 0.8995 | 0.8765 | 0.9449 | 19,063.6 ms |
| 2 | Logistic Regression | 0.8721 | 0.8590 | 0.8915 | 0.8750 | 0.9461 | 2,266.1 ms |
| 3 | Linear SVM | 0.8715 | 0.8609 | 0.8872 | 0.8738 | 0.9431 | 3,940.3 ms |
| 4 | **Custom LDA** | **0.8697** | **0.8540** | **0.8931** | **0.8731** | **0.9434** | **3,602.0 ms** |
| 5 | Naive Bayes (BNB) | 0.8452 | 0.8260 | 0.8762 | 0.8503 | 0.9204 | 636.8 ms |
| 6 | Random Forest | 0.8399 | 0.8418 | 0.8384 | 0.8401 | 0.9172 | 26,074.5 ms |
| 7 | Gradient Boosting | 0.8079 | 0.7782 | 0.8631 | 0.8184 | 0.8942 | 617,668.4 ms |

The custom implementation places 4th of 7 models on test accuracy, within 0.3 points of Sklearn's own LDA and within 0.2 points of Logistic Regression and Linear SVM, while training over 5x faster than Sklearn LDA's adaptive shrinkage solver.

**Custom vs. Sklearn LDA, matched TF-IDF features (`solver='lsqr'`, `shrinkage='auto'`):**

| Metric | Custom LDA | Sklearn LDA | Δ |
|---|---|---|---|
| Accuracy | 0.8697 | 0.8728 | −0.0031 |
| Precision | 0.8540 | 0.8546 | −0.0007 |
| Recall | 0.8931 | 0.8995 | −0.0064 |
| F1 Score | 0.8731 | 0.8765 | −0.0034 |
| ROC-AUC | 0.9434 | 0.9449 | −0.0015 |
| Train Time | 3,602.0 ms | 19,063.6 ms | −15,461.7 ms |
| Inference Time | 30.9 ms (4.2 µs/sample) | 59.4 ms | −28.5 ms |

**Prediction agreement: 97.49%** of test samples — both implementations reach the same class assignment for the large majority of the 7,437-sample test set, with the small remaining gap attributable to Sklearn's adaptive Ledoit-Wolf shrinkage versus the notebook's fixed-$\alpha$ ridge regularisation.

**5-fold stratified cross-validation** (vectoriser refit independently per fold):

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.8799 ± 0.0022 |
| F1 Score | 0.8821 ± 0.0024 |
| ROC-AUC | 0.9501 ± 0.0023 |

The CV mean accuracy runs about 1 point above the single held-out test accuracy (0.8697) — with the vectoriser refit per fold, this reflects normal variance across five different partitions rather than leakage.

**PCA + LDA experiment** (dimensionality reduction before fitting):

| n_components | Accuracy | F1 | AUC | Variance Explained |
|---|---|---|---|---|
| 50 | 0.8429 | 0.8476 | 0.9227 | 10.5% |
| 100 | 0.8538 | 0.8577 | 0.9313 | 16.9% |
| 150 | 0.8573 | 0.8608 | 0.9340 | 22.2% |
| 200 | 0.8602 | 0.8638 | 0.9349 | 26.5% |

None of the tested component counts matched direct regularised LDA's 0.8697 test accuracy.

**Error analysis:** 570 false positives (avg. confidence 0.7438) and 399 false negatives (avg. confidence 0.7156) out of 7,437 test samples. Correct predictions cluster at high confidence, while incorrect predictions concentrate near the 0.5 decision boundary, consistent with a well-discriminating (if imperfectly calibrated) classifier.

**Data pipeline:** 50,000 raw reviews → 49,582 after removing 418 exact duplicate (review, sentiment) pairs → 49,574 after removing rows that collided post-cleaning → split 70/15/15 into 34,701 train / 7,436 validation / 7,437 test samples, each stratified to ~50.2% positive.

---

## What I learned:

1. **A Rank-Deficient Scatter Matrix Is the First Thing High-Dimensional Text Breaks.**

With 2,000 TF-IDF features and a 96.7% sparse matrix, the within-class scatter matrix $\mathbf{S}_W$ is rank-deficient the moment feature count approaches sample count per class. Watching the regularisation sweep collapse from a near-singular, noise-amplifying inversion at $\alpha=10^{-6}$ to an over-shrunk, prior-dominated one at $\alpha=1.0$, with a clear peak at $\alpha=10^{-4}$, made the bias-variance trade-off in matrix inversion tangible in a way the textbook derivation alone doesn't.

2. **Matching Sklearn Closely, Not Exactly, Is the Correct Bar for a Regularised Model.**

Unlike a lazy learner such as KNN, where identical hyperparameters guarantee identical predictions, LDA's regularisation strategy is itself a design choice: a fixed ridge $\alpha$ versus Sklearn's adaptive Ledoit-Wolf shrinkage will not produce bit-identical output. 97.49% prediction agreement and metrics within 0.3–0.6 points across the board was the right signal to look for, an exact match would have actually been suspicious given the differing regularisation.

3. **An Honest Assumption-Violation Section Is More Useful Than a Silent One.**

TF-IDF features are bounded, sparse, and right-skewed, none of which is Gaussian. Rather than skipping past LDA's core assumption or quietly hoping it doesn't matter, stating the violation explicitly and then explaining why LDA still performs competitively (large-sample robustness, near-linear separability of TF-IDF text) produced a more defensible and more transferable finding than simply reporting the accuracy number.

4. **Cleaning Text Creates New Duplicates That Raw-Text Deduplication Cannot Catch.**

Two reviews that differ only in punctuation, capitalisation, or HTML markup are distinct strings before cleaning and identical after it. Deduplicating once on the raw `(review, sentiment)` pair and calling it done would have silently left near-duplicate rows in the training set; a second, explicit dedup pass on `clean_review` after the cleaning pipeline caught 8 additional collisions the first pass structurally could not.

5. **Testing a "Should Help" Idea Is Different From Assuming It Does.**

PCA preprocessing is the textbook fix for a rank-deficient scatter matrix, it produces a full-rank $\mathbf{S}_W$ without any regularisation at all. But at every tested component count up to 200, PCA+LDA underperformed direct regularised LDA on test accuracy. Running the actual experiment rather than defaulting to the theoretically cleaner approach was what surfaced that gap.

6. **Cross-Validation Must Refit the Full Preprocessing Pipeline, Not Just the Model.**

Reusing the training-split TF-IDF vectoriser's vocabulary and IDF weights across all five CV folds would have leaked information from the original split into every fold's evaluation. Refitting the vectoriser from raw cleaned text inside each fold, adding real per-fold preprocessing cost, was necessary for the resulting 0.8799 ± 0.0022 CV accuracy to be a trustworthy generalisation estimate rather than an optimistic one.

7. **Discriminant Weights Are Interpretability That Naive Bayes Doesn't Give You for Free.**

Because $w = \Sigma^{-1}(\mu_1 - \mu_0)$ passes the raw class-mean difference through the inverse pooled covariance, LDA's per-feature weights are already correlation-adjusted, co-occurring words don't double-count their evidence the way they would under Naive Bayes' independence assumption. Inspecting the top positive and negative-sentiment features directly showed which vocabulary was actually driving the decision boundary, not just which words were frequent.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)