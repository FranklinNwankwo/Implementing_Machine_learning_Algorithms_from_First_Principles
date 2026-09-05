# Multinomial Naive Bayes from Scratch — 20 Newsgroups Text Classification

An end-to-end machine learning project implementing a Multinomial Naive Bayes classifier using only NumPy and Pandas, applied to the classic [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/).

Built to understand the fundamentals of generative, probabilistic text classification; Bayes' theorem, the conditional independence assumption, and Laplace smoothing, without relying on high-level abstractions.

---

## Project Overview

Naive Bayes takes a fundamentally different approach from every other classifier in this repository: instead of learning a decision boundary directly, it models how each class *generates* its features, then uses Bayes' theorem to invert that into a posterior over classes. For text, "generates its features" means a class-conditional word distribution — the "naive" independence assumption treats every word in a document as conditionally independent given its class, which is false in practice but remains a surprisingly strong baseline for bag-of-words text classification. This project implements that idea from first principles:

1. **Class Priors** — `P(C_k) = N_k / N`, the maximum-likelihood estimate of how common each class is in the training data
2. **Laplace-Smoothed Likelihoods** — `P(word_j | C_k) = (count(word_j, C_k) + α) / (total_words(C_k) + α·V)`, additive smoothing to prevent any unseen word from assigning a document zero probability outright
3. **Log-Space Posterior** — `log P(C_k | x) ∝ log P(C_k) + x · log P(w | C_k)`, computed as a single batched matrix multiply rather than per-document loops, with predictions numerically stabilized via the log-sum-exp trick

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — class balance across 20 categories, document length distribution, vocabulary size and Zipf's-law word-frequency analysis, top words per category before stopword removal
- **Preprocessing** — a from-scratch text-cleaning pipeline (lowercasing, email/URL/digit stripping, custom 166-word stopword list including newsgroup-header artifacts like `subject`/`organization`/`writes`), applied identically to train and test
- **Vocabulary & Document-Term Matrix** — a hand-built vocabulary (document-frequency filtered, capped at 50,000 words) fit on training data only, and a from-scratch DTM builder applied to both splits using that fixed vocabulary
- **Model Implementation** — a `MultinomialNaiveBayes` class (vectorized NumPy fit/predict, batched inference to avoid a hidden dtype-promotion memory blowup) built entirely from scratch
- **Validation Against Sklearn** — matched-hyperparameter comparison against `sklearn.naive_bayes.MultinomialNB`, including a full prediction-by-prediction agreement check
- **Error Analysis** — most-confused class pairs, top discriminative words per class, and individual misclassification inspection
- **Hyperparameter Sensitivity Analysis** — a Laplace-smoothing (`α`) sweep and a vocabulary-size sweep, both evaluated on a held-out validation split carved from the training data (not the test set)
- **TF-IDF vs. Raw Counts** — a controlled comparison of feature weighting schemes on the same vocabulary
- **Comparative Modeling** — benchmarked against sklearn's Naive Bayes, Logistic Regression, Linear SVM, SGD, Random Forest, and Gradient Boosting
- **Leakage Ablation** — the full pipeline rebuilt on the same data with headers, footers, and quoted reply text stripped, to directly measure how much of the headline accuracy comes from topic content vs. metadata leakage
- **Hypothesis Testing** — six pre-registered hypotheses about the data and model, checked explicitly against final results rather than assumed

Key issues encountered and resolved during the project:

- **Two hyperparameter sweeps were entirely test-set-driven, with no validation split anywhere in the notebook** — the Laplace-smoothing (`α`) sweep and vocabulary-size sweep both selected and characterized "best" values by evaluating directly against the test set in a loop, with no cross-validation or held-out validation data used anywhere else in the notebook to fall back on. Fixed by carving a stratified 15% validation split out of the training data specifically for these two sections, leaving the true test set completely unseen until final reporting. This changed the actual sweep results (the corrected smoothing sweep shows a smooth monotonic decline as `α` increases, not the U-shaped curve the original test-set-driven sweep suggested), which meant the hypothesis conclusion drawn from it needed rewriting to match, not just the code.
- **A hardcoded confusion-pair example that wasn't in the model's own confusion data** — the error-analysis write-up cited `talk.politics.misc` ↔ `talk.politics.mideast` as a confused pair, but the notebook's own top-15 most-confused-pairs table has no such entry; the actual second-most-confused pair overall is `talk.politics.misc` → `talk.politics.guns`. Corrected to match the table.
- **A hypothesis summary table that understated its own strongest result** — the final hypothesis-validation table described sklearn agreement as ">97% prediction agreement; <0.5% accuracy delta," when the actual measured result was **100% agreement (0 disagreements out of 7,532 predictions) and a 0.0000 accuracy delta** — a strictly better result than what was reported.
- **A misleading sklearn training-time comparison, caught and explained rather than left as a false signal** — the validation section's sklearn baseline showed a training time of ~119 seconds, dramatically slower than the custom implementation, which would wrongly suggest sklearn is inefficient. Traced to sklearn being fit on this notebook's dense `int32` document-term matrix rather than a sparse one; 20 Newsgroups' DTM is 99.71% sparse, and Section 14's benchmark (built on a proper sparse `CountVectorizer` matrix) shows sklearn training in ~75ms, consistent with its reputation. The accuracy and agreement numbers from that section were unaffected; only the training-time figure needed the caveat.
- **An initial hypothesis about a data artifact, tested and retracted once evidence didn't support it** — the `ax` token's unusually high frequency was initially suspected to be a header/footer/quote artifact. The leakage-ablation experiment (stripping headers, footers, and quotes) showed `ax` persisting at nearly its original frequency (62,485 occurrences, barely changed from 62,520), ruling that hypothesis out; it's a body-text encoding artifact instead, most likely from quoted-printable content in Windows-related posts.

---

## Repository Structure

```
07. naive_bayes_newsgroups/
├── data/
│   └── .gitkeep
├── notebook/
│   └── naive_bayes_newsgroups.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via scikit-learn's `fetch_20newsgroups` (official by-date train/test partition) inside the notebook, so no manual download is needed. The first run downloads and caches the corpus locally; subsequent runs use the cache.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/07. naive_bayes_newsgroups"
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
| `numpy` | Vectorized fit/predict, log-space posterior computation, batched inference |
| `pandas` | Data framing, structural audit, EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots, confusion matrix heatmap |
| `scikit-learn` | Dataset loader (`fetch_20newsgroups`), `CountVectorizer`/`TfidfVectorizer` for feature comparisons, validation-phase benchmark models, evaluation metrics |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `alpha` (Laplace smoothing) | 1.0 |
| Vocabulary — `min_freq` | 3 |
| Vocabulary — `max_vocab` cap | 50,000 |
| Vocabulary — actual size after filtering | 30,563 |
| Stopword list size | 166 words |
| Header/footer/quote text | Kept (see Leakage Ablation in Results) |

---

## Limitations

- **The headline accuracy is inflated by metadata leakage.** With headers, footers, and quoted reply text kept, the model scores 79.69% accuracy; with them stripped, accuracy drops to 64.79% — a 14.9-point gap. The higher figure should be read as an upper bound reflecting how easy this particular dataset's headers make classification, not a clean measure of topic-classification ability from message content alone.
- **The conditional independence assumption is false and shows up in specific failure modes.** Classes with overlapping vocabulary (e.g., `alt.atheism` and `talk.religion.misc`, or the various `comp.sys.*.hardware` categories) are the most confused pairs, exactly where word co-occurrence patterns that Naive Bayes can't model would be most informative.
- **The Laplace-smoothing sweep does not actually validate the smoothing hypothesis as originally framed.** On the corrected, validation-set-based sweep, accuracy declined monotonically from `α=0.001` (0.8787) to `α=10` (0.8015) — less smoothing was consistently better across the entire tested range. True zero smoothing (`α=0`, which would cause `-∞` log-likelihoods for unseen words) was never tested, so the underlying theoretical justification for smoothing isn't ruled out, but the specific claim that `α=1` sits near an optimal middle ground is not supported by this data.
- **A body-text encoding artifact (`ax`) inflates one class's error rate independent of metadata leakage.** `comp.os.ms-windows.misc` collapses to near-zero recall partly because of this artifact, which persists even after headers/footers/quotes are stripped — a bag-of-words model has no mechanism to distinguish genuine topic signal from this kind of noise.
- **No context or word order.** The bag-of-words representation can't distinguish semantically different phrasings that share the same words.

---

## Results

**Headline result (Section 8 — headers/footers/quotes kept, `alpha=1.0`, full 11,314/7,532 train/test split):**

| Metric | Value |
|---|---|
| Accuracy | 0.7969 |
| Macro Precision | 0.7986 |
| Macro Recall | 0.7889 |
| Macro F1 | 0.7732 |
| Weighted F1 | 0.7787 |
| Inference throughput | 5,456 docs/sec |

**Custom vs. sklearn `MultinomialNB`, matched hyperparameters (`alpha=1.0`):**

| Metric | Custom NB | Sklearn NB | Δ |
|---|---|---|---|
| Accuracy | 0.7969 | 0.7969 | +0.0000 |
| Macro F1 | 0.7732 | 0.7732 | +0.0000 |

**Prediction agreement: 7,532 / 7,532 test documents (100%)** — 0 disagreements with sklearn's implementation.

**Full benchmark comparison (sklearn `CountVectorizer` features, same 30,563-word vocabulary, sorted by accuracy):**

| Model | Accuracy | Macro F1 | Weighted F1 | Train (ms) | Infer (ms) |
|---|---|---|---|---|---|
| Sklearn Naive Bayes | 0.8035 | 0.7824 | 0.7860 | 74.7 | 19.7 |
| **Custom Naive Bayes** | **0.7969** | **0.7732** | **0.7787** | **896.7** | **724.2** |
| Logistic Regression | 0.7963 | 0.7893 | 0.7954 | 6,577.4 | 15.6 |
| Linear SVM | 0.7869 | 0.7810 | 0.7864 | 5,792.0 | 12.5 |
| Random Forest | 0.7831 | 0.7721 | 0.7796 | 19,474.2 | 403.5 |
| Gradient Boosting | 0.7456 | 0.7548 | 0.7597 | 221,424.2 | 150.7 |
| SGD Classifier | 0.7400 | 0.7350 | 0.7400 | 423.9 | 22.7 |

Naive Bayes trains 5–250x faster than every other model in the comparison, at accuracy competitive with, and macro F1 close behind, linear classifiers that take orders of magnitude longer to fit.

**TF-IDF vs. raw word counts (same vocabulary):**

| Feature type | Accuracy | Macro F1 |
|---|---|---|
| Raw counts | 0.7969 | 0.7732 |
| TF-IDF | 0.8301 | 0.8254 |
| **Δ** | **+0.0332** | **+0.0522** |

**Header/footer/quote leakage ablation:**

| Metric | With headers/footers/quotes | Without | Δ |
|---|---|---|---|
| Vocabulary size | 30,563 | 23,138 | −7,425 |
| Accuracy | 0.7969 | 0.6479 | −0.1490 |
| Macro F1 | 0.7732 | 0.6190 | −0.1542 |

**Most confused class pairs (top 3 by error count):**

| True class | Predicted class | Errors | % of true class |
|---|---|---|---|
| `comp.os.ms-windows.misc` | `comp.sys.ibm.pc.hardware` | 161 | 40.9% |
| `talk.politics.misc` | `talk.politics.guns` | 95 | 30.6% |
| `comp.os.ms-windows.misc` | `comp.graphics` | 87 | 22.1% |

**Hypothesis validation summary:**

| Hypothesis | Result | Evidence |
|---|---|---|
| H1: Custom NB ≈ sklearn NB accuracy | Confirmed | 100% prediction agreement (0/7,532 disagreements); 0.0000 accuracy delta |
| H2: NB trains faster than complex models | Confirmed | Orders of magnitude faster vs. Random Forest / Gradient Boosting |
| H3: TF-IDF improves over raw counts | Confirmed | +3.32 pts accuracy, +5.22 pts macro F1 |
| H4: Laplace smoothing critical for generalization | Not supported by sweep as run | Accuracy declined monotonically from α=0.001 (0.8787) to α=10 (0.8015) on a validation split; true zero smoothing was never tested |
| H5: Linear classifiers outperform NB on macro F1, not raw accuracy | Partially confirmed | LR/Linear SVM win macro F1; Custom NB narrowly wins raw accuracy over LR (0.7969 vs. 0.7963) |
| H6: Header/footer/quote metadata inflates reported accuracy | Confirmed | Accuracy 79.69% → 64.79% (−14.90 pts) with metadata removed |

---

## What I learned:

1. **A Model That Assumes Something False Can Still Be Extremely Useful.**

Multinomial Naive Bayes assumes every word in a document is independent given its class, obviously untrue for real language, and still lands within a couple of points of Logistic Regression and Linear SVM on this task, at a fraction of the training cost. Watching a deliberately wrong assumption still produce a competitive, genuinely useful classifier was a more convincing lesson in "all models are wrong, some are useful" than any textbook statement of it.

2. **A Hyperparameter Sweep Without a Validation Set Is Just the Test Set Wearing a Disguise.**

Both the smoothing sweep and the vocabulary-size sweep evaluated directly against the test set in a loop, and it took building an actual validation split, something this notebook had never needed before, since it only ever had a single train/test partition to notice that neither sweep had anywhere leakage-safe to report to. The fix wasn't subtle once named, but the notebook ran for a long time without a validation set ever being necessary anywhere else, which is exactly how this kind of gap goes unnoticed.

3. **Fixing a Leaky Evaluation Can Change the Actual Finding, Not Just Clean Up the Method.**

The original, test-set-driven smoothing sweep suggested a U-shaped curve with a peak around `α=0.1`. The corrected, validation-set-based sweep showed something completely different: a smooth, monotonic decline from `α=0.001` all the way to `α=10`. That's not a rounding difference, it reverses the conclusion the sweep supports. A methodology fix that doesn't change any downstream numbers is easy to shrug off as pedantry; one that flips the actual result is a reminder that the two are not always separable.

4. **Every Number in a Write-Up Should Be Traceable to a Specific Cell, Not Just "Roughly Right."**

Two separate mistakes made it into this notebook's prose despite being directly checkable against code two cells away: a confused-pair example (`talk.politics.mideast`) that wasn't actually in the confusion table, and a hypothesis summary ("97% agreement") that undersold an actual 100% result. Neither was a modeling error — both were narration drifting slightly from what the numbers actually said, in different directions. The only reliable check is re-reading the specific cell a claim is supposed to rest on, not trusting that a plausible-sounding claim is probably fine.

5. **An Absurd-Looking Number Is a Bug Report, Not Always a Bug.**

Sklearn's benchmark training time briefly looked 1,600x slower than the custom implementation, which would have been a strange thing to publish uncommented. Tracing it to a dense-vs-sparse matrix mismatch (this notebook's DTM is 99.71% sparse, and sklearn was accidentally handed a dense copy) turned a number that looked like a mistake into an accurate, explainable artifact of a specific design choice made earlier in the notebook, not a flaw in sklearn or in the comparison.

6. **Retracting Your Own Hypothesis Mid-Notebook Is a Feature, Not a Failure.**

The leakage ablation was originally run partly to confirm that the `ax` token artifact was a header/footer/quote leftover. It wasn't, `ax` persisted almost unchanged after removing all three. Reporting that reversal directly, rather than quietly dropping the original hypothesis or forcing the data to fit it, produced a more accurate final write-up than either alternative would have.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)