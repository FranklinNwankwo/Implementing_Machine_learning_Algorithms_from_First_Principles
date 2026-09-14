# Proximal Policy Optimization (PPO) from First Principles — LunarLander-v3 (Discrete)

An end-to-end reinforcement learning project implementing **Proximal Policy Optimization** using only NumPy, a manual dense-layer network, manual backpropagation, a manual Adam optimizer, manual Generalized Advantage Estimation (GAE), and a manually-derived clipped-surrogate PPO gradient, applied to the discrete-action [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) Gymnasium environment.

Built to understand PPO's full mathematical machinery, the policy-gradient theorem, importance sampling, the clipped-surrogate objective, and GAE's bias-variance trade-off, with no automatic differentiation framework (PyTorch/TensorFlow/JAX) used anywhere inside the primary agent, and no use of Stable-Baselines3 until the custom implementation was fully derived, unit-tested, and trained.

---

## Project Overview

Unlike a supervised model fit once to a fixed dataset, PPO is trained through repeated interaction with a simulator: collect on-policy experience, estimate advantages, take a small, trust-region-constrained optimization step, and repeat. This project implements that loop from first principles:

1. **Mathematical Derivation** — the MDP formalism, returns and value functions, the policy-gradient theorem, importance sampling, and PPO's clipped-surrogate-plus-entropy objective, worked out in full (including the by-hand derivation of the clipped-objective gradient's four sign/clip-region cases) before any training code is written

2. **From-Scratch Implementation** — two separate NumPy MLPs (`8 → 64 → 64 → output`, `tanh` hidden activations), a manual forward/backward pass, a manual Adam optimizer with global gradient-norm clipping, a categorical policy head, a GAE implementation with correct `terminated`/`truncated` episode-boundary handling, and a rollout buffer that freezes "old" log-probabilities at collection time

3. **Numerical Gradient Checking** — the analytic policy-gradient and entropy-gradient derivations are validated by finite-difference checking against the *actual* combined clipped-objective-plus-entropy loss (not a generic MSE toy loss) before any large-scale training is trusted

4. **External Validation, Not Internal Dependency** — Stable-Baselines3 PPO, A2C, and DQN are trained under a matched, documented protocol strictly *after* the from-scratch agent is complete, used only as independent reference points, never as a hidden component of the custom algorithm

This project walks through the full RL experimentation pipeline:

- **Environment Understanding** — programmatic (not assumed) verification of `LunarLander-v3`'s 8-dimensional observation space, 4-action discrete action space, and reward structure, plus an explicit `terminated`-vs-`truncated` distinction carried through to the GAE implementation
- **Exploratory Data Analysis** — a random-policy episode batch collected purely for exploratory analysis (never used as PPO training data), establishing the random-policy performance floor
- **Model Implementation** — `DenseLayer`/`MLP` (manual forward/backward, Xavier init), `AdamOptimizer`, a categorical policy network, a value network, `compute_gae`, and `PPOAgent.update()` implementing the clipped-surrogate-plus-entropy loss
- **Unit Tests & Gradient Checks** — probability normalization, log-probability consistency, entropy non-negativity, a hand-computed GAE toy-trajectory check (plus a dedicated episode-boundary-leakage regression test, see below), and finite-difference verification of the PPO clipped-objective gradient
- **Smoke Test** — a short run confirming the full train → update → evaluate loop is free of NaN/Inf instability before committing to the main training budget
- **Main Training Run** — 300,000 timesteps (~146 PPO updates), with periodic deterministic-policy evaluation kept strictly separate from training rollouts
- **Multi-Seed Evaluation** — 3 independent seeds at a reduced budget (150,000 timesteps each), reporting variance rather than a single number (H5)
- **Ablation Studies** — advantage normalization, clip epsilon, and entropy coefficient, each varied independently at a reduced budget (100,000 timesteps), holding the seed fixed
- **Cross-Algorithm Benchmarking** — Stable-Baselines3 PPO, A2C, and DQN trained under the same 300,000-timestep budget and evaluation protocol, alongside a random-policy floor
- **Statistical Analysis** — mean, median, standard deviation, min/max, and IQR of evaluation-episode returns per algorithm, deliberately *not* a formal cross-seed hypothesis test given the small per-algorithm seed count
- **Hypothesis Assessment** — six hypotheses stated before any training (H1–H6), revisited against the actual results
- **Failure Analysis** — training-return non-monotonicity, seed sensitivity, and ablation instability discussed explicitly, plus the exact debugging assertions checked *before* trusting any training result

Key issues encountered and resolved during the project:

- **The GAE recursion was leaking advantage information across an unrelated episode boundary.** The original `collect_rollout` correctly bootstrapped a `truncated` transition's reward with `gamma * V(next_obs)`, but only marked `dones[t]=1` for a true `terminated` transition. Since `LunarLander-v3` auto-resets on truncation, this left the GAE backward recursion unmasked at truncation boundaries, so it silently propagated advantage signal from a *different, unrelated* next episode backward into the truncated one, and simultaneously double-counted the bootstrap (once via the injected reward, once via the recursion's own `next_value` term). This is exactly the class of bug the notebook's own "important warning" on `terminated` vs. `truncated` was written to guard against, and the original unit tests didn't catch it because they only exercised pure-termination toy trajectories. The fix masks the recursion on **any** episode boundary (terminated *or* truncated) while keeping the reward-bootstrap injection for truncations, and a dedicated regression test was added: a toy 4-step trajectory spanning a truncation boundary, asserting that perturbing a value estimate belonging to the *next* episode has zero effect on the *previous* episode's advantages.

- **Narrative markdown cells silently went stale after the fix changed real training dynamics.** Because the leak was corrected (not just cosmetically renamed), the same three seeds produced materially different multi-seed results post-fix (e.g. seed 0's mean eval return moved from 6.4 to −50.6). A "Failure Analysis" cell that hardcoded the pre-fix numbers had to be caught and updated by re-running the notebook end-to-end and diffing the fresh output against the prose, rather than trusting that a code fix alone was sufficient.

- **Several section-header markdown cells contained unrendered variable placeholders.** Text like `` `{CONFIG.main_run_timesteps}` `` inside a *markdown* cell is not interpolated by Jupyter (unlike an f-string in a code cell) and was rendering to readers as literal, un-evaluated template syntax instead of the actual number, quietly undermining the notebook's own stated principle that "every number in this notebook comes from actually executing the code." Fixed by converting those cells to code cells that `display(Markdown(f"..."))`, so the values are recomputed live on every run and can't drift out of sync with `CONFIG` again.

---

## Repository Structure

```
3. PPO_lunarlander/
├── notebook/
│   └── PPO_LunarLander.ipynb
├── README.md
└── requirements.txt
```

> **Note on artifacts:** no dataset is used (transitions are generated live by the `LunarLander-v3` simulator) and no trained model checkpoints are committed to the repo; every result in the README below comes from executing the notebook fully, in order, in a clean environment.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/3. Reinforcement_Learning/3. PPO_LunarLander"
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
| `numpy` | Dense-layer forward/backward pass, Adam optimizer, softmax/log-softmax, GAE, PPO clipped-objective gradient |
| `pandas` | Experiment tracking (per-update training diagnostics, evaluation logs, ablation/multi-seed result tables) |
| `matplotlib` | Training diagnostics and comparison visualizations |
| `seaborn` | Plot styling |
| `gymnasium[box2d]` | `LunarLander-v3` environment |
| `stable-baselines3` | External validation baselines (PPO, A2C, DQN) — never used inside the custom agent |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| Network architecture | `[64, 64]`, `tanh` hidden activations (separate policy and value MLPs) |
| Learning rate | 3e-4 |
| `gamma` (discount factor) | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_epsilon` | 0.2 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Rollout length | 2048 |
| Minibatch size | 64 |
| Epochs per update | 10 |
| Gradient-norm clip | 0.5 |
| Advantage normalization | Enabled |
| Main run budget | 300,000 timesteps (~146 updates) |

Not tuned to reproduce any single Stable-Baselines3 number; training, evaluation, and validation trajectories are kept strictly separate throughout.

---

## Limitations

- **A NumPy-only implementation is educational, not production-grade**: substantially slower per gradient step than a vectorized autodiff framework, and manual backpropagation carries real implementation risk, mitigated, but not eliminated, by the numerical gradient checks (and, concretely, by the truncation-boundary bug caught and fixed during development, see above).

- **Training budgets are deliberately small** (hundreds of thousands of timesteps) relative to published PPO benchmarks that often use millions, a conscious choice to keep the notebook runnable in minutes, and the direct reason the from-scratch agent does not fully cross the 200-return "solved" threshold even though it clearly learns; this is not claimed as a solved environment.

- **LunarLander is a benchmark simulator**, not a real physical system; none of these results establish real-world deployment readiness for the business analogies drawn in the notebook.

- **Performance is seed-sensitive** and hyperparameters materially affect outcomes; a single run of any algorithm here should not be over-interpreted.

- **Stable-Baselines3 may contain implementation optimizations** (advantage-normalization details, orthogonal initialization, learning-rate annealing, vectorized environments) not replicated in the from-scratch agent, exact numerical equality between the two was never expected and was not observed.

- **A2C and DQN solve the discrete-control problem via fundamentally different optimization paradigms**; their comparison to PPO here establishes a *reference range* of achievable performance under a shared budget, not a claim of universal superiority for any one algorithm.

- **A single benchmark environment** cannot establish general superiority of PPO over other methods, or of this implementation over any other from-scratch implementation.

---

## Results

**Cross-algorithm comparison** (deterministic evaluation, 20 episodes each, matched 300,000-timestep budget for every trained agent):

| Algorithm | Implementation | On/Off-Policy | Mean Return | Median Return | Return Std | Mean Ep. Length | Success Rate |
|---|---|---|---|---|---|---|---|
| Random Policy | N/A | N/A | −190.4 | −155.3 | 115.2 | 97.2 | 0.00 |
| **From-Scratch PPO** | NumPy (this notebook) | On-policy | **155.5** | **220.1** | 103.7 | 446.4 | **0.60** |
| Stable-Baselines3 PPO | PyTorch (SB3) | On-policy | 94.1 | 147.5 | 117.7 | 696.3 | 0.20 |
| Stable-Baselines3 A2C | PyTorch (SB3) | On-policy | −90.5 | −108.1 | 73.9 | 457.9 | 0.00 |
| Stable-Baselines3 DQN | PyTorch (SB3) | Off-policy | −113.7 | −116.8 | 20.6 | 1000.0 | 0.00 |

The from-scratch agent landed in the same broad performance regime as SB3 PPO under an identical timestep budget and comparable architecture, in this run it outperformed SB3 PPO on mean/median return and success rate, which is a useful validation signal but not a general claim (SB3's defaults are not tuned for this specific environment either, and a single run per algorithm is weak evidence per H5/H6).

**Main training run** (300,000 timesteps, deterministic evaluation): best rolling eval mean return **188.2**, final-checkpoint eval mean return **176.8**, final-checkpoint success rate **0.75**, the best checkpoint seen during training was not the final one, which is why both are reported separately rather than only the last.

**Multi-seed evaluation** (3 seeds, 150,000 timesteps each, final deterministic eval):

| Seed | Mean Return | Median Return | Std Return | Success Rate |
|---|---|---|---|---|
| 0 | −50.6 | −65.7 | 123.1 | 0.00 |
| 1 | 53.9 | 109.6 | 130.1 | 0.10 |
| 2 | 150.4 | 179.9 | 87.5 | 0.25 |

A concrete, observed instance of seed sensitivity (H5): seed 0 ended with a much lower deterministic-eval return than seeds 1 and 2 at the identical reduced budget.

**Ablation studies** (100,000-timestep budget, seed 0, deterministic eval):

| Ablation | Mean Eval Return | Success Rate |
|---|---|---|
| Advantage normalization (default: on) | −196.7 | 0.00 |
| Advantage normalization: off | 78.4 | 0.00 |
| `clip_epsilon=0.1` | −68.7 | 0.00 |
| `clip_epsilon=0.2` (default) | −196.7 | 0.00 |
| `clip_epsilon=0.3` | 29.9 | 0.00 |
| Entropy coefficient: 0.0 | −87.5 | 0.00 |
| Entropy coefficient: 0.01 (default) | −196.7 | 0.00 |
| Entropy coefficient: 0.05 | −24.6 | 0.00 |

Disabling advantage normalization and zeroing the entropy coefficient both visibly changed outcomes at this reduced budget/single seed, consistent with their mathematical role (an un-normalized advantage signal's scale depends on the arbitrary reward magnitude; zero entropy regularization removes the exploration pressure that prevents early collapse onto a near-deterministic, possibly suboptimal policy), not presented as a statistically powered claim given the single-seed-per-ablation design (see Limitations).

**Hypothesis assessment:**

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | PPO beats a random/untrained policy | **Supported** — random floor mean return −190.4 vs. from-scratch PPO 155.5 |
| H2 | PPO is more stable than naive (unclipped) policy gradient | **Not directly tested** — no unclipped policy-gradient baseline was implemented (see Future Work) |
| H3 | From-scratch PPO reaches the same broad regime as SB3 PPO | **Supported** — 155.5 vs. 94.1 mean eval return, comparable architecture and budget |
| H4 | A2C/DQN provide useful independent reference points | **Supported** — SB3 A2C −90.5, SB3 DQN −113.7, distinct from PPO's regime |
| H5 | Multiple seeds reveal meaningful variance | **Supported** — multi-seed means: −50.6 / 53.9 / 150.4 |
| H6 | Reward variance/stability is informative beyond peak reward | **Supported** — best rolling eval (188.2) vs. final checkpoint (176.8) diverge |

**Failure analysis:** no full training collapse occurred in the main run, but training-return non-monotonicity is visible throughout the learning curve (e.g. a dip around 100k–125k steps), an expected feature of policy-gradient optimization on a moderately stochastic environment rather than a bug, a reminder not to read any single local dip or peak as a definitive signal.

---

## What I learned:

1. **A Correct-Looking Fix Can Still Leave a Gap If It Only Half-Addresses the Bug.**

The original truncation handling got the *reward* side right (bootstrapping with `gamma * V(next_obs)`) but left the *recursion-masking* side wrong (`dones[t]` still only tracked true termination). Both halves have to agree, or GAE will silently propagate advantage information across an episode boundary into a completely unrelated trajectory, and a single hand-derived toy-trajectory unit test that only exercises pure terminations will not catch it. The fix wasn't a new idea, it was making the existing docstring's *intent* actually match the code's *behavior*.

2. **A Regression Test Should Try to Break the Specific Failure Mode, Not Just Re-Check the Nominal Case.**

The added test doesn't just assert GAE matches a hand-computed number, it perturbs a value estimate that belongs to the *next, unrelated* episode and asserts the *previous* episode's advantages don't move. That's the direct, mechanical signature of the leak this bug produces, and is the kind of test that would have caught the original implementation red-handed.

3. **Markdown Narrative Can Go Stale the Instant a Fix Changes Real Behavior, Not Just When Code Changes Syntactically.**

Fixing the GAE bug wasn't a no-op on the numbers, it changed actual training dynamics, so the same three seeds produced different multi-seed results after the fix. A "Failure Analysis" section that hardcoded the old numbers had to be explicitly re-verified against a fresh execution and corrected, a reminder that "the code is right now" and "the notebook's prose is still right" are two separate claims that both need checking after any real fix.

4. **Markdown Cells Don't Execute Python — a Templated-Looking String Is Just a String.**

Several section headers used f-string-style syntax (`` `{CONFIG.main_run_timesteps}` ``) inside plain markdown cells, which Jupyter never evaluates, so they rendered as literal, unfilled placeholder text to any reader. The fix, converting those cells to code cells that render dynamic markdown via `display(Markdown(f"..."))`, is also a more robust pattern going forward: those numbers are now recomputed from `CONFIG` on every execution and structurally cannot drift out of sync the way the Failure Analysis numbers briefly did.

5. **Debugging RL Requires a Different Set of "Is This Actually Correct" Questions Than Supervised Learning.**

There's no held-out label to check predictions against, so correctness has to be established structurally: are `terminated` and `truncated` handled differently, are old log-probabilities frozen across the multi-epoch update, does the analytic clipped-objective gradient match a finite-difference check on the *actual* loss (not a generic toy loss), does clipping actually zero the gradient outside the trust region. Every one of these was checked with an explicit assertion before trusting any large-scale training result, specifically to avoid concluding "correct" just because the reward went up, a signal that, as this project's own bug demonstrated, can look fine on the surface while a real defect sits underneath it.

6. **PPO's Clipped Objective and A2C's Simpler Update Are Not Interchangeable Comparisons Without Saying So.**

A2C and DQN sit under this notebook's shared evaluation protocol and budget as *reference points*, not as claims that PPO must or does dominate every on/off-policy alternative. Reporting SB3 A2C's −90.5 and SB3 DQN's −113.7 alongside PPO's numbers, rather than only reporting PPO's own result, is what makes the "same broad performance regime" validation claim (H3) actually checkable instead of asserted.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo)