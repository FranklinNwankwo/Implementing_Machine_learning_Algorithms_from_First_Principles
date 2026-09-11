# Deep Q-Network from First Principles — CartPole-v1 Control

An end-to-end reinforcement learning project implementing a Deep Q-Network using only NumPy, applied to balancing a pole on a cart in the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment (Gymnasium).

Built to understand DQN's full mathematical machinery, the Bellman optimality equation, temporal-difference targets, backpropagation through a manually implemented neural network, experience replay, and the target network, with no reliance on PyTorch, TensorFlow, or Stable-Baselines3's DQN until the custom implementation was fully derived and validated.

---

## Project Overview

Unlike tabular Q-learning, which stores an explicit value for every state-action pair, DQN approximates the action-value function $Q(s,a;\theta)$ with a neural network, letting it generalise across the continuous 4-dimensional state space CartPole produces. This project implements that idea from first principles:

1. **Mathematical Derivation** — the MDP formulation, Bellman optimality equation, the Q-learning TD target, and the DQN loss, worked out in full before any code is written
2. **From-Scratch Implementation** — a `QNetwork` class (NumPy only, He-initialised 4→64→64→2 MLP) with manual forward and backward passes; `Adam` and `SGD` optimisers implemented from their update rules; a ring-buffer `ReplayBuffer`; a `DQNAgent` combining a target network, epsilon-greedy action selection, and explicit terminal-vs-bootstrapped target construction
3. **Gradient Verification** — a numerical (finite-difference) gradient check against the analytical backward pass, run before the network ever touches the environment
4. **Environment Interface Only** — Gymnasium is used exclusively to create and step `CartPole-v1`; every piece of the learning algorithm itself is hand-implemented

This project walks through the full reinforcement-learning pipeline:

- **Exploratory Environment Analysis (EDA)** — behavioural, not feature/target EDA: random-policy state distributions, a state-variable correlation matrix, a representative trajectory plot, and random-policy return/length distributions
- **Baseline Policies** — a uniform-random policy and a hand-designed heuristic controller (push in the direction the pole is tipping, using angle + 0.5×angular velocity), fixed before any DQN result was seen
- **Model Implementation** — `QNetwork`, `Adam`/`SGD`, `ReplayBuffer`, and `DQNAgent` built from scratch with NumPy, with a synthetic-regression sanity test confirming both optimisers actually reduce loss before being connected to the RL loop
- **Unit and Integration Testing** — replay-buffer insertion/sampling tests, a terminal-vs-non-terminal target construction test, and a short end-to-end integration run before committing to the full experiment
- **Multi-Seed Training and Evaluation** — 5 independent seeds × 350 training episodes each, followed by 50-episode greedy (epsilon = 0) evaluation per seed, never mixing training-time exploration into reported performance
- **Ablation Studies** — DQN trained with the replay buffer removed, the target network removed, and epsilon held constant (no decay), each isolating one component's causal contribution
- **Hyperparameter Sensitivity Analysis** — a sweep over learning rate, discount factor ($\gamma$), and epsilon-decay length
- **Optimizer Comparison** — Adam vs. SGD, both from scratch, run inside the actual RL training loop rather than only on the synthetic sanity check
- **Diagnostics** — sample efficiency (return vs. cumulative environment steps), Q-value inspection on representative states, Bellman/TD-residual tracking, and an explicit convergence-vs-empirical-stabilisation discussion

Key issues encountered and resolved during the project:

- **A silent double-division bug in backpropagation, caught only by the gradient check** — the first draft of `backward()` divided by the batch size a second time (once implicitly via `dOut`, once explicitly inside the layer gradients), which still ran without error and still looked like it was training. The finite-difference gradient check flagged a ~0.6 relative error against the analytical gradient before any RL logic was layered on top, well above the 1e-4 tolerance; removing the redundant division brought the error to ~2e-9. Without that check, this bug would not have surfaced as a crash, only as a network that quietly failed to learn as well as it should have.
- **Truncation was never allowed to look like termination** — CartPole's 500-step cap ends an episode without the pole having actually fallen. The replay buffer stores `terminated` explicitly and separately from the loop's `done` flag (`terminated or truncated`), so the TD target only skips bootstrapping on a genuine terminal state, never on a time-limit truncation.
- **The external library benchmark (Stable-Baselines3) was deliberately not run** — pulling in a full PyTorch-based framework as a "reference" would work against the from-scratch purpose of the project for the sake of a single comparison number, so it is documented as an explicit limitation and left as future work rather than silently included or silently skipped.
- **The target-network ablation did not behave as the textbook story predicts, and the notebook reports that directly** — see *Results* below; in the single-seed ablation run, removing the target network did not clearly hurt performance the way removing replay or epsilon decay did. Rather than forcing the result to match the expected narrative, the notebook reports the number as measured and discusses why CartPole specifically may be too easy a benchmark to expose the instability a target network is meant to fix.

---

## Repository Structure

```
2. Deep Q-Network_CartPole/
├── notebook/
│   └── DQN_CartPole.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** No dataset is required. `CartPole-v1` is a live physics simulator provided by Gymnasium — states are generated by interacting with the environment at runtime, not loaded from a file.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/3. Reinforcement_Learning/2. Deep Q-Network_CartPole"
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
| `numpy` | Neural network forward/backward pass, Adam/SGD optimisers, replay buffer, all DQN math |
| `pandas` | Episode-level logging, multi-seed aggregation, ablation/sweep summary tables |
| `matplotlib` | Visualisations |
| `seaborn` | Dark-theme styling, state-variable correlation heatmap |
| `gymnasium` | `CartPole-v1` environment interface only, never the learning algorithm |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| Network architecture | 4 → 64 (ReLU) → 64 (ReLU) → 2 (linear), He initialisation |
| Optimiser | Adam (`lr=5e-4`, from scratch) |
| Discount factor ($\gamma$) | 0.99 |
| Replay buffer capacity | 20,000 |
| Minimum buffer size before learning | 1,000 |
| Batch size | 64 |
| Target network update frequency | every 250 gradient updates (hard copy) |
| Epsilon schedule | linear decay, 1.0 → 0.05 over 200 episodes |
| Training seeds / episodes | 5 seeds × 350 episodes each |

Chosen for solid performance within a modest compute budget (~2 minutes for all 5 seeds), not exhaustively tuned to be optimal, the hyperparameter sensitivity sweep below is exploratory, not a search for a better configuration to substitute in here.

---

## Limitations

- **DQN carries no formal convergence guarantee**: nonlinear function approximation combined with bootstrapping and off-policy data (the "deadly triad") breaks tabular Q-learning's convergence proof. The results below support empirical stabilisation on this benchmark, not a proof of convergence to $Q^*$.
- **Ablation and hyperparameter-sensitivity runs used a single seed each**, for compute-budget reasons, unlike the 5-seed main evaluation — their qualitative direction is informative, but exact magnitudes carry more sampling noise than the main result.
- **No established framework-based DQN benchmark was run** (see *Key issues* above) — the random and heuristic baselines anchor the low and high ends of "sane" CartPole performance instead.
- **CartPole is a low-dimensional, fully-observable, stationary benchmark.** None of these results should be read as evidence this implementation is ready for real robotic or physical control systems, which involve partial observability, safety constraints, continuous actions, and non-stationary dynamics this project does not address.
- **Performance is meaningfully hyperparameter-sensitive** (see *Results*) — the configuration above reflects one reasonable setting, not a tuned optimum.

---

## Results

**Baseline comparison** (random policy and heuristic controller, 200 episodes each; DQN, greedy evaluation, 250 episodes across 5 seeds):

| Policy | Episodes | Mean Return | Std | Median |
|---|---|---|---|---|
| Random | 200 | 22.69 | 11.38 | 18.0 |
| Heuristic (angle + 0.5 × angular velocity) | 200 | 500.00 | 0.00 | 500.0 |
| **DQN (from scratch, 5-seed greedy eval)** | 250 | **424.68** | **106.05** | **496.5** |

The heuristic reaching a perfect, zero-variance 500 is a known property of CartPole specifically (a simple proportional controller on angle and angular velocity is close to optimal here), not a generally-beatable baseline; the DQN's value is that it reaches strong performance without ever being given that control law, purely from scalar reward feedback.

**Multi-seed greedy evaluation** (50 episodes per seed, $\epsilon=0$):

| Seed | Mean Return | Std | Median | Min | Max | Success Rate (≥475) |
|---|---|---|---|---|---|---|
| 0 | 500.0 | 0.0 | 500.0 | 500 | 500 | 100% |
| 1 | 236.8 | 23.8 | 236.5 | 196 | 290 | 0% |
| 2 | 483.8 | 40.9 | 495.0 | 211 | 500 | 84% |
| 3 | 402.7 | 58.1 | 403.5 | 253 | 500 | 10% |
| 4 | 500.0 | 0.0 | 500.0 | 500 | 500 | 100% |

Two of five seeds reach a perfect, zero-variance policy; one (seed 1) never breaks out of a ~240-return regime within the 350-episode training budget. This spread, not just the pooled mean, is the honest picture of how reliable this configuration is, a single-seed report would have hidden it entirely.

**Ablation studies** (single seed, 150 episodes each, rolling mean of the last 20 episodes):

| Variant | Mean Return (last 20 eps) | Std (last 20 eps) | Mean Return (all episodes) |
|---|---|---|---|
| Full DQN | 163.2 | 43.5 | 39.7 |
| No replay buffer | 9.6 | 0.7 | 13.0 |
| No target network | 274.3 | 107.1 | 58.4 |
| Constant epsilon = 0.1 (no decay) | 9.8 | 0.9 | 10.6 |

Removing the replay buffer or epsilon decay collapses training to near-random within this budget as both are doing real work. Removing the target network is the surprising result: in this single-seed, 150-episode run did not hurt final performance and even edged ahead of the full model on the last-20 mean, at roughly double the variance (107.1 vs. 43.5 std). The likely read is a bias-variance trade-off specific to an easy, low-dimensional benchmark and a short horizon, without a target network the online estimate can chase a better policy faster, at the cost of a noisier one, rather than evidence that target networks are unimportant on harder problems, where the instability they exist to fix is expected to be more pronounced.

**Hyperparameter sensitivity** (single seed, 150 episodes per configuration, mean return over the last 20 episodes):

| Sweep | Value | Mean Return (last 20 eps) |
|---|---|---|
| Learning rate | 1e-3 | 230.4 |
| Learning rate | 5e-4 | 163.2 |
| Learning rate | 1e-4 | 9.7 |
| Discount ($\gamma$) | 0.90 | 240.1 |
| Discount ($\gamma$) | 0.95 | 277.1 |
| Discount ($\gamma$) | 0.99 | 163.2 |
| Epsilon decay length | 40 episodes | 10.2 |
| Epsilon decay length | 100 episodes | 163.2 |
| Epsilon decay length | 180 episodes | 229.7 |

A learning rate that's too small (1e-4) essentially fails to learn within this episode budget. Decaying epsilon too quickly (40 episodes) leaves no time to explore before committing to a policy; a longer decay (180 episodes) does better within this same fixed budget.

**Optimizer comparison** (single seed, 200 episodes, inside the actual RL loop, not just the synthetic sanity test):

| Optimizer | Mean Return (last 20 eps) |
|---|---|
| Adam | 198.6 |
| SGD | 75.3 |

Adam's adaptive per-parameter step sizes handle the noisy, non-stationary bootstrapped targets in DQN considerably better than plain SGD, consistent with Adam being the default for the main experiment.

**Gradient check:** maximum relative error of 2.03e-09 between the analytical backward pass and a central finite-difference approximation, well under the 1e-4 tolerance conventionally used for this test.

**Compute:** all 5 main-experiment seeds trained in ≈103 seconds combined; the full notebook (environment audit, EDA, gradient check, main training, evaluation, all ablations, hyperparameter sweep, optimizer comparison) executes end-to-end in under 2.5 minutes.

---

## What I learned:

1. **A Working Training Loop Is Not Evidence of a Correct Gradient.**

The double-division bug in `backward()` produced a network that ran without crashing, took gradient steps, and even showed some early return improvement; it simply learned worse than it should have. Nothing about watching the training loop run would have surfaced that. The finite-difference gradient check, run once on a small synthetic problem before RL was ever involved, caught a ~0.6 relative error immediately. Debugging RL code by staring at reward curves is often the wrong first move; verifying the parts that can be checked in isolation, before layering on RL-specific complexity, is what made the eventual reward curves trustworthy.

2. **Ablations Reveal Which Components Actually Matter, Not Which Ones Are Supposed To.**

Textbook DQN treats experience replay, the target network, and epsilon-greedy scheduling as roughly co-equal stabilising ingredients. Running the actual ablations showed a real asymmetry on this benchmark: removing replay or epsilon decay was catastrophic, while removing the target network barely hurt (and even improved the last-20 mean, at the cost of variance) within a short, single-seed run. Reporting that honestly, rather than assuming the textbook story would hold and not bothering to check, was more useful than it would have been to simply state "we compared with and without each component" without the actual, sometimes counter-intuitive, numbers.

3. **Multi-Seed Variance Is a Finding, Not Noise to Average Away.**

Two of five seeds reached a perfect 500 return; one seed plateaued around 240 and never escaped it within the training budget. Reporting only the pooled mean (424.7) would have made the method look more reliably strong than it is. The per-seed table is the more honest artifact; it tells a reader what fraction of runs they should actually expect to succeed with this exact configuration, not just what the average run looks like.

4. **Truncation and Termination Look Identical From Inside the Training Loop Unless You Deliberately Separate Them.**

Both end an episode; both trigger a `reset()`. It would have been easy to let a single `done` flag drive the TD target construction too, which would silently teach the network that reaching the 500-step cap is a "failure" state worth zero future value, when it's actually the best possible outcome. Storing `terminated` separately in the replay buffer, and using it (not `done`) to decide whether to bootstrap, is a one-line distinction with a real effect on what the agent is taught to value.

5. **A Baseline That's "Too Good" Is Still the Right Baseline to Report.**

The heuristic controller reaching a perfect, zero-variance 500 could look like a reason to drop it from the comparison, since it makes the DQN look like it isn't adding value. Keeping it in, and explaining *why* it's near-optimal on this specific benchmark (CartPole is close to linearly controllable), is more informative than removing an inconvenient number; the DQN's real contribution is learning a comparably strong policy from reward alone, which the heuristic comparison actually helps make legible.

6. **Diagnostics That "Look Healthy" Still Need an Explicit Ceiling on What They Prove.**

Decreasing TD error and a stabilising Q-value magnitude are genuinely useful signals that training isn't diverging, but they describe internal self-consistency between the online network and its own bootstrapped targets, not correctness of the resulting policy. Stating that distinction directly, rather than letting a smooth-looking loss curve imply more than it demonstrates, keeps the convergence claims in this project limited to what greedy-policy evaluation actually supports.

7. **Deciding Not to Depend on a Framework Is Itself a Design Decision Worth Documenting.**

Skipping the Stable-Baselines3 comparison for dependency reasons was the right call for this project's purpose, but it's a trade-off, not a free win; it means the project has no comparison against a maturely-tuned reference implementation. Writing that limitation down explicitly, instead of letting the benchmark's absence go unremarked, was more useful than either silently including a heavy dependency or silently leaving the gap unexplained.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)