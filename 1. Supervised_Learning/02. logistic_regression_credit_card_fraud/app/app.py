"""
Fraud Detection Decision Lab — interactive dashboard for the from-scratch
logistic regression models in this project.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudlab import metrics as M
from fraudlab.model import (
    MODEL_LABELS,
    amounts_from_features,
    load_eval_data,
    load_meta,
    load_models,
)

st.set_page_config(
    page_title="Fraud Detection Decision Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Colours mirror the notebook's palette: blue = legitimate / Model A, pink = fraud / Model B.
C_LEGIT, C_FRAUD = "#4361EE", "#F72585"
C_MODEL = {"A": "#4361EE", "B": "#F72585"}
C_GOLD, C_TEAL, C_MUTED = "#D4AF37", "#2A9D8F", "#8A93A6"
REPO_URL = "https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles"
NB_PATH = "1. Supervised_Learning/02. logistic_regression_credit_card_fraud/notebook"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .note {color: #8A93A6; font-size: 0.86rem;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
             font-weight:600; margin-right:6px;}
      .pill-fraud {background:#F7258522; color:#F72585;}
      .pill-legit {background:#4361EE22; color:#4361EE;}
      .pill-flag {background:#D4AF3726; color:#B8941F;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_models():
    return load_models()


@st.cache_resource
def get_data():
    return load_eval_data()


@st.cache_data
def get_meta():
    return load_meta()


@st.cache_data
def scores_for(model_key: str, split: str) -> np.ndarray:
    X, _ = get_data().split(split)
    return get_models()[model_key].predict_proba(X)


@st.cache_data
def exact_curves(model_key: str, split: str) -> dict:
    _, y = get_data().split(split)
    s = scores_for(model_key, split)
    return {
        "pr": M.pr_curve(y, s),
        "roc": M.roc_curve(y, s),
        "ap": M.average_precision(y, s),
        "auc": M.roc_auc(y, s),
        "brier": M.brier(y, s),
        "rel": M.reliability(y, s),
    }


@st.cache_data
def validation_thresholds(model_key: str, fixed_fee: float, include_amount: bool, review_cost: float) -> dict:
    """Thresholds chosen on the VALIDATION split only (never on test)."""
    data, model = get_data(), get_models()[model_key]
    s_val = scores_for(model_key, "validation")
    amounts = amounts_from_features(data.X_val, model.feature_names)
    loss = M.fraud_loss_per_sample(amounts, fixed_fee, include_amount)
    f2_tau, f2_val = M.best_fbeta_threshold(data.y_val, s_val, beta=2.0)
    cost_tau = M.min_cost_threshold(data.y_val, s_val, loss, review_cost)
    return {"notebook": model.tau, "f2_exact": f2_tau, "f2_exact_val": f2_val, "min_cost": cost_tau}


models, data, meta = get_models(), get_data(), get_meta()
feature_names = models["A"].feature_names


def fmt_pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or np.isnan(x) else f"{100 * x:.{digits}f}%"


def fmt_eur(x: float) -> str:
    return f"€{x:,.0f}"


def clamp_tau(t: float) -> float:
    """Slider range is [0, 1]; 'flag nothing' (τ = ∞) maps to 1.0."""
    return float(min(max(t, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Session state: one threshold per model, initialised at the notebook's τ*
# ─────────────────────────────────────────────────────────────────────────────
for k, m in models.items():
    st.session_state.setdefault(f"tau_{k}", round(m.tau, 4))


def set_tau(model_key: str, value: float) -> None:
    st.session_state[f"tau_{model_key}"] = round(clamp_tau(value), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — evaluation split and cost assumptions
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    split = st.radio(
        "Evaluate on",
        ["test", "validation"],
        format_func=lambda s: {"test": "Test set (held out)", "validation": "Validation set"}[s],
        help="Thresholds suggested by the app are always chosen on the validation set. "
             "Test-set numbers show how that choice generalises.",
    )
    X_eval, y_eval = data.split(split)
    n_eval, n_fraud_eval = len(y_eval), int(y_eval.sum())
    st.caption(f"{n_eval:,} transactions · {n_fraud_eval} frauds ({100 * n_fraud_eval / n_eval:.3f}%)")

    st.divider()
    st.subheader("Cost assumptions")
    st.caption("Used for the cost view and the minimum-cost threshold. Change them to match your business.")
    review_cost = st.number_input("Cost to review one alert (€)", 0.0, 500.0, 5.0, 1.0,
                                  help="Analyst time plus customer friction for every flagged transaction.")
    include_amount = st.toggle("Missed fraud loses the transaction amount", value=True,
                               help="If on, each missed fraud costs its actual amount (recovered from Amount_log).")
    fixed_fee = st.number_input("Fixed cost per missed fraud (€)", 0.0, 1000.0, 25.0, 5.0,
                                help="Chargeback handling, investigation and admin per fraud that gets through.")
    st.caption(
        f"For scale: in the training set the mean fraud amount is €{meta['train_fraud_amount_mean']:.2f} "
        f"and the median is €{meta['train_fraud_amount_median']:.2f}."
    )

    st.divider()
    st.markdown(
        f"<span class='note'>Models, metrics and SMOTE are implemented from scratch in NumPy. "
        f"Source: <a href='{REPO_URL}' target='_blank'>GitHub repository</a>.</span>",
        unsafe_allow_html=True,
    )

amounts_eval = amounts_from_features(X_eval, feature_names)
loss_eval = M.fraud_loss_per_sample(amounts_eval, fixed_fee, include_amount)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("Fraud Detection Decision Lab")
st.markdown(
    "Logistic regression **written from scratch** (NumPy gradient descent, from-scratch SMOTE) on "
    f"{meta['n_rows_after_dedup']:,} European card transactions, of which only "
    f"{100 * (meta['train_fraud'] + meta['val_fraud'] + meta['test_fraud']) / meta['n_rows_after_dedup']:.2f}% are fraud. "
    "The model only produces a score; **the threshold is a business decision**. Use this lab to see what each choice costs."
)

kcols = st.columns(4)
for col, k in zip(kcols[:2], ["A", "B"]):
    s = scores_for(k, "test")
    c = M.counts_at(data.y_test, s, models[k].tau)
    sm = M.summary(c)
    col.metric(
        f"Model {k} · test recall at τ*",
        f"{c.tp}/{c.tp + c.fn} frauds ({fmt_pct(sm['recall'])})",
        help=f"Test set, τ* = {models[k].tau:.4f} chosen on validation by F2.",
    )
    col.caption(f"{c.fp} false alarms · precision {fmt_pct(sm['precision'])} · F2 {sm['fbeta']:.3f}")
kcols[2].metric("Exact AUC-PR (test)", f"A {exact_curves('A', 'test')['ap']:.3f} · B {exact_curves('B', 'test')['ap']:.3f}",
                help="Average precision over every distinct threshold. Random guessing scores the fraud rate (≈0.0016).")
kcols[3].metric("ROC-AUC (test)", f"A {exact_curves('A', 'test')['auc']:.3f} · B {exact_curves('B', 'test')['auc']:.3f}",
                help="Looks excellent for both — but ROC is flattered by 42,000+ easy negatives. AUC-PR is the honest metric here.")

tab_thr, tab_cmp, tab_txn, tab_hood = st.tabs(
    ["Threshold & cost", "Model A vs Model B", "Transaction explorer", "How it was built"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return fig


def confusion_fig(c: M.Counts) -> go.Figure:
    text = [[f"TN<br><b>{c.tn:,}</b>", f"FP · false alarm<br><b>{c.fp:,}</b>"],
            [f"FN · missed fraud<br><b>{c.fn:,}</b>", f"TP · fraud caught<br><b>{c.tp:,}</b>"]]
    fig = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], text=text, texttemplate="%{text}", showscale=False,
                               colorscale=[[0, "#EEF1FB"], [1, "#FDE7F1"]], hoverinfo="skip",
                               textfont=dict(size=15, color="#16202B")))
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Predicted legitimate", "Predicted fraud"], side="top")
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Actually legitimate", "Actually fraud"], autorange="reversed")
    return base_layout(fig, "Confusion matrix", height=330)


def pr_fig(curves: dict[str, dict], points: dict[str, tuple[float, float, str]], fraud_rate: float) -> go.Figure:
    fig = go.Figure()
    for k, cv in curves.items():
        fig.add_trace(go.Scatter(x=cv["pr"]["recall"], y=cv["pr"]["precision"], mode="lines",
                                 line=dict(color=C_MODEL[k], width=2, shape="hv"),
                                 name=f"{MODEL_LABELS[k]} (AP {cv['ap']:.3f})"))
    for name, (r, p, colour) in points.items():
        fig.add_trace(go.Scatter(x=[r], y=[p], mode="markers", name=name,
                                 marker=dict(size=13, color=colour, line=dict(width=2, color="white"),
                                             symbol="diamond")))
    fig.add_hline(y=fraud_rate, line=dict(color=C_MUTED, dash="dot"),
                  annotation_text=f"random = {fraud_rate:.4f}", annotation_position="top left")
    fig.update_xaxes(title="Recall (share of frauds caught)", range=[0, 1.02])
    fig.update_yaxes(title="Precision (share of alerts that are fraud)", range=[0, 1.04])
    return base_layout(fig, "Precision–recall curve")


def score_hist_fig(scores: np.ndarray, y: np.ndarray, tau: float) -> go.Figure:
    bins = dict(start=0, end=1.0001, size=0.02)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores[y == 0], xbins=bins, name="Legitimate", marker_color=C_LEGIT, opacity=0.7))
    fig.add_trace(go.Histogram(x=scores[y == 1], xbins=bins, name="Fraud", marker_color=C_FRAUD, opacity=0.85))
    fig.add_vline(x=tau, line=dict(color=C_GOLD, width=3), annotation_text=f"τ = {tau:.4f}")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="log", title="Transactions (log scale)")
    fig.update_xaxes(title="Model score P(fraud | x)")
    return base_layout(fig, "Score distribution by true class")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Threshold & cost
# ─────────────────────────────────────────────────────────────────────────────
with tab_thr:
    top = st.columns([1.1, 2.4])
    with top[0]:
        mk = st.radio("Model", ["A", "B"], format_func=lambda k: MODEL_LABELS[k], horizontal=True, key="thr_model")
        vt = validation_thresholds(mk, fixed_fee, include_amount, review_cost)
        st.slider("Decision threshold τ — flag if score ≥ τ", 0.0, 1.0, step=0.0001, format="%.4f",
                  key=f"tau_{mk}")
        st.caption("Presets (all chosen on the **validation** set):")
        b1, b2, b3 = st.columns(3)
        b1.button("Notebook", on_click=set_tau, args=(mk, vt["notebook"]), width="stretch",
                  help=f"F2-optimal on a 300-point grid between 0.01 and 0.99, as in the notebook: τ = {vt['notebook']:.4f}")
        b2.button("Exact F2", on_click=set_tau, args=(mk, vt["f2_exact"]), width="stretch",
                  help=f"F2-optimal over every distinct validation score (no grid): τ = {vt['f2_exact']:.4f}, "
                       f"validation F2 = {vt['f2_exact_val']:.3f}")
        b3.button("Min cost", on_click=set_tau, args=(mk, vt["min_cost"]), width="stretch",
                  help="Threshold that minimises total cost on validation under the sidebar's cost assumptions.")
        if np.isinf(vt["min_cost"]):
            st.info("Under these cost assumptions, flagging nothing is cheapest on validation; the Min cost "
                    "preset sets τ = 1.0. Raise the fixed cost per missed fraud or lower the review cost.")

    tau = float(st.session_state[f"tau_{mk}"])
    s_eval = scores_for(mk, split)
    c = M.counts_at(y_eval, s_eval, tau)
    sm = M.summary(c, beta=2.0)
    cost = M.cost_at(y_eval, s_eval, loss_eval, tau, review_cost)
    rec_ci = M.wilson_interval(c.tp, c.tp + c.fn)
    prec_ci = M.wilson_interval(c.tp, c.tp + c.fp)

    with top[1]:
        r1 = st.columns(3)
        r1[0].metric("Frauds caught (recall)", f"{c.tp} of {c.tp + c.fn}",
                     f"{fmt_pct(sm['recall'])} · 95% CI {fmt_pct(rec_ci[0], 0)}–{fmt_pct(rec_ci[1], 0)}", delta_color="off", delta_arrow="off")
        r1[1].metric("Alerts that are fraud (precision)", fmt_pct(sm["precision"]),
                     f"95% CI {fmt_pct(prec_ci[0], 0)}–{fmt_pct(prec_ci[1], 0)}" if c.alerts else "no alerts",
                     delta_color="off", delta_arrow="off")
        r1[2].metric("F2 score", f"{sm['fbeta']:.3f}", "weights recall 2× precision", delta_color="off", delta_arrow="off")
        r2 = st.columns(3)
        r2[0].metric("Alerts per 10,000 transactions", f"{10000 * sm['alert_rate']:.1f}",
                     f"{c.alerts:,} alerts · {c.fp:,} false", delta_color="off", delta_arrow="off")
        r2[1].metric("False alarm rate (FPR)", f"{100 * sm['fpr']:.3f}%", f"1 in {1 / sm['fpr']:,.0f} legit flagged"
                     if sm["fpr"] else "none", delta_color="off", delta_arrow="off")
        saving = cost["flag_nothing"] - cost["total"]
        r2[2].metric("Total cost on this split", fmt_eur(cost["total"]),
                     f"{'saves' if saving >= 0 else 'costs'} {fmt_eur(abs(saving))} vs no alerts",
                     delta_color="normal" if saving >= 0 else "inverse")
        st.markdown(
            f"<span class='note'>Cost = {fmt_eur(review_cost)} × {c.alerts:,} alerts "
            f"({fmt_eur(cost['review'])}) + missed fraud ({fmt_eur(cost['missed'])}). "
            f"Confidence intervals are Wilson 95% intervals: with only {c.tp + c.fn} frauds on this split, "
            f"one fraud moves recall by {100 / max(c.tp + c.fn, 1):.1f} points.</span>",
            unsafe_allow_html=True,
        )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(confusion_fig(c), key="cm")
    with g2:
        cv = exact_curves(mk, split)
        pts = {f"Current τ = {tau:.4f}": (sm["recall"], 1.0 if np.isnan(sm["precision"]) else sm["precision"], C_GOLD)}
        st.plotly_chart(pr_fig({mk: cv}, pts, n_fraud_eval / n_eval), key="pr")

    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(score_hist_fig(s_eval, y_eval, tau), key="hist")
        st.caption("Everything right of the gold line is flagged. Pink bars left of it are missed frauds; "
                   "blue bars right of it are false alarms.")
    with g4:
        cc = M.cost_curve(y_eval, s_eval, loss_eval, review_cost)
        x_alerts = np.maximum(cc["alerts"], 0.8)  # place "flag nothing" on a log axis
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_alerts, y=cc["cost"], mode="lines", name="Total cost",
                                 line=dict(color=C_TEAL, width=2.5)))
        fig.add_trace(go.Scatter(x=x_alerts, y=cc["missed_loss"], mode="lines", name="Missed-fraud loss",
                                 line=dict(color=C_FRAUD, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=x_alerts, y=cc["review"], mode="lines", name="Review cost",
                                 line=dict(color=C_LEGIT, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=[max(c.alerts, 0.8)], y=[cost["total"]], mode="markers", name="Current τ",
                                 marker=dict(size=13, color=C_GOLD, symbol="diamond", line=dict(width=2, color="white"))))
        imin = int(np.argmin(cc["cost"]))
        fig.add_trace(go.Scatter(x=[x_alerts[imin]], y=[cc["cost"][imin]], mode="markers",
                                 name=f"Cheapest on this split", marker=dict(size=10, color="#16202B", symbol="x")))
        fig.update_xaxes(type="log", title="Number of alerts raised (log scale) — lower threshold → more alerts")
        fig.update_yaxes(title="€ on this split")
        st.plotly_chart(base_layout(fig, "Cost vs. alert volume"), key="cost")
        st.caption("The 'cheapest on this split' marker is for reading only. The Min cost preset uses the "
                   "validation set, so choosing it does not peek at the test set.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Model A vs Model B
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.markdown(
        "Both models are the same from-scratch logistic regression. **A** counters the 599 : 1 imbalance by "
        f"weighting each fraud ≈{meta['class_weight_fraud']:.0f}× in the loss; **B** trains unweighted on "
        f"{meta['smote_fraud'] - meta['train_fraud']:,} synthetic frauds generated by from-scratch SMOTE. "
        "Each uses its own threshold from the *Threshold & cost* tab."
    )
    rows, preds, curves, points = [], {}, {}, {}
    for k in ("A", "B"):
        t = float(st.session_state[f"tau_{k}"])
        s = scores_for(k, split)
        ck = M.counts_at(y_eval, s, t)
        smk = M.summary(ck)
        cv = exact_curves(k, split)
        ck_cost = M.cost_at(y_eval, s, loss_eval, t, review_cost)
        preds[k] = s >= t
        curves[k] = cv
        points[f"{k} at τ = {t:.4f}"] = (smk["recall"], 1.0 if np.isnan(smk["precision"]) else smk["precision"], C_MODEL[k])
        rows.append({
            "Model": MODEL_LABELS[k], "τ": f"{t:.4f}", "Frauds caught": f"{ck.tp}/{ck.tp + ck.fn}",
            "False alarms": ck.fp, "Precision": fmt_pct(smk["precision"]), "Recall": fmt_pct(smk["recall"]),
            "F2": f"{smk['fbeta']:.3f}", "MCC": f"{smk['mcc']:.3f}",
            "AUC-PR (exact)": f"{cv['ap']:.3f}", "ROC-AUC": f"{cv['auc']:.4f}",
            "Brier": f"{cv['brier']:.4f}", "Total cost": fmt_eur(ck_cost["total"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(pr_fig(curves, points, n_fraud_eval / n_eval), key="pr_cmp")
    with g2:
        fig = go.Figure()
        for k, cv in curves.items():
            fig.add_trace(go.Scatter(x=np.maximum(cv["roc"]["fpr"], 1e-5), y=cv["roc"]["tpr"], mode="lines",
                                     line=dict(color=C_MODEL[k], width=2, shape="hv"),
                                     name=f"{MODEL_LABELS[k]} (AUC {cv['auc']:.3f})"))
        fig.update_xaxes(type="log", title="False positive rate (log scale)", range=[-5, 0])
        fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
        st.plotly_chart(base_layout(fig, "ROC curve, zoomed into the low false-alarm region"), key="roc_cmp")
        st.caption("On a linear axis both ROC curves hug the top-left corner. The log axis shows the region that "
                   "matters in practice: a bank can only afford false alarm rates around 0.01–0.1%.")

    fraud = y_eval == 1
    both = int(np.sum(fraud & preds["A"] & preds["B"]))
    only_a = int(np.sum(fraud & preds["A"] & ~preds["B"]))
    only_b = int(np.sum(fraud & ~preds["A"] & preds["B"]))
    neither = int(np.sum(fraud & ~preds["A"] & ~preds["B"]))
    fa_both = int(np.sum(~fraud & preds["A"] & preds["B"]))
    st.subheader("Do they catch the same frauds?")
    v = st.columns(5)
    v[0].metric("Caught by both", both)
    v[1].metric("Only Model A", only_a)
    v[2].metric("Only Model B", only_b)
    v[3].metric("Missed by both", neither)
    v[4].metric("False alarms shared", fa_both)
    st.markdown(
        "<span class='note'>At the notebook thresholds the two models have identical confusion matrices on the test set "
        "(58 caught, 12 missed, 29 false alarms). That is a coincidence of a 70-fraud test set, not proof they are "
        "equivalent: the overlap above shows whether they agree on <i>which</i> transactions, and their PR curves differ "
        "across the rest of the threshold range.</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Transaction explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_txn:
    sA, sB = scores_for("A", split), scores_for("B", split)
    tA, tB = float(st.session_state["tau_A"]), float(st.session_state["tau_B"])
    fA, fB = sA >= tA, sB >= tB
    fraud = y_eval == 1
    groups = {
        "Frauds missed by Model A": np.where(fraud & ~fA)[0],
        "False alarms of Model A": np.where(~fraud & fA)[0],
        "Frauds caught by Model A": np.where(fraud & fA)[0],
        "Models disagree": np.where(fA != fB)[0],
        "All frauds": np.where(fraud)[0],
        "Random legitimate sample": np.random.default_rng(0).choice(np.where(~fraud)[0], 200, replace=False),
    }
    c1, c2, c3 = st.columns([1.2, 2, 1])
    group = c1.selectbox("Show", list(groups), help=f"Uses each model's current threshold (A: {tA:.4f}, B: {tB:.4f}).")
    idx = groups[group]
    idx = idx[np.argsort(-sA[idx])] if len(idx) else idx
    if len(idx) == 0:
        st.info("No transactions in this group at the current thresholds.")
    else:
        pick = c2.selectbox(
            f"Transaction ({len(idx)} in group, sorted by Model A score)", idx,
            format_func=lambda i: (f"#{i} · €{amounts_eval[i]:,.2f} · "
                                   f"{'FRAUD' if y_eval[i] else 'legit'} · A {sA[i]:.4f} · B {sB[i]:.4f}"),
        )
        explain = c3.radio("Explain with", ["A", "B"], format_func=lambda k: MODEL_LABELS[k], horizontal=False)

        x = X_eval[pick]
        info = st.columns(4)
        info[0].markdown(
            f"**Amount** €{amounts_eval[pick]:,.2f}<br>"
            f"<span class='pill {'pill-fraud' if y_eval[pick] else 'pill-legit'}'>"
            f"{'Actually fraud' if y_eval[pick] else 'Actually legitimate'}</span>",
            unsafe_allow_html=True,
        )
        for col, k, s_k, t_k, f_k in [(info[1], "A", sA, tA, fA), (info[2], "B", sB, tB, fB)]:
            pct_rank = 100 * np.mean(s_k <= s_k[pick])
            col.markdown(
                f"**{MODEL_LABELS[k]}**<br>score {s_k[pick]:.6f} · τ {t_k:.4f}<br>"
                f"<span class='pill {'pill-flag' if f_k[pick] else 'pill-legit'}'>"
                f"{'Flagged' if f_k[pick] else 'Not flagged'}</span> "
                f"<span class='note'>higher than {pct_rank:.2f}% of transactions</span>",
                unsafe_allow_html=True,
            )
        m = models[explain]
        contrib = m.contributions(x)
        logit = m.bias + contrib.sum()
        info[3].markdown(
            f"**Logit (log-odds)** {logit:+.2f}<br><span class='note'>= bias {m.bias:+.2f} + Σ feature terms "
            f"{contrib.sum():+.2f}</span>",
            unsafe_allow_html=True,
        )

        order = np.argsort(-np.abs(contrib))
        top_k = order[:10]
        rest = contrib[order[10:]].sum()
        labels = ["Bias"] + [feature_names[i] for i in top_k] + [f"Other {len(order) - 10} features"]
        values = [m.bias] + [float(contrib[i]) for i in top_k] + [float(rest)]
        fig = go.Figure(go.Waterfall(
            x=labels + ["Logit"], y=values + [0], measure=["relative"] * len(values) + ["total"],
            connector=dict(line=dict(color="#C8CDD8")),
            increasing=dict(marker=dict(color=C_FRAUD)), decreasing=dict(marker=dict(color=C_LEGIT)),
            totals=dict(marker=dict(color=C_GOLD)),
            text=[f"{v:+.2f}" for v in values] + [f"{logit:+.2f}"], textposition="outside",
        ))
        t_cur = tA if explain == "A" else tB
        if 0 < t_cur < 1:
            thr_logit = np.log(t_cur / (1 - t_cur))
            fig.add_hline(y=thr_logit, line=dict(color=C_MUTED, dash="dash"),
                          annotation_text=f"current threshold τ = {t_cur:.4f} in log-odds ({thr_logit:+.2f}); "
                                          f"the transaction is flagged if the gold bar ends above this line")
        fig.update_yaxes(title="Contribution to log-odds of fraud")
        st.plotly_chart(base_layout(fig, f"Why {MODEL_LABELS[explain]} scored this transaction as it did", height=430),
                        key="waterfall")
        st.markdown(
            "<span class='note'>For logistic regression this decomposition is exact, not an approximation: "
            "logit = bias + Σ wⱼ·zⱼ, where zⱼ is the feature standardised with training-set mean and standard "
            "deviation. Pink pushes towards fraud, blue towards legitimate. V1–V28 are anonymised PCA components "
            "released by the data owner, so they can be ranked but not named in business terms.</span>",
            unsafe_allow_html=True,
        )
        with st.expander("Raw and standardised feature values"):
            z = m.standardise(x.reshape(1, -1))[0]
            st.dataframe(pd.DataFrame({
                "feature": feature_names, "raw value": x, "standardised z": z,
                "weight w": m.coef, "contribution w·z": contrib,
            }).sort_values("contribution w·z", key=np.abs, ascending=False), hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — How it was built
# ─────────────────────────────────────────────────────────────────────────────
with tab_hood:
    st.markdown(
        f"""
**Pipeline** (identical to the notebook; `build_artifacts.py` re-runs it and refuses to save if the results differ):

1. Drop {meta['n_duplicates_dropped']:,} exact duplicate rows → {meta['n_rows_after_dedup']:,} transactions; `Amount → log1p(Amount)`; drop `Time`.
2. Stratified 70 / 15 / 15 split (seed {meta['seed']}): {meta['n_train']:,} train ({meta['train_fraud']} frauds),
   {meta['n_val']:,} validation ({meta['val_fraud']}), {meta['n_test']:,} test ({meta['test_fraud']}).
3. Z-score normalisation fitted on the training split only.
4. Gradient check before training: relative error **{meta['gradient_check_rel_error']:.1e}** between analytic and finite-difference gradients.
5. **Model A** — balanced class weights (fraud ×{meta['class_weight_fraud']:.1f}, legitimate ×{meta['class_weight_legit']:.3f}),
   batch gradient descent, lr 0.1, up to 2,000 epochs, early stopping on *unweighted* validation loss.
6. **Model B** — from-scratch SMOTE (k = 5) on the training split only, raising fraud to 10% ({meta['smote_fraud']:,} of {meta['smote_rows']:,} rows), then unweighted training with the same settings.
7. Threshold τ* chosen on validation by F2 (recall weighted 2× precision). The test set is used once, for reporting.
"""
    )
    h1, h2 = st.columns(2)
    with h1:
        fig = go.Figure()
        for k in ("A", "B"):
            mm = models[k]
            ep = np.arange(1, len(mm.loss) + 1)
            fig.add_trace(go.Scatter(x=ep, y=mm.loss, name=f"{k} train loss", line=dict(color=C_MODEL[k])))
            fig.add_trace(go.Scatter(x=ep, y=mm.val_loss, name=f"{k} validation loss",
                                     line=dict(color=C_MODEL[k], dash="dot")))
        fig.update_xaxes(title="Epoch"); fig.update_yaxes(title="BCE loss", type="log")
        st.plotly_chart(base_layout(fig, "Training curves"), key="loss")
        st.caption("Validation loss is still falling at epoch 2,000 for both models, so early stopping never "
                   "fired: more epochs (or a larger learning rate) would train them further.")
    with h2:
        rel_fig = go.Figure()
        rel_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                     line=dict(color=C_MUTED, dash="dash")))
        for k in ("A", "B"):
            r = exact_curves(k, split)["rel"]
            rel_fig.add_trace(go.Scatter(
                x=r["mean_pred"], y=r["frac_pos"], mode="lines+markers",
                name=f"{k} (Brier {exact_curves(k, split)['brier']:.4f})", line=dict(color=C_MODEL[k]),
                customdata=np.c_[r["count"], r["lo"], r["hi"]],
                hovertemplate="scores %{customdata[1]:.3f}–%{customdata[2]:.3f}<br>%{customdata[0]:,} transactions"
                              "<br>mean score %{x:.3f}<br>observed fraud rate %{y:.3f}<extra></extra>"))
        rel_fig.update_xaxes(title="Mean predicted probability in bin", range=[0, 1])
        rel_fig.update_yaxes(title="Observed fraud rate in bin", range=[0, 1])
        st.plotly_chart(base_layout(rel_fig, "Calibration (reliability diagram)"), key="rel")
        st.caption("Both models over-predict fraud: class weighting and SMOTE both train on a world where fraud is "
                   "far more common than the real 0.17%. Treat scores as rankings, not probabilities, unless they are "
                   "recalibrated (e.g. Platt scaling on validation, or a prior-shift correction).")

    w = pd.DataFrame({"feature": feature_names, "Model A": models["A"].coef, "Model B": models["B"].coef})
    w = w.reindex(w["Model A"].abs().sort_values(ascending=False).index).head(15)
    wf = go.Figure()
    for k in ("A", "B"):
        wf.add_trace(go.Bar(y=w["feature"], x=w[f"Model {k}"], orientation="h", name=MODEL_LABELS[k],
                            marker_color=C_MODEL[k]))
    wf.update_yaxes(autorange="reversed"); wf.update_xaxes(title="Weight on the standardised feature")
    st.plotly_chart(base_layout(wf, "Largest learned weights (top 15 by |Model A weight|)", height=480), key="weights")

    st.subheader("Numbers you may compare with the notebook")
    st.markdown(
        f"""
- **Operating points match exactly.** At τ* = {models['A'].tau:.4f} (A) and {models['B'].tau:.4f} (B), both models catch
  58 of 70 test frauds with 29 false alarms — the notebook's figures.
- **AUC-PR differs, by design.** The notebook integrates the PR curve over a 300-point threshold grid, which reports
  {meta['notebook_grid_auc_pr_A']:.3f} (A) and {meta['notebook_grid_auc_pr_B']:.3f} (B). Model A's scores pile up near 1.0, between grid points, so the grid
  under-measures it. This app uses every distinct score (exact average precision): **{exact_curves('A', 'test')['ap']:.3f} (A)**
  and **{exact_curves('B', 'test')['ap']:.3f} (B)** on test.
- **The notebook's grid also caps τ at 0.99.** Model A's exact F2 optimum on validation is
  τ = {validation_thresholds('A', fixed_fee, include_amount, review_cost)['f2_exact']:.4f} (the "Exact F2" preset).

**Limitations to keep in mind:** the test set holds only 70 frauds, so single-threshold metrics move in steps of
~1.4 points of recall; features V1–V28 are anonymised PCA components, so the model cannot be explained in business
terms; the data covers two days of September 2013, and real fraud patterns drift.

Source notebook: [`{NB_PATH}`]({REPO_URL})
"""
    )
