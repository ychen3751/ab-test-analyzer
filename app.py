"""Streamlit app: analyze the Cookie Cats A/B test and recommend a decision.

Run locally with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from abtest.analysis import (
    CONTROL,
    METRICS,
    TREATMENT,
    group_sizes_check,
    load_data,
    recommend,
    summarize_metric,
)
from abtest.stats import compare_engagement, minimum_detectable_effect, required_sample_size

COLOR_A = "#4C72B0"   # control
COLOR_B = "#DD8452"   # treatment

st.set_page_config(page_title="A/B Test Analyzer: Cookie Cats", layout="wide")


@st.cache_data
def get_data():
    return load_data()


df = get_data()

# --- Sidebar: analysis settings -------------------------------------------
st.sidebar.header("Settings")
primary_metric = st.sidebar.selectbox(
    "Primary metric", list(METRICS), index=1, format_func=METRICS.get,
    help="The metric the decision is based on. 7-day retention reflects long-term engagement.",
)
alpha = st.sidebar.select_slider("Significance level (alpha)", options=[0.01, 0.05, 0.10], value=0.05)
n_boot = st.sidebar.select_slider("Bootstrap samples", options=[1000, 5000, 10000, 20000], value=5000)
srm_threshold = st.sidebar.select_slider(
    "SRM alert threshold", options=[0.0001, 0.001, 0.01], value=0.001,
    help="Flag the test if the group-size check's p-value falls below this.",
)

# --- Header ----------------------------------------------------------------
st.title("A/B Test Analyzer: Cookie Cats")
st.markdown(
    f"""
In the mobile game *Cookie Cats*, players hit a **gate** that makes them wait or pay
before continuing. The team tested moving the first gate from **level 30**
(`{CONTROL}`, control) to **level 40** (`{TREATMENT}`, treatment) across
**{len(df):,} players**. Did the change affect whether players kept coming back?
"""
)

srm = group_sizes_check(df)
results = {m: summarize_metric(df, m, alpha=alpha, n_boot=n_boot) for m in METRICS}
primary_test, primary_boot = results[primary_metric]
rec = recommend(primary_test, srm, srm_threshold=srm_threshold)

banner = {"Keep the gate at level 30": st.success, "Move the gate to level 40": st.success,
          "No clear winner": st.info, "Investigate before deciding": st.warning}[rec.decision]
banner(f"**Recommendation: {rec.decision}.** {rec.reasoning}")

tab_results, tab_checks, tab_engagement, tab_plan = st.tabs(
    ["Results", "Trust checks", "Engagement", "Plan the next test"]
)

# --- Results tab -------------------------------------------------------------
with tab_results:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{METRICS[primary_metric]}: gate 30", f"{primary_test.rate_a:.2%}")
    c2.metric(f"{METRICS[primary_metric]}: gate 40", f"{primary_test.rate_b:.2%}",
              f"{primary_test.diff * 100:+.2f} pts", delta_color="normal")
    c3.metric("Relative change", f"{primary_test.relative_lift:+.1%}")
    c4.metric("p-value", f"{primary_test.p_value:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Difference with confidence interval")
        st.caption("Gate 40 minus gate 30, in percentage points. "
                   "If the bar crosses zero, the change could be chance.")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        labels = [METRICS[m] for m in METRICS]
        for i, m in enumerate(METRICS):
            t = results[m][0]
            color = "#C44E52" if t.significant else "#8C8C8C"
            ax.errorbar(t.diff * 100, i,
                        xerr=[[(t.diff - t.ci_low) * 100], [(t.ci_high - t.diff) * 100]],
                        fmt="o", color=color, capsize=6, markersize=8, linewidth=2)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_xlabel(f"Change in percentage points ({1 - alpha:.0%} CI)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    with right:
        st.subheader("Bootstrap check")
        st.caption(f"{n_boot:,} resamples of the experiment. "
                   "The spread shows how much the difference could vary by chance.")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.hist(primary_boot.diffs * 100, bins=50, color=COLOR_B, alpha=0.8)
        for edge in (primary_boot.ci_low, primary_boot.ci_high):
            ax.axvline(edge * 100, color="black", linestyle=":", linewidth=1)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f"Difference in {METRICS[primary_metric].lower()} (percentage points)")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        st.write(f"Bootstrap {1 - alpha:.0%} interval: **{primary_boot.ci_low * 100:+.2f} to "
                 f"{primary_boot.ci_high * 100:+.2f} points**, matching the z-test interval "
                 f"({primary_test.ci_low * 100:+.2f} to {primary_test.ci_high * 100:+.2f}). "
                 f"Gate 40 came out ahead in **{primary_boot.prob_b_better:.1%}** of resamples.")

    st.subheader("All metrics")
    rows = []
    for m in METRICS:
        t = results[m][0]
        rows.append({
            "Metric": METRICS[m],
            "Gate 30": f"{t.rate_a:.2%}",
            "Gate 40": f"{t.rate_b:.2%}",
            "Change (pts)": f"{t.diff * 100:+.2f}",
            "Relative": f"{t.relative_lift:+.1%}",
            f"{1 - alpha:.0%} CI (pts)": f"{t.ci_low * 100:+.2f} to {t.ci_high * 100:+.2f}",
            "p-value": f"{t.p_value:.4f}",
            "Significant": "Yes" if t.significant else "No",
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    corrected = alpha / len(METRICS)
    survivors = [METRICS[m] for m in METRICS if results[m][0].p_value < corrected]
    st.caption(
        f"Testing {len(METRICS)} metrics raises the chance of a false positive. With a Bonferroni "
        f"correction (alpha / {len(METRICS)} = {corrected:.3f}), significant metrics: "
        f"{', '.join(survivors) if survivors else 'none'}."
    )

# --- Trust checks tab --------------------------------------------------------
with tab_checks:
    st.subheader("Sample ratio mismatch (SRM)")
    st.write(
        f"Players were meant to be split 50/50. Observed: **{srm.n_a:,}** in gate 30 "
        f"({srm.observed_share_a:.2%}) and **{srm.n_b:,}** in gate 40. "
        f"Chi-square = {srm.chi2:.2f}, p = **{srm.p_value:.4f}**."
    )
    if srm.p_value < srm_threshold:
        st.warning("The split is further from 50/50 than chance would explain at the selected "
                   "threshold. Check how players were assigned before trusting results.")
    elif srm.p_value < 0.05:
        st.info(f"The imbalance is unusual (p < 0.05) but above the {srm_threshold} alert "
                "threshold many teams use to avoid false alarms on large tests. "
                "Worth mentioning, not blocking. The source data does not document the intended split.")
    else:
        st.success("Group sizes are consistent with a 50/50 split.")

    st.subheader("Data quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Players", f"{len(df):,}")
    q2.metric("Duplicate user IDs", f"{df['userid'].duplicated().sum()}")
    q3.metric("Missing values", f"{int(df.isna().sum().sum())}")

# --- Engagement tab ----------------------------------------------------------
with tab_engagement:
    rounds_a = df.loc[df["version"] == CONTROL, "sum_gamerounds"].to_numpy()
    rounds_b = df.loc[df["version"] == TREATMENT, "sum_gamerounds"].to_numpy()
    eng = compare_engagement(rounds_a, rounds_b)

    st.subheader("Game rounds played in the first 14 days")
    e1, e2, e3 = st.columns(3)
    e1.metric("Median rounds: gate 30", f"{eng.median_a:.0f}")
    e2.metric("Median rounds: gate 40", f"{eng.median_b:.0f}")
    e3.metric("Mann-Whitney p-value", f"{eng.p_value:.4f}")

    # Zoom in on 0-100 rounds, where the gates sit; share of each group per 2-round bin.
    fig, ax = plt.subplots(figsize=(10, 3.2))
    bins = np.arange(0, 102, 2)
    for rounds, label, color in [(rounds_a, "Gate 30", COLOR_A), (rounds_b, "Gate 40", COLOR_B)]:
        counts, _ = np.histogram(rounds, bins=bins)
        ax.step(bins[:-1], counts / len(rounds) * 100, where="post",
                label=label, color=color, linewidth=2)
    top = ax.get_ylim()[1]
    for gate, color, height in [(30, COLOR_A, 0.92), (40, COLOR_B, 0.78)]:
        ax.axvline(gate, color=color, linestyle=":", linewidth=1.5)
        ax.text(gate + 0.8, top * height, f"gate at level {gate}", fontsize=8, color=color)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Game rounds played (first 100 shown)")
    ax.set_ylabel("Share of group (%)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    over_100 = (df["sum_gamerounds"] > 100).mean()
    st.caption(
        f"{over_100:.0%} of players played more than 100 rounds, and one played "
        f"{int(df['sum_gamerounds'].max()):,}. Because the metric is so skewed, the comparison "
        "uses medians and a rank-based test instead of means and a t-test."
    )

# --- Planning tab ------------------------------------------------------------
with tab_plan:
    st.subheader("How big does the next test need to be?")
    p1, p2, p3 = st.columns(3)
    base = p1.number_input("Baseline rate (%)", 1.0, 99.0,
                           round(primary_test.rate_a * 100, 1), step=0.5) / 100
    mde = p2.number_input("Smallest change worth detecting (percentage points)",
                          0.1, 20.0, 1.0, step=0.1) / 100
    power = p3.select_slider("Power", options=[0.7, 0.8, 0.9, 0.95], value=0.8)

    n_needed = required_sample_size(base, mde, alpha=alpha, power=power)
    st.metric("Players needed per group", f"{n_needed:,}")

    actual_mde = minimum_detectable_effect(primary_test.rate_a, min(srm.n_a, srm.n_b), alpha, power)
    st.write(
        f"With about {min(srm.n_a, srm.n_b):,} players per group, this experiment could reliably "
        f"detect a change of **{actual_mde * 100:.2f} points** in {METRICS[primary_metric].lower()} "
        f"at {power:.0%} power. The observed change was {abs(primary_test.diff) * 100:.2f} points."
    )

st.divider()
st.caption("Data: Cookie Cats mobile game A/B test (90,189 players), a public dataset "
           "originally released for a DataCamp project and widely shared on Kaggle.")
