# A/B Test Analyzer: Cookie Cats

**Live demo:** https://cookie-cats-ab-test-analyzer.streamlit.app

An interactive Streamlit app that analyzes a real mobile-game A/B test with 90,189 players and turns the statistics into a plain-English product decision.

**Finding:** moving the game's first gate from level 30 to level 40 **lowered 7-day retention by 0.82 percentage points** (19.02% to 18.20%, a 4.3% relative drop; 95% CI −1.33 to −0.31 points, p = 0.0016). **Recommendation: keep the gate at level 30.** 1-day retention also dipped (−0.59 points) but that change was not statistically significant (p = 0.074).

## The experiment

In *Cookie Cats*, players reach a "gate" that forces them to wait or pay before continuing. The developers randomly assigned new players to one of two versions:

| Group | Gate position | Players |
|---|---|---|
| `gate_30` (control) | Level 30 | 44,700 |
| `gate_40` (treatment) | Level 40 | 45,489 |

Each row records a player's group, how many rounds they played in their first 14 days, and whether they came back 1 day and 7 days after installing.

## What the app does

1. **Recommendation banner**: a decision based on the selected primary metric, with the effect size and confidence interval.
2. **Results**: two-proportion z-tests for 1-day and 7-day retention, a confidence-interval chart, a bootstrap distribution that independently checks the z-test, and a Bonferroni note for testing two metrics.
3. **Trust checks**: a sample ratio mismatch (SRM) test on group sizes, plus data-quality counts.
4. **Engagement**: game rounds compared with medians and a Mann-Whitney U test, because the metric is heavily skewed (one player logged 49,854 rounds).
5. **Plan the next test**: a power calculator for required sample size, and the minimum detectable effect of this experiment.

Alpha, bootstrap size, the SRM alert threshold, and the primary metric can all be changed from the sidebar.

## Results

| Metric | Gate 30 | Gate 40 | Change | 95% CI | p-value |
|---|---|---|---|---|---|
| 1-day retention | 44.82% | 44.23% | −0.59 pts | −1.24 to +0.06 | 0.074 |
| 7-day retention | 19.02% | 18.20% | −0.82 pts | −1.33 to −0.31 | 0.0016 |

- The 5,000-sample bootstrap gives nearly the same 7-day interval (−1.33 to −0.32), and gate 40 beat gate 30 in only 0.2% of resamples.
- 7-day retention stays significant after a Bonferroni correction for two metrics (p < 0.025).
- **Caveat:** the groups are not exactly 50/50 (SRM chi-square p = 0.0086). That is below 0.05 but above the stricter 0.001 threshold commonly used for large tests, and the dataset does not document the intended split. The app reports it rather than hiding it.
- Median rounds played: 17 (gate 30) vs 16 (gate 40); Mann-Whitney p = 0.050, borderline.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pytest`.

## Project structure

```
app.py               Streamlit app (charts, tabs, sidebar settings)
abtest/stats.py      Statistical functions: SRM, z-test, bootstrap, Mann-Whitney, power
abtest/analysis.py   Data loading, per-metric summaries, recommendation logic
tests/test_stats.py  Unit tests, including hand-calculated z-test values
data/cookie_cats.csv The experiment data
```

The statistics live in plain functions with no Streamlit code, so they are unit-tested on their own and could be reused in a notebook or a pipeline.

## Data source

The Cookie Cats dataset was published for a DataCamp project and is widely shared on Kaggle and GitHub. This copy has 90,189 rows with columns `userid`, `version`, `sum_gamerounds`, `retention_1`, and `retention_7`.
