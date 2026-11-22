"""Loads the Cookie Cats experiment and turns the statistics into a decision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from abtest.stats import (
    BootstrapResult,
    ProportionTestResult,
    SRMResult,
    bootstrap_proportion_diff,
    srm_check,
    two_proportion_ztest,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cookie_cats.csv"

CONTROL = "gate_30"     # A: first gate at level 30 (the original game)
TREATMENT = "gate_40"   # B: first gate moved to level 40

METRICS = {
    "retention_1": "1-day retention",
    "retention_7": "7-day retention",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if not df["userid"].is_unique:
        raise ValueError("Each user should appear once; found duplicate user IDs.")
    for col in METRICS:
        df[col] = df[col].astype(bool)
    return df


def summarize_metric(df: pd.DataFrame, metric: str, alpha: float = 0.05,
                     n_boot: int = 5000) -> tuple[ProportionTestResult, BootstrapResult]:
    a = df.loc[df["version"] == CONTROL, metric]
    b = df.loc[df["version"] == TREATMENT, metric]
    x_a, n_a, x_b, n_b = int(a.sum()), len(a), int(b.sum()), len(b)
    test = two_proportion_ztest(x_a, n_a, x_b, n_b, alpha=alpha)
    boot = bootstrap_proportion_diff(x_a, n_a, x_b, n_b, n_boot=n_boot, alpha=alpha)
    return test, boot


def group_sizes_check(df: pd.DataFrame) -> SRMResult:
    counts = df["version"].value_counts()
    return srm_check(int(counts[CONTROL]), int(counts[TREATMENT]))


@dataclass
class Recommendation:
    decision: str
    reasoning: str


def recommend(primary: ProportionTestResult, srm: SRMResult,
              srm_threshold: float = 0.001) -> Recommendation:
    """Turns the primary metric's result into a plain-English call."""
    pts = abs(primary.diff) * 100
    ci = f"{primary.ci_low * 100:+.2f} to {primary.ci_high * 100:+.2f} percentage points"

    if srm.p_value < srm_threshold:
        return Recommendation(
            "Investigate before deciding",
            "Group sizes differ more than random assignment would explain "
            f"(SRM p = {srm.p_value:.4f}). Check the assignment and logging "
            "before trusting any metric.",
        )
    if primary.significant and primary.diff < 0:
        return Recommendation(
            "Keep the gate at level 30",
            f"Moving the gate to level 40 lowered the rate by {pts:.2f} percentage "
            f"points ({primary.relative_lift:+.1%} relative; {1 - primary.alpha:.0%} CI {ci}).",
        )
    if primary.significant and primary.diff > 0:
        return Recommendation(
            "Move the gate to level 40",
            f"Moving the gate raised the rate by {pts:.2f} percentage points "
            f"({primary.relative_lift:+.1%} relative; {1 - primary.alpha:.0%} CI {ci}).",
        )
    return Recommendation(
        "No clear winner",
        f"The observed change ({primary.diff * 100:+.2f} points) is within what "
        f"chance alone could produce ({1 - primary.alpha:.0%} CI {ci}). "
        "Keeping the current design is the low-risk default.",
    )
