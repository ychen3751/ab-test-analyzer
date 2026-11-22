"""Statistical tools for analyzing a two-group A/B test.

Every function takes plain numbers or NumPy arrays and returns a small
dataclass, so the same code powers the Streamlit app, the command-line
report, and the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Sample ratio mismatch (SRM): was traffic split the way we intended?
# ---------------------------------------------------------------------------

@dataclass
class SRMResult:
    n_a: int
    n_b: int
    expected_share_a: float
    chi2: float
    p_value: float

    @property
    def observed_share_a(self) -> float:
        return self.n_a / (self.n_a + self.n_b)


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> SRMResult:
    """Chi-square goodness-of-fit test on group sizes.

    A very small p-value means the split is unlikely to come from the
    intended assignment ratio, which can point to a bug in randomization or
    logging. In that case the test results should be treated with caution.
    """
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    chi2, p_value = stats.chisquare([n_a, n_b], f_exp=expected)
    return SRMResult(n_a, n_b, expected_share_a, float(chi2), float(p_value))


# ---------------------------------------------------------------------------
# 2. Two-proportion z-test: did the conversion/retention rate change?
# ---------------------------------------------------------------------------

@dataclass
class ProportionTestResult:
    x_a: int
    n_a: int
    x_b: int
    n_b: int
    alpha: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def rate_a(self) -> float:
        return self.x_a / self.n_a

    @property
    def rate_b(self) -> float:
        return self.x_b / self.n_b

    @property
    def diff(self) -> float:
        """Absolute difference, B minus A (in proportion units)."""
        return self.rate_b - self.rate_a

    @property
    def relative_lift(self) -> float:
        """Difference as a share of the control rate."""
        return self.diff / self.rate_a

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int,
                         alpha: float = 0.05) -> ProportionTestResult:
    """Two-sided z-test for a difference between two proportions.

    The test statistic uses the pooled rate (the standard choice under the
    null hypothesis that both groups share one rate). The confidence
    interval uses the unpooled standard error, because it describes the
    difference we actually observed rather than the null.
    """
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)

    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return ProportionTestResult(
        x_a, n_a, x_b, n_b, alpha,
        z=float(z), p_value=float(p_value),
        ci_low=float(diff - z_crit * se_unpooled),
        ci_high=float(diff + z_crit * se_unpooled),
    )


# ---------------------------------------------------------------------------
# 3. Bootstrap: a distribution-free check on the same question
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    diffs: np.ndarray
    alpha: float

    @property
    def ci_low(self) -> float:
        return float(np.quantile(self.diffs, self.alpha / 2))

    @property
    def ci_high(self) -> float:
        return float(np.quantile(self.diffs, 1 - self.alpha / 2))

    @property
    def prob_b_better(self) -> float:
        """Share of bootstrap samples where B's rate beat A's."""
        return float(np.mean(self.diffs > 0))


def bootstrap_proportion_diff(x_a: int, n_a: int, x_b: int, n_b: int,
                              n_boot: int = 5000, alpha: float = 0.05,
                              seed: int = 42) -> BootstrapResult:
    """Percentile bootstrap for the difference in two proportions.

    Resampling n users with replacement from a 0/1 column and counting the
    1s is exactly a Binomial(n, observed rate) draw, so we sample the counts
    directly. That gives the same result as resampling rows, much faster.
    """
    rng = np.random.default_rng(seed)
    boot_a = rng.binomial(n_a, x_a / n_a, size=n_boot) / n_a
    boot_b = rng.binomial(n_b, x_b / n_b, size=n_boot) / n_b
    return BootstrapResult(diffs=boot_b - boot_a, alpha=alpha)


# ---------------------------------------------------------------------------
# 4. Engagement metric: compare a skewed continuous metric
# ---------------------------------------------------------------------------

@dataclass
class EngagementResult:
    median_a: float
    median_b: float
    mean_a: float
    mean_b: float
    u_stat: float
    p_value: float


def compare_engagement(values_a: np.ndarray, values_b: np.ndarray) -> EngagementResult:
    """Mann-Whitney U test, which suits heavily skewed metrics like game rounds.

    A t-test on means would be dominated by a handful of extreme players;
    Mann-Whitney compares the groups by rank instead.
    """
    u_stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    return EngagementResult(
        median_a=float(np.median(values_a)), median_b=float(np.median(values_b)),
        mean_a=float(np.mean(values_a)), mean_b=float(np.mean(values_b)),
        u_stat=float(u_stat), p_value=float(p_value),
    )


# ---------------------------------------------------------------------------
# 5. Power analysis: how big a test do we need?
# ---------------------------------------------------------------------------

def required_sample_size(base_rate: float, mde: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Users needed per group to detect an absolute change of `mde`.

    Standard normal-approximation formula for a two-sided, two-proportion
    test with equal group sizes.
    """
    p1, p2 = base_rate, base_rate + mde
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(numerator / mde ** 2))


def minimum_detectable_effect(base_rate: float, n_per_group: int,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute change the test could reliably detect at this size."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * np.sqrt(2 * base_rate * (1 - base_rate) / n_per_group))
