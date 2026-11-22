import numpy as np
import pytest

from abtest.analysis import Recommendation, group_sizes_check, load_data, recommend
from abtest.stats import (
    bootstrap_proportion_diff,
    compare_engagement,
    minimum_detectable_effect,
    required_sample_size,
    srm_check,
    two_proportion_ztest,
)


def test_srm_balanced_split_passes():
    result = srm_check(5000, 5000)
    assert result.chi2 == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_srm_lopsided_split_is_flagged():
    assert srm_check(5600, 4400).p_value < 1e-10


def test_ztest_matches_hand_calculation():
    # 100/1000 vs 130/1000: pooled p = 0.115, SE = sqrt(.115*.885*(2/1000)) = 0.014267
    result = two_proportion_ztest(100, 1000, 130, 1000)
    assert result.diff == pytest.approx(0.03)
    assert result.z == pytest.approx(0.03 / 0.0142670, rel=1e-4)
    assert result.p_value == pytest.approx(0.0353, abs=5e-4)
    assert result.significant


def test_ztest_identical_groups_not_significant():
    result = two_proportion_ztest(450, 1000, 450, 1000)
    assert result.p_value == pytest.approx(1.0)
    assert result.ci_low < 0 < result.ci_high


def test_bootstrap_agrees_with_normal_interval():
    z = two_proportion_ztest(4480, 10000, 4420, 10000)
    boot = bootstrap_proportion_diff(4480, 10000, 4420, 10000, n_boot=20000)
    assert boot.ci_low == pytest.approx(z.ci_low, abs=0.002)
    assert boot.ci_high == pytest.approx(z.ci_high, abs=0.002)


def test_bootstrap_is_reproducible_with_seed():
    a = bootstrap_proportion_diff(10, 100, 20, 100, seed=7).diffs
    b = bootstrap_proportion_diff(10, 100, 20, 100, seed=7).diffs
    assert np.array_equal(a, b)


def test_sample_size_and_mde_are_consistent():
    n = required_sample_size(base_rate=0.19, mde=0.01)
    # Detecting a 1-point change on a 19% base rate needs roughly 24-25k users per group.
    assert 23000 < n < 26000
    assert minimum_detectable_effect(0.19, n) == pytest.approx(0.01, rel=0.05)


def test_engagement_detects_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.exponential(10, 2000)
    b = rng.exponential(10, 2000) + 5
    assert compare_engagement(a, b).p_value < 1e-6


def test_recommendation_flags_srm_first():
    primary = two_proportion_ztest(100, 1000, 200, 1000)
    rec = recommend(primary, srm_check(5600, 4400))
    assert isinstance(rec, Recommendation)
    assert rec.decision == "Investigate before deciding"


def test_real_dataset_loads_cleanly():
    df = load_data()
    assert len(df) == 90189
    srm = group_sizes_check(df)
    assert srm.n_a + srm.n_b == 90189
