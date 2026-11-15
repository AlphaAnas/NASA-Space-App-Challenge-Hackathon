"""Synthetic validation for transparent Earth-science trend analysis."""

import numpy as np
import pandas as pd
import pytest

import analyze as az


def monthly(start="2000-01-01", years=20):
    """Make a month-start index with a repeatable seasonal cycle."""
    return pd.date_range(start, periods=years * 12, freq="MS")


def seasonal_noise(n, rng, scale=0.4):
    """Make noisy values with a stable monthly climatology."""
    return np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, scale, n)


def test_known_trend_ci_and_validation_plot():
    """Recover a known rate and save a presentation validation plot."""
    rng = np.random.default_rng(15); ix = monthly(years=20); x = az._decimal_year(ix)
    y = seasonal_noise(len(ix), rng) + 0.03 * (x - x[0])
    series = pd.Series(y, index=ix)
    result = az.analyze_series(series, "file unit", "synthetic", 2000, 2019)
    assert result.ci_low <= result.slope_per_decade <= result.ci_high
    assert result.slope_per_decade == pytest.approx(0.3, abs=0.2)
    az.save_plot(result, "validation/known_trend.png")
    assert __import__("pathlib").Path("validation/known_trend.png").exists()


def test_false_positive_rate_including_autocorrelated_noise():
    """Check the nominal false-positive rate on independent and AR(1) null series."""
    rng = np.random.default_rng(234); false = 0; runs = 500
    for run in range(runs):
        noise = rng.normal(size=240)
        if run % 2:
            for i in range(1, len(noise)):
                noise[i] += 0.5 * noise[i - 1]
        false += az.test_significance(noise)["p_value"] < az.ALPHA
    assert 0.015 <= false / runs <= 0.12


def test_pure_seasonal_cycle_has_no_trend():
    """A repeated seasonal cycle should not be mistaken for long-term change."""
    ix = monthly(years=20); y = np.sin(2 * np.pi * np.arange(len(ix)) / 12)
    result = az.analyze_series(pd.Series(y, index=ix), "u", "cycle", 2000, 2019)
    assert result.p_value >= az.ALPHA
    assert "No significant trend found" in result.verdict


def test_step_change_flagged_by_pettitt():
    """A persistent level shift should trigger the change-point flag."""
    y = np.r_[np.zeros(100), np.ones(100) * 3]
    result = az.check_linearity(y, np.arange(len(y), dtype=float))
    assert result["changepoint_p"] < az.ALPHA
    assert result["nonlinear"]


@pytest.mark.parametrize("series,message", [
    (pd.Series(np.arange(60), index=monthly(years=5)), "too short"),
    (pd.Series([np.nan] * 60 + [1.0] * 60, index=monthly(years=10)), "missing"),
])
def test_short_or_missing_refused(series, message):
    """Quality checks explain why inadequate records cannot be analyzed."""
    with pytest.raises(ValueError, match=message): az.check_quality(series)


def test_missing_values_never_become_zero(tmp_path):
    """Blank CSV values and absent months remain NaN after loading."""
    path = tmp_path / "input.csv"
    path.write_text("date,place,variable,value,unit\n2000-01,X,T,2,K\n2000-03,X,T,,K\n", encoding="utf-8")
    series, unit = az.load_series(path)
    assert unit == "K"
    assert series.iloc[1] != 0 and np.isnan(series.iloc[1])
    assert np.isnan(series.iloc[2])


def test_shared_trend_does_not_create_detrended_association():
    """Independent residuals should not appear strongly associated after trend removal."""
    rng = np.random.default_rng(22); ix = monthly(years=20); x = np.arange(len(ix)) / 120
    a = pd.Series(x + rng.normal(size=len(ix)), index=ix, name="a")
    b = pd.Series(x + rng.normal(size=len(ix)), index=ix, name="b")
    assert az.compare_series(a, b).strength != "strong"


def test_shared_noise_is_strong():
    """Two series driven by the same noise should be reported as strongly associated."""
    rng = np.random.default_rng(31); ix = monthly(years=15); shared = rng.normal(size=len(ix))
    a = pd.Series(shared + rng.normal(0, .05, len(ix)), index=ix, name="a")
    b = pd.Series(shared + rng.normal(0, .05, len(ix)), index=ix, name="b")
    assert az.compare_series(a, b).strength == "strong"


def test_lagged_pair_recovers_two_month_lead():
    """Search direction reports a positive lag when B leads A by two months."""
    rng = np.random.default_rng(42); ix = monthly(years=15); b = rng.normal(size=len(ix))
    a = np.r_[0, 0, b[:-2]]
    result = az.compare_series(pd.Series(a, index=ix), pd.Series(b, index=ix), max_lag_months=4)
    assert result.best_lag_months == 2


def test_difference_distinguishes_extra_and_shared_change():
    """Control subtraction isolates an extra trend and cancels shared changes."""
    rng = np.random.default_rng(12); ix = monthly(years=20); x = np.arange(len(ix)) / 120
    outside = pd.Series(x + rng.normal(0, .15, len(ix)), index=ix)
    city_extra = outside + x
    diff_extra = az.difference_series(city_extra, outside)
    extra = az.analyze_series(diff_extra, "u", "city", 2000, 2019)
    assert extra.slope_per_decade > 0 and extra.p_value < az.ALPHA
    city_same = outside + rng.normal(0, .02, len(ix))
    diff_same = az.difference_series(city_same, outside)
    flat = az.analyze_series(diff_same, "u", "city", 2000, 2019)
    assert flat.p_value >= az.ALPHA

