# Load monthly observations and make gaps explicit.
# Select the requested years or aggregate a selected season by year.
# Check record length and missingness before any statistical test.
# Remove the calendar-month climatology for monthly trend analysis.
# Estimate a robust Theil-Sen slope and its confidence interval.
# Test monotonic trend and correct for meaningful lag-one dependence.
# Compare LOWESS with the robust line and test for a step change.
# Describe the measured trend and its statistical uncertainty in plain English.
# Extend the fitted line conditionally and quantify a simple held-out check.
# Compare aligned series only as association evidence, never proof of cause.
"""Tools for transparent trend analysis of monthly Earth-science observations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats

ALPHA = 0.05
MIN_YEARS = 10
MAX_MISSING_FRAC = 0.2
STRONG_R = 0.6
MODERATE_R = 0.3


def _json_value(value: Any) -> Any:
    """Convert numpy, pandas, and non-finite values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    return value


@dataclass
class AnalysisResult:
    """Computed single-series analysis, including plot-ready observations."""
    label: str
    unit: str
    verdict: str
    warnings: list[str]
    slope_per_decade: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    test_used: str
    seasonal_p_value: float | None
    nonlinear: bool
    changepoint: str | None
    slope_before: float | None
    slope_after: float | None
    dates: list[str]
    raw: list[float | None]
    deseasonalized: list[float | None]
    trend_line: list[float | None]
    lowess: list[float | None]
    forecast_dates: list[str]
    forecast: list[float | None]
    forecast_low: list[float | None]
    forecast_high: list[float | None]
    backtest_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a JSON-safe dictionary."""
        return _json_value(asdict(self))


@dataclass
class ComparisonResult:
    """Aligned association statistics and plain-language interpretation."""
    label_a: str
    label_b: str
    unit_a: str
    unit_b: str
    rho: float | None
    p_value: float | None
    detrended_rho: float | None
    detrended_p_value: float | None
    strength: str
    best_lag_months: int
    best_lag_rho: float | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a JSON-safe dictionary."""
        return _json_value(asdict(self))


def load_series(path: str | Path, place: str | None = None,
                variable: str | None = None) -> tuple[pd.Series, str]:
    """Read validated CSV observations; return a complete monthly Series and its file-provided unit."""
    frame = pd.read_csv(path, dtype={"date": "string", "place": "string", "variable": "string", "unit": "string"})
    required = {"date", "place", "variable", "value", "unit"}
    missing_cols = required - set(frame.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}")
    if place is not None: frame = frame[frame.place == place]
    if variable is not None: frame = frame[frame.variable == variable]
    if frame.empty:
        raise ValueError("No rows match the requested place and variable")
    parsed = pd.to_datetime(frame.date + "-01", format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any(): raise ValueError("date values must use YYYY-MM")
    raw_value = frame["value"]
    numeric_value = pd.to_numeric(raw_value, errors="coerce")
    # A blank value is missing data; invalid nonblank values are rejected to avoid silently hiding bad input.
    invalid = numeric_value.isna() & raw_value.notna() & raw_value.astype(str).str.strip().ne("")
    if invalid.any(): raise ValueError("value contains non-numeric entries")
    frame = frame.assign(date=parsed.dt.to_period("M").dt.to_timestamp(), value=numeric_value)
    if frame.date.duplicated().any(): raise ValueError("Expected at most one row per month after filtering")
    units = frame.unit.dropna().unique()
    if len(units) != 1: raise ValueError("Selected rows must have exactly one consistent unit")
    series = frame.set_index("date").value.astype(float).sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    series = series.reindex(full_index)
    series.index.name = "date"
    series.attrs["place"] = str(frame.place.iloc[0])
    series.attrs["variable"] = str(frame.variable.iloc[0])
    series.attrs["unit"] = str(units[0])
    return series, str(units[0])


def select_period(series: pd.Series, start_year: int, end_year: int,
                  months: Sequence[int] | None = None) -> pd.Series:
    """Crop to inclusive calendar years, optionally averaging selected months within each year."""
    if end_year < start_year: raise ValueError("end_year must be greater than or equal to start_year")
    if months is not None and (not months or any(m < 1 or m > 12 for m in months)):
        raise ValueError("months must contain month numbers from 1 to 12")
    cropped = series[(series.index.year >= start_year) & (series.index.year <= end_year)]
    if months is None: return cropped.copy()
    chosen = cropped[cropped.index.month.isin(months)]
result = chosen.groupby(chosen.index.year).agg(lambda x: x.mean() if x.notna().sum() >= (len(months) + 1) // 2 else np.nan)
    result = result.reindex(range(start_year, end_year + 1))
    result.index = pd.to_datetime(result.index.astype(str) + "-01-01")
    result.index.name = "date"
    result.attrs.update(series.attrs)
    result.attrs["yearly"] = True
    return result


def check_quality(series: pd.Series) -> list[str]:
    """Return data-quality caveats, raising ValueError when length or missingness is unacceptable."""
    monthly = not bool(series.attrs.get("yearly", False))
    observed_years = series.dropna().shape[0] / (12 if monthly else 1)
    missing_fraction = float(series.isna().mean()) if len(series) else 1.0
    if missing_fraction > MAX_MISSING_FRAC:
        raise ValueError(f"Too much missing data: {missing_fraction:.1%}; maximum is {MAX_MISSING_FRAC:.0%}")
    if observed_years < MIN_YEARS:
        raise ValueError(f"Record is too short: {observed_years:.1f} observed years; need at least {MIN_YEARS}")
    warnings = []
    if missing_fraction > 0: warnings.append(f"{missing_fraction:.1%} of months/years are missing; no values were interpolated.")
    if observed_years < 15: warnings.append("Short record: uncertainty in trend estimates may be substantial.")
    return warnings


def deseasonalize(series: pd.Series) -> pd.Series:
    """Subtract the calendar-month mean for monthly data; leave yearly aggregates unchanged."""
    if bool(series.attrs.get("yearly", False)): return series.copy()
    means = series.groupby(series.index.month).transform("mean")
    result = series - means
    result.attrs.update(series.attrs)
    return result


def _decimal_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert dates to decimal calendar years, preserving monthly spacing."""
    return index.year.to_numpy(dtype=float) + (index.month.to_numpy(dtype=float) - 1) / 12


def estimate_trend(values: Sequence[float], years: Sequence[float]) -> tuple[float, float, float]:
    """Return Theil-Sen slope and 95% confidence limits, all expressed per decade."""
    x, y = np.asarray(years, float), np.asarray(values, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2: raise ValueError("At least two finite observations are needed to estimate a trend")
    fit = stats.theilslopes(y[valid], x[valid], alpha=0.95)
    return tuple(float(v * 10) for v in (fit.slope, fit.low_slope, fit.high_slope))


def test_significance(values: Sequence[float], monthly: bool = True) -> dict[str, Any]:
    """Run Mann-Kendall with pre-whitening for significant lag-one dependence."""
    y = np.asarray(values, float); y = y[np.isfinite(y)]
    if len(y) < 2: raise ValueError("At least two finite observations are needed for significance testing")
    rho = float(pd.Series(y).autocorr(lag=1)) if len(y) > 2 else 0.0
    # A nominal lag-one test can be over-sensitive for long series, so require both p<alpha and a nontrivial correlation.
    ac_p = stats.pearsonr(y[:-1], y[1:]).pvalue if len(y) > 3 and np.std(y[:-1]) and np.std(y[1:]) else 1.0
    corrected = np.isfinite(rho) and abs(rho) >= 0.1 and ac_p < ALPHA
    test = mk.pre_whitening_modification_test(y) if corrected else mk.original_test(y)
    seasonal_p = None
    if monthly and len(y) >= 24:
        seasonal_p = float(mk.seasonal_test(y, period=12).p)
    return {"p_value": float(test.p), "test_used": "Pre-whitened Mann-Kendall" if corrected else "Mann-Kendall",
            "autocorrelation": rho, "seasonal_p_value": seasonal_p}


def _pettitt(values: np.ndarray) -> tuple[int | None, float]:
    """Return the Pettitt change-point index and approximate two-sided p-value."""
    n = len(values)
    if n < 3: return None, 1.0
    ranks = stats.rankdata(values)
    # U_t compares every value before the split with every value after it.
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    k = int(np.argmax(np.abs(u)))
    p = min(1.0, 2 * np.exp((-6 * float(u[k] ** 2)) / (n ** 3 + n ** 2)))
    return k, p


def check_linearity(values: Sequence[float], years: Sequence[float]) -> dict[str, Any]:
    """Fit LOWESS, test a Pettitt step change, and report pre/post slopes when nonlinearity is flagged."""
    x, y = np.asarray(years, float), np.asarray(values, float)
    good = np.isfinite(x) & np.isfinite(y)
    original_positions = np.flatnonzero(good)
    x, y = x[good], y[good]
    if len(y) < 3: return {"lowess": np.full(len(good), np.nan), "nonlinear": False, "changepoint_index": None, "changepoint_p": 1.0, "slope_before": None, "slope_after": None}
    span = max(float(np.ptp(x)), 1e-9)
    lowess = sm.nonparametric.lowess(y, x, frac=min(0.67, max(0.25, 10 / len(y))), return_sorted=False)
    slope, intercept, _, _ = stats.theilslopes(y, x)
    line = intercept + slope * x
    scale = max(float(np.nanstd(y)), 1e-12)
    far = float(np.nanmax(np.abs(lowess - line))) > 0.25 * scale
    cp, cp_p = _pettitt(y)
    nonlinear = far or cp_p < ALPHA
    before = after = None
    if nonlinear and cp is not None and cp >= 2 and len(y) - cp >= 2:
        before = estimate_trend(y[:cp], x[:cp])[0]
        after = estimate_trend(y[cp:], x[cp:])[0]
    lowess_full = np.full(len(good), np.nan); lowess_full[original_positions] = lowess
    cp_original = int(original_positions[cp]) if cp is not None else None
    return {"lowess": lowess_full, "nonlinear": nonlinear, "changepoint_index": cp_original if cp_p < ALPHA else None,
            "changepoint_p": cp_p, "slope_before": before, "slope_after": after}


def write_verdict(label: str, unit: str, slope: float, low: float, high: float,
                  p_value: float, warnings: Sequence[str] = (), nonlinear: bool = False,
                  corrected: bool = False, difference: bool = False) -> str:
    """Build a plain-English verdict solely from supplied computed estimates and flags."""
    direction = "increased" if slope > 0 else "decreased" if slope < 0 else "changed"
    subject = f"Extra change in {label} compared with its control" if difference else label
    base = f"{subject} {direction} by {abs(slope):.3g} {unit} per decade (95% range {low:.3g} to {high:.3g})."
    if p_value < ALPHA: base += f" This is a statistically significant trend (p={p_value:.3g})."
    elif p_value < 0.10: base += f" There is weak evidence of a trend (p={p_value:.3g})."
    else: base += f" No significant trend found (p={p_value:.3g})."
    if difference and p_value < ALPHA and slope > 0: base += " A is changing faster than B."
    if difference and p_value >= ALPHA: base += " The areas change together within the uncertainty of this difference series."
    if nonlinear: base += " The pattern is flagged as non-linear; one constant rate may not describe the whole record."
    if corrected: base += " The p-value uses an autocorrelation-corrected test."
    if warnings: base += " " + " ".join(warnings)
    return base


def forecast_if_continues(values: Sequence[float], years: Sequence[float], horizon_years: int = 10) -> dict[str, Any]:
    """Extend a Theil-Sen line conditionally with a residual-spread band; this is not a prediction."""
    x, y = np.asarray(years, float), np.asarray(values, float)
    good = np.isfinite(x) & np.isfinite(y); x, y = x[good], y[good]
    slope, intercept, _, _ = stats.theilslopes(y, x)
    resid = y - (intercept + slope * x)
    spread = float(np.sqrt(np.nanmean(resid ** 2)))
    future = np.arange(np.floor(x.max()) + 1, np.floor(x.max()) + horizon_years + 1, dtype=float)
    center = intercept + slope * future
    return {"years": future, "values": center, "low": center - 1.96 * spread, "high": center + 1.96 * spread}


def backtest_forecast(values: Sequence[float], years: Sequence[float], holdout_years: int = 5) -> float | None:
    """Return the fraction of final held-out observations inside a training residual-spread band."""
    x, y = np.asarray(years, float), np.asarray(values, float)
    valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]
    cut = x.max() - holdout_years
    train = x < cut; test = ~train
    if train.sum() < 3 or not test.any(): return None
    slope, intercept, _, _ = stats.theilslopes(y[train], x[train])
    spread = float(np.sqrt(np.mean((y[train] - (intercept + slope * x[train])) ** 2)))
    pred = intercept + slope * x[test]
    return float(np.mean(np.abs(y[test] - pred) <= 1.96 * spread))


def analyze_series(series: pd.Series, unit: str, label: str, start_year: int,
                   end_year: int, months: Sequence[int] | None = None) -> AnalysisResult:
    """Run quality checks and trend analysis on an in-memory series, including difference series."""
    selected = select_period(series, start_year, end_year, months)
    warnings = check_quality(selected)
    clean = selected.dropna()
    monthly = months is None and not bool(selected.attrs.get("yearly", False))
    analyzed = deseasonalize(selected) if monthly else selected.copy()
    vals = analyzed.to_numpy(float); x = _decimal_year(pd.DatetimeIndex(analyzed.index))
    slope, low, high = estimate_trend(vals, x)
    significance = test_significance(vals, monthly=monthly)
    linear = check_linearity(vals, x)
    verdict = write_verdict(label, unit, slope, low, high, significance["p_value"], warnings,
                            linear["nonlinear"], significance["test_used"] != "Mann-Kendall",
                            bool(series.attrs.get("difference", False)))
    forecast = forecast_if_continues(vals, x)
    trend = np.full(len(x), np.nan)
    valid = np.isfinite(vals)
    fit = stats.theilslopes(vals[valid], x[valid])
    trend[valid] = fit.intercept + fit.slope * x[valid]
    return AnalysisResult(label, unit, verdict, warnings, slope, low, high, significance["p_value"],
        significance["test_used"], significance["seasonal_p_value"], linear["nonlinear"],
        str(analyzed.index[linear["changepoint_index"]].date()) if linear["changepoint_index"] is not None else None,
        linear["slope_before"], linear["slope_after"], [d.isoformat() for d in analyzed.index],
        _json_value(selected.to_numpy(float).tolist()), _json_value(vals.tolist()), _json_value(trend.tolist()),
        _json_value(np.asarray(linear["lowess"]).tolist()),
        [pd.Timestamp(year=int(y), month=1, day=1).isoformat() for y in forecast["years"]],
        _json_value(forecast["values"].tolist()), _json_value(forecast["low"].tolist()),
        _json_value(forecast["high"].tolist()), backtest_forecast(vals, x))


def analyze(path: str | Path, place: str | None, variable: str | None,
            start_year: int = 2005, end_year: int = 2024,
            months: Sequence[int] | None = None) -> AnalysisResult:
    """Load one CSV series, select the requested period, and analyze it."""
    series, unit = load_series(path, place, variable)
    label = " ".join(x for x in (variable, place) if x) or Path(path).stem
    return analyze_series(series, unit, label, start_year, end_year, months)


def difference_series(a: pd.Series, b: pd.Series) -> pd.Series:
    """Align two series by date and return A minus B, tagged for extra-change interpretation."""
    unit_a, unit_b = a.attrs.get("unit"), b.attrs.get("unit")
    if unit_a and unit_b and unit_a != unit_b:
        raise ValueError(f"Cannot subtract series with different units ({unit_a!r} and {unit_b!r})")
    aligned_a, aligned_b = a.align(b, join="inner")
    result = aligned_a - aligned_b
    result.name = "difference"
    result.attrs["difference"] = True
    result.attrs["yearly"] = bool(a.attrs.get("yearly", False) and b.attrs.get("yearly", False))
    result.attrs["unit"] = unit_a or unit_b or ""
    return result


def compare_series(a: pd.Series, b: pd.Series, months: Sequence[int] | None = None,
                   max_lag_months: int = 0) -> ComparisonResult:
    """Align, deseasonalize, and compare two series with raw and detrended Spearman statistics."""
    aa, bb = a.align(b, join="inner")
    if months is not None:
        aa = select_period(aa, int(aa.index.year.min()), int(aa.index.year.max()), months)
        bb = select_period(bb, int(bb.index.year.min()), int(bb.index.year.max()), months)
    aa, bb = aa.align(bb, join="inner")
    da, db = deseasonalize(aa), deseasonalize(bb)
    pair = pd.concat([da, db], axis=1).dropna()
    if len(pair) < 3: raise ValueError("At least three paired observations are needed for comparison")
    rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    x = _decimal_year(pd.DatetimeIndex(pair.index))
    ra = pair.iloc[:, 0].to_numpy() - np.polyval(np.polyfit(x, pair.iloc[:, 0], 1), x)
    rb = pair.iloc[:, 1].to_numpy() - np.polyval(np.polyfit(x, pair.iloc[:, 1], 1), x)
    dr, dp = stats.spearmanr(ra, rb)
    best_lag, best_r = 0, float(dr)
    for lag in range(1, max_lag_months + 1):
        # Positive lag shifts B forward in time, corresponding to B leading A.
        shifted = pd.concat([da.rename("a"), db.shift(lag).rename("b")], axis=1).dropna()
        if len(shifted) >= 3:
            lr = float(stats.spearmanr(shifted.a, shifted.b).statistic)
            if abs(lr) > abs(best_r): best_lag, best_r = lag, lr
    strength = "strong" if abs(dr) >= STRONG_R else "moderate" if abs(dr) >= MODERATE_R else "weak/none"
    verdict = f"{a.name or 'Series A'} {strength}ly goes together with {b.name or 'Series B'} (detrended Spearman r={dr:.3g}, p={dp:.3g}). Correlation is not proof of cause."
    return ComparisonResult(str(a.name or "A"), str(b.name or "B"), str(a.attrs.get("unit", "")),
                            str(b.attrs.get("unit", "")), float(rho), float(p), float(dr), float(dp),
                            strength, best_lag, best_r, verdict)


def save_plot(result: AnalysisResult, out_path: str | Path) -> None:
    """Save raw observations, robust trend, LOWESS curve, and conditional forecast band as a PNG."""
    dates = pd.to_datetime(result.dates)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, result.raw, ".", alpha=0.55, label="Observed")
    ax.plot(dates, result.trend_line, label="Theil-Sen trend")
    ax.plot(dates, result.lowess, label="LOWESS")
    if result.forecast_dates:
        fd = pd.to_datetime(result.forecast_dates)
        ax.plot(fd, result.forecast, "--", label="If this trend continues")
        ax.fill_between(fd, result.forecast_low, result.forecast_high, alpha=0.2, label="Residual-spread band")
    ax.set(title=result.verdict, xlabel="Date", ylabel=result.unit)
    ax.legend(); fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def _read_cli_series(path: str, months: Sequence[int] | None, start: int, end: int) -> tuple[pd.Series, str]:
    """Load a CLI input and apply its requested date/season selection."""
    series, unit = load_series(path)
    return select_period(series, start, end, months), unit


def main(argv: Sequence[str] | None = None) -> None:
    """Parse trend, comparison, or control-difference CLI requests and print JSON results."""
    parser = argparse.ArgumentParser(description="Transparent monthly Earth-science trend analysis")
    sub = parser.add_subparsers(dest="mode", required=True)
    trend = sub.add_parser("trend"); trend.add_argument("path"); trend.add_argument("--start", type=int, default=2005); trend.add_argument("--end", type=int, default=2024); trend.add_argument("--months", type=int, nargs="+"); trend.add_argument("--place"); trend.add_argument("--variable"); trend.add_argument("--json"); trend.add_argument("--plot")
    comp = sub.add_parser("compare"); comp.add_argument("path_a"); comp.add_argument("path_b"); comp.add_argument("--months", type=int, nargs="+"); comp.add_argument("--max-lag", type=int, default=0); comp.add_argument("--start", type=int, default=2005); comp.add_argument("--end", type=int, default=2024); comp.add_argument("--json")
    diff = sub.add_parser("diff"); diff.add_argument("path_a"); diff.add_argument("path_b"); diff.add_argument("--start", type=int, default=2005); diff.add_argument("--end", type=int, default=2024); diff.add_argument("--months", type=int, nargs="+"); diff.add_argument("--json"); diff.add_argument("--plot")
    args = parser.parse_args(argv)
    if args.mode == "trend": result = analyze(args.path, args.place, args.variable, args.start, args.end, args.months)
    elif args.mode == "compare":
        a, ua = _read_cli_series(args.path_a, args.months, args.start, args.end); b, ub = _read_cli_series(args.path_b, args.months, args.start, args.end)
        a.attrs["unit"] = ua; b.attrs["unit"] = ub
        result = compare_series(a, b, max_lag_months=args.max_lag)
    else:
        a, ua = load_series(args.path_a); b, _ = load_series(args.path_b)
        d = difference_series(a, b)
        result = analyze_series(d, ua, "A compared with B", args.start, args.end, args.months)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, allow_nan=False))
    json_path = getattr(args, "json", None)
    if json_path: Path(json_path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    plot_path = getattr(args, "plot", None)
    if plot_path and isinstance(result, AnalysisResult): save_plot(result, plot_path)


if __name__ == "__main__":
    main()
