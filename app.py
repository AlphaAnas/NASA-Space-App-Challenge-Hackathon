from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Keep the analysis module as the single source of statistical logic.
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from analyze import (  # noqa: E402
    analyze_series,
    check_quality,
    compare_series,
    difference_series,
    load_series,
    select_period,
)

DATA_DIRS = [ROOT / "data" / "nasa_data_clean", BACKEND / "data" / "nasa_data_clean"]
NOTES = ROOT / "science_notes.md"
DEFINITIONS = {
    "aerosol": "Aerosol optical depth measures haze in the whole air column and is not the same as ground-level PM2.5.",
    "no2": "Nitrogen dioxide (NO₂) is an atmospheric gas associated with combustion and urban air pollution.",
    "fires": "Fire activity reported as the count in the source dataset.",
    "landtemp": "Land surface temperature estimated from satellite observations.",
    "temperature": "Temperature reported by the source dataset.",
    "rainimerg": "Precipitation estimate from the IMERG dataset.",
    "rainpower": "Precipitation estimate from the POWER dataset.",
}
SEASONS = {"All months": None, "Oct-Dec": [10, 11, 12], "Jul-Sep": [7, 8, 9]}
COLORS = {"raw": "#8C8576", "trend": "#1F5C5C", "smooth": "#34434A", "band": "#B9D8D2"}


st.set_page_config(page_title="Trend Detective", page_icon=None, layout="wide")


st.markdown("""
<style>
.block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1420px;}
[data-testid="stMetric"] {background:#F0EADB; padding: .8rem 1rem; border-radius:8px;}
[data-testid="stMetricLabel"] {font-size:.82rem;}
.stTabs [data-baseweb="tab-list"] {gap:1.1rem;}
.stCaption {color:#5f5a50;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def catalog() -> pd.DataFrame:
    """Read dataset identity and metadata from CSV rows, not file naming conventions."""
    frames = []
    for folder in DATA_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
                if {"date", "place", "variable", "value", "unit"}.issubset(frame.columns):
                    frame["_file"] = str(path)
                    frames.append(frame)
            except (OSError, pd.errors.ParserError):
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_series(path: str, place: str, variable: str):
    return load_series(path, place, variable)


@st.cache_data(show_spinner=False)
def trend_result(path: str, place: str, variable: str, start: int, end: int, season: tuple | None):
    series, unit = load_series(path, place, variable)
    return analyze_series(series, unit, f"{variable} at {place}", start, end, season)


@st.cache_data(show_spinner=False)
def comparison_result(path_a: str, place_a: str, var_a: str, path_b: str, place_b: str,
                      var_b: str, start: int, end: int, season: tuple | None, lag: int):
    a, _ = load_series(path_a, place_a, var_a)
    b, _ = load_series(path_b, place_b, var_b)
    a = select_period(a, start, end)
    b = select_period(b, start, end)
    a.name, b.name = f"{var_a} at {place_a}", f"{var_b} at {place_b}"
    return a, b, compare_series(a, b, season, lag)


def file_for(frame: pd.DataFrame, variable: str, place: str) -> str:
    found = frame[(frame.variable == variable) & (frame.place == place)]
    return str(found.iloc[0]._file)


def select_story(variable: str, place: str) -> None:
    """Apply a guided-story preset before Streamlit recreates its widgets."""
    st.session_state["variable"] = variable
    st.session_state["place"] = place


def chart_layout(fig: go.Figure, title: str, y_title: str, caption: str, x_title: str = "Date") -> None:
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, paper_bgcolor="#FAF7F0",
                      plot_bgcolor="#FAF7F0", font=dict(family="Arial, sans-serif", color="#2A2A2A"),
                      margin=dict(l=12, r=12, t=55, b=18), legend=dict(orientation="h", y=1.12),
                      height=400, showlegend=True)
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E4DED2", showline=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)


def trend_chart(result, modes: list[str], title: str | None = None) -> go.Figure:
    dates = pd.to_datetime(result.dates)
    fig = go.Figure()
    if "Raw" in modes:
        fig.add_trace(go.Scatter(x=dates, y=result.raw, mode="lines+markers", name="Observed",
                                 line=dict(color=COLORS["raw"], width=1)))
    if "Deseasonalized" in modes:
        fig.add_trace(go.Scatter(x=dates, y=result.deseasonalized, mode="lines", name="Deseasonalized",
                                 line=dict(color=COLORS["raw"], width=1.2, dash="dot")))
    if "Trend line" in modes:
        fig.add_trace(go.Scatter(x=dates, y=result.trend_line, mode="lines", name="Theil-Sen trend",
                                 line=dict(color=COLORS["trend"], width=2)))
    if "Smoothed curve" in modes:
        fig.add_trace(go.Scatter(x=dates, y=result.lowess, mode="lines", name="LOWESS",
                                 line=dict(color=COLORS["smooth"], width=1.8)))
    if "if this trend continues" in modes:
        future = pd.to_datetime(result.forecast_dates)
        fig.add_trace(go.Scatter(x=future, y=result.forecast_high, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future, y=result.forecast_low, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(185,216,210,.48)", name="95% residual-spread band"))
        fig.add_trace(go.Scatter(x=future, y=result.forecast, mode="lines", name="if this trend continues",
                                 line=dict(color=COLORS["trend"], width=1.5, dash="dash")))
    fig.update_layout(title=title or result.label)
    return fig


def download_result(result, key_prefix: str, file_prefix: str) -> None:
    st.download_button("Download result as JSON", json.dumps(result.to_dict(), indent=2, allow_nan=False),
                       f"{file_prefix}_result.json", "application/json", key=f"{key_prefix}_json")
    rows = pd.DataFrame({"date": result.dates, "raw": result.raw,
                         "deseasonalized": result.deseasonalized, "trend_line": result.trend_line,
                         "smoothed": result.lowess})
    st.download_button("Download series as CSV", rows.to_csv(index=False),
                       f"{file_prefix}_series.csv", "text/csv", key=f"{key_prefix}_csv")


st.title("Trend Detective")
st.write("Explore environmental time series, describe the pattern, and check what the statistical evidence supports.")
data = catalog()
if data.empty:
    st.warning("No readable datasets were found. Add CSV files with date, place, variable, value, and unit columns under data/nasa_data_clean/ or backend/data/nasa_data_clean/.")
    st.stop()

variables = sorted(data.variable.dropna().astype(str).unique())
places = sorted(data.place.dropna().astype(str).unique())
date_years = pd.to_datetime(data.date.astype(str) + "-01", errors="coerce").dt.year.dropna()
if not variables or not places or date_years.empty:
    st.warning("The available CSV files do not contain usable variable, place, and date selections.")
    st.stop()
year_bounds = (int(date_years.min()), int(date_years.max()))

with st.sidebar:
    st.subheader("5-step flow")
    variable = st.selectbox("1 Variable", variables, key="variable")
    candidate_places = sorted(data.loc[data.variable == variable, "place"].dropna().astype(str).unique())
    place = st.selectbox("2 Place", candidate_places, key="place")
    year_range = st.slider("3 Period · year range", min_value=year_bounds[0], max_value=year_bounds[1],
                           value=year_bounds)
    season_name = st.selectbox("Season (optional)", list(SEASONS))
    season = SEASONS[season_name]
    st.markdown("**Guided stories**")
    st.button("Lahore and Delhi smog", use_container_width=True,
              on_click=select_story, args=("aerosol", "lahore"))
    st.button("Karachi heat", use_container_width=True,
              on_click=select_story, args=("landtemp", "karachicity"))
    st.button("Karachi rain", use_container_width=True,
              on_click=select_story, args=("rainimerg", "karachi"))

tabs = st.tabs(["Trend Explorer", "Possible Reasons", "City vs Surroundings", "Method and Validation", "Data and Definitions"])
start_year, end_year = year_range
selected_path = file_for(data, variable, place)

with tabs[0]:
    st.subheader("4 · Explore the time series")
    st.caption(f"{variable} · {place} · {start_year}–{end_year} · {season_name}")
    try:
        result = trend_result(selected_path, place, variable, start_year, end_year,
                              tuple(season) if season else None)
    except ValueError as exc:
        st.warning(str(exc))
        st.info("The selected series is not suitable for a trend analysis in this period.")
    else:
        mode = st.multiselect("Chart layers", ["Raw", "Deseasonalized", "Trend line", "Smoothed curve", "if this trend continues"],
                              default=["Raw", "Trend line"])
        fig = trend_chart(result, mode)
        unit = result.unit or "Value"
        chart_layout(fig, result.label, unit, "Observations and selected analysis layers for the chosen period.")
        st.markdown("**5 · What trend do you see?**")
        guess = st.radio("What trend do you see?", ["Rising", "Falling", "No clear trend"], horizontal=True,
                         label_visibility="collapsed", key="trend_guess")
        if st.button("Show statistical result", type="primary"):
            st.session_state["show_statistical_result"] = True
        if st.session_state.get("show_statistical_result"):
            if result.slope_per_decade is not None:
                cols = st.columns(4)
                cols[0].metric("Rate per decade", f"{result.slope_per_decade:.3g} {unit}")
                cols[1].metric("95% range", f"{result.ci_low:.3g} to {result.ci_high:.3g} {unit}")
                cols[2].metric("p-value", f"{result.p_value:.3g}")
                cols[3].metric("Test used", result.test_used)
            verdict_color = "#3C6E47" if result.p_value < .05 else "#B58A2B" if result.p_value < .10 else "#6B6B6B"
            verdict_label = "Statistically significant trend" if result.p_value < .05 else "Weak evidence" if result.p_value < .10 else "No significant trend"
            st.markdown(f"<span style='background:{verdict_color};color:white;padding:.35rem .65rem;border-radius:4px'>{verdict_label}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='border-left:4px solid {verdict_color};padding:.5rem .8rem;margin:.8rem 0'>{result.verdict}</div>", unsafe_allow_html=True)
            if result.warnings:
                for warning in result.warnings:
                    st.warning(warning)
            if result.nonlinear:
                st.info("The pattern is flagged as non-linear; a single rate may not describe the full period.")
            actual = "Rising" if (result.slope_per_decade or 0) > 0 else "Falling" if (result.slope_per_decade or 0) < 0 else "No clear trend"
            st.caption(f"Your guess: {guess}. Statistical direction: {actual}.")
            download_result(result, "trend", "trend")

with tabs[1]:
    st.subheader("Possible Reasons")
    st.write("Compare two environmental series as association evidence.")
    options = data[["variable", "place", "_file"]].drop_duplicates().sort_values(["variable", "place"])
    pairs = [(str(row["variable"]), str(row["place"]), str(row["_file"])) for _, row in options.iterrows()]
    default_a = next((i for i, p in enumerate(pairs) if p[:2] == ("aerosol", "lahore")), 0)
    default_b = next((i for i, p in enumerate(pairs) if p[:2] == ("fires", "punjab")), min(1, len(pairs)-1))
    c1, c2 = st.columns(2)
    selected_a = c1.selectbox("Series A", pairs, index=default_a, format_func=lambda x: f"{x[0]} · {x[1]}")
    selected_b = c2.selectbox("Series B", pairs, index=default_b, format_func=lambda x: f"{x[0]} · {x[1]}")
    compare_season_name = st.selectbox("Comparison season", list(SEASONS), index=1, key="compare_season")
    lag = st.slider("Lag option · months", 0, 12, 0)
    try:
        a, b, comp = comparison_result(selected_a[2], selected_a[1], selected_a[0], selected_b[2], selected_b[1], selected_b[0],
                                       start_year, end_year,
                                       tuple(SEASONS[compare_season_name]) if SEASONS[compare_season_name] else None, lag)
        av = select_period(a, start_year, end_year, SEASONS[compare_season_name])
        bv = select_period(b, start_year, end_year, SEASONS[compare_season_name])
        aligned = pd.concat([av.rename(comp.label_a), bv.rename(comp.label_b)], axis=1).dropna()
        line = go.Figure()
        line.add_trace(go.Scatter(x=av.index, y=av, mode="lines", name=comp.label_a, line=dict(color=COLORS["trend"], width=1.5)))
        line.add_trace(go.Scatter(x=bv.index, y=bv, mode="lines", name=comp.label_b, yaxis="y2", line=dict(color=COLORS["smooth"], width=1.5)))
        line.update_layout(yaxis2=dict(title=comp.unit_b, overlaying="y", side="right", showgrid=False), title="Selected series", yaxis_title=comp.unit_a)
        chart_layout(line, "Selected series", comp.unit_a, "Both series are displayed on their own units.")
        scatter = go.Figure(go.Scatter(x=aligned.iloc[:, 0], y=aligned.iloc[:, 1], mode="markers", marker=dict(color=COLORS["raw"])))
        chart_layout(scatter, "Series association", comp.unit_b, "Each point is a paired observation.", x_title=comp.unit_a)
        st.write(f"Correlation: **{comp.strength}** · Spearman ρ {comp.rho:.3g} · p={comp.p_value:.3g}")
        st.write(f"Detrended correlation: Spearman ρ {comp.detrended_rho:.3g} · p={comp.detrended_p_value:.3g}")
        st.write(comp.verdict)
    except (ValueError, IndexError) as exc:
        st.warning(str(exc))
    st.info("Correlation shows that two series go together. It does not prove cause.")

with tabs[2]:
    st.subheader("City vs Surroundings")
    st.write("Karachi city land temperature minus Karachi outside land temperature.")
    city = data[(data.variable == "landtemp") & (data.place == "karachicity")]
    rural = data[(data.variable == "landtemp") & (data.place == "karachirural")]
    if city.empty or rural.empty:
        st.info("The city and surroundings land temperature series are not both available.")
    else:
        try:
            a, unit_a = read_series(str(city.iloc[0]._file), "karachicity", "landtemp")
            b, _ = read_series(str(rural.iloc[0]._file), "karachirural", "landtemp")
            diff = difference_series(a, b)
            diff.attrs["place"] = "Karachi city compared with surroundings"
            diff_result = analyze_series(diff, unit_a, "Karachi city compared with surroundings", start_year,
                                         end_year, tuple(season) if season else None)
            diff_chart = trend_chart(diff_result, ["Raw", "Trend line", "Smoothed curve"], "City minus surroundings")
            chart_layout(diff_chart, "City minus surroundings", unit_a, "Difference between aligned city and outside-city land temperature observations.")
            st.write("This series measures the extra change in the city compared with its surroundings.")
            st.write(diff_result.verdict)
            download_result(diff_result, "city_difference", "city_difference")
        except ValueError as exc:
            st.warning(str(exc))

with tabs[3]:
    st.subheader("Method and Validation")
    st.markdown("""
1. Load monthly observations and make missing months visible.
2. Select the requested years and, when chosen, aggregate the selected season by year.
3. Check record length and missingness before testing.
4. Remove the monthly calendar climatology, estimate a robust Theil-Sen rate and 95% range, and test monotonic trend with a dependence correction when warranted.
5. Compare a LOWESS curve and possible step change, describe uncertainty, and extend the line conditionally with a held-out check.
""")
    validation_dir = BACKEND / "validation"
    if validation_dir.exists():
        images = sorted(validation_dir.glob("*.png"))
        for image_path in images:
            st.image(str(image_path), caption=image_path.stem.replace("_", " ").title())
    else:
        st.info("No validation plots are available in validation/.")
    validation_table = BACKEND / "validation" / "test_results.csv"
    if validation_table.exists():
        st.dataframe(pd.read_csv(validation_table), use_container_width=True, hide_index=True)
    else:
        test_cases = pd.DataFrame([
            {"Synthetic check": "Known trend", "Expected outcome": "Estimated rate and confidence range recover the injected trend; validation plot is saved."},
            {"Synthetic check": "False positive rate", "Expected outcome": "Independent and autocorrelated null series stay within the configured acceptance interval."},
            {"Synthetic check": "Seasonal cycle", "Expected outcome": "A repeated annual cycle is not reported as a long-term trend."},
            {"Synthetic check": "Step change", "Expected outcome": "A persistent level shift is flagged as non-linear."},
            {"Synthetic check": "Quality refusal", "Expected outcome": "Short records and excessive missingness are refused with an explanation."},
            {"Synthetic check": "Missing observations", "Expected outcome": "Blank values and absent months remain missing, not zero."},
            {"Synthetic check": "Series comparison", "Expected outcome": "Shared trends, common noise, and a lagged pair exercise association handling."},
            {"Synthetic check": "Difference series", "Expected outcome": "Subtraction identifies extra change and cancels shared change."},
        ])
        st.caption("The repository contains synthetic test definitions, but no recorded test-run results table.")
        st.dataframe(test_cases, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Data and Definitions")
    records = []
    for path, subset in data.groupby("_file", sort=True):
        dates = pd.to_datetime(subset.date.astype(str) + "-01", errors="coerce").dropna()
        for (var, plc, unit), _ in subset.groupby(["variable", "place", "unit"], dropna=False):
            records.append({"Variable": var, "Place": plc, "Source": Path(path).name,
                            "Period": f"{dates.min():%Y-%m} to {dates.max():%Y-%m}" if not dates.empty else "Unavailable", "Unit": unit})
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    if NOTES.exists():
        st.markdown(NOTES.read_text(encoding="utf-8"))
    else:
        for var in variables:
            st.markdown(f"**{var}** — {DEFINITIONS.get(var, 'Definition is not available in the project notes.')}")
    st.info("Aerosol optical depth measures haze in the whole air column and is not the same as ground-level PM2.5.")
