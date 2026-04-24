#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy>=2.1",
#   "openpyxl>=3.1",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.5",
#   "xlrd>=2.0",
# ]
# ///

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


YIELD_SHEET = "Data"
GDP_SHEET = "Data1"
F17_SHEET = "Yields"
F17_SOURCE_URL = "https://www.rba.gov.au/statistics/tables/xls-hist/zcr-analytical-series-hist.xls"
F17_DEFAULT_FILE = "zcr-analytical-series-hist.xls"
GAP_THRESHOLD_DAYS = 10
ROLLING_3M_DAYS = 63
ROLLING_6M_DAYS = 126
TARGET_FORWARD_QUARTERS = 2
TARGET_COLUMN = "recession_through_next_2q"

YIELD_2Y_TITLE = "Australian Government 2 year bond"
YIELD_10Y_TITLE = "Australian Government 10 year bond"
GDP_SERIES_TITLE = "GDP per capita: Chain volume measures - Percentage changes ;"
GDP_SERIES_TYPE = "Seasonally Adjusted"

FEATURE_COLUMNS = [
    "spread_3m_mean",
    "spread_6m_min",
    "spread_6m_inversion_share",
]

FEATURE_LABELS = {
    "spread_3m_mean": "63-day average 10Y-2Y spread",
    "spread_6m_min": "126-day minimum 10Y-2Y spread",
    "spread_6m_inversion_share": "126-day inversion share",
}


@dataclass(frozen=True)
class ValidationSummary:
    folds_used: int
    auc: float | None
    brier: float | None


@dataclass(frozen=True)
class ModelArtifacts:
    model: Pipeline
    training_quarters: pd.DataFrame
    daily_scores: pd.DataFrame
    gdp_quarters: pd.DataFrame
    validation: ValidationSummary
    coefficients: pd.Series


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Australia daily per-capita recession indicator HTML dashboard."
    )
    parser.add_argument("--yield-file", default="f02d.xlsx", help="Path to the yield workbook.")
    parser.add_argument(
        "--gdp-file",
        default="5206001_Key_Aggregates.xlsx",
        help="Path to the GDP workbook.",
    )
    parser.add_argument(
        "--html-out",
        default="australia_recession_indicator.html",
        help="Path for the standalone HTML output.",
    )
    parser.add_argument(
        "--f17-file",
        default=F17_DEFAULT_FILE,
        help="Path to the local RBA F17 historical workbook.",
    )
    return parser.parse_args()


def excel_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    return pd.NaT


def require_column(mask: pd.Series, label: str, options: pd.Series) -> int:
    matches = mask[mask].index.tolist()
    if not matches:
        available = ", ".join(str(value) for value in options.dropna().tolist())
        raise ValueError(f"Could not find {label}. Available options: {available}")
    return int(matches[0])


def load_current_f2_yield_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=YIELD_SHEET, header=None)
    titles = raw.iloc[1]

    date_col = 0
    two_year_col = require_column(titles.eq(YIELD_2Y_TITLE), "2-year yield series", titles)
    ten_year_col = require_column(titles.eq(YIELD_10Y_TITLE), "10-year yield series", titles)

    data = raw.iloc[11:, [date_col, two_year_col, ten_year_col]].copy()
    data.columns = ["date", "yield_2y", "yield_10y"]
    data["date"] = data["date"].map(excel_timestamp)
    data["yield_2y"] = pd.to_numeric(data["yield_2y"], errors="coerce")
    data["yield_10y"] = pd.to_numeric(data["yield_10y"], errors="coerce")
    data = (
        data.dropna(subset=["date", "yield_2y", "yield_10y"])
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )
    data["yield_source"] = "RBA F2"
    return data


def load_f17_yield_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing local F17 workbook at {path}. Download it from {F17_SOURCE_URL}."
        )

    raw = pd.read_excel(path, sheet_name=F17_SHEET, header=None)
    maturity_row_idx = None
    two_year_col = None
    ten_year_col = None

    for idx in range(min(20, len(raw))):
        numeric_row = pd.to_numeric(raw.iloc[idx], errors="coerce")
        if numeric_row.notna().sum() < 10:
            continue
        if (numeric_row == 2).any() and (numeric_row == 10).any():
            maturity_row_idx = idx
            two_year_col = int(numeric_row[numeric_row == 2].index[0])
            ten_year_col = int(numeric_row[numeric_row == 10].index[0])
            break

    if maturity_row_idx is None or two_year_col is None or ten_year_col is None:
        raise ValueError("Could not locate the 2-year and 10-year maturity columns in the F17 workbook.")

    date_series = raw.iloc[:, 0].map(excel_timestamp)
    data_start = date_series.iloc[maturity_row_idx + 1 :].first_valid_index()
    if data_start is None:
        raise ValueError("Could not locate dated yield rows in the F17 workbook.")

    data = pd.DataFrame(
        {
            "date": date_series.iloc[data_start:],
            "yield_2y": pd.to_numeric(raw.iloc[data_start:, two_year_col], errors="coerce"),
            "yield_10y": pd.to_numeric(raw.iloc[data_start:, ten_year_col], errors="coerce"),
        }
    )
    data = (
        data.dropna(subset=["date", "yield_2y", "yield_10y"])
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )
    data["yield_source"] = "RBA F17"
    return data


def add_yield_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["spread"] = data["yield_10y"] - data["yield_2y"]
    data["quarter"] = data["date"].dt.to_period("Q")
    data["spread_3m_mean"] = data["spread"].rolling(ROLLING_3M_DAYS, min_periods=ROLLING_3M_DAYS).mean()
    data["spread_6m_min"] = data["spread"].rolling(ROLLING_6M_DAYS, min_periods=ROLLING_6M_DAYS).min()
    data["spread_6m_inversion_share"] = (
        (data["spread"] < 0)
        .astype(float)
        .rolling(ROLLING_6M_DAYS, min_periods=ROLLING_6M_DAYS)
        .mean()
    )
    data["quarter_label"] = data["quarter"].map(format_quarter)
    return data


def load_yield_data(current_f2_path: Path, f17_path: Path) -> pd.DataFrame:
    historical = load_f17_yield_data(f17_path)
    current = load_current_f2_yield_data(current_f2_path)

    combined = (
        pd.concat([historical, current], ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    return add_yield_features(combined)


def load_gdp_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=GDP_SHEET, header=None)

    titles = raw.iloc[0]
    series_types = raw.iloc[2]
    target_mask = titles.eq(GDP_SERIES_TITLE) & series_types.eq(GDP_SERIES_TYPE)
    gdp_col = require_column(target_mask, "seasonally adjusted GDP per capita q/q series", titles)

    date_series = raw.iloc[:, 0].map(excel_timestamp)
    data_start = date_series.first_valid_index()
    if data_start is None:
        raise ValueError("Could not locate quarterly date rows in the GDP workbook.")

    data = pd.DataFrame(
        {
            "date": date_series.iloc[data_start:],
            "gdp_per_capita_qoq": pd.to_numeric(raw.iloc[data_start:, gdp_col], errors="coerce"),
        }
    )
    data = (
        data.dropna(subset=["date", "gdp_per_capita_qoq"])
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )
    data["quarter"] = data["date"].dt.to_period("Q")

    negative = data["gdp_per_capita_qoq"] < 0
    data["per_capita_recession"] = negative & (
        negative.shift(1, fill_value=False) | negative.shift(-1, fill_value=False)
    )
    forward_window = pd.concat(
        [data["per_capita_recession"].shift(-offset).astype(float) for offset in range(TARGET_FORWARD_QUARTERS + 1)],
        axis=1,
    )
    data[TARGET_COLUMN] = forward_window.max(axis=1)
    data.loc[~forward_window.notna().all(axis=1), TARGET_COLUMN] = np.nan
    data["quarter_label"] = data["quarter"].map(format_quarter)
    return data


def build_training_quarters(yields: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    complete_feature_rows = yields.dropna(subset=FEATURE_COLUMNS).copy()
    quarter_features = (
        complete_feature_rows.groupby("quarter", as_index=False)
        .agg(
            feature_date=("date", "last"),
            yield_sources=("yield_source", lambda values: ", ".join(sorted(set(values)))),
            spread_3m_mean=("spread_3m_mean", "last"),
            spread_6m_min=("spread_6m_min", "last"),
            spread_6m_inversion_share=("spread_6m_inversion_share", "last"),
        )
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    training = quarter_features.merge(
        gdp[
            [
                "quarter",
                "date",
                "gdp_per_capita_qoq",
                "per_capita_recession",
                TARGET_COLUMN,
                "quarter_label",
            ]
        ],
        on="quarter",
        how="inner",
        suffixes=("", "_gdp"),
    )
    training = training.dropna(subset=[TARGET_COLUMN]).copy()
    training[TARGET_COLUMN] = training[TARGET_COLUMN].astype(int)
    if training[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training sample does not contain both recession and non-recession quarters.")
    return training


def validate_model(training_quarters: pd.DataFrame) -> ValidationSummary:
    X = training_quarters[FEATURE_COLUMNS].to_numpy()
    y = training_quarters[TARGET_COLUMN].astype(int).to_numpy()

    if len(training_quarters) < 10:
        return ValidationSummary(folds_used=0, auc=None, brier=None)

    splitter = TimeSeriesSplit(n_splits=min(5, len(training_quarters) - 1))
    predictions: list[float] = []
    outcomes: list[int] = []
    folds_used = 0

    for train_idx, test_idx in splitter.split(X):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        fold_model = build_model_pipeline()
        fold_model.fit(X[train_idx], y_train)
        fold_predictions = fold_model.predict_proba(X[test_idx])[:, 1]
        predictions.extend(fold_predictions.tolist())
        outcomes.extend(y_test.tolist())
        folds_used += 1

    if folds_used == 0 or len(set(outcomes)) < 2:
        return ValidationSummary(folds_used=folds_used, auc=None, brier=None)

    return ValidationSummary(
        folds_used=folds_used,
        auc=float(roc_auc_score(outcomes, predictions)),
        brier=float(brier_score_loss(outcomes, predictions)),
    )


def fit_indicator(yields: pd.DataFrame, gdp: pd.DataFrame) -> ModelArtifacts:
    training_quarters = build_training_quarters(yields, gdp)
    validation = validate_model(training_quarters)

    model = build_model_pipeline()

    X_train = training_quarters[FEATURE_COLUMNS]
    y_train = training_quarters[TARGET_COLUMN].astype(int)
    model.fit(X_train, y_train)

    scored_daily = yields.copy()
    complete_feature_mask = scored_daily[FEATURE_COLUMNS].notna().all(axis=1)
    scored_daily["recession_probability"] = np.nan
    scored_daily.loc[complete_feature_mask, "recession_probability"] = model.predict_proba(
        scored_daily.loc[complete_feature_mask, FEATURE_COLUMNS]
    )[:, 1]
    scored_daily["recession_probability_pct"] = scored_daily["recession_probability"] * 100

    coefficients = pd.Series(
        model.named_steps["logit"].coef_[0],
        index=FEATURE_COLUMNS,
        name="scaled_logit_coefficient",
    )

    return ModelArtifacts(
        model=model,
        training_quarters=training_quarters,
        daily_scores=scored_daily,
        gdp_quarters=gdp,
        validation=validation,
        coefficients=coefficients,
    )


def quarter_bounds(period: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = period.start_time.normalize()
    end = period.end_time.normalize()
    return start, end


def format_quarter(period: pd.Period) -> str:
    return f"{period.year} Q{period.quarter}"


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def probability_state(value: float) -> tuple[str, str]:
    if value >= 0.60:
        return (
            "Elevated",
            "The curve is signaling a meaningfully higher chance that Australia is in or enters a per-capita recession over the current-plus-next-two-quarter window.",
        )
    if value >= 0.30:
        return (
            "Watch",
            "The curve is soft enough to justify caution, but it is not yet a high-risk forward recession reading.",
        )
    return "Low", "The curve is not currently flashing a strong forward per-capita recession signal."


def spread_state(value: float) -> str:
    if value < 0:
        return "The curve is inverted, with 10-year yields below 2-year yields."
    if value < 0.5:
        return "The curve is only mildly upward sloping, so the buffer over inversion is thin."
    return "The curve is still clearly upward sloping."


def chart_theme() -> dict:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(255,255,255,0)",
        "font": {"family": "Avenir Next, Segoe UI, Arial, sans-serif", "color": "#14312b"},
        "margin": {"l": 52, "r": 28, "t": 56, "b": 44},
        "hovermode": "x unified",
    }


def find_chart_gaps(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    gap_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    ordered_dates = pd.Series(dates).dropna().sort_values().tolist()
    for previous, current in zip(ordered_dates, ordered_dates[1:]):
        if current - previous > pd.Timedelta(days=GAP_THRESHOLD_DAYS):
            gap_windows.append((previous + pd.Timedelta(days=1), current - pd.Timedelta(days=1)))
    return gap_windows


def add_gap_bands(fig: go.Figure, gap_windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    for start, end in gap_windows:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(20, 49, 43, 0.06)",
            line_width=0,
            layer="below",
        )


def insert_gap_rows(scores: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None
    base_columns = ["date", *value_columns]

    for row in scores[base_columns].itertuples(index=False):
        current_date = row[0]
        if previous_date is not None and current_date - previous_date > pd.Timedelta(days=GAP_THRESHOLD_DAYS):
            gap_row = {"date": previous_date + pd.Timedelta(days=1)}
            for column in value_columns:
                gap_row[column] = np.nan
            rows.append(gap_row)
        rows.append({column: value for column, value in zip(base_columns, row)})
        previous_date = current_date

    return pd.DataFrame(rows)


def add_recession_bands(
    fig: go.Figure,
    recession_quarters: pd.DataFrame,
    visible_start: pd.Timestamp,
    visible_end: pd.Timestamp,
) -> None:
    for period in recession_quarters.loc[recession_quarters["per_capita_recession"], "quarter"]:
        start, end = quarter_bounds(period)
        if end < visible_start or start > visible_end:
            continue
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(188, 96, 35, 0.14)",
            line_width=0,
            layer="below",
        )


def make_probability_figure(artifacts: ModelArtifacts) -> go.Figure:
    scores = artifacts.daily_scores.dropna(subset=["recession_probability_pct"]).copy()
    chart_scores = insert_gap_rows(scores, ["recession_probability_pct"])
    visible_start = scores["date"].min()
    visible_end = scores["date"].max()
    gap_windows = find_chart_gaps(scores["date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_scores["date"],
            y=chart_scores["recession_probability_pct"],
            mode="lines",
            name="Probability",
            line={"color": "#0b6e4f", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(11, 110, 79, 0.10)",
            connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>",
        )
    )
    add_recession_bands(fig, artifacts.gdp_quarters, visible_start, visible_end)
    add_gap_bands(fig, gap_windows)
    fig.update_layout(
        **chart_theme(),
        title="Daily Probability of a Per-Capita Recession Through the Current + Next 2 Quarters",
        yaxis={
            "title": "Probability",
            "ticksuffix": "%",
            "range": [0, max(100, float(scores["recession_probability_pct"].max()) * 1.15)],
            "gridcolor": "rgba(20, 49, 43, 0.10)",
            "zeroline": False,
        },
        xaxis={
            "title": "",
            "gridcolor": "rgba(20, 49, 43, 0.07)",
            "range": [visible_start, visible_end],
        },
        showlegend=False,
    )
    return fig


def make_spread_figure(artifacts: ModelArtifacts) -> go.Figure:
    scores = artifacts.daily_scores
    chart_scores = insert_gap_rows(scores, ["spread"])
    visible_start = scores["date"].min()
    visible_end = scores["date"].max()
    gap_windows = find_chart_gaps(scores["date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_scores["date"],
            y=chart_scores["spread"],
            mode="lines",
            name="10Y-2Y spread",
            line={"color": "#184e77", "width": 2.6},
            connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} pp<extra></extra>",
        )
    )
    add_recession_bands(fig, artifacts.gdp_quarters, visible_start, visible_end)
    add_gap_bands(fig, gap_windows)
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="rgba(188, 96, 35, 0.8)")
    fig.update_layout(
        **chart_theme(),
        title="Australia 10Y-2Y Yield Spread",
        yaxis={
            "title": "Percentage points",
            "gridcolor": "rgba(20, 49, 43, 0.10)",
            "zeroline": False,
        },
        xaxis={
            "title": "",
            "gridcolor": "rgba(20, 49, 43, 0.07)",
            "range": [visible_start, visible_end],
        },
        showlegend=False,
    )
    return fig


def render_html(artifacts: ModelArtifacts) -> str:
    scores = artifacts.daily_scores
    scored_window = scores.dropna(subset=["recession_probability"]).copy()
    training = artifacts.training_quarters

    latest = scored_window.iloc[-1]
    state_label, state_summary = probability_state(float(latest["recession_probability"]))
    sample_start = scored_window["date"].min().strftime("%-d %b %Y")
    sample_end = scored_window["date"].max().strftime("%-d %b %Y")
    yield_start = scores["date"].min().strftime("%-d %b %Y")
    latest_date = latest["date"].strftime("%-d %b %Y")
    training_end = training["quarter"].max()
    target_positive_quarters = int(training[TARGET_COLUMN].sum())
    latest_horizon_end = latest["quarter"] + TARGET_FORWARD_QUARTERS

    probability_fig = make_probability_figure(artifacts)
    spread_fig = make_spread_figure(artifacts)

    probability_chart = to_html(
        probability_fig,
        full_html=False,
        include_plotlyjs="inline",
        config={"displayModeBar": False, "responsive": True},
    )
    spread_chart = to_html(
        spread_fig,
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )

    auc_text = "N/A"
    brier_text = "N/A"
    if artifacts.validation.auc is not None:
        auc_text = f"{artifacts.validation.auc:.2f}"
    if artifacts.validation.brier is not None:
        brier_text = f"{artifacts.validation.brier:.3f}"

    strongest_feature = artifacts.coefficients.abs().sort_values(ascending=False).index[0]
    strongest_coef = FEATURE_LABELS[strongest_feature]
    current_window_label = (
        f"{format_quarter(latest['quarter'])} to {format_quarter(latest_horizon_end)}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Australia Recession Probability</title>
    <style>
      :root {{
        --ink: #14312b;
        --muted: #50655f;
        --surface: rgba(255, 250, 241, 0.82);
        --surface-strong: rgba(255, 252, 247, 0.92);
        --line: rgba(20, 49, 43, 0.12);
        --green: #0b6e4f;
        --blue: #184e77;
        --amber: #bc6023;
        --cream: #fffaf1;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        color: var(--ink);
        font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
        background:
          radial-gradient(circle at top left, rgba(11, 110, 79, 0.18), transparent 34%),
          radial-gradient(circle at 85% 15%, rgba(188, 96, 35, 0.18), transparent 24%),
          linear-gradient(180deg, #f6efe2 0%, #f2f7f1 48%, #eef3f5 100%);
      }}

      .page {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 40px 20px 56px;
      }}

      .hero {{
        display: grid;
        grid-template-columns: 1.25fr 1fr;
        gap: 18px;
        align-items: stretch;
      }}

      .card {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(20, 49, 43, 0.08);
        backdrop-filter: blur(18px);
      }}

      .hero-main {{
        padding: 28px 28px 26px;
      }}

      .eyebrow {{
        margin: 0 0 14px;
        font-size: 12px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      h1 {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-size: clamp(34px, 6vw, 56px);
        line-height: 0.96;
        letter-spacing: -0.03em;
      }}

      .hero-copy {{
        margin: 18px 0 0;
        max-width: 52ch;
        color: var(--muted);
        line-height: 1.6;
        font-size: 16px;
      }}

      .hero-highlight {{
        margin-top: 26px;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
      }}

      .pill {{
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(11, 110, 79, 0.10);
        color: var(--green);
        font-size: 13px;
        font-weight: 700;
      }}

      .score-card {{
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
      }}

      .score-card h2 {{
        margin: 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
      }}

      .score-number {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-size: clamp(52px, 8vw, 84px);
        line-height: 0.95;
      }}

      .score-state {{
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(24, 78, 119, 0.10);
        color: var(--blue);
        font-weight: 700;
        font-size: 13px;
        justify-self: start;
      }}

      .metrics {{
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
      }}

      .metric {{
        padding: 18px;
        border-radius: 20px;
        background: var(--surface-strong);
        border: 1px solid var(--line);
      }}

      .metric-label {{
        margin: 0 0 8px;
        color: var(--muted);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}

      .metric-value {{
        margin: 0;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.03em;
      }}

      .metric-note {{
        margin: 8px 0 0;
        color: var(--muted);
        line-height: 1.5;
        font-size: 14px;
      }}

      .charts {{
        margin-top: 18px;
        display: grid;
        gap: 18px;
      }}

      .chart-card {{
        padding: 14px 16px 10px;
      }}

      .chart {{
        min-height: 410px;
      }}

      .footer {{
        margin-top: 22px;
        color: var(--muted);
        font-size: 13px;
      }}

      @media (max-width: 920px) {{
        .hero {{
          grid-template-columns: 1fr;
        }}

        .metrics {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}

      @media (max-width: 640px) {{
        .page {{
          padding: 22px 14px 40px;
        }}

        .hero-main,
        .score-card {{
          padding: 20px;
        }}

        .metrics {{
          grid-template-columns: 1fr;
        }}

        .chart {{
          min-height: 340px;
        }}
      }}
    </style>
  </head>
  <body data-dashboard-ready="true">
    <main class="page">
      <section class="hero">
        <article class="card hero-main">
          <p class="eyebrow">Australia Macro Signal</p>
          <h1>Forward Per-Capita Recession Probability</h1>
          <p class="hero-copy">
            A daily yield-curve estimate of whether Australia is in or enters a per-capita
            recession over the <strong>current quarter plus the next two quarters</strong>.
            Amber bands mark realized quarters where seasonally adjusted real GDP per capita
            fell for at least two consecutive quarters.
          </p>
          <div class="hero-highlight">
            <span class="pill">Probability window: {html.escape(sample_start)} to {html.escape(sample_end)}</span>
            <span class="pill">Yield history: {html.escape(yield_start)} to {html.escape(sample_end)}</span>
            <span class="pill">Model: standardized logistic regression</span>
            <span class="pill">Signal: 10Y-2Y curve only</span>
          </div>
        </article>

        <aside class="card score-card">
          <h2>Current Forward Window</h2>
          <p class="score-number">{html.escape(format_probability(float(latest["recession_probability"])))}</p>
          <div class="score-state">{html.escape(state_label)}</div>
          <p class="metric-note">{html.escape(state_summary)}</p>
          <p class="metric-note">Forecast window: {html.escape(current_window_label)}</p>
          <p class="metric-note">Latest market close: {html.escape(latest_date)}</p>
        </aside>
      </section>

      <section class="metrics">
        <article class="metric">
          <p class="metric-label">Latest 10Y-2Y Spread</p>
          <p class="metric-value">{latest["spread"]:.2f} pp</p>
          <p class="metric-note">{html.escape(spread_state(float(latest["spread"])))}</p>
        </article>
        <article class="metric">
          <p class="metric-label">Training Quarters</p>
          <p class="metric-value">{len(training)}</p>
          <p class="metric-note">{target_positive_quarters} quarter-end observations are positive under the forward window target.</p>
        </article>
        <article class="metric">
          <p class="metric-label">Validation Snapshot</p>
          <p class="metric-value">AUC {html.escape(auc_text)}</p>
          <p class="metric-note">Brier score {html.escape(brier_text)} across {artifacts.validation.folds_used} usable forward-time folds.</p>
        </article>
        <article class="metric">
          <p class="metric-label">Strongest Model Driver</p>
          <p class="metric-value">{html.escape(strongest_coef)}</p>
          <p class="metric-note">The fitted model is most sensitive to this standardized feature in the final sample fit.</p>
        </article>
      </section>

      <section class="charts">
        <article class="card chart-card">
          <div class="chart">{probability_chart}</div>
        </article>
        <article class="card chart-card">
          <div class="chart">{spread_chart}</div>
        </article>
      </section>

      <p class="footer">
        Built from local RBA F17, RBA F2, and ABS workbooks through {html.escape(latest_date)}.
        Latest fully labeled training quarter: {html.escape(format_quarter(training_end))}.
        Probabilities begin once a full 126-trading-day yield window is available.
      </p>
    </main>
  </body>
</html>
"""


def write_html(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    yield_path = Path(args.yield_file)
    f17_path = Path(args.f17_file)
    gdp_path = Path(args.gdp_file)
    html_out = Path(args.html_out)

    yields = load_yield_data(yield_path, f17_path)
    gdp = load_gdp_data(gdp_path)
    artifacts = fit_indicator(yields, gdp)

    html_contents = render_html(artifacts)
    write_html(html_out, html_contents)

    print(f"Wrote dashboard: {html_out.resolve()}")
    print(
        "Yield history:",
        yields["date"].min().date().isoformat(),
        "to",
        yields["date"].max().date().isoformat(),
    )
    for source_name, source_data in yields.groupby("yield_source"):
        print(
            f"  {source_name}:",
            source_data["date"].min().date().isoformat(),
            "to",
            source_data["date"].max().date().isoformat(),
        )
    print(
        "GDP labels:",
        gdp["quarter"].min(),
        "to",
        gdp["quarter"].max(),
        "| recession quarters in full GDP history:",
        int(gdp["per_capita_recession"].sum()),
    )
    print(
        "Training sample:",
        len(artifacts.training_quarters),
        "quarters | positive forward-window quarters:",
        int(artifacts.training_quarters[TARGET_COLUMN].sum()),
    )
    scored_window = artifacts.daily_scores.dropna(subset=["recession_probability"]).copy()
    latest = scored_window.iloc[-1]
    print(
        "Probability history:",
        scored_window["date"].min().date().isoformat(),
        "to",
        scored_window["date"].max().date().isoformat(),
    )
    print(
        "Latest forward probability:",
        f"{latest['recession_probability_pct']:.1f}%",
        "| forecast window:",
        f"{format_quarter(latest['quarter'])} to {format_quarter(latest['quarter'] + TARGET_FORWARD_QUARTERS)}",
        "| latest spread:",
        f"{latest['spread']:.2f} pp",
        "| market date:",
        latest["date"].date().isoformat(),
    )
    return html_out


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
