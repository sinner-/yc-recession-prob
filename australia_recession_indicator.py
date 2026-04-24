#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy>=2.1",
#   "openpyxl>=3.1",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.5",
# ]
# ///

from __future__ import annotations

import argparse
import html
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

YIELD_2Y_TITLE = "Australian Government 2 year bond"
YIELD_10Y_TITLE = "Australian Government 10 year bond"
GDP_SERIES_TITLE = "GDP per capita: Chain volume measures - Percentage changes ;"
GDP_SERIES_TYPE = "Seasonally Adjusted"

FEATURE_COLUMNS = [
    "spread_today",
    "spread_qtd_mean",
    "spread_qtd_min",
    "spread_qtd_inversion_share",
]

FEATURE_LABELS = {
    "spread_today": "Today's 10Y-2Y spread",
    "spread_qtd_mean": "Quarter-to-date average spread",
    "spread_qtd_min": "Quarter-to-date minimum spread",
    "spread_qtd_inversion_share": "Quarter-to-date inversion share",
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


def load_yield_data(path: Path) -> pd.DataFrame:
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

    data["spread"] = data["yield_10y"] - data["yield_2y"]
    data["quarter"] = data["date"].dt.to_period("Q")
    data["spread_today"] = data["spread"]

    grouped_spread = data.groupby("quarter")["spread"]
    data["spread_qtd_mean"] = (
        grouped_spread.expanding().mean().reset_index(level=0, drop=True).to_numpy()
    )
    data["spread_qtd_min"] = (
        grouped_spread.expanding().min().reset_index(level=0, drop=True).to_numpy()
    )
    inversions = (data["spread"] < 0).astype(float)
    data["spread_qtd_inversion_share"] = (
        inversions.groupby(data["quarter"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    data["quarter_label"] = data["quarter"].map(format_quarter)
    return data


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
    data["quarter_label"] = data["quarter"].map(format_quarter)
    return data


def build_training_quarters(yields: pd.DataFrame, gdp: pd.DataFrame) -> pd.DataFrame:
    quarter_features = (
        yields.groupby("quarter", as_index=False)
        .agg(
            feature_date=("date", "last"),
            spread_today=("spread_today", "last"),
            spread_qtd_mean=("spread_qtd_mean", "last"),
            spread_qtd_min=("spread_qtd_min", "last"),
            spread_qtd_inversion_share=("spread_qtd_inversion_share", "last"),
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
                "quarter_label",
            ]
        ],
        on="quarter",
        how="inner",
        suffixes=("", "_gdp"),
    )
    if training["per_capita_recession"].nunique() < 2:
        raise ValueError("Training sample does not contain both recession and non-recession quarters.")
    return training


def validate_model(training_quarters: pd.DataFrame) -> ValidationSummary:
    X = training_quarters[FEATURE_COLUMNS].to_numpy()
    y = training_quarters["per_capita_recession"].astype(int).to_numpy()

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

        fold_model = Pipeline(
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

    model = Pipeline(
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

    X_train = training_quarters[FEATURE_COLUMNS]
    y_train = training_quarters["per_capita_recession"].astype(int)
    model.fit(X_train, y_train)

    scored_daily = yields.copy()
    scored_daily["recession_probability"] = model.predict_proba(scored_daily[FEATURE_COLUMNS])[:, 1]
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
        return "Elevated", "The curve is signaling a meaningfully higher chance of a per-capita recession quarter."
    if value >= 0.30:
        return "Watch", "The curve is soft enough to justify caution, but it is not yet a high-risk reading."
    return "Low", "The curve is not currently flashing a strong per-capita recession signal."


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
    scores = artifacts.daily_scores
    visible_start = scores["date"].min()
    visible_end = scores["date"].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores["date"],
            y=scores["recession_probability_pct"],
            mode="lines",
            name="Probability",
            line={"color": "#0b6e4f", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(11, 110, 79, 0.10)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>",
        )
    )
    add_recession_bands(fig, artifacts.gdp_quarters, visible_start, visible_end)
    fig.update_layout(
        **chart_theme(),
        title="Daily Probability of a Current-Quarter Per-Capita Recession",
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
    visible_start = scores["date"].min()
    visible_end = scores["date"].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores["date"],
            y=scores["spread"],
            mode="lines",
            name="10Y-2Y spread",
            line={"color": "#184e77", "width": 2.6},
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} pp<extra></extra>",
        )
    )
    add_recession_bands(fig, artifacts.gdp_quarters, visible_start, visible_end)
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
    training = artifacts.training_quarters

    latest = scores.iloc[-1]
    state_label, state_summary = probability_state(float(latest["recession_probability"]))
    sample_start = scores["date"].min().strftime("%-d %b %Y")
    sample_end = scores["date"].max().strftime("%-d %b %Y")
    latest_date = latest["date"].strftime("%-d %b %Y")
    training_end = training["quarter"].max()
    recession_quarters = int(training["per_capita_recession"].sum())

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
    current_quarter_label = format_quarter(latest["quarter"])

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
        .score-card,
        .notes .card {{
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
          <h1>Daily Per-Capita Recession Probability</h1>
          <p class="hero-copy">
            A yield-curve-based nowcast for whether <strong>{html.escape(current_quarter_label)}</strong>
            will eventually be recorded as a per-capita recession quarter in Australia.
            Amber bands mark historical quarters where seasonally adjusted real GDP per capita
            fell for at least two consecutive quarters.
          </p>
          <div class="hero-highlight">
            <span class="pill">Sample window: {html.escape(sample_start)} to {html.escape(sample_end)}</span>
            <span class="pill">Model: standardized logistic regression</span>
            <span class="pill">Signal: 10Y-2Y curve only</span>
          </div>
        </article>

        <aside class="card score-card">
          <h2>Current-Quarter Nowcast</h2>
          <p class="score-number">{html.escape(format_probability(float(latest["recession_probability"])))}</p>
          <div class="score-state">{html.escape(state_label)}</div>
          <p class="metric-note">{html.escape(state_summary)}</p>
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
          <p class="metric-note">{recession_quarters} quarters are labeled recessionary in the usable sample.</p>
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
        Built from local RBA and ABS workbooks through {html.escape(latest_date)}. Latest GDP-labeled training quarter: {html.escape(format_quarter(training_end))}.
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
    gdp_path = Path(args.gdp_file)
    html_out = Path(args.html_out)

    yields = load_yield_data(yield_path)
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
        "quarters | recession quarters:",
        int(artifacts.training_quarters["per_capita_recession"].sum()),
    )
    latest = artifacts.daily_scores.iloc[-1]
    print(
        "Latest nowcast:",
        f"{latest['recession_probability_pct']:.1f}%",
        "| latest spread:",
        f"{latest['spread']:.2f} pp",
        "| market date:",
        latest["date"].date().isoformat(),
    )
    return html_out


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
