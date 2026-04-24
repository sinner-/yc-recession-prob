# Australia Recession Probability

This repo builds a daily Australia recession-probability dashboard from a local yield workbook and a local GDP workbook.

The indicator is designed to mirror the familiar `2Y/10Y` yield-curve idea, but for Australia and with the recession target defined using **real GDP per capita** rather than headline GDP.

## What It Does

The script:

- reads daily Australian government bond yields from the RBA workbook
- reads quarterly GDP per capita growth from the ABS workbook
- labels a quarter as recessionary when seasonally adjusted real GDP per capita is negative for at least two consecutive quarters
- trains a logistic regression on quarter-end yield-curve features
- scores the full daily yield history
- writes a standalone HTML dashboard you can open locally

The current model uses:

- signal: `10Y - 2Y`
- target: probability that the **current quarter** will later be classified as a per-capita recession quarter

## Files

- `australia_recession_indicator.py`: all-in-one `uv` script
- `f02d.xlsx`: RBA yield data workbook
- `5206001_Key_Aggregates.xlsx`: ABS GDP workbook
- `australia_recession_indicator.html`: generated dashboard output

## Run It

```bash
uv run australia_recession_indicator.py
```

Optional arguments:

```bash
uv run australia_recession_indicator.py \
  --yield-file f02d.xlsx \
  --gdp-file 5206001_Key_Aggregates.xlsx \
  --html-out australia_recession_indicator.html
```

The script uses an inline `uv` dependency block, so you do not need to create or manage a separate virtualenv for normal use.

## Output

Running the script produces:

- `australia_recession_indicator.html`

The HTML is standalone and embeds Plotly inline, so it can be opened directly from disk without external asset links.

## Updating the Data

When you replace the spreadsheets with fresher versions and rerun the script, it:

- reloads the full contents of both workbooks
- rebuilds the recession labels from the GDP file
- rebuilds the yield features from the yield file
- retrains the model from scratch on the full available joined sample
- regenerates the full HTML output

Nothing is cached between runs, and no fitted model is saved separately.

## Method Summary

Quarter-end training features:

- today’s `10Y - 2Y` spread
- quarter-to-date average spread
- quarter-to-date minimum spread
- quarter-to-date inversion share

Recession definition:

- a quarter is shaded as recessionary if it belongs to a run of at least two consecutive negative quarterly prints in seasonally adjusted real GDP per capita growth

## Notes

- The script expects the current workbook structure and series names used by the supplied RBA and ABS files.
- If those source files change shape or naming conventions, the script will fail loudly rather than guessing.
- The HTML dashboard is the intended deliverable; the PNG screenshot used during development was only for visual validation.
