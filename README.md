# Australia Recession Probability

This repo builds a daily Australia recession-probability dashboard from local RBA yield workbooks and a local ABS GDP workbook.

The indicator is designed to mirror the familiar `2Y/10Y` yield-curve idea, but for Australia and with the recession target defined using **real GDP per capita** rather than headline GDP.

## What It Does

The script:

- reads historical daily Australian government zero-coupon yields from local RBA F17 data
- reads current daily Australian government bond yields from local RBA F2 data
- reads quarterly GDP per capita growth from the ABS workbook
- labels a quarter as recessionary when seasonally adjusted real GDP per capita is negative for at least two consecutive quarters
- trains a logistic regression on quarter-end rolling yield-curve regime features
- scores the full daily yield history
- writes a standalone HTML dashboard you can open locally

The current model uses:

- signal: `10Y - 2Y`
- target: probability that Australia is in or enters a per-capita recession over the **current quarter plus the next 2 quarters**

## Files

- `australia_recession_indicator.py`: all-in-one `uv` script
- `zcr-analytical-series-hist.xls`: local RBA F17 historical zero-coupon yield workbook
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
  --f17-file zcr-analytical-series-hist.xls \
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
- reloads the local F17 history file
- rebuilds the recession labels from the GDP file
- rebuilds the yield features from the yield file
- retrains the model from scratch on the full available joined sample
- regenerates the full HTML output

Nothing is cached between runs, and no fitted model is saved separately.

## Method Summary

Quarter-end training features:

- `63` trading-day average `10Y - 2Y` spread
- `126` trading-day minimum `10Y - 2Y` spread
- `126` trading-day inversion share for `10Y - 2Y`

Forward recession target:

- a quarter is labeled positive if that quarter or either of the next `2` quarters is classified as a per-capita recession quarter
- the daily line is made more stable by using continuous rolling yield-state features rather than quarter-to-date features that reset each quarter

Recession definition:

- a quarter is shaded as recessionary if it belongs to a run of at least two consecutive negative quarterly prints in seasonally adjusted real GDP per capita growth

Yield history:

- `1992-07-01` to `2013-05-17`: RBA F17 zero-coupon `2Y` and `10Y`
- `2013-09-02` onward: current RBA F2 `2Y` and `10Y`
- the gap between May and September 2013 is kept as a real missing-data gap and is shown as a break in the charts
- probability scoring begins only after the first full `126`-trading-day lookback window is available

## Notes

- The script expects the current workbook structure and series names used by the supplied RBA and ABS files.
- The local F17 workbook must be present; the script does not download it.
- If those source files change shape or naming conventions, the script will fail loudly rather than guessing.
- The HTML dashboard is the intended deliverable; the PNG screenshot used during development was only for visual validation.
