# Tadawul Compass · Group 5 capstone

A simple financial-quality dashboard: change your definition of quality and watch company rankings change. Uses the supplied Tadawul dataset and the four course days, including the existing Colab work.

## Run the dashboard

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). On this workspace, `.venv` is already prepared with the dashboard dependencies.

## Deliverables

- `app.py`: interactive dashboard with year selection, six weight sliders, strategy presets, search, sector filters, company histories, size-matched peers, correlation heatmap, score contributions, sensitivity comparisons, and CSV downloads.
- `Capstone_G5.ipynb`: standalone submission notebook, including a live ipywidgets dashboard. Upload this notebook and `data/tadawul_merged.csv` to Colab; run all cells. The notebook also works locally.
- `analytics.py`: shared, inspectable ranking calculations.
- `docs/METHODOLOGY.md`: ratio formulas, reasons for ratios and weights, data rules and course mapping.
- `docs/PITCH.md`: a four-minute presentation with verified findings.
- `outputs/all_year_rankings.csv`: all 3,963 observed firm-year records with default scores and ranks.
- `outputs/all_firms_all_years.csv`: all 8,085 company/year combinations; missing reports are labeled explicitly.
- `tests/test_analytics.py`: calculation and coverage checks.

## Default definition of quality

| Ratio | Weight | Reason |
|---|---:|---|
| Return on assets | 25% | Profit earned from the asset base |
| Net profit margin | 20% | Revenue retained as profit |
| Revenue growth | 20% | Expansion, without letting growth dominate |
| Total liabilities / assets | 15% | Lower obligations relative to assets |
| Current ratio | 10% | Short-term liquidity, capped at 3× for scoring |
| Operating cash / assets | 10% | A cash-based check on accounting earnings |

The weights are explicit team judgments, not optimized investment weights. The dashboard explains each choice.

Ratios are standardized within each year and sector by default. Outlier clipping is switchable. Lower liabilities/assets is rewarded. Finance current ratios are excluded and their weight is redistributed. Missing ratios contribute neutral zero and reduce visible coverage; below 70% is provisional. No usable ratios means no defensible rank.

## Coverage and limitations

The source contains 385 firms across 2006–2026. Fourteen repeated firm-year rows are resolved using the most-complete record, with latest report-end date as a tie-break. All 3,963 observed firm-years remain visible; 3,956 receive default ranks and seven lack usable weighted ratios. An additional 4,122 firm-year combinations have no source report and are labeled rather than given invented ranks. Historical sector labels are profile metadata. 2026 includes only eight records and is explicitly marked partial.

Yearly ranks order peer-relative scores when sector mode is selected. These scores describe relative financial quality under a chosen model, not valuation, expected returns, or a recommended portfolio. Finance analysis needs more specific metrics than this dataset supplies.

## Verify or regenerate rankings

```bash
python -m unittest discover -s tests -v
python analytics.py
```

The downloaded rankings use your current slider settings; the supplied output CSVs use balanced weights, sector benchmarks and outlier clipping.
