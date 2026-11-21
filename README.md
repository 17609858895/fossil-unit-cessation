# Forecasting US fossil-unit cessation from operating histories under delayed event confirmation

Core analysis code for the US coal, gas and oil unit study. The scripts turn hourly monitoring reports into year-end operating histories and evaluate one-, three- and five-year forecasts. At each historical forecast origin, the most recent training outcomes are reconstructed from the quarters observable at that date. Eight quarters without recorded operation confirm cessation. A subsequent restart can revise the earlier classification.

## Install and check

Use Python 3.12 in a virtual environment. From this directory run:

```sh
python -m pip install -r requirements.txt
python scripts/smoke_check.py
```

The check uses two synthetic units to exercise the eight-quarter rule, restart revisions, historical fleet capacity and cumulative risk. It runs without the original archives.

## Public input data

1. Download the 32 annual ZIP files named `epacems-1995.zip` through `epacems-2026.zip` from [PUDL EPA CEMS raw archive v79.0.0](https://doi.org/10.5281/zenodo.23982717). Place the ZIPs unopened in `data_raw/cems/`. This release ends at 2026 Q3 and includes 127 quarterly CSV members.
2. For the plant registry comparison, place `globalpowerplantdatabasev130.zip` in `data_raw/wri/`. The year-end risk model uses CEMS inputs. The WRI input supplies a separate facility-level coverage and generation check. [Global Power Plant Database v1.3.0](https://datasets.wri.org/datasets/global-power-plant-database)

To reuse existing downloads or panels, set `CEMS_ARCHIVES` to the annual ZIP directory, `CEMS_WRI_ARCHIVE` to the WRI ZIP file, `CEMS_PANEL` to the directory of derived Parquet files, and/or `CEMS_RESULTS` to an output directory. The defaults for the latter two are `data_processed/` and `results/`. The `CESSATION_THREADS` environment variable controls the LightGBM thread count for the main model (default 4).

## Reproduce the core analyses

Run the following commands in order. The extraction reads a quarterly CSV into memory per worker. Start with one worker and increase it if memory permits. This is a billion-row archive.

```sh
python scripts/extract_hourly.py --workers 1
python scripts/build_panel.py
python scripts/operating_histories.py
python scripts/backtest.py
python scripts/fleet_intervals.py --threads 4
python scripts/fleet_outlook.py
python scripts/plot_summary.py
```

Two companion analyses use the same processed panel:

```sh
python scripts/registry_check.py
python scripts/scrubber_comparison.py --threads 4
```

| Output | Interpretation |
| --- | --- |
| `pre_retirement_trajectories.csv` | Medians of differences from same-year fuel and technology peers and pointwise unit-bootstrap intervals. |
| `backtest.csv` | AUROC, average precision, Brier score, expected and observed counts and GW for each origin, model, eligibility group and horizon. The fourth-quarter subset uses the same model and prediction date as the annual cohort. |
| `fleet_interval_calibration_summary.csv` | Historical coverage and mean width. Use the independent-unit and sequential wave-calibrated rows for the primary comparison. Historical calibration uses earlier fully resolved cohorts and their apparent cessation capacity as of each forecast date. |
| `forecast_2026_2030_units.csv`, `fleet_scenario_intervals.csv` | Unit probabilities for the 2025 operating cohort fitted using historical labels available through 2026 Q3, plus expected GW and central 90% prediction intervals. |
| `core_comparisons.pdf` | Selected trajectory, count-ratio and capacity-coverage comparisons from the tables above. |
| `causal_scrubber_summary.json` | Adjusted five-year contrasts by existing scrubber status in the fully observable 2008–2018 coal cohorts. |

A quick real-data fit can be requested using `python scripts/backtest.py --origins 2008`. Choose a separate `CEMS_RESULTS` directory for a partial run to preserve any full-period results. For full-period totals, select `population == "Any operation"` and the requested horizon and model, then divide the sum of expected counts by the sum of observed counts across origins. Multiple forecast origins can include the same unit. With the archive above, the full data include 4,353 eligible units and 81,083 unit–year landmarks. At five years the pooled hazard accounts for approximately 97% of the observed count and 88% of the observed cessation-associated GW across 11 origins.

The fixed-horizon classifier and pooled hazard share the same primary operating features. Their difference includes both the additional resolved intervals used by the hazard model and its pooling across follow-up years. The equal-weight hybrid combines those forecasts with a recency-weighted hazard and an elastic-net classifier on the log-odds scale. This release fixes the elastic-net solver seed for repeatability; its last decimal can differ from the original unseeded solver run.
