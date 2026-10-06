# Qiyas Hackathon Project Ground Truth

This project must follow `Qiyas-AI-Hackathon_Ride_Demand_Instructions.pdf` as the authoritative source of truth. This file is a working reference distilled from that PDF, not a replacement for it. When requirements are ambiguous, use the PDF first, then document any reasonable engineering assumption.

## Challenge

Forecast hourly ride demand (`trips`) for 12 Addis Ababa zones for every hour from 2025-11-01 through 2025-11-14. Training history covers 2025-01-01 through 2025-10-31.

Core principle: this is forecasting. Use the past to predict the future, validate chronologically, and only use features known at forecast time.

## Raw Inputs

- `data/raw/ride_demand_train.csv`: zone-hour training history with `trips` target plus historical-only operational columns.
- `data/raw/ride_demand_test.csv`: 4,032 forecast rows, 12 zones x 336 hours, without `trips`.
- `data/raw/submission_template.csv`: required `row_id` order for submission.
- `data/raw/weather_hourly.csv`: hourly city weather, observed during history and forecast for 2025-11-01 to 2025-11-14.
- `data/raw/events_calendar.csv`: event intervals requiring interval/window joins.

Never manually edit `data/raw/`.

## Non-Negotiable Rules

- Final model must use at least one weather-derived feature and at least one event-derived feature.
- Do not use train-only operational outcome columns as model inputs unless the feature availability audit proves they would be known at forecast time. By default, exclude `avg_fare_birr`, `avg_wait_min`, and `active_drivers` from forecast model inputs; keep them for analysis/demo outputs where appropriate.
- Validate by time only. Reported scores must use chronological splits; random splits may only be shown as contrast.
- Never score on the test file.
- Fit all learned statistics, imputers, caps, category lists, scalers, and encoders on training data only, then apply to validation/test.
- Use relative paths only.
- Every reported number, table, and figure must be produced by code in the project.
- Test pipeline must go through the same cleaning, joining, and feature engineering as train.
- Preserve the original `row_id` order in the submission file.

## Time And Joining

- Treat Addis Ababa local time as `Africa/Addis_Ababa` / EAT / UTC+3 with no daylight saving.
- Prove the clock/timezone alignment from the data before joining weather to trips.
- Standardize all zone labels to the same 12 cleaned labels across trip, weather-related joins, and event data.
- Weather join should be many-to-one by standardized hour and must not change row counts.
- Events are intervals, not point timestamps. Define and document before/during/after event windows, affected-zone logic, and exclusion logic.

## Required Deliverables

- Prediction file: `submission/team_<NAME>_submission.csv` with exactly `row_id,predicted_trips`, 4,032 rows, no blanks, negative values, or duplicate row IDs.
- A: Data Cleaning & Integration Pipeline.
- B: Data Analysis Report with all 14 numbered tasks.
- C: Visualization Pack with 12 exact PNG filenames plus `figures/figure_captions.md`.
- D: Modeling & Evaluation with baselines, model comparison, rolling-origin validation, leakage audit, ablation, tuning, error analysis, and plain-language metric explanation.
- E: Working forecast demo app.
- F: Five-slide presentation.
- G: Project structure, README, requirements, reproducibility, valid submission, and method hygiene.

## Required Processed Outputs

- `data/processed/master_train.csv`
- `data/processed/master_test.csv`
- `data/processed/data_dictionary_master.csv`
- `models/final_model.joblib`
- `app/assets/` containing the cleaned lookup tables and model artifacts needed by the demo.

## Required Figure Filenames

- `fig01_gaps_and_missingness.png`
- `fig02_before_after_cleaning.png`
- `fig03_demand_trend_with_holidays.png`
- `fig04_hour_by_weekday_heatmap.png`
- `fig05_zone_profiles.png`
- `fig06_weather_timezone_check.png`
- `fig07_rain_effect.png`
- `fig08_event_study.png`
- `fig09_holiday_effects.png`
- `fig10_model_comparison.png`
- `fig11_forecast_vs_actual.png`
- `fig12_feature_importance.png`

## Modeling Requirements

- Baselines: mean predictor and seasonal naive, scored with RMSE and MAE on a chronological validation fortnight.
- Compare at least three model families beyond baselines on the same split/features.
- Use rolling-origin validation with at least four folds for the final model and seasonal naive baseline.
- Include an ablation table with:
  - calendar + zone + trend only
  - plus weather
  - plus events
  - plus both
- Include a leakage/feature availability audit for every candidate feature.
- Include error analysis by zone, hour, day type, and top 10 largest-error zone-hours.

## Demo Requirements

- User inputs only: zone and date, where date must be in 2025-11-01 through 2025-11-14.
- App must look up weather and events automatically from bundled assets.
- Output the 24-hour forecast table and curve, peak hour, estimated drivers needed using forecast trips / about 1.3 trips per driver-hour, expected gross fares using zone average fare from history, and a short lookup summary.
- Include at least one in-app visual, such as forecast curve versus typical profile with event windows shaded.

## Expected Final Layout

```text
team_<NAME>/
+-- README.md
+-- requirements.txt
+-- submission/
+-- data/
|   +-- raw/
|   +-- processed/
+-- notebooks/
+-- src/
+-- models/
+-- figures/
+-- reports/
+-- app/
+-- presentation/
```

