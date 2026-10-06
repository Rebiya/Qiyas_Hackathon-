# Qiyas AI Hackathon - Addis Ride Demand Forecasting

End-to-end workflow for the Qiyas AI Hackathon #2 ride demand forecasting challenge.

## Assignment Ground Truth

The authoritative assignment is `Qiyas-AI-Hackathon_Ride_Demand_Instructions.pdf`.
Use `docs/project_ground_truth.md` as the working reference for rules, deliverables,
required filenames, validation constraints, and demo requirements.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current Pipeline

Clean raw source tables:

```bash
python -m src.cleaning
```

This writes phase-2 cleaned tables and audits to `data/processed/`:

- `clean_train.csv`
- `clean_test.csv`
- `clean_weather_hourly.csv`
- `clean_events_calendar.csv`
- `cleaning_log.csv`
- `unique_value_audit.csv`

Raw files live in `data/raw/` and should never be edited manually.
