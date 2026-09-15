Automated Data Quality & Validation System

An end-to-end pipeline that profiles, cleans, and validates messy datasets before they are used downstream — a "data gatekeeper" for financial data.

Built as part of the CadetX AI virtual work experience programme (AI Engineering track).

Problem

Raw data arriving at an organisation is usually messy: missing values, duplicates, inconsistent formats, and errors that are invisible until they corrupt a model or a report. This system detects and fixes those problems automatically, and flags what it cannot fix.

Domain & Dataset

Domain: Finance (fraud and lending)

Dataset: Finance Fraud & Loans Dataset (TestDataBox, Kaggle) — a 50k-record synthetic finance dataset covering customers, accounts, loans, and transactions.

Approach

The validation layer uses a hybrid fraud detection strategy:

Rule-based checks — format validation (emails, phone numbers, dates), range checks (age, loan amounts), and cross-field consistency. Fast, explainable, catches the obvious problems first.
XGBoost (supervised) — learns from labelled fraud examples to catch known fraud patterns.
Isolation Forest (unsupervised) — flags unusual records without needing labels, catching novel patterns that supervised models have not seen.

Combining supervised and unsupervised detection covers both known and previously unseen fraud.

Architecture
raw data
   │
   ▼
Module 1 — Profile ──────► profiling_report.json
   │
   ▼
Module 2 — Clean ────────► cleaned_data.csv + cleaning_log.json
   │
   ▼
Module 3 — Validate ─────► validation_report.json
   │
   ▼
Module 4 — Automate ─────► end-to-end pipeline (CLI + Docker)
Module 1 — Profiling & Metadata Intelligence

Infers column types and semantic meaning, maps missing values, detects mixed-type columns, flags potential PII, and generates distribution and correlation summaries.

Output: reports/profiling_report.json

Module 2 — Cleaning & Transformation

Imputes missing values (statistical and ML-based: KNN, regression, iterative), detects exact and fuzzy duplicates, normalises types, dates, and categories, and scores data quality before and after so the improvement is measurable.

Output: data/processed/cleaned_data.csv, reports/cleaning_log.json

Module 3 — AI Validation & Anomaly Detection

Applies rule-based validation, then hybrid fraud detection (XGBoost + Isolation Forest). Scores error and anomaly severity and produces an overall dataset health score. Evaluated with precision, recall, and confusion matrices.

Output: reports/validation_report.json

Module 4 — Integration & Production Engineering

Orchestrates Modules 1 to 3 into a single configurable pipeline with YAML config, logging, automated tests, a CLI, and Docker packaging.

Usage:

bash
python pipeline.py --input data/raw/transactions.csv --output reports/
Tech Stack

Python · pandas · scikit-learn · XGBoost · pytest · Docker

Setup
bash
git clone https://github.com/FalakShahid66/automated-data-quality-system.git
cd automated-data-quality-system
pip install -r requirements.txt
Repository Structure
├── data/
│   ├── raw/                 # source dataset
│   └── processed/           # cleaned output
├── module1_profiling/
├── module2_cleaning/
├── module3_validation/
├── module4_pipeline/
├── reports/                 # generated JSON reports
├── tests/
└── docs/                    # architecture and API documentation
Team
Name	Role
Falak Shahid	AI Engineer — validation & anomaly detection
—	ML Engineer
—	Data Scientist

Scrum Master rotates weekly.

Status

Module 1 in progress.
