# OpsLens

OpsLens investigates sales and operations data by combining deterministic Python analytics with an LLM that selects from a small set of supported tools. The model does not calculate business metrics itself and cannot execute arbitrary Python or SQL.

## What it does

- Validates and prepares a retail operations CSV.
- Calculates revenue, estimated profit, units, order counts, average delivery time, cancellation rate, and profit margin.
- Compares metrics by region, category, product, or order status.
- Builds weekly or monthly trends with period-over-period change.
- Ranks regions for further delivery and cancellation investigation.
- Uses Groq tool calling to select analyses in response to natural-language questions.
- Displays tool results alongside the generated explanation so findings can be checked.
- Includes a sample dataset and tests for core calculations and validation rules.

## Project structure

```text
opslens/
├── app.py
├── data/
│   └── retail_operations.csv
├── src/
│   └── opslens/
│       ├── __init__.py
│       ├── agent.py
│       └── analytics.py
├── tests/
│   └── test_analytics.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Run locally

Python 3.10 or later is required.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add a Groq API key to `.env` to enable natural-language investigations. Keep the key private and do not commit `.env`.

```powershell
streamlit run app.py
```

The deterministic comparison and trend panels work without an API key. The natural-language agent requires a valid key and model access.

## Dataset contract

Uploaded CSV files must contain these columns:

`order_date`, `order_id`, `region`, `category`, `product`, `units`, `unit_price`, `discount_pct`, `cost_per_unit`, `delivery_days`, `status`.

Rows with invalid dates, missing order identifiers, or invalid core numeric values are excluded. Units, prices, and costs must be non-negative. Missing discounts default to zero and discounts are constrained to 0–100. The interface displays the resulting valid-row count and data quality summary.

Revenue is calculated as `units × unit_price × (1 − discount_pct / 100)`. Profit is an estimate calculated as revenue minus `units × cost_per_unit`. It does not include shipping, overhead, tax, returns, or other costs unless those are represented in the source data. Cancellation rate is the percentage of rows whose status equals `Cancelled`, so the input must represent the intended order population consistently.

## Design choices

### Calculations stay in Python

Metrics and aggregations are implemented in `analytics.py`. This makes their definitions testable and prevents the language model from inventing calculations. The LLM interprets the question and requests one of the predefined tools.

### Tool access is constrained

`agent.py` exposes four tools: dataset profiling, metric comparison, metric trends, and delivery-risk ranking. Tool arguments are constrained by JSON schemas and validated again by the analytics layer. The model cannot execute generated code or issue arbitrary database queries. A turn is limited to four tool-call rounds.

### Findings remain inspectable

The UI displays the tool names and structured outputs used to prepare an answer. A reviewer can inspect the figures rather than relying only on generated prose.

## Run tests

```bash
python -m pytest
```

## Current limitations

- The analysis contract is designed for retail sales and operations data with the columns listed above, not arbitrary CSV schemas.
- The sample dataset is synthetic and should not be treated as a real business benchmark.
- Profit is an estimate based on the available cost fields.
- A regional risk ranking identifies where to investigate; it does not establish why delays or cancellations occur.
- Natural-language output depends on the configured model. Tool outputs are the source of numeric evidence, but generated explanations still require review.
- This is a portfolio project, not a production service. Authentication, deployment hardening, audit retention, and enterprise access controls are outside the current scope.


**What would you improve next?** Add dataset-specific schema mapping, stronger evaluation cases for tool selection and answer faithfulness, data-size limits, request timeouts, and a deployment model with authentication and audit logs.
