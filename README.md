# Automated AI Agent Evaluation & Regression Framework

A modular, pytest-based evaluation and regression benchmarking suite for LLM agents. Evaluates task completion, tool-call accuracy, schema conformance, latency percentiles, and token consumption with structured failure categorization.

## Features

- **Pydantic Schemas**: Strict data contracts for evaluation cases, tool invocations, agent outputs, and summary metrics.
- **Agent with Built-in Tools**: Extensible agent architecture with sample tools (`calculate`, `get_weather`, `analyze_text`).
- **Comprehensive Dataset**: 16 curated evaluation cases covering arithmetic, real-time tool routing, text analysis, edge cases, and conversational queries.
- **Latency & Token Telemetry**: Tracks execution duration in milliseconds, calculates P95 percentiles, and tracks token consumption.
- **Failure Categorization**: Automatically categorizes errors into `tool_selection_error`, `argument_error`, `schema_validation_error`, `timeout_error`, or `output_mismatch`.
- **Markdown Reporting**: Automatically outputs structured regression reports to `reports/`.
- **CI / CD Ready**: Preconfigured GitHub Actions workflow for automated testing.

## Directory Structure

```
01-agent-eval-framework/
├── .github/workflows/
│   └── ci.yml               # Automated CI test pipeline
├── datasets/
│   └── cases.json           # Evaluation benchmark cases (16 cases)
├── reports/                 # Generated evaluation & regression reports
├── src/
│   ├── __init__.py
│   ├── agent.py             # Agent runtime with tool execution & token tracker
│   ├── evaluator.py         # Evaluation engine and report generator
│   └── schemas.py           # Pydantic models and FailureCategory enum
├── tests/
│   ├── test_agent.py        # Agent and tool unit tests
│   └── test_evaluator.py    # Evaluator and metric aggregation tests
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt         # Project dependencies
├── run_eval.py              # CLI evaluation runner
└── README.md
```

## Quick Start

### 1. Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Evaluation Suite

Run the CLI runner to execute all benchmark cases and produce a report:

```bash
python run_eval.py
```

### 3. Run Pytest Suite

```bash
pytest -v
```
