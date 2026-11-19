"""Unit tests for AgentEvaluator and regression reporting."""

from pathlib import Path
from src.evaluator import AgentEvaluator
from src.agent import EvaluationAgent
from src.schemas import TestCase, FailureCategory


def test_load_and_run_dataset():
    evaluator = AgentEvaluator()
    cases_path = Path(__file__).resolve().parent.parent / "datasets" / "cases.json"
    cases = evaluator.load_cases(str(cases_path))
    assert len(cases) >= 15

    summary = evaluator.run_suite(cases)
    assert summary.total_cases == len(cases)
    assert summary.passed_cases > 0
    assert summary.accuracy_rate >= 80.0
    assert summary.avg_latency_ms >= 0.0


def test_failure_categorization_tool_mismatch():
    evaluator = AgentEvaluator()
    # Expects weather tool, but query is a calculation
    bad_case = TestCase(
        case_id="FAIL-001",
        description="Intent mismatch",
        query="Calculate 5 * 5",
        expected_tool="get_weather",
    )
    res = evaluator.evaluate_case(bad_case)
    assert res.passed is False
    assert res.failure_category == FailureCategory.TOOL_SELECTION_ERROR


def test_markdown_report_generation(tmp_path):
    evaluator = AgentEvaluator()
    case = TestCase(
        case_id="TEST-REPORT",
        description="Simple pass case",
        query="What is 10 + 10",
        expected_tool="calculate",
        expected_keywords=["20"],
    )
    summary = evaluator.run_suite([case])
    report_file = tmp_path / "test_report.md"
    content = evaluator.generate_markdown_report(summary, str(report_file))

    assert "Agent Evaluation & Regression Report" in content
    assert report_file.exists()
    assert "TEST-REPORT" in report_file.read_text(encoding="utf-8")
