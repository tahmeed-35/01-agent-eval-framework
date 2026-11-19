"""Evaluation runner that benchmarks an LLM agent against test suites and produces regression reports."""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.schemas import (
    TestCase,
    TestCaseResult,
    EvaluationSummary,
    FailureCategory,
    AgentInput,
)
from src.agent import EvaluationAgent


class AgentEvaluator:
    """Orchestrates test suite execution and evaluation metrics aggregation."""

    def __init__(self, agent: Optional[EvaluationAgent] = None):
        self.agent = agent or EvaluationAgent()

    def load_cases(self, file_path: str) -> List[TestCase]:
        """Loads and validates test cases from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Test cases file not found: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return [TestCase(**item) for item in raw_data]

    def evaluate_case(self, case: TestCase) -> TestCaseResult:
        """Executes a single test case and categorizes any failure."""
        inp = AgentInput(query=case.query)
        output = self.agent.run(inp)

        # Check for runtime error
        if output.raw_status == "error":
            return TestCaseResult(
                case_id=case.case_id,
                passed=False,
                failure_category=FailureCategory.EXECUTION_ERROR,
                failure_reason=f"Agent returned error status: {output.response}",
                output=output,
                latency_ms=output.latency_ms,
                token_usage=output.token_usage,
            )

        # Check tool selection
        called_tools = [tc.name for tc in output.tool_calls]
        if case.expected_tool:
            if case.expected_tool not in called_tools:
                return TestCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    failure_category=FailureCategory.TOOL_SELECTION_ERROR,
                    failure_reason=f"Expected tool '{case.expected_tool}', but called {called_tools}",
                    output=output,
                    latency_ms=output.latency_ms,
                    token_usage=output.token_usage,
                )
            
            # Check tool arguments if specified
            if case.expected_args:
                matched_tc = next((tc for tc in output.tool_calls if tc.name == case.expected_tool), None)
                if matched_tc:
                    for k, v in case.expected_args.items():
                        actual_val = matched_tc.arguments.get(k)
                        if actual_val is None or str(v).lower() not in str(actual_val).lower():
                            return TestCaseResult(
                                case_id=case.case_id,
                                passed=False,
                                failure_category=FailureCategory.ARGUMENT_ERROR,
                                failure_reason=f"Argument mismatch for key '{k}': expected '{v}', got '{actual_val}'",
                                output=output,
                                latency_ms=output.latency_ms,
                                token_usage=output.token_usage,
                            )
        else:
            # No tool was expected
            if len(called_tools) > 0:
                return TestCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    failure_category=FailureCategory.TOOL_SELECTION_ERROR,
                    failure_reason=f"Expected no tool calls, but called {called_tools}",
                    output=output,
                    latency_ms=output.latency_ms,
                    token_usage=output.token_usage,
                )

        # Check expected keywords in response
        if case.expected_keywords:
            missing_keywords = [
                kw for kw in case.expected_keywords if kw.lower() not in output.response.lower()
            ]
            if missing_keywords:
                return TestCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    failure_category=FailureCategory.OUTPUT_MISMATCH,
                    failure_reason=f"Missing expected keywords: {missing_keywords}",
                    output=output,
                    latency_ms=output.latency_ms,
                    token_usage=output.token_usage,
                )

        # Check latency SLA if specified
        if case.max_latency_ms is not None and output.latency_ms > case.max_latency_ms:
            return TestCaseResult(
                case_id=case.case_id,
                passed=False,
                failure_category=FailureCategory.TIMEOUT_ERROR,
                failure_reason=f"Latency {output.latency_ms:.2f}ms exceeded max {case.max_latency_ms:.2f}ms",
                output=output,
                latency_ms=output.latency_ms,
                token_usage=output.token_usage,
            )

        # All assertions passed
        return TestCaseResult(
            case_id=case.case_id,
            passed=True,
            failure_category=FailureCategory.NONE,
            output=output,
            latency_ms=output.latency_ms,
            token_usage=output.token_usage,
        )

    def run_suite(self, cases: List[TestCase]) -> EvaluationSummary:
        """Runs the entire test suite and compiles summary metrics."""
        results: List[TestCaseResult] = []
        failure_breakdown: Dict[FailureCategory, int] = {cat: 0 for cat in FailureCategory}

        latencies = []
        total_tokens = 0

        for case in cases:
            res = self.evaluate_case(case)
            results.append(res)
            latencies.append(res.latency_ms)
            total_tokens += res.token_usage.total_tokens
            if not res.passed:
                failure_breakdown[res.failure_category] += 1

        total_cases = len(cases)
        passed_cases = sum(1 for r in results if r.passed)
        failed_cases = total_cases - passed_cases
        accuracy = (passed_cases / total_cases * 100.0) if total_cases > 0 else 0.0

        latencies.sort()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_index = int(math.ceil(0.95 * len(latencies))) - 1 if latencies else 0
        p95_latency = latencies[max(0, p95_index)] if latencies else 0.0

        return EvaluationSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            accuracy_rate=round(accuracy, 2),
            avg_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            total_tokens_used=total_tokens,
            failure_breakdown=failure_breakdown,
            results=results,
        )

    def generate_markdown_report(self, summary: EvaluationSummary, output_path: Optional[str] = None) -> str:
        """Formats the evaluation summary into a human-readable markdown regression report."""
        lines = [
            "# Agent Evaluation & Regression Report",
            "",
            f"- **Total Cases**: {summary.total_cases}",
            f"- **Passed**: {summary.passed_cases} ({summary.accuracy_rate}%)",
            f"- **Failed**: {summary.failed_cases}",
            f"- **Avg Latency**: {summary.avg_latency_ms:.2f} ms",
            f"- **P95 Latency**: {summary.p95_latency_ms:.2f} ms",
            f"- **Total Tokens**: {summary.total_tokens_used}",
            "",
            "## Failure Breakdown",
            "",
            "| Failure Category | Count |",
            "|---|---|",
        ]

        for cat, cnt in summary.failure_breakdown.items():
            if cnt > 0:
                lines.append(f"| {cat.value} | {cnt} |")

        lines.extend([
            "",
            "## Detailed Results",
            "",
            "| Case ID | Status | Latency (ms) | Tokens | Failure Reason |",
            "|---|---|---|---|---|",
        ])

        for r in summary.results:
            status_icon = "✅ PASS" if r.passed else f"❌ FAIL ({r.failure_category.value})"
            reason = r.failure_reason or "-"
            lines.append(f"| {r.case_id} | {status_icon} | {r.latency_ms:.1f} | {r.token_usage.total_tokens} | {reason} |")

        report_content = "\n".join(lines) + "\n"

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content
