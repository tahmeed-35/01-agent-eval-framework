#!/usr/bin/env python3
"""CLI script to run agent evaluation suite and write benchmark report."""

import sys
from pathlib import Path
from src.evaluator import AgentEvaluator


def main():
    base_dir = Path(__file__).resolve().parent
    dataset_file = base_dir / "datasets" / "cases.json"
    report_file = base_dir / "reports" / "latest_report.md"

    print(f"🚀 Loading evaluation suite from: {dataset_file}")
    evaluator = AgentEvaluator()
    cases = evaluator.load_cases(str(dataset_file))
    print(f"Loaded {len(cases)} evaluation test cases.")

    print("Running evaluations...")
    summary = evaluator.run_suite(cases)

    print("\n" + "=" * 50)
    print(f"📊 Evaluation Results:")
    print(f"   Accuracy: {summary.accuracy_rate}% ({summary.passed_cases}/{summary.total_cases})")
    print(f"   Average Latency: {summary.avg_latency_ms:.2f} ms")
    print(f"   P95 Latency: {summary.p95_latency_ms:.2f} ms")
    print(f"   Total Tokens: {summary.total_tokens_used}")
    print("=" * 50)

    evaluator.generate_markdown_report(summary, str(report_file))
    print(f"📄 Report generated at: {report_file}")

    if summary.failed_cases > 0:
        print(f"⚠️  {summary.failed_cases} test cases failed.")
        sys.exit(1)
    else:
        print("✅ All evaluation cases passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
