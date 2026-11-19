"""Unit tests for EvaluationAgent and tool execution."""

import pytest
from src.agent import EvaluationAgent, calculate, get_weather, analyze_text
from src.schemas import AgentInput, FailureCategory


def test_calculate_valid_expressions():
    assert calculate("2 + 2") == 4.0
    assert calculate("10 * 5 + 2") == 52.0
    assert calculate("(15 - 5) / 2") == 5.0
    assert calculate("2^4") == 16.0


def test_calculate_invalid_expression():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('dir')")


def test_weather_lookup():
    res = get_weather("Tokyo")
    assert res["location"] == "Tokyo"
    assert res["condition"] == "Sunny"
    assert res["temperature_c"] == 22


def test_analyze_text():
    res = analyze_text("This is great and awesome!")
    assert res["word_count"] == 5
    assert res["sentiment"] == "positive"


def test_agent_run_calculator():
    agent = EvaluationAgent()
    out = agent.run(AgentInput(query="Calculate 40 + 60"))
    assert out.raw_status == "success"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "calculate"
    assert "100" in out.response
    assert out.latency_ms >= 0.0
    assert out.token_usage.total_tokens > 0


def test_agent_run_weather():
    agent = EvaluationAgent()
    out = agent.run(AgentInput(query="What is the weather in London?"))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "get_weather"
    assert "London" in out.response


def test_agent_run_conversational():
    agent = EvaluationAgent()
    out = agent.run(AgentInput(query="Good morning!"))
    assert len(out.tool_calls) == 0
    assert "Good morning!" in out.response
