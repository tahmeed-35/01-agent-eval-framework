"""Evaluation Agent with built-in tools, latency measurement, and token logging."""

import time
import math
import re
from typing import Dict, Any, Tuple
from src.schemas import AgentInput, AgentOutput, ToolCall, TokenUsage


def calculate(expression: str) -> float:
    """Safe basic math evaluator."""
    clean_expr = expression.replace(" ", "").replace("x", "*").replace("^", "**")
    if not re.match(r"^[\d\+\-\*\/\.\(\)\%]+$", clean_expr):
        raise ValueError(f"Invalid or unsafe math expression: {expression}")
    # Controlled local evaluation for basic arithmetic
    allowed_names = {"__builtins__": None, "math": math}
    return float(eval(clean_expr, allowed_names, {}))


def get_weather(location: str) -> Dict[str, Any]:
    """Mock weather service returning structured weather data."""
    loc_clean = location.strip().title()
    mock_db = {
        "New York": {"temperature_c": 18, "condition": "Partly Cloudy", "humidity": 65},
        "London": {"temperature_c": 12, "condition": "Rainy", "humidity": 85},
        "Tokyo": {"temperature_c": 22, "condition": "Sunny", "humidity": 50},
        "Berlin": {"temperature_c": 14, "condition": "Overcast", "humidity": 70},
        "Paris": {"temperature_c": 16, "condition": "Clear", "humidity": 55},
        "San Francisco": {"temperature_c": 17, "condition": "Foggy", "humidity": 78},
    }
    data = mock_db.get(loc_clean, {"temperature_c": 20, "condition": "Clear", "humidity": 60})
    return {"location": loc_clean, **data}


def analyze_text(text: str) -> Dict[str, Any]:
    """Analyzes text properties such as word count, reading time, and basic sentiment tone."""
    words = re.findall(r"\w+", text)
    word_count = len(words)
    char_count = len(text)
    positive_words = {"good", "great", "excellent", "fast", "love", "awesome", "effective", "best"}
    negative_words = {"bad", "slow", "terrible", "poor", "hate", "bug", "failing", "worst"}
    
    pos_score = sum(1 for w in words if w.lower() in positive_words)
    neg_score = sum(1 for w in words if w.lower() in negative_words)
    
    sentiment = "neutral"
    if pos_score > neg_score:
        sentiment = "positive"
    elif neg_score > pos_score:
        sentiment = "negative"
        
    return {
        "word_count": word_count,
        "char_count": char_count,
        "estimated_reading_seconds": round(word_count / 3.5, 1),
        "sentiment": sentiment,
    }


class EvaluationAgent:
    """Agent that can parse user intent, dispatch to tools, and track latency & tokens."""

    def __init__(self, name: str = "eval-agent-v1"):
        self.name = name

    def _estimate_tokens(self, prompt: str, completion: str) -> TokenUsage:
        """Estimates token counts using standard heuristic (~4 characters per token)."""
        prompt_tokens = max(1, math.ceil(len(prompt) / 4))
        completion_tokens = max(1, math.ceil(len(completion) / 4))
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Processes the input query, invokes relevant tool if required, and returns structured AgentOutput."""
        start_time = time.perf_counter()
        query = input_data.query.strip()
        tool_calls = []
        response_text = ""
        status = "success"

        try:
            # 1. Routing for Calculator Tool
            math_match = re.search(r"(?:calculate|what is|compute)\s+([\d\s\+\-\*\/\.\(\)\%\^x]+)", query, re.IGNORECASE)
            if math_match or (re.search(r"[\d]+[\s]*[\+\-\*\/][\s]*[\d]+", query) and not "weather" in query.lower()):
                expr = math_match.group(1).strip() if math_match else query.strip()
                result = calculate(expr)
                tool_calls.append(ToolCall(name="calculate", arguments={"expression": expr}, output=result))
                response_text = f"The result of {expr} is {result}."

            # 2. Routing for Weather Tool
            elif "weather" in query.lower():
                loc_match = re.search(r"weather in ([a-zA-Z\s]+)", query, re.IGNORECASE)
                location = loc_match.group(1).strip() if loc_match else "San Francisco"
                weather_info = get_weather(location)
                tool_calls.append(ToolCall(name="get_weather", arguments={"location": location}, output=weather_info))
                response_text = f"The weather in {weather_info['location']} is {weather_info['condition']} with temperature {weather_info['temperature_c']}°C."

            # 3. Routing for Text Analyzer Tool
            elif any(k in query.lower() for k in ["analyze text", "word count", "sentiment of", "analyze:"]):
                target_text = re.sub(r"^(?:analyze text|sentiment of|word count|analyze:)\s*", "", query, flags=re.IGNORECASE)
                analysis = analyze_text(target_text)
                tool_calls.append(ToolCall(name="analyze_text", arguments={"text": target_text}, output=analysis))
                response_text = (
                    f"Analysis: {analysis['word_count']} words, {analysis['char_count']} characters. "
                    f"Detected sentiment: {analysis['sentiment']}."
                )

            # 4. Fallback / Direct Conversational Answer
            else:
                response_text = f"Direct answer: I understood your query: '{query}'."

        except Exception as exc:
            status = "error"
            response_text = f"Error executing task: {str(exc)}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tokens = self._estimate_tokens(query, response_text)

        return AgentOutput(
            response=response_text,
            tool_calls=tool_calls,
            latency_ms=round(elapsed_ms, 2),
            token_usage=tokens,
            raw_status=status,
        )
