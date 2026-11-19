"""Data schemas and type definitions for Agent Evaluation Framework."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    NONE = "none"
    TOOL_SELECTION_ERROR = "tool_selection_error"
    ARGUMENT_ERROR = "argument_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    TIMEOUT_ERROR = "timeout_error"
    OUTPUT_MISMATCH = "output_mismatch"
    EXECUTION_ERROR = "execution_error"


class ToolCall(BaseModel):
    name: str = Field(..., description="Name of the invoked tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    output: Optional[Any] = Field(None, description="Output returned by the tool")


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentInput(BaseModel):
    query: str = Field(..., description="User query or task prompt")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional extra context")


class AgentOutput(BaseModel):
    response: str = Field(..., description="Final text response from the agent")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of executed tool calls")
    latency_ms: float = Field(0.0, description="End-to-end execution latency in milliseconds")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Token consumption metrics")
    raw_status: str = Field("success", description="Status string: success, error, or fallback")


class TestCase(BaseModel):
    case_id: str = Field(..., description="Unique test case identifier")
    description: str = Field(..., description="Description of the test scenario")
    query: str = Field(..., description="User input to pass to the agent")
    expected_tool: Optional[str] = Field(None, description="Expected tool name if a tool should be called")
    expected_args: Optional[Dict[str, Any]] = Field(None, description="Expected arguments or subset of arguments")
    expected_keywords: Optional[List[str]] = Field(default_factory=list, description="Keywords expected in the response")
    max_latency_ms: Optional[float] = Field(None, description="Maximum acceptable latency in ms")


class TestCaseResult(BaseModel):
    case_id: str
    passed: bool
    failure_category: FailureCategory = FailureCategory.NONE
    failure_reason: Optional[str] = None
    output: AgentOutput
    latency_ms: float
    token_usage: TokenUsage


class EvaluationSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens_used: int
    failure_breakdown: Dict[FailureCategory, int]
    results: List[TestCaseResult]
