from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    ticker: str = Field(default="SPY", min_length=1, max_length=20, description="Ticker symbol to analyze (e.g. SPY, AAPL, 7203.T)")
    analysis_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="Analysis date (YYYY-MM-DD)")
    analysts: List[str] = Field(default=["market", "social", "news", "fundamentals"], description="Analyst types: market, social, news, fundamentals")
    research_depth: int = Field(default=1, ge=1, le=5, description="Research depth: 1=shallow, 3=medium, 5=deep")
    llm_provider: str = Field(default="openai", description="LLM provider: openai, google, anthropic, xai, deepseek, qwen, glm, openrouter, azure, ollama")
    backend_url: Optional[str] = Field(default=None, description="Custom LLM API endpoint URL")
    shallow_thinker: str = Field(default="gpt-4o-mini", description="Model for shallow/fast thinking tasks")
    deep_thinker: str = Field(default="gpt-4o", description="Model for deep/complex reasoning tasks")
    google_thinking_level: Optional[str] = Field(default=None, description="Gemini thinking level: high, minimal")
    openai_reasoning_effort: Optional[str] = Field(default=None, description="OpenAI reasoning effort: low, medium, high")
    anthropic_effort: Optional[str] = Field(default=None, description="Anthropic effort level: low, medium, high")
    output_language: str = Field(default="English", description="Output language for reports")


class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    model_config = ConfigDict(from_attributes=True)


class JobStatus(BaseModel):
    id: str
    ticker: str
    analysis_date: str
    status: str
    created_at: Any
    updated_at: Any
    error_message: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class ReportResponse(BaseModel):
    job_id: str
    ticker: str
    analysis_date: str
    decision: Optional[str] = None
    report: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
