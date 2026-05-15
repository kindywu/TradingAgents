import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, JobStatus, ReportResponse
from app.services.analysis_service import analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisResponse, status_code=202)
def start_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    """Start a new trading analysis job. Returns job_id for tracking.

    Connect to GET /{job_id}/stream for real-time SSE progress events.
    """
    if not request.analysts:
        raise HTTPException(status_code=400, detail="At least one analyst must be selected")
    job_id = analysis_service.start_analysis(db, request)
    return AnalysisResponse(job_id=job_id, status="started")


@router.get("/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the current status of an analysis job."""
    job = analysis_service.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/stream")
async def stream_analysis(job_id: str):
    """SSE endpoint: stream real-time analysis progress events.

    Events emitted:
      - agent_status: {"agent_name": "status", ...}
      - message: {"type": "Agent|User|Data|Control", "content": "..."}
      - tool_call: {"name": "...", "args": {...}}
      - report: {"section": "...", "title": "...", "content": "..."}
      - stats: {"agents_completed": N, "llm_calls": N, "tokens_in": N, ...}
      - complete: {"status": "completed", "decision": "...", ...}
      - error: {"message": "..."}
    """
    try:
        q = analysis_service.get_event_queue(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found or already finished")

    loop = asyncio.get_event_loop()

    async def event_generator():
        while True:
            event = await loop.run_in_executor(None, q.get)
            if event is None:
                break
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/report", response_model=ReportResponse)
def get_report(job_id: str, db: Session = Depends(get_db)):
    """Get the final analysis report for a completed job."""
    job = analysis_service.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        raise HTTPException(status_code=425, detail="Analysis still in progress")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {job.error_message}")
    if job.result is None:
        raise HTTPException(status_code=404, detail="No report available")

    result = job.result
    sections = result.get("report_sections", {})
    # Build a single markdown report from sections
    report_parts = []
    if sections.get("market_report"):
        report_parts.append(f"## Market Analysis\n{sections['market_report']}")
    if sections.get("sentiment_report"):
        report_parts.append(f"## Social Sentiment\n{sections['sentiment_report']}")
    if sections.get("news_report"):
        report_parts.append(f"## News Analysis\n{sections['news_report']}")
    if sections.get("fundamentals_report"):
        report_parts.append(f"## Fundamentals Analysis\n{sections['fundamentals_report']}")
    if sections.get("investment_plan"):
        report_parts.append(f"## Research Team Decision\n{sections['investment_plan']}")
    if sections.get("trader_investment_plan"):
        report_parts.append(f"## Trading Team Plan\n{sections['trader_investment_plan']}")
    if sections.get("final_trade_decision"):
        report_parts.append(f"## Portfolio Management Decision\n{sections['final_trade_decision']}")

    return ReportResponse(
        job_id=job.id,
        ticker=job.ticker,
        analysis_date=job.analysis_date,
        decision=result.get("decision"),
        report="\n\n".join(report_parts) if report_parts else None,
        sections=sections if sections else None,
    )
