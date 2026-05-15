import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.analysis import AnalysisJob
from app.schemas.analysis import AnalysisRequest
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler

logger = logging.getLogger(__name__)

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}
FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}
SECTION_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}


def _utc_now():
    return datetime.now(timezone.utc)


class AnalysisService:
    """Manages analysis jobs and provides event streaming for SSE."""

    def __init__(self, session_factory=None):
        self._queues: Dict[str, queue.Queue] = {}
        self._session_factory = session_factory

    def _new_session(self):
        if self._session_factory:
            return self._session_factory()
        from app.db.session import SessionLocal
        return SessionLocal()

    def start_analysis(self, db: Session, params: AnalysisRequest) -> str:
        job = AnalysisJob(
            ticker=params.ticker,
            analysis_date=params.analysis_date,
            status="running",
            params=params.model_dump(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        q: queue.Queue = queue.Queue()
        self._queues[job.id] = q

        thread = threading.Thread(
            target=self._run_analysis,
            args=(job.id, params, q, db),
            daemon=True,
        )
        thread.start()

        return job.id

    def get_event_queue(self, job_id: str) -> queue.Queue:
        q = self._queues.get(job_id)
        if q is None:
            raise KeyError(f"Job {job_id} not found or already completed")
        return q

    def get_job(self, db: Session, job_id: str) -> Optional[AnalysisJob]:
        return db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()

    def get_report(self, db: Session, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get_job(db, job_id)
        if job is None or job.result is None:
            return None
        return job.result

    # ------------------------------------------------------------------
    # Internal: run analysis in background thread
    # ------------------------------------------------------------------

    def _run_analysis(
        self, job_id: str, params: AnalysisRequest, q: queue.Queue, db: Session
    ):
        try:
            self._execute_analysis(job_id, params, q)
        except Exception as exc:
            logger.exception("Analysis job %s failed", job_id)
            q.put({"type": "error", "data": {"message": str(exc)}})
            self._update_job(db, job_id, status="failed", error=str(exc))
        finally:
            q.put(None)  # sentinel
            # Keep queue for a short time then clean up
            self._queues.pop(job_id, None)

    def _execute_analysis(
        self, job_id: str, params: AnalysisRequest, q: queue.Queue
    ):
        start_time = time.time()
        config = self._build_config(params)
        stats_handler = StatsCallbackHandler()

        graph = TradingAgentsGraph(
            params.analysts,
            config=config,
            debug=True,
            callbacks=[stats_handler],
        )

        # Initialize agent statuses
        agent_status = {}
        for analyst_key in params.analysts:
            if analyst_key in ANALYST_AGENT_NAMES:
                agent_status[ANALYST_AGENT_NAMES[analyst_key]] = "pending"
        for team_agents in FIXED_AGENTS.values():
            for agent in team_agents:
                agent_status[agent] = "pending"

        # Set first analyst as in_progress
        if params.analysts:
            first = ANALYST_AGENT_NAMES.get(params.analysts[0])
            if first:
                agent_status[first] = "in_progress"

        # Track report sections and processed message IDs
        report_sections: Dict[str, Optional[str]] = {}
        section_map = {
            "market_report": ("market", "Market Analyst"),
            "sentiment_report": ("social", "Social Analyst"),
            "news_report": ("news", "News Analyst"),
            "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        }
        for section, (analyst_key, _) in section_map.items():
            if analyst_key in params.analysts:
                report_sections[section] = None
        # Fixed sections (always included)
        for section in ("investment_plan", "trader_investment_plan", "final_trade_decision"):
            report_sections[section] = None

        seen_msg_ids: set = set()
        reports_done = 0

        # Emit initial state
        self._emit(q, "agent_status", dict(agent_status))
        self._emit_stats(q, agent_status, report_sections, stats_handler, start_time)

        # Stream through graph
        init_state = graph.propagator.create_initial_state(
            params.ticker, params.analysis_date
        )
        graph_args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        for chunk in graph.graph.stream(init_state, **graph_args):
            # --- Messages ---
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in seen_msg_ids:
                        continue
                    seen_msg_ids.add(msg_id)

                msg_type, content = self._classify_message(message)
                if content and content.strip():
                    q.put({"type": "message", "data": {"type": msg_type, "content": content}})

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        name = tc["name"] if isinstance(tc, dict) else tc.name
                        args = tc["args"] if isinstance(tc, dict) else tc.args
                        q.put({"type": "tool_call", "data": {"name": name, "args": args}})

            # --- Analyst status ---
            self._update_analyst_statuses(q, agent_status, report_sections, chunk, params.analysts)
            self._emit(q, "agent_status", dict(agent_status))

            # --- Research Team ---
            if chunk.get("investment_debate_state"):
                debate = chunk["investment_debate_state"]
                bull = (debate.get("bull_history") or "").strip()
                bear = (debate.get("bear_history") or "").strip()
                judge = (debate.get("judge_decision") or "").strip()

                if bull or bear:
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        agent_status[a] = "in_progress"
                if bull:
                    self._update_section(q, report_sections, "investment_plan", f"### Bull Researcher Analysis\n{bull}")
                if bear:
                    self._update_section(q, report_sections, "investment_plan", f"### Bear Researcher Analysis\n{bear}")
                if judge:
                    self._update_section(q, report_sections, "investment_plan", f"### Research Manager Decision\n{judge}")
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        agent_status[a] = "completed"
                    agent_status["Trader"] = "in_progress"
                    self._emit(q, "agent_status", dict(agent_status))

            # --- Trading Team ---
            if chunk.get("trader_investment_plan"):
                self._update_section(q, report_sections, "trader_investment_plan", chunk["trader_investment_plan"])
                if agent_status.get("Trader") != "completed":
                    agent_status["Trader"] = "completed"
                    agent_status["Aggressive Analyst"] = "in_progress"
                    self._emit(q, "agent_status", dict(agent_status))

            # --- Risk Management ---
            if chunk.get("risk_debate_state"):
                risk = chunk["risk_debate_state"]
                agg = (risk.get("aggressive_history") or "").strip()
                con = (risk.get("conservative_history") or "").strip()
                neu = (risk.get("neutral_history") or "").strip()
                rjudge = (risk.get("judge_decision") or "").strip()

                if agg:
                    agent_status["Aggressive Analyst"] = "in_progress"
                    self._update_section(q, report_sections, "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg}")
                if con:
                    agent_status["Conservative Analyst"] = "in_progress"
                    self._update_section(q, report_sections, "final_trade_decision", f"### Conservative Analyst Analysis\n{con}")
                if neu:
                    agent_status["Neutral Analyst"] = "in_progress"
                    self._update_section(q, report_sections, "final_trade_decision", f"### Neutral Analyst Analysis\n{neu}")
                if rjudge:
                    agent_status["Portfolio Manager"] = "in_progress"
                    self._update_section(q, report_sections, "final_trade_decision", f"### Portfolio Manager Decision\n{rjudge}")
                    for a in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"):
                        agent_status[a] = "completed"
                    self._emit(q, "agent_status", dict(agent_status))

            # --- Stats ---
            self._emit_stats(q, agent_status, report_sections, stats_handler, start_time)

        # Mark all completed
        for agent in agent_status:
            agent_status[agent] = "completed"
        self._emit(q, "agent_status", dict(agent_status))
        self._emit_stats(q, agent_status, report_sections, stats_handler, start_time)

        # Process final decision
        final_decision = graph.process_signal(report_sections.get("final_trade_decision", ""))

        # Emit complete
        self._emit(q, "complete", {
            "status": "completed",
            "ticker": params.ticker,
            "analysis_date": params.analysis_date,
            "decision": final_decision,
            "report_sections": report_sections,
            "stats": stats_handler.get_stats(),
            "elapsed": round(time.time() - start_time, 1),
        })

        # Save result to DB
        self._save_result(job_id, params, report_sections, final_decision, stats_handler)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_config(self, params: AnalysisRequest) -> Dict[str, Any]:
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = params.research_depth
        config["max_risk_discuss_rounds"] = params.research_depth
        config["quick_think_llm"] = params.shallow_thinker
        config["deep_think_llm"] = params.deep_thinker
        config["backend_url"] = params.backend_url
        config["llm_provider"] = params.llm_provider.lower()
        config["google_thinking_level"] = params.google_thinking_level
        config["openai_reasoning_effort"] = params.openai_reasoning_effort
        config["anthropic_effort"] = params.anthropic_effort
        config["output_language"] = params.output_language
        config["checkpoint_enabled"] = False
        return config

    def _emit(self, q: queue.Queue, event_type: str, data: dict):
        q.put({"type": event_type, "data": data})

    def _emit_stats(self, q, agent_status, report_sections, stats_handler, start_time):
        stats = stats_handler.get_stats()
        completed = sum(1 for s in agent_status.values() if s == "completed")
        total = len(agent_status)
        reports = sum(1 for v in report_sections.values() if v is not None)
        self._emit(q, "stats", {
            "agents_completed": completed,
            "agents_total": total,
            "llm_calls": stats["llm_calls"],
            "tool_calls": stats["tool_calls"],
            "tokens_in": stats["tokens_in"],
            "tokens_out": stats["tokens_out"],
            "reports_completed": reports,
            "reports_total": len(report_sections),
            "elapsed": round(time.time() - start_time, 1),
        })

    def _update_section(self, q, sections, key, content):
        sections[key] = content
        title = SECTION_TITLES.get(key, key)
        q.put({"type": "report", "data": {"section": key, "title": title, "content": content}})

    def _update_analyst_statuses(self, q, agent_status, report_sections, chunk, selected):
        found_active = False
        for analyst_key in ANALYST_ORDER:
            if analyst_key not in selected:
                continue
            agent_name = ANALYST_AGENT_NAMES[analyst_key]
            report_key = ANALYST_REPORT_MAP[analyst_key]
            if chunk.get(report_key):
                report_sections[report_key] = chunk[report_key]
                q.put({"type": "report", "data": {"section": report_key, "title": SECTION_TITLES[report_key], "content": chunk[report_key]}})
            if report_sections.get(report_key):
                agent_status[agent_name] = "completed"
            elif not found_active:
                agent_status[agent_name] = "in_progress"
                found_active = True
            else:
                agent_status[agent_name] = "pending"
        if not found_active and selected:
            if agent_status.get("Bull Researcher") == "pending":
                agent_status["Bull Researcher"] = "in_progress"

    @staticmethod
    def _classify_message(message) -> tuple:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        content = AnalysisService._extract_content(message)
        if isinstance(message, HumanMessage):
            if content and content.strip() == "Continue":
                return ("Control", content)
            return ("User", content)
        if isinstance(message, ToolMessage):
            return ("Data", content)
        if isinstance(message, AIMessage):
            return ("Agent", content)
        return ("System", content)

    @staticmethod
    def _extract_content(message) -> Optional[str]:
        import ast

        def is_empty(val):
            if val is None or val == "":
                return True
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return True
                try:
                    return not bool(ast.literal_eval(s))
                except (ValueError, SyntaxError):
                    return False
            return not bool(val)

        content = getattr(message, "content", None)
        if is_empty(content):
            return None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            text = content.get("text", "")
            return text.strip() if not is_empty(text) else None
        if isinstance(content, list):
            parts = [
                item.get("text", "").strip() if isinstance(item, dict) and item.get("type") == "text"
                else (item.strip() if isinstance(item, str) else "")
                for item in content
            ]
            result = " ".join(t for t in parts if t and not is_empty(t))
            return result if result else None
        return str(content).strip() if not is_empty(content) else None

    def _save_result(self, job_id, params, report_sections, decision, stats_handler):
        db = self._new_session()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "completed"
                job.result = {
                    "ticker": params.ticker,
                    "analysis_date": params.analysis_date,
                    "decision": decision,
                    "report_sections": report_sections,
                    "stats": stats_handler.get_stats(),
                }
                job.updated_at = _utc_now()
                db.commit()
        finally:
            db.close()

    def _update_job(self, _db, job_id, status, error=None):
        db = self._new_session()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = status
                if error:
                    job.error_message = error
                job.updated_at = _utc_now()
                db.commit()
        except Exception:
            pass
        finally:
            db.close()


# Singleton
analysis_service = AnalysisService()
