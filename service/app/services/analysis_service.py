import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown
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

    def start_analysis(self, db: Session, params: AnalysisRequest) -> tuple[str, str]:
        """Start or retrieve an analysis job.

        Returns (job_id, status) — status is "started" for new jobs,
        "completed" for cached results, or "running" for in-progress jobs.
        """
        existing = (
            db.query(AnalysisJob)
            .filter(AnalysisJob.ticker == params.ticker)
            .filter(AnalysisJob.analysis_date == params.analysis_date)
            .first()
        )

        if existing is not None:
            return existing.id, existing.status

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

        return job.id, "started"

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
        # Save report files to disk
        report_dir = self._save_report_to_disk(
            params.ticker, params.analysis_date, report_sections, decision
        )

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
                    "report_dir": str(report_dir),
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

    # ------------------------------------------------------------------
    # Report file storage
    # ------------------------------------------------------------------

    _HTML_CSS = """\
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem 1rem; line-height: 1.6; color: #1a1a1a; background: #fff; }
        h1 { border-bottom: 2px solid #0366d6; padding-bottom: .3em; }
        h2 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; margin-top: 2em; }
        h3 { margin-top: 1.5em; }
        code { background: #f6f8fa; padding: .2em .4em; border-radius: 3px; font-size: 90%; }
        pre { background: #f6f8fa; padding: 1em; border-radius: 6px; overflow-x: auto; }
        pre code { background: none; padding: 0; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0; }
        th, td { border: 1px solid #dfe2e5; padding: .5em .8em; text-align: left; }
        th { background: #f6f8fa; }
        blockquote { border-left: 4px solid #0366d6; color: #555; padding-left: 1em; margin: 1em 0; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .nav { margin-bottom: 1.5em; font-size: .9em; }
        .nav a { padding: .2em .5em; }
        .toc { background: #f6f8fa; padding: 1em 2em; border-radius: 6px; margin: 1.5em 0; }
        .toc ul { padding-left: 1.5em; }
        .toc li { margin: .3em 0; }
    """

    @staticmethod
    def _md_to_html(md_text: str, title: str = "", back_link: str | None = None) -> str:
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        nav = f'<p class="nav"><a href="{back_link}">Back to Index</a></p>' if back_link else ""
        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{AnalysisService._HTML_CSS}
</style>
</head>
<body>
{nav}
<article>
{body}
</article>
</body>
</html>"""

    @staticmethod
    def _save_report_to_disk(ticker: str, analysis_date: str, report_sections: Dict[str, Optional[str]], decision: str) -> Path:
        base_dir = Path(__file__).resolve().parent.parent.parent / "reports" / ticker / analysis_date
        base_dir.mkdir(parents=True, exist_ok=True)

        section_files = {
            "market_report": ("market_analysis", "Market Analysis"),
            "sentiment_report": ("sentiment", "Social Sentiment"),
            "news_report": ("news", "News Analysis"),
            "fundamentals_report": ("fundamentals", "Fundamentals Analysis"),
            "investment_plan": ("research_decision", "Research Team Decision"),
            "trader_investment_plan": ("trader_plan", "Trading Team Plan"),
            "final_trade_decision": ("portfolio_decision", "Portfolio Management Decision"),
        }

        file_map: Dict[str, list] = {}
        report_parts: list = []

        for section_key, (filename, display_name) in section_files.items():
            content = report_sections.get(section_key)
            if not content:
                continue

            md_path = base_dir / f"{filename}.md"
            html_path = base_dir / f"{filename}.html"

            md_path.write_text(content, encoding="utf-8")
            html = AnalysisService._md_to_html(content, title=display_name, back_link="index.html")
            html_path.write_text(html, encoding="utf-8")

            file_map[display_name] = [(display_name, f"{filename}.md", f"{filename}.html")]
            report_parts.append(f"## {display_name}\n{content}")

        # Complete report
        header = f"# Trading Analysis Report: {ticker}\n\nAnalysis Date: {analysis_date}\nDecision: **{decision}**\n\n---\n\n"
        complete_md = header + "\n\n".join(report_parts)
        (base_dir / "complete_report.md").write_text(complete_md, encoding="utf-8")
        html = AnalysisService._md_to_html(complete_md, title=f"Report: {ticker}", back_link="index.html")
        (base_dir / "complete_report.html").write_text(html, encoding="utf-8")

        # Index
        AnalysisService._generate_index_html(base_dir, ticker, file_map)

        return base_dir

    @staticmethod
    def _generate_index_html(save_path: Path, ticker: str, file_map: dict) -> None:
        toc_items = []
        for section_name, files in file_map.items():
            toc_items.append(f'<h3>{section_name}</h3>')
            toc_items.append('<ul>')
            for label, md_file, html_file in files:
                toc_items.append(f'<li><a href="{html_file}">{label}</a> (<a href="{md_file}">md</a>)</li>')
            toc_items.append('</ul>')

        body = f"""\
<h1>Trading Analysis Report: {ticker}</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<div class="toc">
<h2>Contents</h2>
{"".join(toc_items)}
</div>
"""
        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Analysis Report: {ticker}</title>
<style>
{AnalysisService._HTML_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""
        (save_path / "index.html").write_text(html, encoding="utf-8")


# Singleton
analysis_service = AnalysisService()
