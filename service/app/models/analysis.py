import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, JSON, Text, UniqueConstraint
from app.models.base import Base


def _utc_now():
    return datetime.now(timezone.utc)


def _new_uuid():
    return str(uuid.uuid4())


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        UniqueConstraint("ticker", "analysis_date", name="uq_ticker_date"),
    )

    id = Column(String, primary_key=True, default=_new_uuid)
    ticker = Column(String, nullable=False, index=True)
    analysis_date = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    params = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utc_now)
    updated_at = Column(DateTime, nullable=False, default=_utc_now, onupdate=_utc_now)
