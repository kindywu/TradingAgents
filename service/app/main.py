from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.session import engine
from app.models.base import Base
from app.routers import health, analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="TradingAgents API",
    description="REST API for the TradingAgents multi-agent financial trading framework",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(analysis.router)

settings = get_settings()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
