import os
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


class Settings:
    PORT: int = int(os.getenv("PORT", "8000"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    ENV: str = os.getenv("ENV", "development")


@lru_cache
def get_settings() -> Settings:
    return Settings()
