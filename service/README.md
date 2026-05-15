### Run

uvicorn app.main:app --reload --port 8000 --app-dir service

# Set environment variables
export TRADINGAGENTS_CACHE_DIR="$PWD/.tradingagents/cache"
export TRADINGAGENTS_RESULTS_DIR="$PWD/.tradingagents/logs"
export TRADINGAGENTS_MEMORY_LOG_PATH="$PWD/.tradingagents/memory/trading_memory.md"