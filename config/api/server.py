#!/usr/bin/env python3
"""
Main FastAPI server for AI Day Trader Agent API Layer.
Implements secure, standards-compliant REST endpoints for portfolio management and trading.
"""

from dotenv import load_dotenv

from core.env_store import env_path

# Load the same .env the API Keys tab writes to (DOTENV_PATH or the project's .env).
load_dotenv(env_path())

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import logging
import os
import pathlib
from contextlib import asynccontextmanager

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config.api.auth import router as auth_router, limiter
from config.api.portfolios import router as portfolios_router
from config.api.analysis import router as analysis_router
from config.api.trading import router as trading_router
from config.api.settings import router as settings_router
from config.api.control import router as control_router
from config.api.websockets import websocket_endpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_day_trader_api")

async def _supervisor_loop() -> None:
    """Keep bots that should be running alive (after crashes or a reboot)."""
    import asyncio

    from fastapi.concurrency import run_in_threadpool

    from config.api.control import get_bot_manager, supervise_once

    interval = float(os.getenv("SUPERVISOR_INTERVAL_SECONDS", "30"))
    while True:
        try:
            await run_in_threadpool(supervise_once, get_bot_manager())
        except Exception as exc:  # never let supervision take the server down
            logger.error(f"Bot supervisor error: {exc}")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    task = asyncio.create_task(_supervisor_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="AI Day Trader Agent API",
    description="Secure REST API for portfolio management, trading, and analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - restrict origins in production
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",")]

# Warn if using wildcard in production
if os.getenv("ENVIRONMENT") == "production" and "*" in CORS_ORIGINS:
    logger.warning("⚠️  CORS allows all origins in production! Set CORS_ORIGINS environment variable.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)

# Register routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(portfolios_router, prefix="/api/portfolios", tags=["portfolios"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(trading_router, prefix="/api/trading", tags=["trading"])
app.include_router(settings_router, prefix="/api/settings", tags=["settings"])
app.include_router(control_router, prefix="/api/control", tags=["control"])

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please contact support."}
    )

@app.get("/api/health", tags=["system"])
async def health_check():
    return {"status": "ok", "message": "API is running."}

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)

# Dashboard
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the trading dashboard."""
    html_file = PROJECT_ROOT / "static" / "index.html"
    return HTMLResponse(content=html_file.read_text())

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
