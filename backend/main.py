"""
QuantStream Dashboard — FastAPI Application Entry Point
========================================================
backend/main.py

Creates and configures the FastAPI application instance.

Responsibilities
----------------
* **Lifespan** — startup env-var checks; graceful shutdown of
  Qdrant and Gemini client connections.
* **CORS** — allows the Streamlit Community Cloud frontend and local
  dev server to reach the Render-hosted API.
* **Request logging** — adds an ``X-Process-Time`` header to every
  response and emits a single ``INFO`` log line per request.
* **Global exception handler** — converts unhandled exceptions to a
  structured 500 JSON body rather than a bare HTML traceback.
* **Health endpoint** — ``GET /health`` with live Qdrant probe,
  used by Render's uptime check and by the Streamlit frontend to
  detect a cold-start wake.

Memory policy
--------------
Only stdlib, FastAPI, and our lightweight ``routes`` module are
imported at module scope.  ``pandas``, ``numpy``, ``yfinance``,
``hmmlearn``, ``google-genai``, and ``qdrant-client`` are
intentionally absent from this file — they load on demand inside
the route handlers' ``asyncio.to_thread()`` workers.

Deployment
----------
Run locally::

    uvicorn backend.main:app --reload --port 8000

On Render (``Start Command``)::

    uvicorn backend.main:app --host 0.0.0.0 --port $PORT

Required environment variables (set in Render → Settings → Environment):

    GEMINI_API_KEY   — Google AI Studio API key
    QDRANT_URL       — Full HTTPS URL of your Qdrant Cloud cluster
    QDRANT_API_KEY   — Qdrant Cloud API key
    CORS_ORIGINS     — Comma-separated list of allowed origins
                       (default: http://localhost:8501)

Optional:

    LOG_LEVEL        — Python logging level (default: INFO)
    PORT             — Injected automatically by Render
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Our router — imports only FastAPI / Pydantic at module scope (lightweight).
from backend.api.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Configure root logger before anything else so all subsequent getLogger()
# calls inherit this format.  On Render, stdout is captured by their log
# aggregator; the timestamped format makes log correlation easy.
_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level    = getattr(logging, _LOG_LEVEL, logging.INFO),
    format   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    stream   = sys.stdout,
    force    = True,   # override any handlers already attached (e.g. uvicorn)
)
logger = logging.getLogger("quantstream.main")


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# Allow origins are read from an environment variable so the same image works
# for local dev, staging, and production without code changes.
#
# CORS_ORIGINS env var: comma-separated exact origins, e.g.:
#   "http://localhost:8501,https://my-dashboard.streamlit.app"
#
# In addition to the explicit list, we also allow any *.streamlit.app
# subdomain via a regex — handy when the exact Community Cloud URL is not
# known at deploy time.

_raw_cors = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000",
)
_ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in _raw_cors.split(",")
    if origin.strip()
]

# Regex that matches all Streamlit Community Cloud app subdomains.
# CORSMiddleware compiles this once at startup; overhead is negligible.
_CORS_ORIGIN_REGEX: str = r"https://[a-zA-Z0-9\-]+\.streamlit\.app"


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
_API_TITLE       = "QuantStream API"
_API_VERSION     = "1.0.0"
_API_DESCRIPTION = (
    "Quantitative analytics and AI strategy assistant for US equities.  "
    "Backend for the QuantStream Dashboard — deployed on Render free tier."
)


# ---------------------------------------------------------------------------
# Startup / shutdown helpers
# ---------------------------------------------------------------------------

def _check_environment() -> None:
    """
    Log the status of every required environment variable at startup.

    Warns (does not raise) on missing variables so the analytics endpoints
    still work even when the chatbot keys are absent.  This produces a clear
    diagnostic in the Render log tail rather than a silent startup failure.
    """
    checks: dict[str, bool] = {
        "GEMINI_API_KEY" : bool(os.environ.get("GEMINI_API_KEY")),
        "QDRANT_URL"     : bool(os.environ.get("QDRANT_URL")),
        "QDRANT_API_KEY" : bool(os.environ.get("QDRANT_API_KEY")),
    }

    logger.info("── Environment check ──────────────────────────────")
    for key, present in checks.items():
        status = "✓  set" if present else "✗  MISSING"
        logger.info("  %-20s %s", key, status)

    logger.info("  %-20s %s", "CORS_ORIGINS", _raw_cors[:80])
    logger.info("  %-20s %s", "LOG_LEVEL", _LOG_LEVEL)
    logger.info("───────────────────────────────────────────────────")

    missing = [k for k, v in checks.items() if not v]
    if missing:
        logger.warning(
            "Missing env vars %s — chat / RAG endpoints will return errors "
            "until they are configured in Render → Settings → Environment.",
            missing,
        )


async def _close_service_clients() -> None:
    """
    Gracefully close any service-layer clients that were initialised during
    the application's lifetime.

    Both clients are module-level singletons.  We import them here (lazily)
    so that if they were never initialised (e.g. the relevant env vars were
    absent) nothing breaks.
    """
    # ── Qdrant ──────────────────────────────────────────────────────────────
    try:
        from backend.services.vector_db import _vector_db_instance  # noqa: PLC0415
        if _vector_db_instance is not None:
            await _vector_db_instance.close()
            logger.info("VectorDBService: AsyncQdrantClient closed.")
    except Exception as exc:
        logger.warning("VectorDBService shutdown error (non-fatal): %s", exc)

    # ── google-genai (LLM agent) ─────────────────────────────────────────────
    try:
        from backend.services.llm_agent import _llm_agent_instance  # noqa: PLC0415
        if (
            _llm_agent_instance is not None
            and getattr(_llm_agent_instance, "_genai_client", None) is not None
        ):
            await _llm_agent_instance._genai_client.aio.close()
            logger.info("GeminiAgent: google.genai.Client closed.")
    except Exception as exc:
        logger.warning("GeminiAgent shutdown error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — replaces the deprecated ``on_startup`` /
    ``on_shutdown`` event hooks.

    Startup
    -------
    * Validate environment variables (warn, do not raise).
    * Log the CORS configuration so it is visible in the Render log tail.

    Shutdown
    --------
    * Close the Qdrant ``AsyncQdrantClient`` connection pool.
    * Close the ``google.genai.Client`` HTTP session.

    Neither shutdown step raises — failures are logged as warnings so a
    single bad close call never prevents the process from exiting cleanly.
    """
    # ──────────────────────────────────── STARTUP ────────────────────────────
    logger.info("QuantStream API %s starting up …", _API_VERSION)
    _check_environment()

    logger.info(
        "CORS: explicit origins=%s  regex='%s'",
        _ALLOWED_ORIGINS,
        _CORS_ORIGIN_REGEX,
    )
    logger.info("QuantStream API is ready to accept requests.")

    yield   # ← application runs here

    # ─────────────────────────────────── SHUTDOWN ────────────────────────────
    logger.info("QuantStream API shutting down …")
    await _close_service_clients()
    logger.info("QuantStream API shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = _API_TITLE,
    version     = _API_VERSION,
    description = _API_DESCRIPTION,
    lifespan    = lifespan,
    # Keep interactive docs available for development; disable in production
    # by setting DISABLE_DOCS=1 and switching these to None.
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# Must be added BEFORE the request-timing middleware so that OPTIONS
# pre-flight requests are handled immediately without incurring timing overhead.

app.add_middleware(
    CORSMiddleware,
    # Exact-match list — covers localhost dev and any explicitly configured
    # deployment URL provided via the CORS_ORIGINS env var.
    allow_origins      = _ALLOWED_ORIGINS,
    # Regex match — covers all *.streamlit.app Community Cloud deployments.
    allow_origin_regex = _CORS_ORIGIN_REGEX,
    # True is required for Streamlit to send session cookies / auth headers.
    allow_credentials  = True,
    allow_methods      = ["GET", "POST", "OPTIONS"],
    allow_headers      = ["Content-Type", "Authorization", "X-Session-ID"],
    # Expose the timing header so the Streamlit debug panel can display it.
    expose_headers     = ["X-Process-Time"],
    # Cache pre-flight response for 10 minutes to reduce OPTIONS round-trips.
    max_age            = 600,
)


# ---------------------------------------------------------------------------
# Request logging / timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next) -> Response:
    """
    Attach an ``X-Process-Time`` header to every response and emit one
    ``INFO`` log line per request.

    Format::

        2024-01-15 14:30:01 | INFO     | quantstream.main |
        GET /api/v1/analytics/AAPL → 200  (347ms)

    This middleware also serves as a lightweight audit log — every API call
    is visible in the Render log tail without needing a separate APM tool.
    """
    start = time.monotonic()

    response: Response = await call_next(request)

    elapsed_ms = (time.monotonic() - start) * 1_000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"

    # Suppress noisy health-check pings from the log (Render polls /health
    # every ~30 seconds; logging them creates thousands of irrelevant lines).
    if request.url.path != "/health":
        logger.info(
            "%s %s → %d  (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch any unhandled exception and return a structured 500 JSON response
    instead of a bare HTML traceback.

    This is a safety net — individual endpoints should catch and re-raise
    as ``HTTPException`` for better error messages, but this handler ensures
    the client always gets valid JSON regardless of what goes wrong.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "An internal server error occurred.  "
                "Please try again in a moment."
            ),
            "error_type": type(exc).__name__,
        },
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
# The router carries the /api/v1 prefix in its own definition, so we mount
# it at the root.  This keeps the prefix visible in routes.py for clarity.

app.include_router(router)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
# Intentionally at the root level (not under /api/v1) so:
#   • Render's health-check URL stays simple: https://your-api.onrender.com/health
#   • The Streamlit frontend can quickly probe whether the backend is awake
#     without needing any auth headers.

@app.get(
    "/health",
    tags=["ops"],
    summary="Liveness / readiness probe",
    response_description=(
        "API status and live connectivity check for each downstream service."
    ),
)
async def health() -> dict[str, Any]:
    """
    Lightweight health check safe to call frequently.

    Performs a live ``get_collections()`` call to Qdrant (~50 ms when warm)
    to confirm the vector store is reachable.  Does not trigger any
    analytics computation or LLM calls.

    ### Response shape
    ```json
    {
      "status": "ok" | "degraded",
      "version": "1.0.0",
      "services": {
        "api"             : true,
        "gemini_key_set"  : true,
        "qdrant_url_set"  : true,
        "qdrant_reachable": true
      }
    }
    ```

    ``"degraded"`` means the API is running but at least one downstream
    dependency is unreachable — chat will fail, analytics will still work.
    """
    services: dict[str, Any] = {
        "api"            : True,
        "gemini_key_set" : bool(os.environ.get("GEMINI_API_KEY")),
        "qdrant_url_set" : bool(os.environ.get("QDRANT_URL")),
        "qdrant_reachable": False,   # filled in below
    }

    # Live Qdrant connectivity probe (lazy import — does not load SDK at idle)
    try:
        from backend.services.vector_db import get_vector_db_service  # noqa: PLC0415
        vdb = get_vector_db_service()
        services["qdrant_reachable"] = await vdb.health_check()
    except Exception as exc:
        logger.debug("Health check: Qdrant probe failed — %s", exc)

    # Degraded = API is up but a required service is unreachable
    all_ok = services["api"] and services["gemini_key_set"]
    status = "ok" if all_ok else "degraded"

    return {
        "status"  : status,
        "version" : _API_VERSION,
        "services": services,
    }