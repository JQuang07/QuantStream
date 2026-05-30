"""
QuantStream Dashboard — API Routes
====================================
backend/api/routes.py

Defines the two primary REST endpoints served by the FastAPI backend:

  GET  /api/v1/analytics/{ticker}  — Download OHLCV data, run the full
                                     QuantEngine + HMM regime pipeline,
                                     and return a serialised JSON payload
                                     ready for Plotly charts.

  POST /api/v1/chat/stream         — Accept a user message and the current
                                     dashboard state, then stream the Gemini
                                     1.5 Flash response token-by-token as
                                     Server-Sent Events.

Memory / concurrency design
----------------------------
*Nothing heavy is imported at module scope.*  ``pandas``, ``numpy``,
``yfinance``, ``hmmlearn``, ``QuantEngine``, and ``RegimeClassifier`` are
all imported inside helper functions that run inside ``asyncio.to_thread()``.
This means:

  • At startup, the process idles at ~80 MB (FastAPI + Pydantic only).
  • Libraries load on the first request and stay resident in the module
    cache — subsequent requests pay only the dict-lookup cost.
  • CPU-bound work (HMM fit, rolling metrics) never blocks the async event
    loop; it executes in the default ``ThreadPoolExecutor``.

SSE event format
-----------------
Every frame in the ``/chat/stream`` response is a valid SSE event::

    data: {"type": "token",  "content": "<text fragment>"}\\n\\n
    data: {"type": "error",  "content": "<description>"}\\n\\n
    data: {"type": "done"}\\n\\n

The Streamlit frontend iterates the raw byte stream, splits on ``\\n\\n``,
strips the leading ``data: ``, and JSON-parses each frame.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants  (no heavy libs here — stdlib / typing only)
# ---------------------------------------------------------------------------

# Allowed ticker symbols: 1–12 chars, uppercase alphanumeric + . - ^
# Covers NYSE / NASDAQ equities, BRK.B, GOOGL, ^GSPC index symbols, etc.
_TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,12}$")

_VALID_PERIODS: frozenset[str] = frozenset(
    {"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
)

# 2y default gives ≥ 504 trading days:
#   • enough for the 200-day SMA z-score (needs 200)
#   • enough for HMM training (needs ≥ 100)
_DEFAULT_PERIOD = "2y"

# Maximum data-points returned per time-series to cap JSON response size.
# 504 ≈ 2 years of daily bars; Plotly renders this without issue.
_MAX_CHART_PTS = 504

router = APIRouter(prefix="/api/v1", tags=["quantstream"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# All heavy-library types (pd.Series, np.ndarray) never appear here.
# ---------------------------------------------------------------------------

class DashboardStateIn(BaseModel):
    """
    Snapshot of the Streamlit dashboard's live metric state.

    The frontend populates this from session-state variables and silently
    attaches it to every chat request so the LLM is always grounded in the
    current chart without the user having to describe it.
    """

    ticker: str
    current_regime: str
    regime_state_id: int
    rolling_vol_21d: float
    gk_vol_21d: float
    rsi_14: float
    macd_histogram: float
    cvar_95: float
    last_close: float
    data_as_of: str
    sma_zscore_200d: Optional[float] = None


class ConversationTurnIn(BaseModel):
    """Single message turn from the Streamlit chat history."""

    # Accept both Gemini-native "model" and OpenAI-style "assistant"
    role: str = Field(..., pattern=r"^(user|model|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Request body for ``POST /api/v1/chat/stream``."""

    message: str = Field(..., min_length=1, max_length=4_000,
                         description="The user's current message.")
    session_id: str = Field(..., min_length=1, max_length=128,
                            description="UUID identifying this conversation session.")
    dashboard_state: DashboardStateIn
    conversation_history: list[ConversationTurnIn] = Field(
        default_factory=list,
        description=(
            "Prior turns, oldest-first.  Must NOT include the current message. "
            "Trimmed to the last 20 items server-side before sending to the LLM."
        ),
    )

    @field_validator("message")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be blank or whitespace-only.")
        return stripped


# ---------------------------------------------------------------------------
# Pure serialisation helpers
# These functions import nothing at definition time.
# They ARE called from inside _run_analytics() which runs in a thread, so
# imports inside them are safe and cheap (module-cache hits after first call).
# ---------------------------------------------------------------------------

def _f(val: Any, prec: int = 8) -> Optional[float]:
    """
    Coerce *val* to a JSON-safe Python ``float``.

    Returns ``None`` for ``NaN``, ``±Inf``, ``None``, and anything that
    cannot be cast — so downstream JSON serialisation never raises.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, prec)
    except (TypeError, ValueError):
        return None


def _i(val: Any) -> Optional[int]:
    """Coerce *val* to Python ``int``, or ``None`` on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _ts(idx: Any) -> str:
    """
    Convert a pandas ``Timestamp`` (tz-aware or naive) to ``'YYYY-MM-DD'``.

    Falls back to ``str(idx)[:10]`` for non-Timestamp objects so this
    helper never raises regardless of the index type yfinance returns.
    """
    try:
        return idx.strftime("%Y-%m-%d")
    except AttributeError:
        return str(idx)[:10]


def _series_to_records(
    series: Any,  # pd.Series with DatetimeIndex
    max_pts: int = _MAX_CHART_PTS,
    prec: int = 6,
) -> list[dict[str, Any]]:
    """
    Convert a pandas Series with DatetimeIndex to a Plotly-ready list.

    Output schema: ``[{"date": "YYYY-MM-DD", "value": float | null}, ...]``

    Only the most recent ``max_pts`` rows are included to cap payload size.
    """
    items = list(series.items())
    start = max(0, len(items) - max_pts)
    result: list[dict[str, Any]] = []
    for idx, val in items[start:]:
        result.append({"date": _ts(idx), "value": _f(val, prec)})
    return result


def _df_to_records(
    df: Any,  # pd.DataFrame with DatetimeIndex
    cols: list[str],
    max_pts: int = _MAX_CHART_PTS,
    prec: int = 6,
) -> list[dict[str, Any]]:
    """
    Convert selected DataFrame columns to a Plotly-ready record list.

    Output schema: ``[{"date": "YYYY-MM-DD", col1: float|null, ...}, ...]``
    """
    n = len(df)
    start = max(0, n - max_pts)
    result: list[dict[str, Any]] = []
    for i in range(start, n):
        row = df.iloc[i]
        record: dict[str, Any] = {"date": _ts(df.index[i])}
        for col in cols:
            record[col] = _f(row[col], prec) if col in df.columns else None
        result.append(record)
    return result


def _last_valid(series: Optional[Any]) -> Optional[float]:
    """
    Return the most recent non-NaN value from a pandas Series as a Python float.
    Returns ``None`` if the series is ``None`` or entirely NaN.
    """
    if series is None:
        return None
    dropped = series.dropna()
    return _f(dropped.iloc[-1]) if len(dropped) > 0 else None


# ---------------------------------------------------------------------------
# Synchronous analytics pipeline  (dispatched to thread pool)
# ---------------------------------------------------------------------------

def _run_analytics(ticker: str, period: str) -> dict[str, Any]:
    """
    Download OHLCV data, run the full ``QuantEngine`` + ``RegimeClassifier``
    pipeline, and serialise all results to a JSON-safe Python dict.

    **This function is intentionally synchronous** and is called exclusively
    via ``asyncio.to_thread()``.  All heavy imports live here to keep the
    FastAPI worker process RAM footprint near-zero at idle.

    Parameters
    ----------
    ticker : str
        Pre-validated, uppercase ticker symbol.
    period : str
        yfinance period string, e.g. ``'2y'``.

    Returns
    -------
    dict
        Fully JSON-serialisable analytics payload consumed by the Streamlit
        frontend to populate all four Plotly chart panels and the status bar.

    Raises
    ------
    ValueError
        Invalid ticker (empty yfinance response) or insufficient data rows.
    RuntimeError
        Unexpected failure in any computation step.
    """
    # ── Lazy imports ────────────────────────────────────────────────────────
    import pandas as pd                              # noqa: PLC0415
    import yfinance as yf                            # noqa: PLC0415
    from backend.analytics.engine import QuantEngine          # noqa: PLC0415
    from backend.analytics.regimes import RegimeClassifier    # noqa: PLC0415

    # ── 1. Download OHLCV ───────────────────────────────────────────────────
    logger.info("Fetching %s  period=%s …", ticker, period)

    # yfinance ≥1.0 returns MultiIndex columns for download(); QuantEngine
    # _validate() normalises them automatically via droplevel().
    df: pd.DataFrame = yf.download(
        tickers      = ticker,
        period       = period,
        interval     = "1d",
        auto_adjust  = True,   # prices adjusted for splits & dividends
        progress     = False,
        threads      = False,  # single-thread avoids spawning extra workers
    )

    if df is None or df.empty:
        raise ValueError(
            f"No market data found for '{ticker}'. "
            "Verify it is a valid US equity symbol (e.g. 'AAPL', 'MSFT', 'SPY')."
        )

    if len(df) < 30:
        raise ValueError(
            f"'{ticker}' returned only {len(df)} trading days for period "
            f"'{period}'.  At least 30 rows are required for base metrics."
        )

    # ── Flatten MultiIndex columns ──────────────────────────────────────────
    # yfinance ≥ 0.2 returns MultiIndex columns for single-ticker downloads,
    # e.g. [('Close', 'AAPL'), ('High', 'AAPL'), ...].
    # Drop the ticker level so downstream code uses plain string keys.
    # QuantEngine._validate() does the same internally, but we need flat
    # column names here too for the OHLCV serialisation loop and for
    # df["Close"].iloc[-1] below.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(level=1, axis=1)
        # After flattening, keep only canonical OHLCV columns to discard
        # any extra columns yfinance may inject (e.g. 'Adj Close', 'Dividends').
        canonical = [c for c in ("Open", "High", "Low", "Close", "Volume")
                     if c in df.columns]
        df = df[canonical]

    # Normalise timezone: drop tz info (yfinance may return tz-aware index
    # depending on the OS locale; our serialiser handles both, but stripping
    # here prevents pd.Series.update() alignment issues later).
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    data_as_of   = _ts(df.index[-1])
    last_close_f = _f(df["Close"].iloc[-1], 4) if "Close" in df.columns else None
    logger.info("%s: %d rows  %s → %s", ticker, len(df), _ts(df.index[0]), data_as_of)

    # ── 2. Quantitative engine ──────────────────────────────────────────────
    engine = QuantEngine(df)
    metrics = engine.compute_all(
        vol_window      = 21,
        macd_fast       = 12,
        macd_slow       = 26,
        macd_signal     = 9,
        rsi_period      = 14,
        sma_window      = 200,
        gk_window       = 21,
        cvar_confidence = 0.95,
    )

    log_returns: pd.Series           = metrics["log_returns"]
    rolling_vol: pd.Series           = metrics["rolling_vol"]
    rsi_series:  pd.Series           = metrics["rsi"]
    gk_vol:      pd.Series           = metrics["gk_vol"]
    macd_df:     Optional[pd.DataFrame] = metrics.get("macd")
    sma_df:      Optional[pd.DataFrame] = metrics.get("sma_zscore")
    cvar_dict:   Optional[dict]      = metrics.get("cvar")

    # ── 3. Regime classification ────────────────────────────────────────────
    regime_payload: dict[str, Any]     = {"available": False}
    regime_label_series: Optional[pd.Series] = None

    clean_returns = log_returns.dropna()
    if len(clean_returns) >= 100:
        try:
            clf    = RegimeClassifier(n_states=3, n_iter=200, random_state=42)
            result = clf.fit_predict(clean_returns)

            regime_label_series = result.regime_labels
            regime_payload = {
                "available"         : True,
                "current_regime"    : result.current_regime_name,
                "current_regime_id" : result.current_regime,
                "regime_names"      : result.regime_names,
                "state_means"       : [_f(m) for m in result.state_means],
                "state_stds"        : [_f(s) for s in result.state_stds],
                "transition_matrix" : [[_f(p) for p in row]
                                       for row in result.transition_matrix],
                "log_likelihood"    : _f(result.log_likelihood, 4),
                "n_training_samples": result.n_training_samples,
                "feature_columns"   : result.feature_columns,
            }
            logger.info(
                "%s: HMM fitted — current regime '%s'.",
                ticker, result.current_regime_name,
            )
        except Exception as exc:
            logger.warning("%s: HMM skipped — %s", ticker, exc)
            regime_payload = {
                "available": False,
                "reason"   : f"Regime classification failed: {exc}",
            }
    else:
        regime_payload["reason"] = (
            f"Only {len(clean_returns)} return observations; "
            f"HMM needs ≥ 100."
        )

    # ── 4. Latest-value snapshot (for the dashboard status bar) ────────────
    technicals_latest: dict[str, Any] = {
        "log_return"      : _last_valid(log_returns),
        "rolling_vol_21d" : _last_valid(rolling_vol),
        "rsi_14"          : _last_valid(rsi_series),
        "gk_vol_21d"      : _last_valid(gk_vol),
        "macd_line"       : _last_valid(macd_df["macd_line"]    if macd_df is not None else None),
        "macd_signal"     : _last_valid(macd_df["signal_line"]  if macd_df is not None else None),
        "macd_histogram"  : _last_valid(macd_df["histogram"]    if macd_df is not None else None),
        "sma_200d"        : _last_valid(sma_df["sma"]           if sma_df is not None else None),
        "sma_zscore_200d" : _last_valid(sma_df["zscore"]        if sma_df is not None else None),
    }

    # ── 5. CVaR / risk metrics ──────────────────────────────────────────────
    def _cvar_block(d: Optional[dict]) -> dict[str, Any]:
        if not d:
            return {}
        return {
            "var_95"        : _f(d.get("var"),          4),
            "cvar_95"       : _f(d.get("cvar"),         4),
            "confidence"    : _f(d.get("confidence"),   4),
            "horizon_days"  : d.get("horizon_days", 1),
            "n_observations": d.get("n_observations"),
        }

    risk_payload: dict[str, Any] = {"historical": _cvar_block(cvar_dict)}

    try:
        cvar_param = engine.cvar(confidence=0.95, method="parametric")
        risk_payload["parametric"] = _cvar_block(cvar_param)
    except Exception as exc:
        logger.debug("%s: parametric CVaR skipped — %s", ticker, exc)
        risk_payload["parametric"] = {}

    # ── 6. OHLCV records for the candlestick chart ──────────────────────────
    n       = len(df)
    start_i = max(0, n - _MAX_CHART_PTS)
    ohlcv: list[dict[str, Any]] = []
    for i in range(start_i, n):
        row = df.iloc[i]
        ohlcv.append({
            "date"  : _ts(df.index[i]),
            "open"  : _f(row.get("Open"),   4),
            "high"  : _f(row.get("High"),   4),
            "low"   : _f(row.get("Low"),    4),
            "close" : _f(row.get("Close"),  4),
            "volume": _i(row.get("Volume", 0)),
        })

    # ── 7. Time-series payloads for all Plotly indicator subplots ───────────
    time_series: dict[str, Any] = {
        "log_returns" : _series_to_records(log_returns, _MAX_CHART_PTS),
        "rolling_vol" : _series_to_records(rolling_vol, _MAX_CHART_PTS),
        "rsi"         : _series_to_records(rsi_series,  _MAX_CHART_PTS),
        "gk_vol"      : _series_to_records(gk_vol,      _MAX_CHART_PTS),
    }

    if macd_df is not None:
        time_series["macd"] = _df_to_records(
            macd_df, ["macd_line", "signal_line", "histogram"], _MAX_CHART_PTS
        )

    if sma_df is not None:
        time_series["sma_zscore"] = _df_to_records(
            sma_df, ["sma", "zscore"], _MAX_CHART_PTS
        )

    # Regime label overlay: re-index onto the full price index so every
    # OHLCV bar has a corresponding regime colour.  Pre-HMM warmup bars
    # receive label_id = -1 ("Insufficient Data").
    if regime_label_series is not None:
        regime_names_map: dict[int, str] = {
            i: name
            for i, name in enumerate(
                regime_payload.get("regime_names", [])
            )
        }
        aligned = pd.Series(-1, index=df.index, dtype=int)
        aligned.update(regime_label_series)

        regime_ts: list[dict[str, Any]] = []
        n_al  = len(aligned)
        st_al = max(0, n_al - _MAX_CHART_PTS)
        for i in range(st_al, n_al):
            lid = int(aligned.iloc[i])
            regime_ts.append({
                "date"    : _ts(aligned.index[i]),
                "label_id": lid,
                "label"   : (
                    regime_names_map.get(lid, "Unknown")
                    if lid >= 0 else "Insufficient Data"
                ),
            })
        time_series["regime_labels"] = regime_ts

    # ── 8. Assemble final response ──────────────────────────────────────────
    return {
        "ticker"           : ticker,
        "data_as_of"       : data_as_of,
        "last_close"       : last_close_f,
        "n_trading_days"   : len(df),
        "period_requested" : period,
        "ohlcv"            : ohlcv,
        "technicals_latest": technicals_latest,
        "time_series"      : time_series,
        "regime"           : regime_payload,
        "risk"             : risk_payload,
    }


# ---------------------------------------------------------------------------
# Endpoint 1 — Quantitative Analytics
# ---------------------------------------------------------------------------

@router.get(
    "/analytics/{ticker}",
    summary="Full quantitative analytics for a US equity ticker",
    response_description=(
        "OHLCV bars, latest indicator snapshot, time-series for all "
        "Plotly charts, 3-state HMM regime, and CVaR risk metrics."
    ),
)
async def get_analytics(
    ticker: str,
    period: str = Query(
        default=_DEFAULT_PERIOD,
        description=(
            "yfinance historical window.  "
            f"Allowed: {sorted(_VALID_PERIODS)}.  "
            "Use '2y' (default) to ensure 200-day SMA and HMM convergence."
        ),
    ),
) -> dict[str, Any]:
    """
    Fetch and compute the complete QuantStream analytics suite for *ticker*.

    ### Processing pipeline (runs in ``asyncio.to_thread()``)

    1. Download OHLCV via ``yfinance.download()`` (adjusted prices)
    2. Instantiate ``QuantEngine`` → ``compute_all()`` to produce:
       log returns, 21-day rolling vol, MACD, RSI, 200-day SMA z-score,
       Garman-Klass vol, and 95% CVaR (historical + parametric)
    3. Fit a 3-state Gaussian HMM via ``RegimeClassifier.fit_predict()``
    4. Serialise all pandas / numpy outputs to JSON-safe Python dicts

    ### Memory note
    ``pandas``, ``numpy``, ``yfinance``, and ``hmmlearn`` are imported lazily
    inside the thread-pool worker.  The FastAPI process stays near 80 MB RAM
    at idle and peaks at ~300 MB during the first request.
    """
    # ── Input validation (fast, synchronous) ────────────────────────────────
    ticker_upper = ticker.upper().strip()

    if not _TICKER_RE.match(ticker_upper):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid ticker '{ticker}'.  Must be 1–12 uppercase "
                "alphanumeric characters (dots, hyphens, and ^ allowed)."
            ),
        )

    if period not in _VALID_PERIODS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid period '{period}'.  Allowed: {sorted(_VALID_PERIODS)}",
        )

    # ── Dispatch to thread pool ──────────────────────────────────────────────
    try:
        payload: dict[str, Any] = await asyncio.to_thread(
            _run_analytics, ticker_upper, period
        )
    except ValueError as exc:
        # User-facing error: bad ticker, insufficient data, etc.
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Analytics pipeline failed for '%s': %s",
            ticker_upper, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Analytics computation failed for '{ticker_upper}'.  "
                "This is likely a transient yfinance issue — please retry."
            ),
        )

    return payload


# ---------------------------------------------------------------------------
# Endpoint 2 — SSE Chat Stream
# ---------------------------------------------------------------------------

@router.post(
    "/chat/stream",
    summary="Stream an AI strategy assistant response via Server-Sent Events",
    response_description=(
        "A ``text/event-stream`` response.  "
        "Each frame: ``data: {type, content}\\n\\n``.  "
        "Types: ``token`` | ``error`` | ``done``."
    ),
)
async def stream_chat(body: ChatRequest) -> StreamingResponse:
    """
    Accept a user message with the live dashboard state, build a
    quantitative-analyst system prompt grounded in those metrics, and
    stream the Gemini 1.5 Flash reply token-by-token as SSE.

    ### Silent state injection
    The ``dashboard_state`` payload is attached by the Streamlit frontend
    on every request without the user doing anything.  The backend uses it
    to write a system prompt like:
    > *"RSI is 71 (overbought). Rolling vol is 28%. Current HMM regime is
    > Low-Volatility Bull — momentum strategies have edge in this state..."*

    This grounds the LLM in the live chart data rather than letting it
    confabulate generic advice.

    ### RAG context
    Before calling Gemini, the backend embeds the user's message and queries
    Qdrant for the most semantically similar past exchanges.  Up to 4 snippets
    are injected into the system prompt as "Past Session Context".

    ### SSE frame format
    ```
    data: {"type": "token",  "content": "The RSI is"}\\n\\n
    data: {"type": "token",  "content": " signalling ..."}\\n\\n
    data: {"type": "done"}\\n\\n
    ```
    """
    # ── Lazy service-layer imports ───────────────────────────────────────────
    # These modules are cheap once cached but import google-genai and
    # qdrant-client the first time — keep them out of module scope.
    from backend.services.llm_agent import (  # noqa: PLC0415
        DashboardState,
        get_llm_agent,
        STREAM_ERROR_PREFIX,
    )
    from backend.services.vector_db import get_vector_db_service  # noqa: PLC0415

    # ── Convert Pydantic schema → service-layer dataclass ───────────────────
    ds = body.dashboard_state
    dashboard_state = DashboardState(
        ticker          = ds.ticker,
        current_regime  = ds.current_regime,
        regime_state_id = ds.regime_state_id,
        rolling_vol_21d = ds.rolling_vol_21d,
        gk_vol_21d      = ds.gk_vol_21d,
        rsi_14          = ds.rsi_14,
        macd_histogram  = ds.macd_histogram,
        cvar_95         = ds.cvar_95,
        last_close      = ds.last_close,
        data_as_of      = ds.data_as_of,
        sma_zscore_200d = ds.sma_zscore_200d,
    )

    conversation_history = [
        {"role": t.role, "content": t.content}
        for t in body.conversation_history
    ]

    # ── Obtain the singleton GeminiAgent (RAG enabled if Qdrant is set up) ──
    agent = get_llm_agent(vector_db=get_vector_db_service())

    # ── SSE generator ────────────────────────────────────────────────────────
    async def event_stream():
        """
        Wraps ``GeminiAgent.stream_response`` and emits one SSE frame per
        yielded token.  Handles all three terminal conditions:

        * **Clean end** — emits ``{"type": "done"}`` and returns.
        * **LLM error** — the agent yields a token prefixed with
          ``STREAM_ERROR_PREFIX``; translated to ``{"type": "error", ...}``.
        * **Client disconnect** — ``asyncio.CancelledError`` is caught,
          logged, and swallowed; no further yields needed.
        """
        try:
            async for token in agent.stream_response(
                user_message         = body.message,
                dashboard_state      = dashboard_state,
                session_id           = body.session_id,
                conversation_history = conversation_history,
            ):
                if token.startswith(STREAM_ERROR_PREFIX):
                    # Strip the sentinel prefix the agent injects on failure
                    err_content = token[len(STREAM_ERROR_PREFIX):].strip()
                    frame = json.dumps({"type": "error", "content": err_content})
                    logger.warning(
                        "LLM stream error [session=%s]: %s",
                        body.session_id, err_content,
                    )
                else:
                    frame = json.dumps({"type": "token", "content": token})

                yield f"data: {frame}\n\n"

            # Signal clean stream end to the frontend
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except asyncio.CancelledError:
            # The Streamlit client closed the connection (tab closed, re-run, etc.)
            logger.info(
                "SSE stream cancelled by client [session=%s].",
                body.session_id,
            )
            # No yield — connection is already gone.

        except Exception as exc:
            logger.error(
                "Unexpected SSE generator error [session=%s]: %s",
                body.session_id, exc, exc_info=True,
            )
            frame = json.dumps({
                "type"   : "error",
                "content": f"Unexpected server error: {type(exc).__name__}",
            })
            yield f"data: {frame}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent any proxy (nginx, Render's load balancer) from
            # buffering the stream — essential for real-time SSE delivery.
            "Cache-Control"     : "no-cache",
            "X-Accel-Buffering" : "no",
            "Connection"        : "keep-alive",
        },
    )