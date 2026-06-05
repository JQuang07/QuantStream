"""
QuantStream Dashboard — API Client
====================================
frontend/api_client.py

All HTTP communication between the Streamlit frontend and the FastAPI
backend is centralised here.  No Streamlit-specific code lives in this
file — it is pure Python so it can be unit-tested independently.

Endpoints consumed
------------------
  GET  {BACKEND_URL}/api/v1/analytics/{ticker}?period={period}
  POST {BACKEND_URL}/api/v1/chat/stream          (SSE)
  GET  {BACKEND_URL}/health

SSE frame contract  (from ``backend/api/routes.py``)
------------------------------------------------------
  data: {"type": "token",  "content": "<text fragment>"}\\n\\n
  data: {"type": "error",  "content": "<description>"}\\n\\n
  data: {"type": "done"}\\n\\n

Usage
-----
In ``app.py`` the module-level ``BACKEND_URL`` is overridden once at
startup so every subsequent call in the same process sees the correct URL::

    import api_client
    api_client.BACKEND_URL = st.secrets["BACKEND_URL"]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Generator

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

#: Override this after importing to point at a different backend,
#: e.g. ``api_client.BACKEND_URL = st.secrets["BACKEND_URL"]``.
BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Analytics: allow up to 120 s read for cold-start wake + yfinance + HMM fit.
_ANALYTICS_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=10.0, pool=5.0)

# Stream: LLM responses can be verbose; 300 s ceiling prevents zombie connections.
_STREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=15.0, pool=5.0)

# Health probe: fast, fail-fast.
_HEALTH_TIMEOUT = httpx.Timeout(8.0)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class BackendError(Exception):
    """
    Raised when the FastAPI backend returns a non-200 HTTP status, times
    out, or is unreachable.

    Attributes
    ----------
    status_code : int
        HTTP status code, or a synthetic code:
        ``503`` — connection refused / DNS failure
        ``504`` — timeout
    detail : str
        Human-readable error message suitable for display in the UI.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class StreamError(Exception):
    """
    Raised when the SSE stream emits ``{"type": "error", "content": "..."}``
    — meaning the LLM call itself failed server-side after the HTTP
    connection was already established.
    """


# ---------------------------------------------------------------------------
# Analytics endpoint
# ---------------------------------------------------------------------------

def get_analytics(ticker: str, period: str = "2y") -> dict:
    """
    Fetch the full quantitative analytics payload for *ticker*.

    Parameters
    ----------
    ticker : str
        Uppercase US equity or ETF symbol (e.g. ``'AAPL'``, ``'SPY'``).
    period : str, default ``'2y'``
        yfinance period string.  Allowed: ``1mo``, ``3mo``, ``6mo``,
        ``1y``, ``2y``, ``5y``, ``max``.

    Returns
    -------
    dict
        Full analytics payload.  Key structure::

            {
              "ticker": str,
              "data_as_of": str,          # "YYYY-MM-DD"
              "last_close": float,
              "n_trading_days": int,
              "ohlcv": [{"date","open","high","low","close","volume"}],
              "technicals_latest": {
                  "log_return", "rolling_vol_21d", "rsi_14",
                  "gk_vol_21d", "macd_line", "macd_signal",
                  "macd_histogram", "sma_200d", "sma_zscore_200d"
              },
              "time_series": {
                  "log_returns", "rolling_vol", "rsi", "gk_vol",
                  "macd",        # [{"date","macd_line","signal_line","histogram"}]
                  "sma_zscore",  # [{"date","sma","zscore"}]
                  "regime_labels" # [{"date","label_id","label"}]
              },
              "regime": {
                  "available": bool,
                  "current_regime": str, "current_regime_id": int,
                  "regime_names": list[str],
                  "state_means": list[float], "state_stds": list[float],
                  "transition_matrix": list[list[float]],
                  "log_likelihood": float, "n_training_samples": int
              },
              "risk": {
                  "historical": {"var_95","cvar_95","confidence","horizon_days","n_observations"},
                  "parametric": {"var_95","cvar_95"}
              }
            }

    Raises
    ------
    BackendError
        On any HTTP, connection, or timeout failure.
    """
    url = f"{BACKEND_URL}/api/v1/analytics/{ticker}"

    try:
        with httpx.Client(timeout=_ANALYTICS_TIMEOUT) as client:
            response = client.get(url, params={"period": period})

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text[:400])
            except Exception:
                detail = response.text[:400]
            raise BackendError(response.status_code, detail)

        return response.json()

    except BackendError:
        raise
    except httpx.TimeoutException:
        raise BackendError(
            504,
            "The request timed out.  If this is the first request after a period "
            "of inactivity, the Render free-tier service may be waking from sleep "
            "(typically ~30 s).  Please wait a moment and try again.",
        )
    except httpx.ConnectError:
        raise BackendError(
            503,
            f"Cannot connect to the backend at {BACKEND_URL!r}.  "
            "Verify the Render service is running and that BACKEND_URL is set "
            "correctly in Streamlit secrets or the environment.",
        )


# ---------------------------------------------------------------------------
# Chat / SSE streaming endpoint
# ---------------------------------------------------------------------------

def stream_chat(
    message: str,
    dashboard_state: dict,
    session_id: str,
    history: list[dict],
) -> Generator[str, None, None]:
    """
    POST to the chat endpoint and yield response text tokens one by one.

    This is a **synchronous generator** compatible with ``st.write_stream``.
    The HTTP connection stays open for the duration of the stream; each
    SSE frame is parsed on arrival and the ``"content"`` of every
    ``{"type": "token"}`` frame is yielded immediately.

    Parameters
    ----------
    message : str
        The user's current message (already stripped of leading/trailing
        whitespace by the caller).
    dashboard_state : dict
        A ``DashboardStateIn``-compatible dict built from live session
        state.  Injected silently — the user never sees it.
    session_id : str
        UUID string identifying this conversation session.
    history : list[dict]
        Previous turns, oldest-first, in ``{"role": str, "content": str}``
        format.  The current message must NOT be included.
        Roles ``"assistant"`` and ``"model"`` are both accepted; they are
        normalised to ``"model"`` before sending to the backend (Gemini's
        required role name).

    Yields
    ------
    str
        Individual text tokens from the LLM response.

    Raises
    ------
    BackendError
        On HTTP-level failures (non-200 status, timeout, connection error).
    StreamError
        When the backend emits ``{"type": "error", ...}`` mid-stream,
        meaning the LLM call itself failed after the connection was open.
    """
    # ── Normalise history roles ────────────────────────────────────────────
    # Streamlit stores "assistant"; Gemini expects "model".
    # Unknown roles are silently dropped to keep the turn list well-formed.
    normalised_history: list[dict] = []
    for turn in history:
        role = turn.get("role", "user")
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            continue
        content = turn.get("content", "").strip()
        if content:
            normalised_history.append({"role": role, "content": content})

    payload = {
        "message"              : message,
        "session_id"           : session_id,
        "dashboard_state"      : dashboard_state,
        "conversation_history" : normalised_history,
    }

    url = f"{BACKEND_URL}/api/v1/chat/stream"
    headers = {
        "Content-Type"  : "application/json",
        "Accept"        : "text/event-stream",
        "Cache-Control" : "no-cache",
    }

    try:
        with httpx.Client(timeout=_STREAM_TIMEOUT) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:

                # ── HTTP-level error (e.g. 422 validation, 500 crash) ──────
                if resp.status_code != 200:
                    resp.read()  # drain the body before leaving the context
                    try:
                        detail = resp.json().get("detail", resp.text[:400])
                    except Exception:
                        detail = resp.text[:400] or f"HTTP {resp.status_code}"
                    raise BackendError(resp.status_code, detail)

                # ── Parse SSE frames line-by-line ──────────────────────────
                for raw_line in resp.iter_lines():
                    # Standard SSE format: "data: <payload>"
                    # Blank lines are frame separators — skip them.
                    if not raw_line:
                        continue

                    if not raw_line.startswith("data:"):
                        # Comment lines, retry directives, etc. — ignore.
                        continue

                    data_str = raw_line[5:].strip()  # strip "data:" + whitespace
                    if not data_str:
                        continue

                    try:
                        frame = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(
                            "stream_chat: unparseable SSE frame: %r",
                            data_str[:120],
                        )
                        continue

                    frame_type = frame.get("type")

                    if frame_type == "token":
                        content = frame.get("content", "")
                        if content:
                            yield content

                    elif frame_type == "error":
                        err_msg = frame.get("content", "Unknown LLM error.")
                        logger.error("stream_chat: server-side LLM error: %s", err_msg)
                        raise StreamError(err_msg)

                    elif frame_type == "done":
                        return

                    # Unknown frame types are silently skipped.

    except (BackendError, StreamError):
        raise
    except httpx.TimeoutException:
        raise BackendError(504, "The chat stream timed out waiting for the LLM response.")
    except httpx.ConnectError:
        raise BackendError(503, f"Cannot connect to the backend at {BACKEND_URL!r}.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """
    GET ``/health`` and return the status dict.

    **Never raises** — always returns a dict with at least a ``"status"``
    key.  Safe to call on every Streamlit rerun for connection monitoring.

    Returns
    -------
    dict
        One of:
        - ``{"status": "ok", "version": "...", "services": {...}}``
        - ``{"status": "degraded", ...}``
        - ``{"status": "offline", "detail": "..."}``
        - ``{"status": "timeout"}``
    """
    try:
        with httpx.Client(timeout=_HEALTH_TIMEOUT) as client:
            resp = client.get(f"{BACKEND_URL}/health")
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "http_code": resp.status_code}
    except httpx.ConnectError:
        return {"status": "offline", "detail": f"Cannot reach {BACKEND_URL}"}
    except httpx.TimeoutException:
        return {"status": "timeout"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}