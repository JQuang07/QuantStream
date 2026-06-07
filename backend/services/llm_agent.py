"""
QuantStream Dashboard — LLM Strategy Agent
==========================================
backend/services/llm_agent.py

Integrates Google Gemini 1.5 Flash via the ``google-genai`` SDK (v2+) to
provide a streaming AI quantitative strategy assistant grounded in the live
dashboard state.

Core Flow (per user message)
-----------------------------
1. **RAG retrieval** — Embeds the user query and pulls the ``top_k`` most
   semantically similar past exchanges from Qdrant for context.
2. **System prompt construction** — Builds a rich, metric-aware system
   prompt that injects the live ``DashboardState`` (ticker, regime, vol,
   RSI, etc.) with contextual interpretations and the RAG snippets.
3. **Async streaming** — Calls ``client.aio.models.generate_content_stream``
   and yields tokens one by one.  The FastAPI endpoint wraps these tokens
   in SSE frames.
4. **Fire-and-forget storage** — After streaming completes, schedules
   ``asyncio.Task`` calls to persist the full exchange in Qdrant for future
   RAG retrieval.  Storage failures are logged but never surface to the user.

Key Design Decisions
--------------------
* **Lazy imports**: ``google.genai`` is imported inside methods; at idle the
  module adds ~0 MB to the Render free-tier budget.
* **New SDK (google-genai ≥ 2.0)**: uses ``genai.Client`` and
  ``client.aio.models.generate_content_stream``.  The deprecated
  ``google.generativeai`` package is intentionally *not* used.
* **Dashboard state as a table in the system prompt**: the LLM receives
  human-readable metric values *and* a concise interpretation of each value,
  so it never has to independently decide "is 72 RSI overbought?".
* **Conversation role mapping**: Gemini uses ``"model"`` (not ``"assistant"``)
  for assistant turns; the helper ``_format_history_for_gemini`` normalises
  both conventions from the Streamlit frontend.

Environment Variables Required
-------------------------------
    GEMINI_API_KEY  — Google AI Studio API key
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Literal,
    Optional,
    TypedDict,
)

if TYPE_CHECKING:
    from backend.services.vector_db import VectorDBService  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME: str          = "gemini-1.5-flash"
_EMBEDDING_MODEL: str     = "text-embedding-004"
_DEFAULT_TEMPERATURE: float = 0.65
_DEFAULT_MAX_TOKENS: int   = 2048
_RAG_TOP_K: int            = 4          # past entries injected as RAG context
_MAX_HISTORY_TURNS: int    = 10         # conversation turns kept in context window

# Prefix injected into the stream when the LLM call itself errors out.
# The Streamlit frontend detects this prefix and renders an error banner.
STREAM_ERROR_PREFIX: str = "__STREAM_ERROR__:"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DashboardState:
    """
    Snapshot of the QuantStream dashboard's current metric state.

    This dataclass is populated by the Streamlit frontend and sent to the
    backend as part of every chat request body.  It mirrors the FastAPI
    request schema; FastAPI will deserialise the JSON body directly into
    this class when used with ``pydantic`` or as a plain dataclass.

    Attributes
    ----------
    ticker : str
        The active equity ticker symbol, e.g. ``'AAPL'``.
    current_regime : str
        HMM regime label, e.g. ``'Low-Volatility Bull'``.
    regime_state_id : int
        Raw integer HMM state index (0, 1, or 2).
    rolling_vol_21d : float
        21-day annualised close-to-close volatility (e.g. ``0.182`` = 18.2%).
    gk_vol_21d : float
        21-day annualised Garman-Klass volatility.
    rsi_14 : float
        14-period RSI value in range [0, 100].
    macd_histogram : float
        MACD histogram value (MACD line minus signal line).
    sma_zscore_200d : float, optional
        Z-score of price relative to 200-day SMA.  ``None`` if fewer than
        200 trading days of data are available.
    cvar_95 : float
        1-day 95% Conditional Value-at-Risk as a *negative* return fraction
        (e.g. ``-0.024`` means the expected tail loss is 2.4%).
    last_close : float
        Most recent closing price in USD.
    data_as_of : str
        ISO date string of the most recent data point, e.g. ``'2024-01-15'``.
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
    sma_zscore_200d: Optional[float] = None  # None when < 200 days available


class ConversationTurn(TypedDict):
    """
    A single turn in the multi-turn conversation history.

    Attributes
    ----------
    role : str
        ``'user'`` for user messages; ``'model'`` or ``'assistant'`` for
        assistant responses.  Both ``'model'`` and ``'assistant'`` are
        accepted — they are normalised to ``'model'`` for the Gemini API.
    content : str
        The full text of this turn.
    """

    role: Literal["user", "model", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class GeminiAgent:
    """
    Stateless AI strategy assistant backed by Google Gemini 1.5 Flash.

    The agent is designed to be instantiated once (see ``get_llm_agent()``)
    and called concurrently from multiple FastAPI requests.  It holds *no*
    mutable per-request state: all state is passed as arguments.

    Parameters
    ----------
    vector_db : VectorDBService, optional
        If provided, the agent performs RAG retrieval before each response
        and stores exchanges after streaming completes.  If ``None``,
        the agent runs in a RAG-less mode that is fully functional but
        lacks long-term memory.
    temperature : float, default 0.65
        Gemini generation temperature.  0.65 balances precision (needed
        for quantitative reasoning) with fluency (needed for prose advice).
    max_output_tokens : int, default 2048
        Hard ceiling on generated tokens per response.
    """

    def __init__(
        self,
        vector_db: Optional["VectorDBService"] = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_output_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._vector_db = vector_db
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._genai_client: Any = None  # google.genai.Client, lazily created

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _require_api_key() -> str:
        """
        Read ``GEMINI_API_KEY`` from the environment.

        Raises
        ------
        EnvironmentError
            With a deployment-friendly message if the key is absent.
        """
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it in Render → Settings → Environment Variables."
            )
        return key

    async def _get_client(self) -> Any:
        """
        Return the shared ``google.genai.Client``, creating it on first call.

        The client is intentionally *not* created in ``__init__`` to keep
        the google-genai SDK out of memory when the service is idle.
        """
        if self._genai_client is not None:
            return self._genai_client

        import google.genai as genai_sdk  # lazy

        api_key = self._require_api_key()
        self._genai_client = genai_sdk.Client(api_key=api_key)
        logger.debug("GeminiAgent: google.genai.Client initialised.")
        return self._genai_client

    # ------------------------------------------------------------------
    # Metric interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_metrics(state: DashboardState) -> dict[str, str]:
        """
        Map raw dashboard metric values to concise interpretation strings.

        These strings are embedded directly in the system prompt table so
        that the LLM never has to infer contextual meaning from raw numbers.
        Each interpretation is written in professional analyst language.

        Returns
        -------
        dict[str, str]
            Keys: ``'rsi'``, ``'vol'``, ``'zscore'``, ``'regime'``,
            ``'macd'``, ``'cvar'``.
        """
        interp: dict[str, str] = {}

        # --- RSI (Wilder 14) ---
        r = state.rsi_14
        if r >= 75:
            interp["rsi"] = f"Severely overbought ({r:.1f}) — high reversal risk; do not initiate new longs"
        elif r >= 70:
            interp["rsi"] = f"Overbought ({r:.1f}) — stretched momentum; tighten stops on existing longs"
        elif r >= 58:
            interp["rsi"] = f"Bullish momentum ({r:.1f}) — healthy trend strength, not extreme"
        elif r <= 25:
            interp["rsi"] = f"Severely oversold ({r:.1f}) — potential exhaustion of sellers; watch for capitulation"
        elif r <= 30:
            interp["rsi"] = f"Oversold ({r:.1f}) — selling pressure stretched; potential mean-reversion bounce"
        elif r <= 42:
            interp["rsi"] = f"Bearish momentum ({r:.1f}) — selling pressure sustained"
        else:
            interp["rsi"] = f"Neutral ({r:.1f}) — no directional edge from RSI alone"

        # --- Rolling vol (21-day, annualised) ---
        v = state.rolling_vol_21d * 100  # as percentage
        if v > 60:
            interp["vol"] = f"Crisis-level ({v:.1f}%) — reduce all positions; market is dislocated"
        elif v > 40:
            interp["vol"] = f"Elevated ({v:.1f}%) — stressed regime; cut position size 40–60%"
        elif v > 25:
            interp["vol"] = f"Moderate-high ({v:.1f}%) — worth monitoring for further expansion"
        elif v > 15:
            interp["vol"] = f"Normal ({v:.1f}%) — standard risk environment for US large-cap"
        else:
            interp["vol"] = f"Compressed ({v:.1f}%) — low-vol regime; watch for vol expansion catalyst"

        # --- SMA 200-day Z-Score ---
        if state.sma_zscore_200d is None:
            interp["zscore"] = "N/A — fewer than 200 trading days of data available"
        else:
            z = state.sma_zscore_200d
            if z > 3.0:
                interp["zscore"] = f"Extreme extension ({z:+.2f}σ) — historically precedes sharp mean reversion"
            elif z > 2.0:
                interp["zscore"] = f"Significantly extended ({z:+.2f}σ) — momentum can persist but risk/reward deteriorates"
            elif z > 1.0:
                interp["zscore"] = f"Above long-term mean ({z:+.2f}σ) — uptrend intact, not overextended"
            elif z < -3.0:
                interp["zscore"] = f"Deep discount ({z:+.2f}σ) — historically strong long-term value zone"
            elif z < -2.0:
                interp["zscore"] = f"Significantly depressed ({z:+.2f}σ) — potential base-forming zone"
            elif z < -1.0:
                interp["zscore"] = f"Below long-term mean ({z:+.2f}σ) — downtrend in place; needs regime shift"
            else:
                interp["zscore"] = f"Near long-term mean ({z:+.2f}σ) — neutral trend context"

        # --- HMM Market Regime ---
        rl = state.current_regime.lower()
        if "bull" in rl:
            interp["regime"] = (
                "Persistent low-volatility uptrend identified by HMM. "
                "Momentum and trend-following strategies historically have positive expectancy in this state."
            )
        elif "bear" in rl:
            interp["regime"] = (
                "Stressed, high-volatility downtrend. HMM signals elevated tail risk. "
                "Prioritise capital preservation; consider defensive rotation or hedges."
            )
        else:
            interp["regime"] = (
                "Transitional / ranging state with no clear directional bias. "
                "Mean-reversion strategies may have edge; avoid momentum sizing until regime resolves."
            )

        # --- MACD Histogram ---
        h = state.macd_histogram
        abs_h = abs(h)
        if h > 0 and abs_h > 0.005:
            interp["macd"] = f"Strengthening bullish momentum (hist={h:+.5f}) — fast EMA accelerating above slow"
        elif h > 0:
            interp["macd"] = f"Weak / nascent bullish crossover (hist={h:+.5f}) — early signal, confirm with price action"
        elif h < 0 and abs_h > 0.005:
            interp["macd"] = f"Strengthening bearish momentum (hist={h:+.5f}) — fast EMA decelerating below slow"
        else:
            interp["macd"] = f"Weak bearish / near-zero histogram ({h:+.5f}) — momentum fading or reversing"

        # --- CVaR (95%, 1-day) ---
        cvar_pct = abs(state.cvar_95) * 100
        interp["cvar"] = (
            f"1-day 95% Expected Shortfall: −{cvar_pct:.2f}%. "
            f"On the worst 5% of days, the expected loss exceeds this threshold. "
            f"Use for Kelly sizing: risk_per_trade = portfolio_value × target_risk% / {cvar_pct:.2f}%."
        )

        return interp

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        state: DashboardState,
        rag_context: str,
    ) -> str:
        """
        Construct the complete LLM system prompt for a single request.

        The prompt is rebuilt fresh on every request so the LLM always
        sees the *current* dashboard state rather than a cached snapshot.

        Structure
        ---------
        1. Persona and role definition
        2. Live dashboard state as a markdown table with interpretations
        3. RAG context from semantically similar past sessions
        4. Behavioural guidelines

        Parameters
        ----------
        state : DashboardState
            The live metrics snapshot injected by the Streamlit frontend.
        rag_context : str
            Pre-formatted string of similar past conversation snippets.
            Empty string if RAG is disabled or no relevant entries exist.

        Returns
        -------
        str
            The complete system instruction string passed to Gemini.
        """
        interp = self._interpret_metrics(state)

        zscore_raw = (
            f"{state.sma_zscore_200d:+.2f}σ"
            if state.sma_zscore_200d is not None
            else "N/A"
        )

        rag_section = (
            f"## Relevant Context from Past Analysis Sessions\n\n"
            f"{rag_context}\n"
            if rag_context.strip()
            else (
                "## Past Context\n\n"
                "*(No semantically similar prior sessions found — "
                "responding from current dashboard state only.)*\n"
            )
        )

        return f"""You are **QuantStream**, the AI quantitative strategy assistant embedded in a professional US equities analytics dashboard.

You operate as a seasoned quant analyst and portfolio strategist. You reason rigorously from the live market data shown below, quantify risk explicitly, and always acknowledge uncertainty. You never manufacture conviction you do not have.

---
## ⚡ Live Dashboard State — {state.ticker} (as of {state.data_as_of})

| Metric | Value | Contextual Interpretation |
|:---|:---:|:---|
| **Ticker** | `{state.ticker}` | Last close: **${state.last_close:,.2f}** |
| **Market Regime (HMM-3)** | State {state.regime_state_id} — *{state.current_regime}* | {interp['regime']} |
| **RSI (Wilder 14)** | {state.rsi_14:.1f} | {interp['rsi']} |
| **Rolling Vol (21d Ann.)** | {state.rolling_vol_21d:.1%} | {interp['vol']} |
| **Garman-Klass Vol (21d)** | {state.gk_vol_21d:.1%} | OHLC-efficient vol estimator; use alongside close-to-close vol |
| **MACD Histogram** | {state.macd_histogram:+.5f} | {interp['macd']} |
| **SMA Z-Score (200d)** | {zscore_raw} | {interp['zscore']} |
| **CVaR (95%, 1-day)** | {state.cvar_95:.3%} | {interp['cvar']} |

---
{rag_section}
---
## Behavioural Guidelines

1. **Ground everything in the live state above.** Reference specific metric values and their interpretations when making claims. Do not invent data.

2. **Quantify risk using CVaR for position sizing.** For any trade suggestion, frame sizing as:
   `Shares = (Portfolio × Risk%) / (CVaR × Price)`
   where CVaR = {abs(state.cvar_95):.3%} for {state.ticker}.

3. **Respect the HMM regime.** If the regime is bearish, lead with capital preservation logic before any bullish case. The HMM state carries more weight than short-term RSI signals.

4. **Explicitly resolve metric conflicts.** If RSI is oversold but the regime is bearish, state the tension directly and weigh the evidence — do not silently pick one.

5. **Use probabilistic language.** Never guarantee outcomes. Use: "historically this setup has resolved by...", "elevated probability of...", "the base case is X, but the bear case is Y if...".

6. **Be concise and structured.** Use markdown tables, short bullet points, and clear headers for multi-part answers. Avoid prose padding. Every sentence should carry information.

7. **Label time horizons explicitly.** Distinguish: *intraday*, *swing (1–10 days)*, and *positional (weeks–months)*. A signal that is bearish on a swing basis may be constructive on a positional basis.

8. **Disclaim when appropriate.** This is a tool for informed decision-making, not financial advice. Include a brief disclaimer on trade suggestions.
"""

    # ------------------------------------------------------------------
    # RAG context retrieval
    # ------------------------------------------------------------------

    async def _fetch_rag_context(
        self,
        user_message: str,
        ticker: str,
        top_k: int = _RAG_TOP_K,
    ) -> str:
        """
        Query Qdrant for past entries semantically similar to the user message.

        Gracefully returns an empty string if:
        * ``self._vector_db`` is ``None`` (RAG disabled).
        * The Qdrant call raises any exception (non-fatal degradation).

        Returns
        -------
        str
            Formatted context snippets joined by horizontal rules, or ``''``.
        """
        if self._vector_db is None:
            return ""

        try:
            entries = await self._vector_db.search_similar(
                query_text=user_message,
                top_k=top_k,
                ticker_filter=ticker,
            )
            if not entries:
                return ""

            snippets = [e.to_context_snippet() for e in entries]
            logger.debug(
                "_fetch_rag_context: ticker=%s, retrieved=%d snippets.",
                ticker, len(snippets),
            )
            return "\n\n---\n\n".join(snippets)

        except Exception as exc:
            logger.warning(
                "_fetch_rag_context: non-fatal failure, proceeding without RAG. "
                "Error: %s", exc,
            )
            return ""

    # ------------------------------------------------------------------
    # Conversation history formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_history_for_gemini(
        history: list[ConversationTurn],
        max_turns: int = _MAX_HISTORY_TURNS,
    ) -> list[dict[str, Any]]:
        """
        Convert the Streamlit conversation history to Gemini's content format.

        Gemini requires alternating ``'user'`` / ``'model'`` turns.  This
        method normalises ``'assistant'`` → ``'model'`` and trims the history
        to the most recent ``max_turns`` exchanges (each exchange = one user
        turn + one model turn = 2 items in the list).

        Parameters
        ----------
        history : list[ConversationTurn]
            Ordered oldest-first.  Should NOT include the current user message
            (that is passed separately to ``generate_content_stream``).
        max_turns : int
            Keep at most this many complete exchanges.

        Returns
        -------
        list[dict]
            List of ``{'role': str, 'parts': [{'text': str}]}`` dicts.
        """
        # Each "turn" in the UI is 1 message; max_turns exchanges = 2× items
        trimmed = history[-(max_turns * 2):]

        formatted = []
        for turn in trimmed:
            role = turn["role"]
            if role == "assistant":
                role = "model"
            if role not in ("user", "model"):
                logger.debug(
                    "_format_history_for_gemini: skipping unknown role '%s'.",
                    role,
                )
                continue
            formatted.append(
                {"role": role, "parts": [{"text": turn["content"]}]}
            )

        return formatted

    # ------------------------------------------------------------------
    # Primary streaming interface
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        user_message: str,
        dashboard_state: DashboardState,
        session_id: str,
        conversation_history: Optional[list[ConversationTurn]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the Gemini response to a user message, token by token.

        This is an **async generator**; the FastAPI endpoint iterates over
        it and wraps each yielded token in an SSE ``data:`` frame::

            async def sse_handler(request):
                async def event_stream():
                    async for token in agent.stream_response(...):
                        yield f"data: {json.dumps({'token': token})}\\n\\n"
                return StreamingResponse(event_stream(), media_type="text/event-stream")

        Yields
        ------
        str
            Individual text tokens from the Gemini response.
            On API failure, yields a single token with the
            ``STREAM_ERROR_PREFIX`` prefix so the frontend can display
            a user-friendly error banner rather than a silent hang.

        Side Effects (fire-and-forget)
        --------------------------------
        * Before streaming: schedules a Qdrant ``upsert_entry`` for the user
          message.
        * After streaming completes: schedules a Qdrant ``upsert_entry`` for
          the full concatenated assistant response.
        Both are ``asyncio.Task`` fire-and-forget; failures are logged, not
        raised.

        Parameters
        ----------
        user_message : str
            The user's current message text.
        dashboard_state : DashboardState
            Current metric snapshot from the Streamlit frontend.
        session_id : str
            UUID string identifying the conversation session (used for
            Qdrant storage and scoped RAG retrieval).
        conversation_history : list[ConversationTurn], optional
            Previous turns in this session, oldest-first.  The current
            user message must NOT be included here.
        """
        from google.genai import types as genai_types  # lazy

        client = await self._get_client()

        # ------ 1. Fetch RAG context (non-blocking, graceful fallback) ------
        rag_context = await self._fetch_rag_context(
            user_message,
            dashboard_state.ticker,
        )

        # ------ 2. Build system prompt with live state ----------------------
        system_prompt = self._build_system_prompt(dashboard_state, rag_context)

        # ------ 3. Configure generation parameters --------------------------
        safety_settings = [
            genai_types.SafetySetting(
                category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                # Financial content (short positions, derivatives) can
                # trigger "dangerous content" filters; disable for this
                # professional context.
                category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ]

        gen_config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            top_p=0.95,
            max_output_tokens=self._max_output_tokens,
            safety_settings=safety_settings,
        )

        # ------ 4. Assemble the full content list for the API call ----------
        # Gemini's multi-turn format: a flat list of {'role', 'parts'} dicts.
        # History does NOT include the current user message; that is appended here.
        gemini_history = self._format_history_for_gemini(
            conversation_history or []
        )
        contents: list[dict[str, Any]] = [
            *gemini_history,
            {"role": "user", "parts": [{"text": user_message}]},
        ]

        # ------ 5. Persist user message (fire-and-forget) -------------------
        self._schedule_storage(
            content   = user_message,
            entry_type= "chat_message",
            role      = "user",
            session_id= session_id,
            state     = dashboard_state,
        )

        # ------ 6. Stream the response token by token -----------------------
        full_response_parts: list[str] = []

        try:
            logger.debug(
                "stream_response: calling generate_content_stream "
                "(ticker=%s, history_turns=%d, rag_snippets=%d chars).",
                dashboard_state.ticker,
                len(gemini_history),
                len(rag_context),
            )

            # generate_content_stream is a coroutine that must be awaited
            # first; the awaited result is the async iterable of chunks.
            stream = await client.aio.models.generate_content_stream(
                model=_MODEL_NAME,
                contents=contents,
                config=gen_config,
            )
            async for chunk in stream:
                try:
                    token: str = chunk.text  # raises ValueError if blocked
                    if token:
                        full_response_parts.append(token)
                        yield token
                except (ValueError, AttributeError):
                    # Chunk was filtered by safety settings — skip silently.
                    # The stream continues; only this chunk is dropped.
                    continue

        except Exception as exc:
            # Unrecoverable API error — emit a structured error token so the
            # frontend can display a banner rather than an infinite spinner.
            error_token = f"{STREAM_ERROR_PREFIX} {type(exc).__name__}: {exc}"
            logger.error(
                "stream_response: fatal error during streaming. "
                "ticker=%s, error=%s",
                dashboard_state.ticker, exc, exc_info=True,
            )
            yield error_token
            return  # Stops the generator; finally block still runs.

        finally:
            # ------ 7. Persist assistant response (fire-and-forget) ---------
            # This runs whether streaming completed normally, errored, or was
            # cancelled by the client (e.g. user closes the browser tab).
            if full_response_parts:
                complete_response = "".join(full_response_parts)
                self._schedule_storage(
                    content    = complete_response,
                    entry_type = "chat_message",
                    role       = "assistant",
                    session_id = session_id,
                    state      = dashboard_state,
                    extra_meta = {"rag_context_chars": len(rag_context)},
                )

    # ------------------------------------------------------------------
    # Non-streaming: strategy summary generation
    # ------------------------------------------------------------------

    async def generate_strategy_summary(
        self,
        ticker: str,
        session_id: str,
        dashboard_state: DashboardState,
    ) -> str:
        """
        Generate and store a concise strategy summary for a completed session.

        This method is called by the frontend's "End Session / Summarise"
        action.  It retrieves the full session history from Qdrant,
        synthesises it into a 150–200 word summary, and then stores that
        summary back as a ``'strategy_summary'`` entry for future RAG use.

        Parameters
        ----------
        ticker : str
            The equity ticker analysed during this session.
        session_id : str
            Session UUID to look up in Qdrant.
        dashboard_state : DashboardState
            The final metric snapshot for context in the summary.

        Returns
        -------
        str
            The generated summary text, or ``''`` if generation fails or
            there is no session history to summarise.
        """
        if self._vector_db is None:
            logger.warning(
                "generate_strategy_summary: vector_db is None, skipping."
            )
            return ""

        try:
            history_entries = await self._vector_db.get_session_history(
                session_id=session_id
            )
        except Exception as exc:
            logger.error("generate_strategy_summary: history fetch failed: %s", exc)
            return ""

        if not history_entries:
            logger.info(
                "generate_strategy_summary: no history for session %s, skipping.",
                session_id,
            )
            return ""

        # Format the session as a readable dialogue
        dialogue = "\n\n".join(e.to_context_snippet() for e in history_entries)
        interp = self._interpret_metrics(dashboard_state)

        summary_prompt = (
            f"You are a senior quantitative analyst summarising a client advisory session.\n\n"
            f"Ticker analysed: {ticker}\n"
            f"Final dashboard state: "
            f"Regime={dashboard_state.current_regime}, "
            f"RSI={dashboard_state.rsi_14:.1f}, "
            f"Vol={dashboard_state.rolling_vol_21d:.1%}, "
            f"SMA-Z={dashboard_state.sma_zscore_200d:+.2f}σ"
            if dashboard_state.sma_zscore_200d is not None
            else f"SMA-Z=N/A"
        )
        summary_prompt += (
            f", CVaR(95%)={dashboard_state.cvar_95:.3%}\n\n"
            f"Conversation:\n{dialogue}\n\n"
            f"Write a professional strategy summary (150–200 words) covering:\n"
            f"1. Key analytical findings (regime interpretation, momentum, risk metrics)\n"
            f"2. Any specific strategy ideas or trade setups discussed\n"
            f"3. Primary risk factors identified\n"
            f"4. Suggested next steps or monitoring triggers\n\n"
            f"Output only the summary text. No preamble, no headings."
        )

        from google.genai import types as genai_types  # lazy

        client = await self._get_client()

        try:
            response = await client.aio.models.generate_content(
                model=_MODEL_NAME,
                contents=summary_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,     # Low temp for factual summaries
                    max_output_tokens=400,
                ),
            )
            summary_text: str = response.text.strip()

            if summary_text and self._vector_db:
                await self._vector_db.upsert_entry(
                    content    = summary_text,
                    entry_type = "strategy_summary",
                    role       = "assistant",
                    session_id = session_id,
                    ticker     = ticker,
                    regime     = dashboard_state.current_regime,
                    metadata   = {
                        "rsi"        : dashboard_state.rsi_14,
                        "rolling_vol": dashboard_state.rolling_vol_21d,
                        "cvar_95"    : dashboard_state.cvar_95,
                        "sma_zscore" : dashboard_state.sma_zscore_200d,
                    },
                )
                logger.info(
                    "generate_strategy_summary: summary stored for session %s.",
                    session_id,
                )

            return summary_text

        except Exception as exc:
            logger.error(
                "generate_strategy_summary: Gemini call failed: %s", exc,
                exc_info=True,
            )
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_storage(
        self,
        content: str,
        entry_type: Literal["chat_message", "strategy_summary"],
        role: Literal["user", "assistant"],
        session_id: str,
        state: DashboardState,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Fire-and-forget: schedule a Qdrant upsert as an ``asyncio.Task``.

        Failures are caught and logged inside the task; they never surface
        to the caller.  This keeps storage I/O off the critical response path.
        """
        if self._vector_db is None:
            return

        metadata: dict[str, Any] = {
            "rsi"        : state.rsi_14,
            "rolling_vol": state.rolling_vol_21d,
            "sma_zscore" : state.sma_zscore_200d,
            "cvar_95"    : state.cvar_95,
            **(extra_meta or {}),
        }

        async def _store() -> None:
            try:
                await self._vector_db.upsert_entry(  # type: ignore[union-attr]
                    content    = content,
                    entry_type = entry_type,
                    role       = role,
                    session_id = session_id,
                    ticker     = state.ticker,
                    regime     = state.current_regime,
                    metadata   = metadata,
                )
            except Exception as exc:
                logger.warning(
                    "_schedule_storage task failed (non-fatal): "
                    "role=%s, ticker=%s, error=%s",
                    role, state.ticker, exc,
                )

        try:
            asyncio.get_running_loop().create_task(_store())
        except RuntimeError:
            # No running event loop (e.g. in synchronous unit tests) — skip.
            logger.debug(
                "_schedule_storage: no running loop; storage skipped."
            )


# ---------------------------------------------------------------------------
# Singleton factory for FastAPI dependency injection
# ---------------------------------------------------------------------------

_llm_agent_instance: Optional[GeminiAgent] = None


def get_llm_agent(
    vector_db: Optional["VectorDBService"] = None,
) -> GeminiAgent:
    """
    Return the process-scoped ``GeminiAgent`` singleton.

    On the first call, the agent is created with the provided
    ``vector_db``.  Subsequent calls return the same instance regardless
    of arguments, so the ``vector_db`` argument only has effect once.

    Intended for use with FastAPI's ``Depends`` pattern::

        from backend.services.llm_agent import get_llm_agent
        from backend.services.vector_db import get_vector_db_service

        @app.post("/chat/stream")
        async def stream_chat(
            request: ChatRequest,
            agent: GeminiAgent = Depends(
                lambda: get_llm_agent(get_vector_db_service())
            ),
        ):
            ...
    """
    global _llm_agent_instance
    if _llm_agent_instance is None:
        _llm_agent_instance = GeminiAgent(vector_db=vector_db)
        logger.debug(
            "GeminiAgent singleton created (rag_enabled=%s).",
            vector_db is not None,
        )
    return _llm_agent_instance
