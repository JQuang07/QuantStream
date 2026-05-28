"""
QuantStream Dashboard — Vector Database Service
================================================
backend/services/vector_db.py

Manages persistent semantic memory for the AI strategy assistant using
Qdrant Cloud (free tier) as the vector store.

Responsibilities
----------------
1. Store every chat message and AI-generated strategy summary as an
   embedding vector alongside its structured metadata payload.
2. Retrieve the ``top_k`` most semantically similar past entries for RAG
   context injection into the LLM system prompt.
3. Provide ordered session history retrieval for conversation continuity.

Architecture
------------
* ``AsyncQdrantClient`` — non-blocking Qdrant operations; shared instance
  per process via a module-level singleton factory.
* ``google.genai.Client.aio.models.embed_content`` — fully async embedding
  calls via Google's ``text-embedding-004`` model (768-dim, free tier).
* Both clients are lazily initialised: the SDK is not imported and no network
  connection is made until the first actual operation, keeping idle memory
  near zero.
* A single ``asyncio.Lock`` guards the collection-creation path to avoid
  race conditions on cold-start with multiple concurrent requests.

Single Qdrant Collection Schema
--------------------------------
Collection : ``quantstream_memory``
Vector size: 768 (cosine distance)
Payload keys:
    entry_type  — ``'chat_message'`` | ``'strategy_summary'``
    role        — ``'user'`` | ``'assistant'``
    content     — raw text
    ticker      — e.g. ``'AAPL'``
    session_id  — UUID string grouping one conversation
    timestamp   — ISO-8601 UTC string
    regime      — HMM regime label at time of entry
    + any extra numeric metrics passed via ``metadata``

Environment Variables Required
-------------------------------
    QDRANT_URL      — Full HTTPS URL of your Qdrant Cloud cluster endpoint
                      (e.g. ``https://abc123.us-east4-0.gcp.cloud.qdrant.io``)
    QDRANT_API_KEY  — API key from the Qdrant Cloud dashboard
    GEMINI_API_KEY  — Google AI Studio API key (reused here for embeddings)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    # Annotation-only imports — never executed at runtime
    import google.genai as _genai_t            # noqa: F401
    from qdrant_client import AsyncQdrantClient  # noqa: F401
    from qdrant_client.models import Record, ScoredPoint  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_COLLECTION_NAME: str = "quantstream_memory"
_VECTOR_SIZE: int = 768                          # text-embedding-004 output dim
_EMBEDDING_MODEL: str = "text-embedding-004"
_MAX_EMBED_RETRIES: int = 3
_EMBED_RETRY_BASE_DELAY: float = 1.0            # seconds; doubled each attempt

EntryType = Literal["chat_message", "strategy_summary"]
MessageRole = Literal["user", "assistant"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """
    A single entry retrieved from the Qdrant vector store.

    Attributes
    ----------
    point_id : str
        The Qdrant point UUID string.
    entry_type : EntryType
        ``'chat_message'`` or ``'strategy_summary'``.
    role : MessageRole
        ``'user'`` or ``'assistant'``.
    content : str
        Raw text of the stored entry.
    ticker : str
        The equity ticker symbol associated with this entry.
    timestamp : str
        ISO-8601 UTC timestamp string (e.g. ``'2024-01-15T14:30:00+00:00'``).
    session_id : str
        UUID string that groups all messages in one conversation session.
    regime : str
        HMM market regime label at the time this entry was stored.
    score : float
        Cosine similarity score (0.0–1.0) from a similarity search.
        Zero when the entry was retrieved via a scroll/filter query.
    metadata : dict[str, Any]
        Additional numeric metrics stored in the payload
        (e.g. ``{'rsi': 65.4, 'rolling_vol': 0.182}``).
    """

    point_id: str
    entry_type: EntryType
    role: MessageRole
    content: str
    ticker: str
    timestamp: str
    session_id: str
    regime: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_snippet(self) -> str:
        """
        Format this entry as a compact, LLM-readable context string.

        Example output::

            [2024-01-15 | AAPL | Low-Volatility Bull]
            User: What's the MACD signal telling us here?
        """
        role_label = "User" if self.role == "user" else "QuantStream"
        date_str = self.timestamp[:10] if self.timestamp else "unknown date"
        return (
            f"[{date_str} | {self.ticker} | {self.regime}]\n"
            f"{role_label}: {self.content}"
        )


# ---------------------------------------------------------------------------
# Payload key constants — centralised to prevent silent typo bugs
# ---------------------------------------------------------------------------
class _PayloadKey:
    ENTRY_TYPE = "entry_type"
    ROLE       = "role"
    CONTENT    = "content"
    TICKER     = "ticker"
    SESSION_ID = "session_id"
    TIMESTAMP  = "timestamp"
    REGIME     = "regime"

    # The set of "structural" keys that are NOT stored in MemoryEntry.metadata
    STRUCTURAL: frozenset[str] = frozenset(
        {ENTRY_TYPE, ROLE, CONTENT, TICKER, SESSION_ID, TIMESTAMP, REGIME}
    )


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class VectorDBService:
    """
    Async service layer for Qdrant Cloud vector storage and retrieval.

    All public methods are coroutines and safe to call from FastAPI async
    route handlers.  The class is designed to be instantiated *once* per
    process (use ``get_vector_db_service()`` for the singleton).

    Parameters
    ----------
    collection_name : str, optional
        Override the default Qdrant collection name.  Useful in tests.
    """

    def __init__(self, collection_name: str = _COLLECTION_NAME) -> None:
        self._collection_name: str = collection_name
        self._qdrant_client: Optional["AsyncQdrantClient"] = None
        self._genai_client: Optional["_genai_t.Client"] = None
        self._collection_ready: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_env(key: str) -> str:
        """
        Read a required environment variable.

        Raises
        ------
        EnvironmentError
            With a clear message if the variable is absent or empty,
            so misconfiguration is diagnosed immediately.
        """
        value = os.environ.get(key, "").strip()
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set or is empty. "
                f"Configure it in your Render dashboard (Settings → Environment)."
            )
        return value

    # ------------------------------------------------------------------
    # Lazy client initialisation
    # ------------------------------------------------------------------

    async def _get_qdrant_client(self) -> "AsyncQdrantClient":
        """
        Return the shared ``AsyncQdrantClient``, creating it on first call.

        The client is NOT created in ``__init__`` to avoid importing
        ``qdrant_client`` (and its ~40 MB of dependencies) until needed.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client

        from qdrant_client import AsyncQdrantClient  # lazy

        url     = self._require_env("QDRANT_URL")
        api_key = self._require_env("QDRANT_API_KEY")

        self._qdrant_client = AsyncQdrantClient(url=url, api_key=api_key)
        logger.info(
            "VectorDBService: AsyncQdrantClient connected to %s…",
            url[:40],
        )
        return self._qdrant_client

    async def _get_genai_client(self) -> "_genai_t.Client":
        """
        Return the shared ``google.genai.Client``, creating it on first call.
        """
        if self._genai_client is not None:
            return self._genai_client

        import google.genai as genai_sdk  # lazy

        api_key = self._require_env("GEMINI_API_KEY")
        self._genai_client = genai_sdk.Client(api_key=api_key)
        logger.debug("VectorDBService: google.genai.Client initialised.")
        return self._genai_client

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        """
        Idempotently create the Qdrant collection if it does not yet exist.

        Protected by an ``asyncio.Lock`` so that the first wave of
        concurrent requests on a cold-start triggers only one creation call.
        The ``_collection_ready`` flag short-circuits the lock on all
        subsequent calls with zero overhead.
        """
        if self._collection_ready:
            return

        async with self._init_lock:
            # Double-checked locking: another coroutine may have initialised
            # the collection while we were waiting for the lock.
            if self._collection_ready:
                return

            from qdrant_client.models import Distance, VectorParams  # lazy

            client = await self._get_qdrant_client()

            try:
                info = await client.get_collection(self._collection_name)
                logger.info(
                    "VectorDBService: collection '%s' already exists "
                    "(vectors_count=%s).",
                    self._collection_name,
                    info.vectors_count,
                )
            except Exception:
                # Collection does not exist — create it.
                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "VectorDBService: collection '%s' created "
                    "(size=%d, distance=COSINE).",
                    self._collection_name,
                    _VECTOR_SIZE,
                )

            self._collection_ready = True

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """
        Generate a 768-dimensional embedding for ``text`` via Google's
        ``text-embedding-004`` model.

        Parameters
        ----------
        text : str
            The text to embed.  Truncated server-side if it exceeds the
            model's token limit (2 048 tokens for text-embedding-004).
        task_type : str
            ``'RETRIEVAL_DOCUMENT'`` when storing a document;
            ``'RETRIEVAL_QUERY'`` when embedding a search query.
            Using the correct task type improves recall quality.

        Returns
        -------
        list[float]
            768-element list of floats (L2-normalised by the model).

        Raises
        ------
        RuntimeError
            After ``_MAX_EMBED_RETRIES`` consecutive failures.
        """
        from google.genai import types as genai_types  # lazy

        genai_client = await self._get_genai_client()

        last_exc: Exception = RuntimeError("Embedding failed before first attempt.")

        for attempt in range(_MAX_EMBED_RETRIES):
            try:
                response = await genai_client.aio.models.embed_content(
                    model=_EMBEDDING_MODEL,
                    contents=text,
                    config=genai_types.EmbedContentConfig(task_type=task_type),
                )
                # response.embeddings is a list; we always pass a single string
                # so we take the first (and only) ContentEmbedding.
                embedding: list[float] = response.embeddings[0].values
                logger.debug(
                    "_embed: task_type=%s, dim=%d, text_len=%d.",
                    task_type, len(embedding), len(text),
                )
                return embedding

            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_EMBED_RETRIES - 1:
                    delay = _EMBED_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "_embed attempt %d/%d failed (%s). "
                        "Retrying in %.1fs…",
                        attempt + 1, _MAX_EMBED_RETRIES, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Embedding failed after {_MAX_EMBED_RETRIES} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_entry(
        self,
        content: str,
        entry_type: EntryType,
        role: MessageRole,
        session_id: str,
        ticker: str,
        regime: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Embed and store a chat message or strategy summary in Qdrant.

        Parameters
        ----------
        content : str
            The raw text to store (user message, assistant response, or
            strategy summary).
        entry_type : EntryType
            ``'chat_message'`` or ``'strategy_summary'``.
        role : MessageRole
            ``'user'`` or ``'assistant'``.
        session_id : str
            UUID string identifying the conversation session.
        ticker : str
            Active equity ticker (e.g. ``'AAPL'``).
        regime : str
            Current HMM regime label (e.g. ``'Low-Volatility Bull'``).
        metadata : dict, optional
            Extra numeric fields to store in the payload for future
            filtering or display (e.g. ``{'rsi': 65.4, 'rolling_vol': 0.18}``).

        Returns
        -------
        str
            The UUID string assigned to this Qdrant point.

        Raises
        ------
        RuntimeError
            If the embedding call fails after all retries.
        """
        from qdrant_client.models import PointStruct  # lazy

        await self._ensure_collection()

        point_id: str = str(uuid.uuid4())
        embedding: list[float] = await self._embed(
            content, task_type="RETRIEVAL_DOCUMENT"
        )

        payload: dict[str, Any] = {
            _PayloadKey.ENTRY_TYPE : entry_type,
            _PayloadKey.ROLE       : role,
            _PayloadKey.CONTENT    : content,
            _PayloadKey.TICKER     : ticker,
            _PayloadKey.SESSION_ID : session_id,
            _PayloadKey.TIMESTAMP  : datetime.now(timezone.utc).isoformat(),
            _PayloadKey.REGIME     : regime,
            **(metadata or {}),
        }

        client = await self._get_qdrant_client()
        await client.upsert(
            collection_name=self._collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )

        logger.debug(
            "upsert_entry: point_id=%s, type=%s, role=%s, ticker=%s.",
            point_id, entry_type, role, ticker,
        )
        return point_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        ticker_filter: Optional[str] = None,
        entry_type_filter: Optional[EntryType] = None,
        min_score: float = 0.35,
    ) -> list[MemoryEntry]:
        """
        Retrieve the ``top_k`` most semantically similar entries for RAG.

        The query is embedded with ``task_type='RETRIEVAL_QUERY'`` (which
        uses a different embedding projection from ``RETRIEVAL_DOCUMENT`` and
        yields significantly better recall).

        Parameters
        ----------
        query_text : str
            The user's current message / search query.
        top_k : int, default 5
            Maximum number of results to return.
        ticker_filter : str, optional
            If set, only entries tagged with this ticker are considered.
        entry_type_filter : EntryType, optional
            Restrict results to ``'chat_message'`` or ``'strategy_summary'``.
        min_score : float, default 0.35
            Cosine similarity threshold; entries below this score are
            discarded to prevent injecting irrelevant context.

        Returns
        -------
        list[MemoryEntry]
            Ordered by descending similarity score.  May be empty.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # lazy

        await self._ensure_collection()

        query_vector: list[float] = await self._embed(
            query_text, task_type="RETRIEVAL_QUERY"
        )

        # Build payload filter — only add conditions that were requested
        conditions = []
        if ticker_filter:
            conditions.append(
                FieldCondition(
                    key=_PayloadKey.TICKER,
                    match=MatchValue(value=ticker_filter),
                )
            )
        if entry_type_filter:
            conditions.append(
                FieldCondition(
                    key=_PayloadKey.ENTRY_TYPE,
                    match=MatchValue(value=entry_type_filter),
                )
            )
        query_filter = Filter(must=conditions) if conditions else None

        client = await self._get_qdrant_client()
        results = await client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=min_score,
        )

        entries = [self._point_to_memory_entry(r, use_score=True) for r in results]
        logger.debug(
            "search_similar: query_len=%d, found=%d entries (ticker=%s).",
            len(query_text), len(entries), ticker_filter,
        )
        return entries

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 30,
    ) -> list[MemoryEntry]:
        """
        Retrieve all stored entries for a given session, sorted oldest-first.

        Used to reconstruct the conversation history for a
        ``generate_strategy_summary()`` call.

        Parameters
        ----------
        session_id : str
            The session UUID to filter on.
        limit : int, default 30
            Maximum number of entries to return.

        Returns
        -------
        list[MemoryEntry]
            Sorted by ascending ``timestamp``.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # lazy

        await self._ensure_collection()

        client = await self._get_qdrant_client()
        records, _ = await client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=_PayloadKey.SESSION_ID,
                        match=MatchValue(value=session_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,  # Vectors not needed for history display
        )

        entries = [self._point_to_memory_entry(r, use_score=False) for r in records]
        # Sort ascending by ISO timestamp string (lexicographic sort is valid here)
        entries.sort(key=lambda e: e.timestamp)

        logger.debug(
            "get_session_history: session_id=%s, returned=%d entries.",
            session_id, len(entries),
        )
        return entries

    # ------------------------------------------------------------------
    # Private conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_to_memory_entry(
        point: Any,
        use_score: bool,
    ) -> "MemoryEntry":
        """
        Convert a raw Qdrant ``ScoredPoint`` or ``Record`` to a ``MemoryEntry``.

        Handles both types transparently: ``ScoredPoint`` (from ``search``)
        carries a ``score`` attribute; ``Record`` (from ``scroll``) does not.
        """
        payload: dict[str, Any] = point.payload or {}

        # Extract structural fields; everything else becomes metadata
        extra_meta = {
            k: v for k, v in payload.items()
            if k not in _PayloadKey.STRUCTURAL
        }

        return MemoryEntry(
            point_id   = str(point.id),
            entry_type = payload.get(_PayloadKey.ENTRY_TYPE, "chat_message"),
            role       = payload.get(_PayloadKey.ROLE, "user"),
            content    = payload.get(_PayloadKey.CONTENT, ""),
            ticker     = payload.get(_PayloadKey.TICKER, ""),
            timestamp  = payload.get(_PayloadKey.TIMESTAMP, ""),
            session_id = payload.get(_PayloadKey.SESSION_ID, ""),
            regime     = payload.get(_PayloadKey.REGIME, ""),
            score      = float(getattr(point, "score", 0.0)) if use_score else 0.0,
            metadata   = extra_meta,
        )

    # ------------------------------------------------------------------
    # Operational utilities
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Verify Qdrant Cloud connectivity.

        Returns
        -------
        bool
            ``True`` if the cluster responds to ``get_collections()``
            within the client's default timeout; ``False`` otherwise.
        """
        try:
            client = await self._get_qdrant_client()
            await client.get_collections()
            return True
        except Exception as exc:
            logger.warning("VectorDBService.health_check failed: %s", exc)
            return False

    async def close(self) -> None:
        """
        Gracefully close all open client connections.

        Call this from the FastAPI ``lifespan`` shutdown handler.
        """
        if self._qdrant_client is not None:
            await self._qdrant_client.close()
            self._qdrant_client = None
            logger.info("VectorDBService: AsyncQdrantClient closed.")

        if self._genai_client is not None:
            await self._genai_client.aio.close()
            self._genai_client = None
            logger.info("VectorDBService: google.genai.Client closed.")


# ---------------------------------------------------------------------------
# Singleton factory for FastAPI dependency injection
# ---------------------------------------------------------------------------

_vector_db_instance: Optional[VectorDBService] = None


def get_vector_db_service() -> VectorDBService:
    """
    Return the process-scoped ``VectorDBService`` singleton.

    Intended for use as a FastAPI ``Depends`` dependency::

        from backend.services.vector_db import get_vector_db_service

        @app.get("/health")
        async def health(vdb: VectorDBService = Depends(get_vector_db_service)):
            ok = await vdb.health_check()
            return {"qdrant": ok}

    The first call creates the instance; subsequent calls return the same
    object.  Thread-safe for the single-threaded async FastAPI runtime.
    """
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBService()
        logger.debug("VectorDBService singleton created.")
    return _vector_db_instance