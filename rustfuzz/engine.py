"""
rustfuzz.engine — Batteries-included retriever.

Provides the :class:`Retriever` class, a single entry-point that
automatically selects the best retrieval pipeline based on what the
user provides:

- **Corpus only** → BM25Plus + Fuzzy via RRF (fast, typo-tolerant)
- **+ Embeddings** → Upgrades to 3-way HybridSearch (BM25 + Fuzzy + Dense)
- **+ Reranker**   → Adds a 2nd-stage cross-encoder reranker for SOTA accuracy

Embedding providers are auto-detected from short names:

- ``"openai"``  → OpenAI ``text-embedding-3-small`` via API
- ``"cohere"``  → Cohere ``embed-english-v3.0`` via API
- ``True``      → HuggingFace ``all-MiniLM-L6-v2`` via ``embed_anything`` (local)
- Any ``"org/model"`` string → loaded via ``embed_anything`` (local, no PyTorch)

Usage::

    from rustfuzz import Retriever

    # Simplest — BM25+ with fuzzy (no extra dependencies)
    r = Retriever(docs)
    results = r.search("wireless headphones", n=10)

    # Auto-embed with OpenAI API
    r = Retriever(docs, embeddings="openai")

    # Full SOTA pipeline (local, no PyTorch, no API keys)
    r = Retriever(docs, embeddings=True, reranker=cross_encoder)
    results = r.search("wireless headphones", n=10)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

from .document import Document  # noqa: F401
from .search import (
    BM25,
    BM25L,
    BM25T,
    BM25Plus,
    HybridSearch,
    Reranker,
    _extract_column,
    _extract_metadata,
)

# Type aliases
_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]

# Default HuggingFace embedding model — small, fast, no PyTorch
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Embedding provider shortcuts ─────────────────────────────────────
_PROVIDER_SHORTCUTS: dict[str, dict[str, str]] = {
    # OpenAI API
    "openai": {"provider": "openai", "model": "text-embedding-3-small"},
    "openai-small": {"provider": "openai", "model": "text-embedding-3-small"},
    "openai-large": {"provider": "openai", "model": "text-embedding-3-large"},
    "openai-ada": {"provider": "openai", "model": "text-embedding-ada-002"},
    # Azure OpenAI
    "azure-openai": {"provider": "azure-openai", "model": "text-embedding-3-small"},
    "azure-openai-small": {
        "provider": "azure-openai",
        "model": "text-embedding-3-small",
    },
    "azure-openai-large": {
        "provider": "azure-openai",
        "model": "text-embedding-3-large",
    },
    # Cohere API
    "cohere": {"provider": "cohere", "model": "embed-english-v3.0"},
    "cohere-english": {"provider": "cohere", "model": "embed-english-v3.0"},
    "cohere-multilingual": {"provider": "cohere", "model": "embed-multilingual-v3.0"},
    "cohere-light": {"provider": "cohere", "model": "embed-english-light-v3.0"},
    # Azure Cohere (serverless endpoint via Azure AI Inference)
    "azure-cohere": {"provider": "azure-cohere", "model": "embed-english-v3.0"},
    "azure-cohere-english": {"provider": "azure-cohere", "model": "embed-english-v3.0"},
    "azure-cohere-multilingual": {
        "provider": "azure-cohere",
        "model": "embed-multilingual-v3.0",
    },
}

# Maps user-friendly algorithm names to BM25 class
_ALGO_MAP: dict[str, type[BM25 | BM25L | BM25Plus | BM25T]] = {
    "bm25": BM25,
    "bm25okapi": BM25,
    "bm25l": BM25L,
    "bm25+": BM25Plus,
    "bm25plus": BM25Plus,
    "bm25t": BM25T,
}

# Maps classes to the algorithm string accepted by HybridSearch
_ALGO_TO_HYBRID: dict[type, str] = {
    BM25: "bm25",
    BM25L: "bm25l",
    BM25Plus: "bm25+",
    BM25T: "bm25t",
}


# ── Config dataclass ─────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RetrieverConfig:
    """
    Configuration for :class:`Retriever`.

    All tuning knobs in one place — pass as ``Retriever(docs, config=cfg)``.

    Parameters
    ----------
    algorithm : str
        BM25 variant: ``"bm25"`` / ``"bm25l"`` / ``"bm25plus"`` / ``"bm25t"``.
    k1 : float
        BM25 term-frequency saturation parameter.
    b : float
        BM25 document-length normalisation parameter.
    delta : float | None
        BM25L / BM25Plus delta. Auto-selected when ``None``.
    normalize : bool
        Apply Unicode NFKD + lowercase text normalisation.
    rerank_top_k : int
        Max results returned after reranking.

    Examples
    --------
    >>> cfg = RetrieverConfig(algorithm="bm25l", k1=1.2, b=0.8, delta=0.5)
    >>> r = Retriever(docs, config=cfg, embeddings="openai")
    """

    algorithm: str = "bm25plus"
    k1: float = 1.5
    b: float = 0.75
    delta: float | None = None
    normalize: bool = True
    rerank_top_k: int = 10


# ── Internal helpers ─────────────────────────────────────────────────


def _search_query(owner: Any) -> Any:
    """Lazy import to avoid circular dependency."""
    from .query import SearchQuery

    return SearchQuery(owner)


# ── Embedding factory functions ──────────────────────────────────────


def _build_embed_fn_hf(
    model_id: str,
) -> Callable[[list[str]], list[list[float]]]:
    """
    Build an embedding callback from a HuggingFace model name.

    Uses ``embed_anything`` (Rust/ONNX backend) — no PyTorch required.
    """
    try:
        import embed_anything
        from embed_anything import EmbeddingModel
    except ImportError:
        raise ImportError(
            "embed-anything is required for HuggingFace embeddings. "
            "Install via: uv add embed-anything"
        ) from None

    model = EmbeddingModel.from_pretrained_hf(model_id=model_id)

    def _embed(texts: list[str]) -> list[list[float]]:
        results = embed_anything.embed_query(texts, embedder=model)
        return [r.embedding for r in results]

    return _embed


def _build_embed_fn_openai(
    model: str = "text-embedding-3-small",
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """
    Build an embedding callback using the OpenAI API.

    Falls back to ``OPENAI_API_KEY`` env var when ``api_key`` is not provided.
    """
    import os

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI embeddings. Install via: uv add openai"
        ) from None

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "api_key or OPENAI_API_KEY environment variable is required "
            "for OpenAI embeddings."
        )

    client = OpenAI(api_key=resolved_key, base_url=api_base)

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    return _embed


def _build_embed_fn_cohere(
    model: str = "embed-english-v3.0",
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """
    Build an embedding callback using the Cohere API.

    Falls back to ``COHERE_API_KEY`` / ``CO_API_KEY`` env var.
    """
    import os

    try:
        import cohere
    except ImportError:
        raise ImportError(
            "cohere is required for Cohere embeddings. Install via: uv add cohere"
        ) from None

    resolved_key = (
        api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
    )
    if not resolved_key:
        raise ValueError(
            "api_key or COHERE_API_KEY environment variable is required "
            "for Cohere embeddings."
        )

    kwargs: dict[str, Any] = {"api_key": resolved_key}
    if api_base:
        kwargs["base_url"] = api_base
    client = cohere.ClientV2(**kwargs)

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embed(
            texts=texts,
            model=model,
            input_type="search_document",
            embedding_types=["float"],
        )
        return [list(emb) for emb in response.embeddings.float_]

    return _embed


def _build_embed_fn_azure_openai(
    model: str = "text-embedding-3-small",
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """
    Build an embedding callback using Azure OpenAI.

    Falls back to ``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_API_KEY`` env vars.
    """
    import os

    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError(
            "openai is required for Azure OpenAI embeddings. Install via: uv add openai"
        ) from None

    endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
    resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not resolved_key:
        raise ValueError(
            "api_base/AZURE_OPENAI_ENDPOINT and api_key/AZURE_OPENAI_API_KEY "
            "are required for Azure OpenAI embeddings."
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=resolved_key,
        api_version=api_version,
    )

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    return _embed


def _build_embed_fn_azure_cohere(
    model: str = "embed-english-v3.0",
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """
    Build an embedding callback using a Cohere model deployed on Azure AI.

    Uses the Azure AI Inference SDK (``azure-ai-inference``).
    Falls back to ``AZURE_COHERE_ENDPOINT`` and ``AZURE_COHERE_API_KEY`` env vars.
    """
    import os

    try:
        from azure.ai.inference import EmbeddingsClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "azure-ai-inference is required for Azure Cohere embeddings. "
            "Install via: uv add azure-ai-inference"
        ) from None

    endpoint = api_base or os.environ.get("AZURE_COHERE_ENDPOINT")
    resolved_key = api_key or os.environ.get("AZURE_COHERE_API_KEY")
    if not endpoint or not resolved_key:
        raise ValueError(
            "api_base/AZURE_COHERE_ENDPOINT and api_key/AZURE_COHERE_API_KEY "
            "are required for Azure Cohere embeddings."
        )

    client = EmbeddingsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(resolved_key),
    )

    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embed(
            input=texts,
            model=model,
            input_type="search_document",
        )
        return [list(item.embedding) for item in response.data]

    return _embed


def _resolve_embeddings(
    embeddings: Any,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> tuple[Any, str | None]:
    """
    Resolve the ``embeddings`` parameter into a usable callback or matrix.

    Returns ``(resolved_embeddings, model_id_for_display)``.
    """
    if embeddings is None or embeddings is False:
        return None, None

    if embeddings is True:
        return _build_embed_fn_hf(_DEFAULT_MODEL), _DEFAULT_MODEL

    if isinstance(embeddings, str):
        key = embeddings.lower().strip()
        provider_info = _PROVIDER_SHORTCUTS.get(key)

        if provider_info is not None:
            provider = provider_info["provider"]
            model_name = provider_info["model"]
            builders: dict[
                str, Callable[..., Callable[[list[str]], list[list[float]]]]
            ] = {
                "openai": _build_embed_fn_openai,
                "azure-openai": _build_embed_fn_azure_openai,
                "cohere": _build_embed_fn_cohere,
                "azure-cohere": _build_embed_fn_azure_cohere,
            }
            builder = builders.get(provider)
            if builder is not None:
                return (
                    builder(model_name, api_key=api_key, api_base=api_base),
                    f"{provider}/{model_name}",
                )

        # Treat as HuggingFace model name
        return _build_embed_fn_hf(embeddings), embeddings

    # Callable or pre-computed matrix — pass through
    return embeddings, None


# ── Main class ───────────────────────────────────────────────────────


class Retriever:
    """
    Batteries-included retriever — SOTA search in 3 lines.

    Automatically selects the optimal pipeline:

    1. **BM25Plus + Fuzzy RRF** — fast, typo-tolerant (default)
    2. **+ Embeddings** → 3-way HybridSearch (BM25 + Fuzzy + Dense)
    3. **+ Reranker**   → Cross-encoder reranking for maximum accuracy

    Parameters
    ----------
    corpus : Iterable[str] | Iterable[Document] | Any
        Text documents. Accepts strings, ``Document`` objects,
        LangChain Documents, or DataFrame columns.
    embeddings : str | matrix | Callable | bool | None
        One of:

        - ``"openai"`` / ``"openai-large"`` — OpenAI API (needs ``OPENAI_API_KEY``)
        - ``"cohere"`` / ``"cohere-multilingual"`` — Cohere API (needs ``COHERE_API_KEY``)
        - ``True`` — default HF model (``all-MiniLM-L6-v2``, local, no PyTorch)
        - Any ``"org/model"`` string — loaded via ``embed_anything`` (local)
        - A pre-computed matrix (list of lists, numpy array)
        - A **callback** ``fn(texts) -> list[list[float]]``
        - ``None`` — BM25 + Fuzzy only (no embeddings)
    reranker : Any | None
        A cross-encoder model or callable ``fn(query, texts) -> list[float]``.
    config : RetrieverConfig | None
        Configuration dataclass. Overrides ``algorithm``, ``k1``, ``b``,
        ``delta``, ``normalize``, ``rerank_top_k`` when provided.
    metadata : Iterable[Any] | None
        Per-document metadata dicts for filtering / sorting.
    **kwargs
        Any field from ``RetrieverConfig`` can also be passed directly
        as a keyword argument (e.g. ``algorithm="bm25l"``).

    Examples
    --------
    >>> from rustfuzz import Retriever
    >>> r = Retriever(["Apple iPhone", "Samsung Galaxy"])
    >>> r.search("iphone", n=1)
    [('Apple iPhone', ...)]

    With config::

        cfg = RetrieverConfig(algorithm="bm25l", k1=1.2, b=0.8)
        r = Retriever(docs, config=cfg, embeddings="openai")

    Full SOTA::

        r = Retriever(docs, embeddings=True, reranker=cross_encoder)
        r.search("query", n=10)
    """

    def __init__(
        self,
        corpus: Iterable[str] | Iterable[Any] | Any,
        *,
        embeddings: Any
        | str
        | bool
        | Callable[[list[str]], list[list[float]]]
        | None = None,
        reranker: Any | None = None,
        config: RetrieverConfig | None = None,
        metadata: Iterable[Any] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        # Config overrides (ignored when config= is provided)
        algorithm: str = "bm25plus",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float | None = None,
        normalize: bool = True,
        rerank_top_k: int = 10,
    ) -> None:
        # ── Merge config ──
        if config is not None:
            self._config = config
        else:
            self._config = RetrieverConfig(
                algorithm=algorithm,
                k1=k1,
                b=b,
                delta=delta,
                normalize=normalize,
                rerank_top_k=rerank_top_k,
            )

        cfg = self._config
        self._api_key = api_key
        self._api_base = api_base

        # ── Resolve embeddings ──
        resolved_embeddings, self._model_id = _resolve_embeddings(
            embeddings, api_key=api_key, api_base=api_base
        )
        self._embeddings_raw = embeddings

        # ── Resolve BM25 variant class ──
        algo_name = cfg.algorithm.lower().strip()
        algo_cls = _ALGO_MAP.get(algo_name)
        if algo_cls is None:
            valid = ", ".join(sorted(_ALGO_MAP))
            raise ValueError(
                f"Unknown algorithm {cfg.algorithm!r}. Choose from: {valid}"
            )

        # ── Reranker setup ──
        self._reranker: Reranker | None = None
        if reranker is not None:
            if isinstance(reranker, Reranker):
                self._reranker = reranker
            else:
                self._reranker = Reranker(reranker)

        # ── Build the underlying index ──
        if resolved_embeddings is not None:
            hybrid_algo = _ALGO_TO_HYBRID.get(algo_cls, "bm25+")
            effective_delta = cfg.delta
            if effective_delta is None and algo_cls in (BM25L, BM25Plus):
                effective_delta = 0.5 if algo_cls is BM25L else 1.0

            self._hybrid: HybridSearch | None = HybridSearch(
                corpus,
                embeddings=resolved_embeddings,
                k1=cfg.k1,
                b=cfg.b,
                metadata=metadata,
                algorithm=hybrid_algo,
                delta=effective_delta,
            )
            self._bm25: BM25 | BM25L | BM25Plus | BM25T | None = None
            self._corpus = self._hybrid._corpus
            self._metadata = self._hybrid._metadata
        else:
            self._hybrid = None
            bm25_kwargs: dict[str, Any] = {
                "k1": cfg.k1,
                "b": cfg.b,
                "normalize": cfg.normalize,
            }
            if algo_cls in (BM25L, BM25Plus) and cfg.delta is not None:
                bm25_kwargs["delta"] = cfg.delta
            elif algo_cls is BM25L:
                bm25_kwargs["delta"] = 0.5
            elif algo_cls is BM25Plus:
                bm25_kwargs["delta"] = 1.0
            if metadata is not None:
                bm25_kwargs["metadata"] = metadata

            self._bm25 = algo_cls(corpus, **bm25_kwargs)  # type: ignore[arg-type]
            self._corpus = self._bm25._corpus
            self._metadata = self._bm25._metadata

    # ── Class methods ────────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Retriever:
        """
        Build a Retriever from a DataFrame column.

        Parameters
        ----------
        df : DataFrame
            Pandas, Polars, or Spark DataFrame.
        column : str
            Name of the text column.
        metadata_columns : list[str] | str | None
            Column(s) to extract as per-row metadata dicts.
        **kwargs
            Forwarded to ``Retriever.__init__``.
        """
        corpus = _extract_column(df, column)
        meta = _extract_metadata(df, metadata_columns) if metadata_columns else None
        return cls(corpus, metadata=meta, **kwargs)

    # ── Properties ───────────────────────────────────────────

    @property
    def config(self) -> RetrieverConfig:
        """The active configuration."""
        return self._config

    @property
    def num_docs(self) -> int:
        """Number of documents in the index."""
        if self._hybrid is not None:
            return self._hybrid.num_docs
        assert self._bm25 is not None
        return self._bm25.num_docs

    @property
    def has_embeddings(self) -> bool:
        """Whether dense embeddings are available."""
        return self._hybrid is not None and self._hybrid.has_vectors

    @property
    def has_reranker(self) -> bool:
        """Whether a reranker is attached."""
        return self._reranker is not None

    # ── Core search ──────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        query_embedding: Any = None,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Search the corpus with the best available pipeline.

        Automatically uses:
        - HybridSearch (BM25 + Fuzzy + Dense) when embeddings are available
        - BM25 + Fuzzy RRF otherwise
        - Reranker post-processing when a reranker is attached

        Parameters
        ----------
        query : str
            The search query.
        n : int, default 10
            Number of results to return.
        query_embedding : Any, optional
            Pre-computed query embedding. If omitted and an embedding
            model/callback was provided, it is generated automatically.
        """
        retrieve_n = n
        if self._reranker is not None:
            retrieve_n = max(n * 5, 50)

        if self._hybrid is not None:
            results = self._hybrid.search(
                query,
                query_embedding=query_embedding,
                n=retrieve_n,
                bm25_candidates=max(retrieve_n * 2, 200),
            )
        else:
            assert self._bm25 is not None
            results = self._bm25.get_top_n_rrf(
                query,
                n=retrieve_n,
                bm25_candidates=max(retrieve_n * 2, 200),
            )

        if self._reranker is not None and results:
            results = self._reranker.rerank(
                query, results, top_k=min(n, self._config.rerank_top_k)
            )
        elif len(results) > n:
            results = results[:n]

        return results

    # ── Fluent query builder ─────────────────────────────────

    def filter(self, expression: str) -> Any:
        """
        Start a filtered query chain.

        Parameters
        ----------
        expression : str
            Meilisearch-style filter, e.g. ``'brand = "Apple"'``.
        """
        owner = self._hybrid if self._hybrid is not None else self._bm25
        return _search_query(owner).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """
        Start a sorted query chain.

        Parameters
        ----------
        expression : str | list[str]
            Sort criteria, e.g. ``"price:asc"`` or ``["price:asc", "name:desc"]``.
        """
        owner = self._hybrid if self._hybrid is not None else self._bm25
        return _search_query(owner).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """
        Execute a search with the query builder (terminal operation).

        Parameters
        ----------
        query : str
            The search query.
        **kwargs
            Forwarded to the underlying search method.
        """
        owner = self._hybrid if self._hybrid is not None else self._bm25
        return _search_query(owner).match(query, **kwargs)

    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any:
        """
        Add a reranker to the query builder chain.

        Parameters
        ----------
        model_or_callable : Any
            Cross-encoder model or callable.
        top_k : int, default 10
            Number of results after reranking.
        """
        owner = self._hybrid if self._hybrid is not None else self._bm25
        return _search_query(owner).rerank(model_or_callable, top_k=top_k)

    def get_top_n(
        self,
        query: str,
        n: int = 5,
        *,
        query_embedding: Any = None,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Return the top N matching documents — alias for :meth:`search`.

        Provides API parity with ``BM25.get_top_n`` for easy migration.

        Parameters
        ----------
        query : str
            The search query.
        n : int, default 5
            Number of results to return.
        query_embedding : Any, optional
            Pre-computed query embedding (only used with HybridSearch).
        """
        return self.search(query, n=n, query_embedding=query_embedding)

    # ── Upgrade methods ──────────────────────────────────────

    def to_hybrid(self, embeddings: Any | str | bool | None = True) -> Retriever:
        """
        Upgrade to HybridSearch by attaching dense embeddings.

        Returns a **new** ``Retriever`` with embeddings enabled.
        """
        return Retriever(
            self._corpus,
            embeddings=embeddings,
            reranker=self._reranker,
            config=self._config,
            metadata=self._metadata,
        )

    # ── Serialisation ────────────────────────────────────────

    def __reduce__(self) -> tuple[Any, ...]:
        emb_for_pickle: Any = None
        if self._model_id is not None:
            emb_for_pickle = self._model_id
        elif self._hybrid is not None:
            emb_for_pickle = self._hybrid._embeddings

        state = {
            "corpus": self._corpus,
            "embeddings": emb_for_pickle,
            "reranker": self._reranker,
            "config": self._config,
            "metadata": self._metadata,
        }
        return (_reconstruct_retriever, (state,))

    def __repr__(self) -> str:
        parts = [f"Retriever(docs={self.num_docs}"]
        parts.append(f"algo={self._config.algorithm!r}")
        if self._model_id:
            parts.append(f"model={self._model_id!r}")
        elif self.has_embeddings:
            parts.append("hybrid=True")
        if self.has_reranker:
            parts.append("reranker=True")
        return ", ".join(parts) + ")"


def _reconstruct_retriever(state: dict[str, Any]) -> Retriever:
    """Pickle reconstruction helper."""
    return Retriever(
        state["corpus"],
        embeddings=state.get("embeddings"),
        reranker=state.get("reranker"),
        config=state.get("config", RetrieverConfig()),
        metadata=state.get("metadata"),
    )


__all__ = ["Retriever", "RetrieverConfig"]
