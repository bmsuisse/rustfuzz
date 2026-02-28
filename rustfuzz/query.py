"""
rustfuzz.query — Fluent query builder for BM25 / HybridSearch with filter & sort.

Provides a lazy, chainable API that mirrors the Meilisearch experience.

Usage::

    from rustfuzz.search import BM25

    bm25 = BM25(corpus, metadata=metadata)

    # Chainable — lazy until .collect()
    results = (
        bm25
        .filter('price > 100 AND brand = "Apple"')
        .sort("price:asc")
        .search("iphone pro max", n=10)
        .collect()
    )

    # Or one-shot — filter/sort applied directly
    results = bm25.filter('category = "phones"').get_top_n("query", n=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .filter import evaluate_filter, parse_filter
from .sort import apply_sort

if TYPE_CHECKING:
    from .filter import FilterNode

# Type aliases for result tuples
_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]


class SearchQuery:
    """
    Lazy query builder for filtered + sorted search.

    Accumulates filter expressions, sort criteria, and search parameters.
    Execution is deferred until a terminal method is called.

    **Terminal methods** (execute immediately):
    - ``.match(query, n)`` — search with filter/sort (works on BM25 + HybridSearch)
    - ``.get_top_n(query, n)`` — BM25 / Hybrid top-N with filter/sort
    - ``.get_top_n_fuzzy(query, ...)`` — BM25+fuzzy with filter/sort
    - ``.get_top_n_rrf(query, ...)`` — BM25+fuzzy RRF with filter/sort
    - ``.get_top_n_phrase(query, ...)`` — BM25 phrase with filter/sort
    - ``.collect()`` — execute a deferred ``.search()`` call

    **Builder methods** (lazy, return self):
    - ``.filter(expr)`` — add a Meilisearch-style filter expression
    - ``.sort(expr)`` — add sort criteria
    - ``.search(query, ...)`` — set the text search query (deferred)
    """

    __slots__ = (
        "_owner",
        "_filters",
        "_sort_expr",
        "_search_query",
        "_search_kwargs",
        "_search_method",
    )

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._filters: list[FilterNode] = []
        self._sort_expr: list[str] | str | None = None
        self._search_query: str | None = None
        self._search_kwargs: dict[str, Any] = {}
        self._search_method: str = "get_top_n"

    # ── Builder methods (return self) ───────────────────────

    def filter(self, expression: str) -> SearchQuery:
        """
        Add a Meilisearch-style filter expression.

        Multiple filters are combined with AND.

        Parameters
        ----------
        expression : str
            Filter string, e.g. ``'price > 100 AND brand = "Apple"'``
        """
        self._filters.append(parse_filter(expression))
        return self

    def sort(self, expression: list[str] | str) -> SearchQuery:
        """
        Set sort criteria (Meilisearch-style).

        Parameters
        ----------
        expression : list[str] | str
            Sort expression(s), e.g. ``["price:asc", "name:desc"]``
            or ``"price:asc"``.
        """
        self._sort_expr = expression
        return self

    def search(
        self,
        query: str,
        *,
        n: int = 5,
        method: str = "get_top_n",
        **kwargs: Any,
    ) -> SearchQuery:
        """
        Set the text search query (lazy — deferred until ``.collect()``).

        Parameters
        ----------
        query : str
            Text query.
        n : int, default 5
            Number of results.
        method : str, default "get_top_n"
            Search method name: ``"get_top_n"``, ``"get_top_n_fuzzy"``,
            ``"get_top_n_rrf"``, ``"get_top_n_phrase"``, ``"search"`` (hybrid).
        **kwargs
            Additional keyword arguments for the search method.
        """
        self._search_query = query
        self._search_method = method
        self._search_kwargs = {"n": n, **kwargs}
        return self

    def match(
        self,
        query: str,
        *,
        n: int = 5,
        query_embedding: list[float] | None = None,
        **kwargs: Any,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Execute a search with accumulated filter/sort and return results immediately.

        Works with both BM25 variants and HybridSearch.

        Parameters
        ----------
        query : str
            Text query.
        n : int, default 5
            Number of results.
        query_embedding : list[float] | None
            Optional dense embedding for HybridSearch.
        **kwargs
            Additional keyword arguments for the search method.
        """
        return self._execute(
            "get_top_n", query, n=n, query_embedding=query_embedding, **kwargs
        )

    # ── Terminal methods ────────────────────────────────────

    def collect(self) -> list[_Result] | list[_MetaResult]:
        """
        Execute the accumulated query and return results.

        Raises
        ------
        ValueError
            If no search query has been set via ``.search()`` or ``.match()``.
        """
        if self._search_query is None:
            raise ValueError(
                "No search query set. Call .search(query) or .match(query) before .collect()"
            )
        return self._execute(
            self._search_method, self._search_query, **self._search_kwargs
        )

    def get_top_n(
        self,
        query: str,
        n: int = 5,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[_Result] | list[_MetaResult]:
        """BM25 / HybridSearch top-N with accumulated filter/sort."""
        return self._execute("get_top_n", query, n=n, query_embedding=query_embedding)

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]:
        """BM25+fuzzy with accumulated filter/sort."""
        return self._execute(
            "get_top_n_fuzzy",
            query,
            n=n,
            bm25_candidates=bm25_candidates,
            fuzzy_weight=fuzzy_weight,
        )

    def get_top_n_rrf(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 100,
        rrf_k: int = 60,
    ) -> list[_Result] | list[_MetaResult]:
        """BM25+fuzzy RRF with accumulated filter/sort."""
        return self._execute(
            "get_top_n_rrf",
            query,
            n=n,
            bm25_candidates=bm25_candidates,
            rrf_k=rrf_k,
        )

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]:
        """BM25 phrase search with accumulated filter/sort."""
        return self._execute(
            "get_top_n_phrase",
            query,
            n=n,
            proximity_window=proximity_window,
            phrase_boost=phrase_boost,
        )

    # ── Internal execution ──────────────────────────────────

    def _build_allowed_mask(self) -> list[bool] | None:
        """
        Evaluate filter expressions against metadata to build a boolean mask.

        Returns None if no filters are set (no filtering).
        """
        if not self._filters:
            return None

        owner = self._owner
        metadata = owner._metadata
        if metadata is None:
            return None

        mask: list[bool] = []
        for meta in metadata:
            if meta is None or not isinstance(meta, dict):
                mask.append(False)
                continue
            # All filters must match (AND)
            passes = all(evaluate_filter(f, meta) for f in self._filters)
            mask.append(passes)
        return mask

    def _parse_sort_keys(self) -> list[tuple[str, bool]] | None:
        """Parse sort expression(s) into (attribute, reverse) tuples for Rust."""
        if self._sort_expr is None:
            return None
        exprs = (
            self._sort_expr if isinstance(self._sort_expr, list) else [self._sort_expr]
        )
        keys: list[tuple[str, bool]] = []
        for expr in exprs:
            expr = expr.strip()
            if ":" in expr:
                attr, direction = expr.rsplit(":", 1)
                keys.append((attr.strip(), direction.strip().lower() == "desc"))
            else:
                keys.append((expr, False))
        return keys if keys else None

    def _execute(
        self, method: str, query: str, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        """Execute the query with filter mask passed to Rust and sort in Python."""
        from .search import _enrich

        owner = self._owner

        # ── Fast-path: HybridSearch with Rust-side metadata ──
        if (
            hasattr(owner._index, "search_filtered_sorted")
            and owner._index.has_metadata
        ):
            filter_json: str | None = None
            if self._filters:
                from .filter import filters_to_json

                filter_json = filters_to_json(self._filters)

            sort_keys = self._parse_sort_keys()

            # Handle embedding callback for query embedding
            query_embedding = kwargs.get("query_embedding")
            if (
                query_embedding is None
                and hasattr(owner, "_embed_fn")
                and owner._embed_fn is not None
            ):
                embs = owner._embed_fn([query])
                if embs and len(embs) > 0:
                    query_embedding = embs[0]

            raw = owner._index.search_filtered_sorted(
                query,
                query_embedding,
                kwargs.get("n", 5),
                kwargs.get("rrf_k", 60),
                kwargs.get("bm25_candidates", 100),
                filter_json,
                sort_keys,
            )

            return _enrich(raw, owner._corpus, owner._metadata, owner._corpus_index)

        # ── Standard path: BM25 variants ──
        allowed = self._build_allowed_mask()

        # HybridSearch without Rust metadata falls back
        if hasattr(owner._index, "search_filtered"):
            raw = owner._index.search_filtered(
                query,
                kwargs.get("query_embedding"),
                kwargs.get("n", 5),
                kwargs.get("rrf_k", 60),
                kwargs.get("bm25_candidates", 100),
                allowed,
            )
        elif method == "get_top_n":
            raw = owner._index.get_top_n_filtered(query, kwargs.get("n", 5), allowed)
        elif method == "get_top_n_rrf":
            raw = owner._index.get_top_n_rrf_filtered(
                query,
                kwargs.get("n", 5),
                kwargs.get("bm25_candidates", 100),
                kwargs.get("rrf_k", 60),
                allowed,
            )
        elif method in ("get_top_n_fuzzy", "get_top_n_phrase"):
            # These don't have filtered variants — use get_top_n_filtered
            raw = owner._index.get_top_n_filtered(query, kwargs.get("n", 5), allowed)
        else:
            raise ValueError(f"Unknown search method: {method!r}")

        # Enrich with metadata
        results = _enrich(raw, owner._corpus, owner._metadata, owner._corpus_index)

        # Apply sort (post-hoc on enriched results)
        if self._sort_expr is not None:
            results = apply_sort(results, self._sort_expr)

        return results

    def __repr__(self) -> str:
        parts = [f"SearchQuery(owner={type(self._owner).__name__}"]
        if self._filters:
            parts.append(f", filters={len(self._filters)}")
        if self._sort_expr:
            parts.append(f", sort={self._sort_expr!r}")
        if self._search_query:
            parts.append(f", query={self._search_query!r}")
        parts.append(")")
        return "".join(parts)
