"""Reranker — 2nd-stage cross-encoder re-ranking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .._types import MetaResult, Result
from ._helpers import _blend_reranked_scores


class Reranker:
    """
    A 2nd-stage Reranker that uses a Cross-Encoder or custom scoring function
    to re-evaluate and re-sort documents retrieved by a 1st-stage search engine.

    Parameters
    ----------
    model_or_callable : Any
        A cross-encoder model object or callable ``fn(query, texts) -> list[float]``.
    blend_alpha : float, default 0.0
        Weight of original retrieval rank (0.0 to 1.0).
    adaptive_blend : bool, default False
        Dynamically adjusts blend_alpha based on reranker score variance.
    """

    def __init__(
        self,
        model_or_callable: Any,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> None:
        self._model = model_or_callable
        self.blend_alpha = blend_alpha
        self.adaptive_blend = adaptive_blend
        self._score_fn = self._resolve_score_fn(model_or_callable)

    @staticmethod
    def _resolve_score_fn(
        model: Any,
    ) -> Callable[[str, list[str]], list[float]]:
        """Auto-detect the scoring interface of a reranker model.

        Dispatch order:
        1. ``.predict(pairs)`` — SentenceTransformers CrossEncoder
        2. ``.compute_scores(queries, docs, n)`` — FlagEmbedding / BGE
        3. ``.score(query, texts)`` — direct score method
        4. Bare callable ``fn(query, texts) -> list[float]``
        """
        if hasattr(model, "predict"):

            def _predict_wrapper(query: str, texts: list[str]) -> list[float]:
                pairs = [(query, t) for t in texts]
                scores = model.predict(pairs)
                return scores.tolist() if hasattr(scores, "tolist") else list(scores)

            return _predict_wrapper

        if hasattr(model, "compute_scores"):

            def _compute_wrapper(query: str, texts: list[str]) -> list[float]:
                scores: list[float] = []
                for t in texts:
                    result = model.compute_scores([query], [t], 1)
                    val = (
                        result[0]
                        if isinstance(result[0], (int, float))
                        else result[0][0]
                    )
                    scores.append(float(val))
                return scores

            return _compute_wrapper

        if hasattr(model, "score"):
            return model.score

        if callable(model):
            return cast("Callable[[str, list[str]], list[float]]", model)

        raise ValueError(
            "Reranker model must be callable or provide a "
            "`.predict()`/`.compute_scores()`/`.score()` method."
        )

    def rerank(
        self,
        query: str,
        results: list[Result] | list[MetaResult],
        top_k: int = 10,
    ) -> list[Result] | list[MetaResult]:
        """Re-score and re-order results from a search engine."""
        if not results:
            return []

        texts = [res[0] for res in results]

        try:
            new_scores = cast("list[float]", self._score_fn(query, texts))
        except Exception as e:
            raise RuntimeError(f"Reranker model failed to score texts: {e}") from e

        if len(new_scores) != len(results):
            raise ValueError(
                f"Reranker returned {len(new_scores)} scores "
                f"for {len(results)} documents."
            )

        # Apply score blending if configured
        alpha = self.blend_alpha
        if self.adaptive_blend and len(new_scores) > 1:
            mean_s = sum(float(x) for x in new_scores) / len(new_scores)
            variance = sum((float(s) - mean_s) ** 2 for s in new_scores) / len(
                new_scores
            )
            std_dev = variance**0.5
            alpha = max(0.0, min(1.0, 0.5 - std_dev * 0.1))

        if alpha > 0.0 and new_scores:
            new_scores = _blend_reranked_scores(texts, results, new_scores, alpha)

        # Re-pack and sort
        reranked = []
        is_meta = len(results[0]) == 3
        for i, new_score in enumerate(new_scores):
            raw = results[i]
            if is_meta:
                raw_meta = cast(MetaResult, raw)
                reranked.append(
                    (raw_meta[0], float(new_score), raw_meta[2])  # type: ignore[misc]
                )
            else:
                reranked.append((raw[0], float(new_score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
