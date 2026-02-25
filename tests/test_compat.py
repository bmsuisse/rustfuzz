"""Tests for rustfuzz.compat — data-framework integration."""

from __future__ import annotations

import pytest

from rustfuzz.compat import _coerce_to_strings, _extract_column
from rustfuzz.search import BM25, BM25L, BM25Plus, BM25T

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
    "a lazy dog",
    "the fast brown fox",
    "jumping over dogs",
]


class TestCoerceToStrings:
    def test_plain_list_passthrough(self) -> None:
        data = ["a", "b", "c"]
        result = _coerce_to_strings(data)
        assert result is data

    def test_generator(self) -> None:
        def gen() -> ...:
            yield "hello"
            yield "world"

        result = _coerce_to_strings(gen())
        assert result == ["hello", "world"]

    def test_empty_list(self) -> None:
        assert _coerce_to_strings([]) == []

    def test_non_string_list_raises(self) -> None:
        with pytest.raises(TypeError, match="list\\[str\\]"):
            _coerce_to_strings([1, 2, 3])

    def test_non_string_iterable_raises(self) -> None:
        with pytest.raises(TypeError, match="Iterable\\[str\\]"):
            _coerce_to_strings(iter([1, 2, 3]))

    def test_tuple_of_strings(self) -> None:
        result = _coerce_to_strings(("a", "b"))
        assert result == ["a", "b"]


class TestPolarsIntegration:
    @pytest.fixture()
    def pl(self) -> ...:
        return pytest.importorskip("polars")

    def test_polars_series(self, pl: ...) -> None:
        s = pl.Series("name", ["apple", "banana", "cherry"])
        result = _coerce_to_strings(s)
        assert result == ["apple", "banana", "cherry"]

    def test_polars_series_with_none(self, pl: ...) -> None:
        s = pl.Series("name", ["apple", None, "cherry"])
        result = _coerce_to_strings(s)
        assert result == ["apple", "", "cherry"]

    def test_bm25_with_polars_series(self, pl: ...) -> None:
        s = pl.Series("docs", CORPUS)
        bm25 = BM25(s)
        assert bm25.num_docs == 5
        scores = bm25.get_scores("fox")
        assert scores[0] > 0
        assert scores[2] == 0

    def test_bm25_from_column_polars(self, pl: ...) -> None:
        df = pl.DataFrame({"name": CORPUS, "id": list(range(5))})
        bm25 = BM25.from_column(df, "name")
        assert bm25.num_docs == 5
        top = bm25.get_top_n("fox", n=2)
        assert len(top) == 2

    def test_bm25l_from_column_polars(self, pl: ...) -> None:
        df = pl.DataFrame({"name": CORPUS})
        bm25l = BM25L.from_column(df, "name", delta=0.5)
        assert bm25l.num_docs == 5

    def test_bm25plus_from_column_polars(self, pl: ...) -> None:
        df = pl.DataFrame({"name": CORPUS})
        bm25p = BM25Plus.from_column(df, "name", delta=1.0)
        assert bm25p.num_docs == 5

    def test_bm25t_from_column_polars(self, pl: ...) -> None:
        df = pl.DataFrame({"name": CORPUS})
        bm25t = BM25T.from_column(df, "name")
        assert bm25t.num_docs == 5

    def test_multijoiner_with_polars(self, pl: ...) -> None:
        from rustfuzz.join import MultiJoiner

        s1 = pl.Series("a", ["Apple Inc.", "Microsoft Corp"])
        s2 = pl.Series("b", ["APPLE INC", "MSFT Corporation"])
        joiner = MultiJoiner()
        joiner.add_array("source", texts=s1)
        joiner.add_array("target", texts=s2)
        results = joiner.join(n=1)
        assert len(results) > 0


class TestPandasIntegration:
    @pytest.fixture()
    def pd(self) -> ...:
        return pytest.importorskip("pandas")

    def test_pandas_series(self, pd: ...) -> None:
        s = pd.Series(["apple", "banana", "cherry"])
        result = _coerce_to_strings(s)
        assert result == ["apple", "banana", "cherry"]

    def test_bm25_with_pandas_series(self, pd: ...) -> None:
        s = pd.Series(CORPUS)
        bm25 = BM25(s)
        assert bm25.num_docs == 5
        top = bm25.get_top_n("quick fox", n=2)
        assert len(top) == 2

    def test_bm25_from_column_pandas(self, pd: ...) -> None:
        df = pd.DataFrame({"name": CORPUS})
        bm25 = BM25.from_column(df, "name")
        assert bm25.num_docs == 5


class TestPyArrowIntegration:
    @pytest.fixture()
    def pa(self) -> ...:
        return pytest.importorskip("pyarrow")

    def test_pyarrow_array(self, pa: ...) -> None:
        arr = pa.array(["apple", "banana", "cherry"])
        result = _coerce_to_strings(arr)
        assert result == ["apple", "banana", "cherry"]

    def test_pyarrow_chunked_array(self, pa: ...) -> None:
        arr1 = pa.array(["apple", "banana"])
        arr2 = pa.array(["cherry"])
        chunked = pa.chunked_array([arr1, arr2])
        result = _coerce_to_strings(chunked)
        assert result == ["apple", "banana", "cherry"]

    def test_pyarrow_with_nulls(self, pa: ...) -> None:
        arr = pa.array(["apple", None, "cherry"])
        result = _coerce_to_strings(arr)
        assert result == ["apple", "", "cherry"]

    def test_bm25_with_pyarrow(self, pa: ...) -> None:
        arr = pa.array(CORPUS)
        bm25 = BM25(arr)
        assert bm25.num_docs == 5
        scores = bm25.get_scores("fox")
        assert scores[0] > 0


class TestExtractColumn:
    def test_extract_from_polars(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"name": ["a", "b"], "id": [1, 2]})
        result = _extract_column(df, "name")
        assert result == ["a", "b"]

    def test_extract_from_pandas(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"name": ["a", "b"], "id": [1, 2]})
        result = _extract_column(df, "name")
        assert result == ["a", "b"]

    def test_extract_invalid_column(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"name": ["a"]})
        with pytest.raises((TypeError, Exception)):
            _extract_column(df, "nonexistent")
