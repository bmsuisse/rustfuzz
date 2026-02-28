"""Tests for rustfuzz.filter, rustfuzz.sort, and the fluent SearchQuery builder.

Uses Faker to generate realistic test data at scale.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from faker import Faker

from rustfuzz.filter import apply_filter, evaluate_filter, parse_filter
from rustfuzz.query import SearchQuery
from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus, HybridSearch
from rustfuzz.sort import apply_sort

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Fixtures: Faker-generated product catalogue ─────────────────

BRANDS = ["Apple", "Samsung", "Google", "Sony", "LG", "Huawei", "Xiaomi", "OnePlus"]
CATEGORIES = ["phone", "tablet", "laptop", "watch", "headphones", "tv", "camera"]
COLORS = ["black", "white", "silver", "gold", "blue", "red", "green"]
TAGS = ["flagship", "budget", "mid-range", "pro", "ultra", "mini", "max", "lite"]


def _make_product(idx: int) -> tuple[str, dict[str, Any]]:
    """Generate a single fake product with realistic metadata."""
    brand = random.choice(BRANDS)
    category = random.choice(CATEGORIES)
    name = f"{brand} {fake.word().capitalize()} {random.choice(TAGS).capitalize()} {category.capitalize()}"
    meta: dict[str, Any] = {
        "brand": brand,
        "category": category,
        "price": round(random.uniform(49.99, 2999.99), 2),
        "year": random.choice([2021, 2022, 2023, 2024]),
        "rating": round(random.uniform(1.0, 5.0), 1),
        "in_stock": random.choice([True, False]),
        "color": random.choice(COLORS),
        "tags": random.sample(TAGS, k=random.randint(1, 3)),
        "specs": {
            "weight_g": random.randint(100, 2000),
            "battery_mah": random.randint(1000, 6000),
        },
        "description": fake.sentence(nb_words=8),
        "reviews": random.randint(0, 5000),
        "discount": random.choice([None, 0.1, 0.15, 0.2, 0.25, 0.3]),
    }
    if random.random() < 0.1:
        meta["brand"] = None  # Some missing brands
    if random.random() < 0.05:
        meta["description"] = ""  # Some empty descriptions
    if random.random() < 0.05:
        meta["tags"] = []  # Some empty tag lists
    return name, meta


def _generate_catalogue(n: int = 200) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate a catalogue of n products."""
    products = [_make_product(i) for i in range(n)]
    corpus = [p[0] for p in products]
    metadata = [p[1] for p in products]
    return corpus, metadata


CORPUS, METADATA = _generate_catalogue(200)


@pytest.fixture()
def bm25() -> BM25:
    return BM25(CORPUS, metadata=METADATA)


@pytest.fixture()
def bm25l() -> BM25L:
    return BM25L(CORPUS, metadata=METADATA)


@pytest.fixture()
def bm25plus() -> BM25Plus:
    return BM25Plus(CORPUS, metadata=METADATA)


@pytest.fixture()
def bm25t() -> BM25T:
    return BM25T(CORPUS, metadata=METADATA)


@pytest.fixture()
def hybrid() -> HybridSearch:
    return HybridSearch(CORPUS, metadata=METADATA)


# ═══════════════════════════════════════════════════════════════
# Filter Parser — Unit Tests
# ═══════════════════════════════════════════════════════════════


class TestFilterParserEquality:
    """Test equality and inequality operators."""

    def test_string_equality(self) -> None:
        ast = parse_filter('brand = "Apple"')
        assert evaluate_filter(ast, {"brand": "Apple"})
        assert not evaluate_filter(ast, {"brand": "Samsung"})

    def test_string_inequality(self) -> None:
        ast = parse_filter('brand != "Apple"')
        assert not evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": "Samsung"})

    def test_numeric_equality(self) -> None:
        ast = parse_filter("year = 2024")
        assert evaluate_filter(ast, {"year": 2024})
        assert not evaluate_filter(ast, {"year": 2023})

    def test_numeric_inequality(self) -> None:
        ast = parse_filter("year != 2024")
        assert evaluate_filter(ast, {"year": 2023})
        assert not evaluate_filter(ast, {"year": 2024})

    def test_boolean_equality(self) -> None:
        ast = parse_filter("in_stock = true")
        assert evaluate_filter(ast, {"in_stock": True})
        assert not evaluate_filter(ast, {"in_stock": False})

    def test_boolean_false(self) -> None:
        ast = parse_filter("in_stock = false")
        assert evaluate_filter(ast, {"in_stock": False})
        assert not evaluate_filter(ast, {"in_stock": True})

    def test_equality_missing_attribute(self) -> None:
        ast = parse_filter('brand = "Apple"')
        assert not evaluate_filter(ast, {"price": 100})

    def test_equality_none_value(self) -> None:
        ast = parse_filter('brand = "Apple"')
        assert not evaluate_filter(ast, {"brand": None})


class TestFilterParserComparison:
    """Test numeric comparison operators."""

    def test_greater_than(self) -> None:
        ast = parse_filter("price > 1000")
        assert evaluate_filter(ast, {"price": 1001})
        assert not evaluate_filter(ast, {"price": 1000})
        assert not evaluate_filter(ast, {"price": 999})

    def test_greater_equal(self) -> None:
        ast = parse_filter("price >= 1000")
        assert evaluate_filter(ast, {"price": 1000})
        assert evaluate_filter(ast, {"price": 1500})
        assert not evaluate_filter(ast, {"price": 999})

    def test_less_than(self) -> None:
        ast = parse_filter("price < 500")
        assert evaluate_filter(ast, {"price": 499})
        assert not evaluate_filter(ast, {"price": 500})
        assert not evaluate_filter(ast, {"price": 501})

    def test_less_equal(self) -> None:
        ast = parse_filter("price <= 500")
        assert evaluate_filter(ast, {"price": 500})
        assert evaluate_filter(ast, {"price": 100})
        assert not evaluate_filter(ast, {"price": 501})

    def test_float_comparison(self) -> None:
        ast = parse_filter("rating > 4.5")
        assert evaluate_filter(ast, {"rating": 4.6})
        assert not evaluate_filter(ast, {"rating": 4.5})

    def test_comparison_missing_attr(self) -> None:
        ast = parse_filter("price > 100")
        assert not evaluate_filter(ast, {"brand": "Apple"})

    def test_negative_numbers(self) -> None:
        ast = parse_filter("temperature > -10")
        assert evaluate_filter(ast, {"temperature": 0})
        assert not evaluate_filter(ast, {"temperature": -20})


class TestFilterParserRange:
    """Test the TO range operator."""

    def test_range_inclusive_both(self) -> None:
        ast = parse_filter("price 100 TO 500")
        assert evaluate_filter(ast, {"price": 100})
        assert evaluate_filter(ast, {"price": 300})
        assert evaluate_filter(ast, {"price": 500})

    def test_range_exclusive_below(self) -> None:
        ast = parse_filter("price 100 TO 500")
        assert not evaluate_filter(ast, {"price": 99})

    def test_range_exclusive_above(self) -> None:
        ast = parse_filter("price 100 TO 500")
        assert not evaluate_filter(ast, {"price": 501})

    def test_range_float(self) -> None:
        ast = parse_filter("rating 3.0 TO 4.5")
        assert evaluate_filter(ast, {"rating": 3.5})
        assert not evaluate_filter(ast, {"rating": 2.9})
        assert not evaluate_filter(ast, {"rating": 4.6})

    def test_range_missing_attr(self) -> None:
        ast = parse_filter("price 100 TO 500")
        assert not evaluate_filter(ast, {"brand": "Apple"})


class TestFilterParserLogicalOps:
    """Test AND, OR, NOT, and parenthesised grouping."""

    def test_and(self) -> None:
        ast = parse_filter('brand = "Apple" AND price > 1000')
        assert evaluate_filter(ast, {"brand": "Apple", "price": 1500})
        assert not evaluate_filter(ast, {"brand": "Apple", "price": 500})
        assert not evaluate_filter(ast, {"brand": "Samsung", "price": 1500})

    def test_or(self) -> None:
        ast = parse_filter('brand = "Apple" OR brand = "Google"')
        assert evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": "Google"})
        assert not evaluate_filter(ast, {"brand": "Samsung"})

    def test_not(self) -> None:
        ast = parse_filter('NOT brand = "Apple"')
        assert not evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": "Samsung"})
        assert evaluate_filter(ast, {"brand": "Google"})

    def test_not_with_comparison(self) -> None:
        ast = parse_filter("NOT price > 1000")
        assert evaluate_filter(ast, {"price": 500})
        assert not evaluate_filter(ast, {"price": 1500})

    def test_and_or_precedence(self) -> None:
        # AND binds tighter than OR
        ast = parse_filter('brand = "Apple" OR brand = "Google" AND price > 1000')
        # Apple matches regardless; Google only if price > 1000
        assert evaluate_filter(ast, {"brand": "Apple", "price": 100})
        assert evaluate_filter(ast, {"brand": "Google", "price": 1500})
        assert not evaluate_filter(ast, {"brand": "Google", "price": 500})

    def test_parenthesised_or_then_and(self) -> None:
        ast = parse_filter('(brand = "Apple" OR brand = "Google") AND price > 500')
        assert evaluate_filter(ast, {"brand": "Apple", "price": 600})
        assert evaluate_filter(ast, {"brand": "Google", "price": 600})
        assert not evaluate_filter(ast, {"brand": "Samsung", "price": 600})
        assert not evaluate_filter(ast, {"brand": "Apple", "price": 100})

    def test_deeply_nested_parens(self) -> None:
        ast = parse_filter(
            '((brand = "Apple" AND price > 500) OR (brand = "Samsung" AND price < 300)) AND in_stock = true'
        )
        assert evaluate_filter(ast, {"brand": "Apple", "price": 600, "in_stock": True})
        assert evaluate_filter(
            ast, {"brand": "Samsung", "price": 200, "in_stock": True}
        )
        assert not evaluate_filter(
            ast, {"brand": "Apple", "price": 600, "in_stock": False}
        )
        assert not evaluate_filter(
            ast, {"brand": "Samsung", "price": 400, "in_stock": True}
        )

    def test_triple_and(self) -> None:
        ast = parse_filter('brand = "Apple" AND price > 500 AND year = 2024')
        assert evaluate_filter(ast, {"brand": "Apple", "price": 600, "year": 2024})
        assert not evaluate_filter(ast, {"brand": "Apple", "price": 600, "year": 2023})

    def test_triple_or(self) -> None:
        ast = parse_filter('brand = "Apple" OR brand = "Samsung" OR brand = "Google"')
        assert evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": "Samsung"})
        assert evaluate_filter(ast, {"brand": "Google"})
        assert not evaluate_filter(ast, {"brand": "Sony"})

    def test_not_or(self) -> None:
        ast = parse_filter('NOT (brand = "Apple" OR brand = "Samsung")')
        assert not evaluate_filter(ast, {"brand": "Apple"})
        assert not evaluate_filter(ast, {"brand": "Samsung"})
        assert evaluate_filter(ast, {"brand": "Google"})


class TestFilterParserSpecialOps:
    """Test IN, EXISTS, IS NULL, IS EMPTY, CONTAINS, STARTS WITH."""

    def test_in_strings(self) -> None:
        ast = parse_filter('brand IN ["Apple", "Google", "Samsung"]')
        assert evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": "Google"})
        assert not evaluate_filter(ast, {"brand": "Sony"})

    def test_in_numbers(self) -> None:
        ast = parse_filter("year IN [2023, 2024]")
        assert evaluate_filter(ast, {"year": 2023})
        assert evaluate_filter(ast, {"year": 2024})
        assert not evaluate_filter(ast, {"year": 2022})

    def test_in_empty_list(self) -> None:
        ast = parse_filter("brand IN []")
        assert not evaluate_filter(ast, {"brand": "Apple"})

    def test_exists_present(self) -> None:
        ast = parse_filter("brand EXISTS")
        assert evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"brand": ""})  # exists but empty
        assert evaluate_filter(ast, {"brand": None})  # exists but null

    def test_exists_missing(self) -> None:
        ast = parse_filter("brand EXISTS")
        assert not evaluate_filter(ast, {"price": 100})
        assert not evaluate_filter(ast, {})

    def test_not_exists(self) -> None:
        ast = parse_filter("NOT brand EXISTS")
        assert not evaluate_filter(ast, {"brand": "Apple"})
        assert evaluate_filter(ast, {"price": 100})

    def test_is_null(self) -> None:
        ast = parse_filter("discount IS NULL")
        assert evaluate_filter(ast, {"discount": None})
        assert not evaluate_filter(ast, {"discount": 0.1})
        assert not evaluate_filter(ast, {"discount": 0})

    def test_is_not_null(self) -> None:
        ast = parse_filter("NOT discount IS NULL")
        assert evaluate_filter(ast, {"discount": 0.1})
        assert not evaluate_filter(ast, {"discount": None})

    def test_is_empty_string(self) -> None:
        ast = parse_filter("description IS EMPTY")
        assert evaluate_filter(ast, {"description": ""})
        assert not evaluate_filter(ast, {"description": "hello"})

    def test_is_empty_list(self) -> None:
        ast = parse_filter("tags IS EMPTY")
        assert evaluate_filter(ast, {"tags": []})
        assert not evaluate_filter(ast, {"tags": ["a"]})

    def test_is_not_empty(self) -> None:
        ast = parse_filter("NOT tags IS EMPTY")
        assert evaluate_filter(ast, {"tags": ["flagship"]})
        assert not evaluate_filter(ast, {"tags": []})


class TestFilterParserDotNotation:
    """Test nested attribute access via dot notation."""

    def test_dot_equality(self) -> None:
        ast = parse_filter("specs.weight_g > 500")
        assert evaluate_filter(ast, {"specs": {"weight_g": 600}})
        assert not evaluate_filter(ast, {"specs": {"weight_g": 400}})

    def test_dot_deep_nesting(self) -> None:
        ast = parse_filter('a.b.c = "deep"')
        assert evaluate_filter(ast, {"a": {"b": {"c": "deep"}}})
        assert not evaluate_filter(ast, {"a": {"b": {"c": "shallow"}}})

    def test_dot_missing_intermediate(self) -> None:
        ast = parse_filter("specs.battery_mah > 3000")
        assert not evaluate_filter(ast, {"specs": None})
        assert not evaluate_filter(ast, {})


class TestFilterParserEdgeCases:
    """Edge cases and error handling."""

    def test_empty_metadata(self) -> None:
        ast = parse_filter("price > 100")
        assert not evaluate_filter(ast, {})

    def test_complex_realistic(self) -> None:
        """A real-world-style complex filter expression."""
        ast = parse_filter(
            '(brand = "Apple" OR brand = "Samsung") AND price 500 TO 2000 AND in_stock = true AND NOT category = "tv"'
        )
        assert evaluate_filter(
            ast,
            {"brand": "Apple", "price": 1000, "in_stock": True, "category": "phone"},
        )
        assert not evaluate_filter(
            ast, {"brand": "Apple", "price": 1000, "in_stock": True, "category": "tv"}
        )
        assert not evaluate_filter(
            ast, {"brand": "Sony", "price": 1000, "in_stock": True, "category": "phone"}
        )

    @pytest.mark.parametrize("val", [0, 0.0, False, "", []])
    def test_falsy_values_not_null(self, val: Any) -> None:
        """Falsy values should NOT be treated as null."""
        ast = parse_filter("NOT x IS NULL")
        assert evaluate_filter(ast, {"x": val})


# ═══════════════════════════════════════════════════════════════
# Sort — Unit Tests
# ═══════════════════════════════════════════════════════════════


class TestSortBasic:
    def test_sort_asc_numeric(self) -> None:
        results = [
            ("c", 0.7, {"price": 300}),
            ("a", 0.9, {"price": 100}),
            ("b", 0.8, {"price": 200}),
        ]
        sorted_r = apply_sort(results, "price:asc")
        assert [r[2]["price"] for r in sorted_r] == [100, 200, 300]

    def test_sort_desc_numeric(self) -> None:
        results = [
            ("a", 0.9, {"price": 100}),
            ("c", 0.7, {"price": 300}),
            ("b", 0.8, {"price": 200}),
        ]
        sorted_r = apply_sort(results, "price:desc")
        assert [r[2]["price"] for r in sorted_r] == [300, 200, 100]

    def test_sort_string_asc(self) -> None:
        results = [
            ("b", 0.8, {"name": "Banana"}),
            ("c", 0.7, {"name": "Cherry"}),
            ("a", 0.9, {"name": "Apple"}),
        ]
        sorted_r = apply_sort(results, "name:asc")
        assert [r[2]["name"] for r in sorted_r] == ["Apple", "Banana", "Cherry"]

    def test_sort_string_desc(self) -> None:
        results = [
            ("a", 0.9, {"name": "Apple"}),
            ("b", 0.8, {"name": "Banana"}),
            ("c", 0.7, {"name": "Cherry"}),
        ]
        sorted_r = apply_sort(results, "name:desc")
        assert [r[2]["name"] for r in sorted_r] == ["Cherry", "Banana", "Apple"]

    def test_sort_none_noop(self) -> None:
        results = [("a", 0.9), ("b", 0.8)]
        assert apply_sort(results, None) is results

    def test_sort_empty_results(self) -> None:
        assert apply_sort([], "price:asc") == []

    def test_sort_default_asc(self) -> None:
        """No direction specified defaults to asc."""
        results = [
            ("c", 0.7, {"price": 300}),
            ("a", 0.9, {"price": 100}),
        ]
        sorted_r = apply_sort(results, "price")
        assert [r[2]["price"] for r in sorted_r] == [100, 300]


class TestSortMultiKey:
    def test_two_keys(self) -> None:
        results = [
            ("a1", 0.9, {"brand": "Apple", "price": 200}),
            ("a2", 0.8, {"brand": "Apple", "price": 100}),
            ("s1", 0.7, {"brand": "Samsung", "price": 50}),
        ]
        sorted_r = apply_sort(results, ["brand:asc", "price:asc"])
        assert sorted_r[0][0] == "a2"  # Apple, 100
        assert sorted_r[1][0] == "a1"  # Apple, 200
        assert sorted_r[2][0] == "s1"  # Samsung, 50

    def test_mixed_directions(self) -> None:
        results = [
            ("a1", 0.9, {"brand": "Apple", "price": 200}),
            ("a2", 0.8, {"brand": "Apple", "price": 100}),
            ("s1", 0.7, {"brand": "Samsung", "price": 50}),
        ]
        sorted_r = apply_sort(results, ["brand:asc", "price:desc"])
        assert sorted_r[0][0] == "a1"  # Apple, 200 (desc)
        assert sorted_r[1][0] == "a2"  # Apple, 100
        assert sorted_r[2][0] == "s1"  # Samsung

    def test_comma_separated_string(self) -> None:
        results = [
            ("a1", 0.9, {"x": 1, "y": 2}),
            ("a2", 0.8, {"x": 1, "y": 1}),
            ("a3", 0.7, {"x": 2, "y": 3}),
        ]
        sorted_r = apply_sort(results, "x:asc, y:desc")
        assert sorted_r[0][0] == "a1"  # x=1, y=2
        assert sorted_r[1][0] == "a2"  # x=1, y=1
        assert sorted_r[2][0] == "a3"  # x=2


class TestSortMissingValues:
    def test_missing_attr_goes_to_end_asc(self) -> None:
        results = [
            ("a", 0.9, {"price": 100}),
            ("b", 0.8, {}),
            ("c", 0.7, {"price": 50}),
        ]
        sorted_r = apply_sort(results, "price:asc")
        assert sorted_r[0][2]["price"] == 50
        assert sorted_r[1][2]["price"] == 100
        assert sorted_r[2][2] == {}  # missing at end

    def test_missing_attr_goes_to_end_desc(self) -> None:
        results = [
            ("b", 0.8, {}),
            ("a", 0.9, {"price": 100}),
            ("c", 0.7, {"price": 50}),
        ]
        sorted_r = apply_sort(results, "price:desc")
        assert sorted_r[0][2]["price"] == 100
        assert sorted_r[1][2]["price"] == 50
        assert sorted_r[2][2] == {}  # missing at end

    def test_null_value_goes_to_end(self) -> None:
        results = [
            ("a", 0.9, {"price": 100}),
            ("b", 0.8, {"price": None}),
            ("c", 0.7, {"price": 50}),
        ]
        sorted_r = apply_sort(results, "price:asc")
        assert sorted_r[2][2]["price"] is None

    def test_no_metadata_results(self) -> None:
        results = [("a", 0.9), ("b", 0.8)]
        sorted_r = apply_sort(results, "price:asc")
        assert len(sorted_r) == 2


class TestSortDotNotation:
    def test_sort_nested_attr(self) -> None:
        results = [
            ("a", 0.9, {"specs": {"weight_g": 300}}),
            ("b", 0.8, {"specs": {"weight_g": 100}}),
            ("c", 0.7, {"specs": {"weight_g": 200}}),
        ]
        sorted_r = apply_sort(results, "specs.weight_g:asc")
        weights = [r[2]["specs"]["weight_g"] for r in sorted_r]
        assert weights == [100, 200, 300]


# ═══════════════════════════════════════════════════════════════
# apply_filter — Integration Tests
# ═══════════════════════════════════════════════════════════════


class TestApplyFilter:
    def test_filter_single_condition(self) -> None:
        results = [
            ("d1", 0.9, {"brand": "Apple", "price": 1199}),
            ("d2", 0.8, {"brand": "Samsung", "price": 1099}),
            ("d3", 0.7, {"brand": "Apple", "price": 599}),
        ]
        filtered = apply_filter(results, 'brand = "Apple"')
        assert len(filtered) == 2
        assert all(r[2]["brand"] == "Apple" for r in filtered)

    def test_filter_complex(self) -> None:
        results = [
            ("d1", 0.9, {"brand": "Apple", "price": 1199, "in_stock": True}),
            ("d2", 0.8, {"brand": "Apple", "price": 599, "in_stock": False}),
            ("d3", 0.7, {"brand": "Samsung", "price": 1099, "in_stock": True}),
        ]
        filtered = apply_filter(
            results, 'brand = "Apple" AND price > 1000 AND in_stock = true'
        )
        assert len(filtered) == 1
        assert filtered[0][0] == "d1"

    def test_filter_preserves_order(self) -> None:
        results = [
            ("d1", 0.9, {"x": 1}),
            ("d2", 0.8, {"x": 2}),
            ("d3", 0.7, {"x": 3}),
        ]
        filtered = apply_filter(results, "x > 1")
        assert [r[0] for r in filtered] == ["d2", "d3"]

    def test_filter_empty_results(self) -> None:
        filtered = apply_filter([], 'brand = "Apple"')
        assert filtered == []

    def test_filter_no_matches(self) -> None:
        results = [("d1", 0.9, {"brand": "Sony"})]
        filtered = apply_filter(results, 'brand = "Apple"')
        assert filtered == []

    def test_filter_with_faker_data(self) -> None:
        """Filter through Faker-generated products."""
        results = [(name, 1.0, meta) for name, meta in zip(CORPUS[:50], METADATA[:50])]
        filtered = apply_filter(results, "price > 1000")
        for _, _, meta in filtered:
            assert meta["price"] > 1000


# ═══════════════════════════════════════════════════════════════
# SearchQuery Builder — Integration Tests with Faker data
# ═══════════════════════════════════════════════════════════════


class TestSearchQueryBuilder:
    """Core builder API tests."""

    def test_filter_returns_search_query(self, bm25: BM25) -> None:
        sq = bm25.filter('brand = "Apple"')
        assert isinstance(sq, SearchQuery)

    def test_sort_returns_search_query(self, bm25: BM25) -> None:
        sq = bm25.sort("price:asc")
        assert isinstance(sq, SearchQuery)

    def test_chaining_filter_then_sort(self, bm25: BM25) -> None:
        sq = bm25.filter('brand = "Apple"').sort("price:asc")
        assert isinstance(sq, SearchQuery)

    def test_chaining_filter_search_collect(self, bm25: BM25) -> None:
        results = bm25.filter("in_stock = true").search("phone", n=10).collect()
        assert isinstance(results, list)
        for _, _, meta in results:
            assert meta["in_stock"] is True

    def test_collect_without_search_raises(self, bm25: BM25) -> None:
        with pytest.raises(ValueError, match="No search query set"):
            bm25.filter('brand = "Apple"').collect()

    def test_match_alias(self, bm25: BM25) -> None:
        results = bm25.filter("in_stock = true").match("phone", n=5)
        assert len(results) > 0

    def test_repr(self, bm25: BM25) -> None:
        sq = bm25.filter('brand = "Apple"').sort("price:asc")
        r = repr(sq)
        assert "SearchQuery" in r
        assert "filters=1" in r


class TestSearchQueryFiltering:
    """Filter expression tests with real BM25 indices."""

    def test_filter_brand(self, bm25: BM25) -> None:
        results = bm25.filter('brand = "Samsung"').get_top_n("Samsung", n=20)
        assert len(results) > 0
        for _, _, meta in results:
            assert meta["brand"] == "Samsung"

    def test_filter_price_range(self, bm25: BM25) -> None:
        results = bm25.filter("price 100 TO 500").get_top_n("pro", n=20)
        for _, _, meta in results:
            assert 100 <= meta["price"] <= 500

    def test_filter_not(self, bm25: BM25) -> None:
        results = bm25.filter('NOT brand = "Apple"').get_top_n("Samsung", n=20)
        for _, _, meta in results:
            assert meta["brand"] != "Apple"

    def test_filter_or(self, bm25: BM25) -> None:
        results = bm25.filter('brand = "Apple" OR brand = "Google"').get_top_n(
            "pro", n=20
        )
        for _, _, meta in results:
            assert meta["brand"] in ("Apple", "Google")

    def test_filter_in(self, bm25: BM25) -> None:
        results = bm25.filter('category IN ["phone", "tablet"]').get_top_n("pro", n=20)
        for _, _, meta in results:
            assert meta["category"] in ("phone", "tablet")

    def test_filter_comparison(self, bm25: BM25) -> None:
        results = bm25.filter("rating >= 4.0").get_top_n("pro", n=20)
        for _, _, meta in results:
            assert meta["rating"] >= 4.0

    def test_filter_boolean(self, bm25: BM25) -> None:
        results = bm25.filter("in_stock = true").get_top_n("pro", n=20)
        for _, _, meta in results:
            assert meta["in_stock"] is True

    def test_filter_complex_and_or(self, bm25: BM25) -> None:
        results = bm25.filter(
            '(brand = "Apple" OR brand = "Samsung") AND price > 500'
        ).get_top_n("pro", n=20)
        for _, _, meta in results:
            assert meta["brand"] in ("Apple", "Samsung")
            assert meta["price"] > 500

    def test_multiple_chained_filters(self, bm25: BM25) -> None:
        results = (
            bm25.filter('brand = "Samsung"')
            .filter("price < 1000")
            .filter("in_stock = true")
            .get_top_n("Samsung", n=20)
        )
        for _, _, meta in results:
            assert meta["brand"] == "Samsung"
            assert meta["price"] < 1000
            assert meta["in_stock"] is True

    def test_filter_dot_notation(self, bm25: BM25) -> None:
        results = bm25.filter("specs.weight_g < 500").get_top_n("phone", n=20)
        for _, _, meta in results:
            assert meta["specs"]["weight_g"] < 500


class TestSearchQuerySorting:
    """Sort tests with real BM25 indices."""

    def test_sort_price_asc(self, bm25: BM25) -> None:
        results = bm25.sort("price:asc").get_top_n("phone", n=10)
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices)

    def test_sort_price_desc(self, bm25: BM25) -> None:
        results = bm25.sort("price:desc").get_top_n("phone", n=10)
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices, reverse=True)

    def test_sort_rating_desc(self, bm25: BM25) -> None:
        results = bm25.sort("rating:desc").get_top_n("phone", n=10)
        ratings = [meta["rating"] for _, _, meta in results]
        assert ratings == sorted(ratings, reverse=True)

    def test_filter_then_sort(self, bm25: BM25) -> None:
        results = (
            bm25.filter("in_stock = true").sort("price:asc").get_top_n("phone", n=10)
        )
        for _, _, meta in results:
            assert meta["in_stock"] is True
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices)

    def test_sort_multi_key(self, bm25: BM25) -> None:
        results = bm25.sort(["brand:asc", "price:desc"]).get_top_n("phone", n=20)
        # Verify that within each brand, prices are descending
        if len(results) >= 2:
            for i in range(len(results) - 1):
                _, _, meta_a = results[i]
                _, _, meta_b = results[i + 1]
                if meta_a["brand"] == meta_b["brand"]:
                    assert meta_a["price"] >= meta_b["price"]


class TestSearchQueryMethods:
    """Test different search methods through the builder."""

    def test_get_top_n(self, bm25: BM25) -> None:
        results = bm25.filter("in_stock = true").get_top_n("phone", n=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_get_top_n_rrf(self, bm25: BM25) -> None:
        results = bm25.filter("in_stock = true").get_top_n_rrf("phone", n=5)
        assert len(results) <= 5
        assert len(results) > 0
        for _, _, meta in results:
            assert meta["in_stock"] is True

    def test_search_collect_method(self, bm25: BM25) -> None:
        results = (
            bm25.filter("price > 500").search("pro", n=5, method="get_top_n").collect()
        )
        assert len(results) > 0

    def test_search_with_rrf_method(self, bm25: BM25) -> None:
        results = (
            bm25.filter("price > 200")
            .search("phone", n=5, method="get_top_n_rrf")
            .collect()
        )
        assert len(results) > 0


class TestSearchQueryNoMetadata:
    """Behaviour when metadata is absent."""

    def test_filter_without_metadata_returns_unfiltered(self) -> None:
        bm25 = BM25(CORPUS)
        results = bm25.filter('brand = "Apple"').get_top_n("phone", n=5)
        # Without metadata, mask is None — Rust returns all matches
        for r in results:
            assert len(r) == 2  # (text, score) — no metadata

    def test_sort_without_metadata_preserves_relevance(self) -> None:
        bm25 = BM25(CORPUS)
        results = bm25.sort("price:asc").get_top_n("phone", n=5)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════
# All BM25 Variants — Parametrised tests
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("cls", [BM25, BM25L, BM25Plus, BM25T])
class TestAllBM25Variants:
    def test_has_filter_and_sort(self, cls: type) -> None:
        assert hasattr(cls, "filter")
        assert hasattr(cls, "sort")

    def test_filter_returns_builder(self, cls: type) -> None:
        bm25 = cls(CORPUS, metadata=METADATA)
        sq = bm25.filter('brand = "Apple"')
        assert isinstance(sq, SearchQuery)

    def test_filter_get_top_n(self, cls: type) -> None:
        bm25 = cls(CORPUS, metadata=METADATA)
        results = bm25.filter('brand = "Samsung"').get_top_n("Samsung", n=5)
        assert len(results) > 0
        for _, _, meta in results:
            assert meta["brand"] == "Samsung"

    def test_sort_get_top_n(self, cls: type) -> None:
        bm25 = cls(CORPUS, metadata=METADATA)
        results = bm25.sort("price:asc").get_top_n("phone", n=5)
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices)

    def test_filter_sort_combo(self, cls: type) -> None:
        bm25 = cls(CORPUS, metadata=METADATA)
        results = (
            bm25.filter("in_stock = true").sort("price:desc").get_top_n("pro", n=5)
        )
        for _, _, meta in results:
            assert meta["in_stock"] is True
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices, reverse=True)

    def test_filter_rrf(self, cls: type) -> None:
        bm25 = cls(CORPUS, metadata=METADATA)
        results = bm25.filter("price > 500").get_top_n_rrf("pro", n=5)
        assert len(results) > 0
        for _, _, meta in results:
            assert meta["price"] > 500


# ═══════════════════════════════════════════════════════════════
# HybridSearch — Filter & Sort tests
# ═══════════════════════════════════════════════════════════════


class TestHybridSearchFilterSort:
    def test_has_filter_and_sort(self) -> None:
        assert hasattr(HybridSearch, "filter")
        assert hasattr(HybridSearch, "sort")

    def test_filter_get_top_n(self, hybrid: HybridSearch) -> None:
        results = hybrid.filter('brand = "Apple"').get_top_n("Apple", n=5)
        assert len(results) > 0
        for _, _, meta in results:
            assert meta["brand"] == "Apple"

    def test_sort_get_top_n(self, hybrid: HybridSearch) -> None:
        results = hybrid.sort("price:asc").get_top_n("phone", n=5)
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices)

    def test_filter_sort_combo(self, hybrid: HybridSearch) -> None:
        results = (
            hybrid.filter("in_stock = true").sort("price:desc").get_top_n("phone", n=5)
        )
        for _, _, meta in results:
            assert meta["in_stock"] is True
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices, reverse=True)


# ═══════════════════════════════════════════════════════════════
# Scale Tests — Faker-generated large data
# ═══════════════════════════════════════════════════════════════


class TestFilterAtScale:
    """Test filtering works correctly on larger datasets."""

    def test_filter_accuracy_on_200_docs(self, bm25: BM25) -> None:
        """Verify all results from filtered search actually match the filter."""
        results = bm25.filter("price > 1500").get_top_n("pro", n=50)
        for _, _, meta in results:
            assert meta["price"] > 1500

    def test_filter_count_consistency(self, bm25: BM25) -> None:
        """Filtered results should never exceed total matching docs."""
        matching_count = sum(1 for m in METADATA if m["in_stock"] is True)
        results = bm25.filter("in_stock = true").get_top_n("phone", n=500)
        assert len(results) <= matching_count

    def test_filter_and_sort_large(self, bm25: BM25) -> None:
        results = bm25.filter("price > 200").sort("price:asc").get_top_n("phone", n=50)
        prices = [meta["price"] for _, _, meta in results]
        assert prices == sorted(prices)
        assert all(p > 200 for p in prices)

    def test_no_filter_returns_more_results(self, bm25: BM25) -> None:
        """Filtering should reduce result count."""
        unfiltered = bm25.get_top_n("phone", n=50)
        filtered = bm25.filter("price > 2000").get_top_n("phone", n=50)
        assert len(filtered) <= len(unfiltered)

    def test_exclusive_filter_returns_empty(self, bm25: BM25) -> None:
        """A filter that no doc satisfies should return empty."""
        results = bm25.filter("price > 99999").get_top_n("phone", n=10)
        assert results == []


# ═══════════════════════════════════════════════════════════════
# Faker helpers — Property-based style tests
# ═══════════════════════════════════════════════════════════════


class TestFilterSortProperties:
    """Property-style tests using random Faker data."""

    def test_filter_then_check_all_pass(self, bm25: BM25) -> None:
        """For any numeric threshold, all returned results must pass."""
        threshold = random.uniform(200, 2000)
        results = bm25.filter(f"price > {threshold}").get_top_n("pro", n=50)
        for _, _, meta in results:
            assert meta["price"] > threshold

    def test_sort_is_stable(self, bm25: BM25) -> None:
        """Sorting the same results twice should give the same order."""
        results1 = bm25.sort("price:asc").get_top_n("phone", n=10)
        results2 = bm25.sort("price:asc").get_top_n("phone", n=10)
        assert [r[0] for r in results1] == [r[0] for r in results2]

    def test_filter_is_idempotent(self, bm25: BM25) -> None:
        """Applying the same filter twice should give the same results."""
        results1 = bm25.filter("price > 500").get_top_n("phone", n=10)
        results2 = (
            bm25.filter("price > 500").filter("price > 500").get_top_n("phone", n=10)
        )
        assert [r[0] for r in results1] == [r[0] for r in results2]

    @pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
    def test_filter_by_year(self, bm25: BM25, year: int) -> None:
        results = bm25.filter(f"year = {year}").get_top_n("phone", n=50)
        for _, _, meta in results:
            assert meta["year"] == year

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_filter_by_category(self, bm25: BM25, category: str) -> None:
        results = bm25.filter(f'category = "{category}"').get_top_n("pro", n=50)
        for _, _, meta in results:
            assert meta["category"] == category
