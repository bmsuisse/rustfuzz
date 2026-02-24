"""
rustfuzz.spark — PySpark UDF factories for fuzzy string matching.

All UDF factories return ``pyspark.sql.functions.udf`` wrappers around
stateless ``rustfuzz`` functions. PySpark is imported lazily so this module
can be imported without pyspark installed (will raise at call time).

Usage
-----
>>> from rustfuzz.spark import ratio_udf, levenshtein_distance_udf
>>> df = df.withColumn("score", ratio_udf()(col("a"), col("b")))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql.functions import UserDefinedFunction


def _make_udf(
    fn: Callable[..., Any],
    return_type: str = "double",
) -> Callable[..., UserDefinedFunction]:
    """Create a factory that returns a PySpark UDF wrapping *fn*."""

    def factory() -> UserDefinedFunction:
        from pyspark.sql.functions import udf as spark_udf
        from pyspark.sql.types import (
            DoubleType,
            IntegerType,
        )

        type_map: dict[str, Any] = {
            "double": DoubleType(),
            "int": IntegerType(),
        }
        return spark_udf(fn, returnType=type_map[return_type])

    factory.__name__ = f"{fn.__name__}_udf"  # type: ignore[attr-defined]
    factory.__doc__ = (
        f"Return a PySpark UDF wrapping ``rustfuzz`` "
        f"``{fn.__module__}.{fn.__name__}``.\n\n"
        f"Call this factory once, then apply the returned UDF to DataFrame columns."
    )
    return factory


# ---------------------------------------------------------------------------
# Fuzzy ratio UDFs
# ---------------------------------------------------------------------------


def ratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.ratio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.ratio, returnType=DoubleType())


def partial_ratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.partial_ratio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.partial_ratio, returnType=DoubleType())


def token_sort_ratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.token_sort_ratio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.token_sort_ratio, returnType=DoubleType())


def token_set_ratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.token_set_ratio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.token_set_ratio, returnType=DoubleType())


def wratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.WRatio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.WRatio, returnType=DoubleType())


def qratio_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.fuzz.QRatio(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz import fuzz

    return spark_udf(fuzz.QRatio, returnType=DoubleType())


# ---------------------------------------------------------------------------
# Distance UDFs
# ---------------------------------------------------------------------------


def levenshtein_distance_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.distance.Levenshtein.distance(s1, s2) -> int``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import IntegerType

    from rustfuzz.distance import Levenshtein

    return spark_udf(Levenshtein.distance, returnType=IntegerType())


def levenshtein_similarity_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.distance.Levenshtein.similarity(s1, s2) -> int``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import IntegerType

    from rustfuzz.distance import Levenshtein

    return spark_udf(Levenshtein.similarity, returnType=IntegerType())


def levenshtein_normalized_similarity_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.distance.Levenshtein.normalized_similarity(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz.distance import Levenshtein

    return spark_udf(Levenshtein.normalized_similarity, returnType=DoubleType())


def jaro_similarity_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.distance.Jaro.similarity(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz.distance import Jaro

    return spark_udf(Jaro.similarity, returnType=DoubleType())


def jaro_winkler_similarity_udf() -> UserDefinedFunction:
    """PySpark UDF for ``rustfuzz.distance.JaroWinkler.similarity(s1, s2) -> float``."""
    from pyspark.sql.functions import udf as spark_udf
    from pyspark.sql.types import DoubleType

    from rustfuzz.distance import JaroWinkler

    return spark_udf(JaroWinkler.similarity, returnType=DoubleType())


__all__ = [
    "ratio_udf",
    "partial_ratio_udf",
    "token_sort_ratio_udf",
    "token_set_ratio_udf",
    "wratio_udf",
    "qratio_udf",
    "levenshtein_distance_udf",
    "levenshtein_similarity_udf",
    "levenshtein_normalized_similarity_udf",
    "jaro_similarity_udf",
    "jaro_winkler_similarity_udf",
]
