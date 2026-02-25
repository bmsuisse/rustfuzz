"""
rustfuzz.compat — Data-framework compatibility helpers.

Converts column-oriented data from Polars, Pandas, PyArrow, and PySpark
into ``list[str]`` using each framework's fastest native path.
All imports are lazy so no new hard dependencies are introduced.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _is_polars_series(data: Any) -> bool:
    try:
        import polars as pl

        return isinstance(data, pl.Series)
    except ImportError:
        return False


def _is_pandas_series(data: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(data, pd.Series)
    except ImportError:
        return False


def _is_pyarrow_array(data: Any) -> bool:
    try:
        import pyarrow as pa

        return isinstance(data, (pa.Array, pa.ChunkedArray))
    except ImportError:
        return False


def _is_spark_dataframe(data: Any) -> bool:
    try:
        from pyspark.sql import DataFrame as SparkDataFrame

        return isinstance(data, SparkDataFrame)
    except ImportError:
        return False


def _coerce_to_strings(data: Any) -> list[str]:
    """
    Convert *data* to a ``list[str]`` using the fastest path available.

    Supported input types
    ---------------------
    * ``list[str]`` — returned as-is (no copy).
    * ``polars.Series`` — ``.cast(Utf8).to_list()``.
    * ``pandas.Series`` — ``.astype(str).tolist()``.
    * ``pyarrow.Array`` / ``pyarrow.ChunkedArray`` — ``.to_pylist()``.
    * ``pyspark.sql.DataFrame`` (single-column) — collected via RDD fast path.
    * Any other ``Iterable[str]`` — ``list(data)``.

    Raises
    ------
    TypeError
        If the resulting list contains non-string elements.
    """
    if isinstance(data, list):
        if data and not isinstance(data[0], str):
            raise TypeError(
                f"Expected list[str], got list[{type(data[0]).__name__}]. "
                "All corpus elements must be strings."
            )
        return data

    if _is_polars_series(data):
        import polars as pl

        s: pl.Series = data.cast(pl.Utf8)
        result = s.to_list()
        return [x if x is not None else "" for x in result]

    if _is_pandas_series(data):
        return data.astype(str).tolist()  # type: ignore[union-attr]

    if _is_pyarrow_array(data):
        result = data.to_pylist()  # type: ignore[union-attr]
        return [x if x is not None else "" for x in result]

    if _is_spark_dataframe(data):
        cols = data.columns  # type: ignore[union-attr]
        if len(cols) != 1:
            raise ValueError(
                f"Expected a single-column Spark DataFrame, got {len(cols)} columns: {cols}. "
                "Use BM25.from_column(df, 'column_name') instead."
            )
        return [row[0] for row in data.collect()]  # type: ignore[union-attr]

    if isinstance(data, Iterable):
        result: list[Any] = list(data)
        if result and not isinstance(result[0], str):
            raise TypeError(
                f"Expected Iterable[str], got elements of type {type(result[0]).__name__}. "
                "All corpus elements must be strings."
            )
        return result  # type: ignore[return-value]

    raise TypeError(
        f"Cannot coerce {type(data).__name__} to list[str]. "
        "Pass a list, Polars Series, Pandas Series, PyArrow Array, "
        "or a single-column Spark DataFrame."
    )


def _extract_column(df: Any, column: str) -> list[str]:
    """
    Extract a named column from a DataFrame as ``list[str]``.

    Supports Polars, Pandas, and PySpark DataFrames.
    """
    if _is_spark_dataframe(df):
        return [row[0] for row in df.select(column).collect()]  # type: ignore[union-attr]

    try:
        col_data = df[column]
    except (KeyError, TypeError) as e:
        raise TypeError(
            f"Cannot extract column '{column}' from {type(df).__name__}."
        ) from e

    return _coerce_to_strings(col_data)


def _extract_metadata(
    df: Any, columns: list[str] | str
) -> list[dict[str, Any]]:
    """
    Extract per-row metadata dicts from one or more DataFrame columns.

    Parameters
    ----------
    df : DataFrame (Polars, Pandas, or PySpark)
    columns : str or list[str]
        Column name(s) to include in each metadata dict.

    Returns
    -------
    list[dict[str, Any]]
        One dict per row, keyed by column name.
    """
    if isinstance(columns, str):
        columns = [columns]

    if _is_spark_dataframe(df):
        rows = df.select(*columns).collect()  # type: ignore[union-attr]
        return [{col: row[col] for col in columns} for row in rows]

    # Polars / Pandas — both support df[col].to_list() or similar
    n_rows: int = len(df)
    col_data: dict[str, list[Any]] = {}
    for col in columns:
        try:
            series = df[col]
        except (KeyError, TypeError) as e:
            raise TypeError(
                f"Cannot extract column '{col}' from {type(df).__name__}."
            ) from e
        if hasattr(series, "to_list"):
            col_data[col] = series.to_list()
        elif hasattr(series, "tolist"):
            col_data[col] = series.tolist()
        else:
            col_data[col] = list(series)

    return [{col: col_data[col][i] for col in columns} for i in range(n_rows)]


__all__ = ["_coerce_to_strings", "_extract_column", "_extract_metadata"]
