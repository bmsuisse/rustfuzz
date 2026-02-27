# Performance Benchmarks

This notebook measures `rustfuzz` performance using Python's `timeit` and visualises results with `plotly`.

All benchmarks run **N = 10,000 iterations** on a fixed string pair to produce stable measurements.


```python
import timeit

import rustfuzz.fuzz as fuzz
from rustfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

N = 10_000
S1, S2 = (
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
)

print(f"Strings:\n  s1 = {S1!r}\n  s2 = {S2!r}\n")
print(f"Iterations: {N:,}")
```


```python
def bench(fn, *args, n=N) -> float:
    """Return milliseconds per call (median of 5 runs)."""
    times = timeit.repeat(lambda: fn(*args), number=n, repeat=5)
    return min(times) / n * 1000  # ms per call


benchmarks = {
    "fuzz.ratio": lambda: fuzz.ratio(S1, S2),
    "fuzz.partial_ratio": lambda: fuzz.partial_ratio(S1, S2),
    "fuzz.token_sort_ratio": lambda: fuzz.token_sort_ratio(S1, S2),
    "fuzz.token_set_ratio": lambda: fuzz.token_set_ratio(S1, S2),
    "fuzz.WRatio": lambda: fuzz.WRatio(S1, S2),
    "Levenshtein.distance": lambda: Levenshtein.distance(S1, S2),
    "Levenshtein.normalized": lambda: Levenshtein.normalized_similarity(S1, S2),
    "JaroWinkler.similarity": lambda: JaroWinkler.similarity(S1, S2),
    "DamerauLevenshtein.dist": lambda: DamerauLevenshtein.distance(S1, S2),
}

results: dict[str, float] = {}

for name, fn in benchmarks.items():
    ms = bench(fn)
    results[name] = ms
    print(f"  {name:35}  {ms * 1000:.3f} μs/call")

print("\n✅ All benchmarks complete")
```

## Results — Bar Chart


```python
try:
    import plotly.graph_objects as go

    ops = list(results.keys())
    times = [v * 1000 for v in results.values()]  # μs

    colors = [
        f"rgba({int(168 + i * 4)},{int(85 - i * 2)},{int(247 - i * 10)},0.85)"
        for i in range(len(ops))
    ]

    fig = go.Figure(
        go.Bar(
            x=ops,
            y=times,
            marker_color=colors,
            text=[f"{t:.2f} μs" for t in times],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="rustfuzz — microseconds per call (lower is better)",
        xaxis_title="Operation",
        yaxis_title="μs / call",
        paper_bgcolor="#0f0319",
        plot_bgcolor="#1a0533",
        font=dict(color="#d8b4fe"),
        xaxis=dict(tickangle=-30),
    )
    fig.show()

except ImportError:
    print("Install plotly: uv pip install plotly")
    for name, ms in results.items():
        bar = "█" * int(ms * 1000 / max(results.values()) * 40)
        print(f"  {name:35} {bar} {ms * 1000:.2f} μs")
```

## Scaling benchmark — string length

How does `Levenshtein.distance` scale with string length?


```python
import random
import string

random.seed(42)


def rand_str(n: int) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


lengths = [10, 50, 100, 250, 500, 1000]
scale_results: dict[int, float] = {}

for length in lengths:
    a, b = rand_str(length), rand_str(length)
    ms = bench(Levenshtein.distance, a, b, n=1000)
    scale_results[length] = ms * 1000  # μs
    print(f"  len={length:5d}  {ms * 1000:.3f} μs/call")
```


```python
try:
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=list(scale_results.keys()),
            y=list(scale_results.values()),
            mode="lines+markers",
            line=dict(color="#a855f7", width=3),
            marker=dict(color="#22c55e", size=10),
        )
    )
    fig.update_layout(
        title="Levenshtein.distance — scaling by string length",
        xaxis_title="String length (chars)",
        yaxis_title="μs / call",
        paper_bgcolor="#0f0319",
        plot_bgcolor="#1a0533",
        font=dict(color="#d8b4fe"),
    )
    fig.show()

except ImportError:
    for length, us in scale_results.items():
        print(f"  len={length:4d}  {us:.3f} μs")
```
