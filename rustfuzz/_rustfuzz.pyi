"""Type stubs for the rustfuzz native extension module."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Edit-operation data classes
# ---------------------------------------------------------------------------

class Editop:
    tag: str
    src_pos: int
    dest_pos: int
    def __repr__(self) -> str: ...

class Editops:
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __getitem__(self, index: int) -> Editop: ...
    def as_opcodes(self) -> Opcodes: ...
    src_len: int
    dest_len: int

class Opcode:
    tag: str
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int
    def __repr__(self) -> str: ...

class Opcodes:
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __getitem__(self, index: int) -> Opcode: ...
    def as_editops(self) -> Editops: ...
    src_len: int
    dest_len: int

class MatchingBlock:
    a: int
    b: int
    size: int
    def __repr__(self) -> str: ...

class ScoreAlignment:
    score: float
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def default_process(s: Any) -> str: ...

# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

def extract(
    query: Any,
    choices: Any,
    scorer_name: str,
    scorer_obj: Any | None,
    processor: Any | None,
    limit: int | None,
    score_cutoff: float | None,
) -> list[tuple[Any, float, int]]: ...
def extract_one(
    query: Any,
    choices: Any,
    scorer_name: str,
    scorer_obj: Any | None,
    processor: Any | None,
    score_cutoff: float | None,
) -> tuple[Any, float, int] | None: ...
def extract_iter(
    query: Any,
    choices: Any,
    scorer_name: str,
    scorer_obj: Any | None,
    processor: Any | None,
    score_cutoff: float | None,
) -> list[tuple[Any, float, int]]: ...
def cdist(
    queries: Any,
    choices: Any,
    scorer_name: str,
    scorer_obj: Any | None,
    processor: Any | None,
    score_cutoff: float | None,
) -> tuple[list[float], int, int]: ...

# ---------------------------------------------------------------------------
# fuzz
# ---------------------------------------------------------------------------

def fuzz_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_partial_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_partial_ratio_alignment(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> ScoreAlignment | None: ...
def fuzz_token_sort_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_token_set_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_token_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_partial_token_sort_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_partial_token_set_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_partial_token_ratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_wratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def fuzz_qratio(
    s1: Any,
    s2: Any,
    *,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...

# ---------------------------------------------------------------------------
# distance — Levenshtein
# ---------------------------------------------------------------------------

def levenshtein_distance(
    s1: Any,
    s2: Any,
    *,
    weights: tuple[int, int, int] = (1, 1, 1),
    processor: Any | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def levenshtein_similarity(
    s1: Any,
    s2: Any,
    *,
    weights: tuple[int, int, int] = (1, 1, 1),
    processor: Any | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def levenshtein_normalized_distance(
    s1: Any,
    s2: Any,
    *,
    weights: tuple[int, int, int] = (1, 1, 1),
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def levenshtein_normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    weights: tuple[int, int, int] = (1, 1, 1),
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def levenshtein_editops(
    s1: Any, s2: Any, *, processor: Any | None = None
) -> Editops: ...
def levenshtein_opcodes(
    s1: Any, s2: Any, *, processor: Any | None = None
) -> Opcodes: ...

# ---------------------------------------------------------------------------
# distance — Hamming
# ---------------------------------------------------------------------------

def hamming_distance(
    s1: Any,
    s2: Any,
    *,
    pad: bool = True,
    processor: Any | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def hamming_similarity(
    s1: Any,
    s2: Any,
    *,
    pad: bool = True,
    processor: Any | None = None,
    score_cutoff: int | None = None,
) -> int: ...
def hamming_normalized_distance(
    s1: Any,
    s2: Any,
    *,
    pad: bool = True,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def hamming_normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    pad: bool = True,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def hamming_editops(s1: Any, s2: Any, *, processor: Any | None = None) -> Editops: ...
def hamming_opcodes(s1: Any, s2: Any, *, processor: Any | None = None) -> Opcodes: ...

# ---------------------------------------------------------------------------
# distance — Indel
# ---------------------------------------------------------------------------

def indel_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def indel_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def indel_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def indel_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def indel_editops(s1: Any, s2: Any, *, processor: Any | None = None) -> Editops: ...
def indel_opcodes(s1: Any, s2: Any, *, processor: Any | None = None) -> Opcodes: ...

# ---------------------------------------------------------------------------
# distance — Jaro / JaroWinkler
# ---------------------------------------------------------------------------

def jaro_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def jaro_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def jaro_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def jaro_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def jaro_winkler_distance(
    s1: Any,
    s2: Any,
    *,
    prefix_weight: float = 0.1,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def jaro_winkler_similarity(
    s1: Any,
    s2: Any,
    *,
    prefix_weight: float = 0.1,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def jaro_winkler_normalized_distance(
    s1: Any,
    s2: Any,
    *,
    prefix_weight: float = 0.1,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def jaro_winkler_normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    prefix_weight: float = 0.1,
    processor: Any | None = None,
    score_cutoff: float | None = None,
) -> float: ...

# ---------------------------------------------------------------------------
# distance — LCSseq
# ---------------------------------------------------------------------------

def lcs_seq_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def lcs_seq_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def lcs_seq_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def lcs_seq_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def lcs_seq_editops(s1: Any, s2: Any, *, processor: Any | None = None) -> Editops: ...
def lcs_seq_opcodes(s1: Any, s2: Any, *, processor: Any | None = None) -> Opcodes: ...

# ---------------------------------------------------------------------------
# distance — OSA
# ---------------------------------------------------------------------------

def osa_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def osa_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def osa_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def osa_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...

# ---------------------------------------------------------------------------
# distance — DamerauLevenshtein
# ---------------------------------------------------------------------------

def damerau_levenshtein_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def damerau_levenshtein_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def damerau_levenshtein_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def damerau_levenshtein_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...

# ---------------------------------------------------------------------------
# distance — Prefix / Postfix
# ---------------------------------------------------------------------------

def prefix_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def prefix_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def prefix_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def prefix_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def postfix_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def postfix_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: int | None = None
) -> int: ...
def postfix_normalized_distance(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...
def postfix_normalized_similarity(
    s1: Any, s2: Any, *, processor: Any | None = None, score_cutoff: float | None = None
) -> float: ...

# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class BM25Index:
    num_docs: int
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None: ...
    def get_scores(self, query: str) -> list[float]: ...
    def get_top_n(self, query: str, n: int) -> list[tuple[str, float]]: ...
    def get_batch_scores(self, queries: list[str]) -> list[list[float]]: ...
    def get_top_n_fuzzy(
        self, query: str, n: int, bm25_candidates: int, fuzzy_weight: float
    ) -> list[tuple[str, float]]: ...
    def get_top_n_rrf(
        self, query: str, n: int, bm25_candidates: int, rrf_k: int
    ) -> list[tuple[str, float]]: ...

def cosine_similarity_matrix(
    queries: list[list[float]],
    corpus: list[list[float]],
) -> tuple[list[float], int, int]: ...

# ---------------------------------------------------------------------------
# BKTree (used by process.dedupe)
# ---------------------------------------------------------------------------

class BKTree:
    def dedupe(self, items: list[str], max_edits: int) -> list[str]: ...
