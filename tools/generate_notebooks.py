import nbformat as nbf
import os

def create_notebooks():
    # 01_introduction.ipynb
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell("# Introduction to rustfuzz\n\n`rustfuzz` is a blazing fast string matching library implemented in Rust. It provides a simple Python API that heavily mirrors the popular RapidFuzz, but powered by Rust for optimal performance.\n\nLet's start with basic matching!"),
        nbf.v4.new_markdown_cell("## Basic String Matching (`fuzz_ratio` and `fuzz_partial_ratio`)\n\nThe simplest way to compare two strings is using `fuzz_ratio`, which calculates the standard Levenshtein distance ratio.\n\nIf you want to compare a short string against a longer string, `fuzz_partial_ratio` is optimal. It finds the best matching substring of the longer string."),
        nbf.v4.new_code_cell("from rustfuzz.fuzz import fuzz_ratio, fuzz_partial_ratio\n\n# Exact match\nprint(f\"Ratio (exact): {fuzz_ratio('apple', 'apple')}\")\n\n# Small edit\nprint(f\"Ratio (edit): {fuzz_ratio('apple', 'bpple')}\")\n\n# Partial match against longer string\nprint(f\"Partial Ratio: {fuzz_partial_ratio('apple', 'I love to eat an apple every day!')}\")")
    ]
    with open("docs/cookbook/01_introduction.ipynb", "w") as f:
        nbf.write(nb1, f)
        
    # 02_advanced_matching.ipynb
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell("# Advanced Matching in rustfuzz\n\nSometimes, string comparison needs to ignore word order, or handle entirely different sets of words gracefully. This is where token sort ratios and set ratios come in."),
        nbf.v4.new_code_cell("from rustfuzz.fuzz import fuzz_ratio, fuzz_token_sort_ratio, fuzz_token_set_ratio\n\nstr1 = 'The quick brown fox'\nstr2 = 'fox brown quick The'\n\n# Standard ratio cares about order\nprint(f\"Ratio: {fuzz_ratio(str1, str2)}\")\n\n# Token sort sorts the words first\nprint(f\"Token Sort Ratio: {fuzz_token_sort_ratio(str1, str2)}\")\n"),
        nbf.v4.new_markdown_cell("## Token Set Ratio\n\nIf one string is a subset of the other (with extra words), `fuzz_token_set_ratio` calculates the ratio based primarily on the intersection of the words."),
        nbf.v4.new_code_cell("str3 = 'apple banana orange'\nstr4 = 'apple orange banana grape kiwi'\n\nprint(f\"Token Set Ratio: {fuzz_token_set_ratio(str3, str4)}\")\n")
    ]
    with open("docs/cookbook/02_advanced_matching.ipynb", "w") as f:
        nbf.write(nb2, f)
        
    # 03_benchmarks.ipynb
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell("# Performance Benchmarks\n\n`rustfuzz` is designed to be extremely fast. Let's visualize its speed relative to a baseline using `plotly` and `pandas`.\n\n*(Note: This is a synthetic dataset generated for illustration purposes)*"),
        nbf.v4.new_code_cell("import pandas as pd\nimport plotly.express as px\n\n# Sample benchmark data points showing rustfuzz performing well\ndata = {\n    'Operation': ['fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_token_sort_ratio', 'fuzz_token_set_ratio'],\n    'Library': ['rustfuzz', 'rustfuzz', 'rustfuzz', 'rustfuzz'],\n    'Time (ms) per 10k calls': [1.2, 1.8, 3.5, 4.1]\n}\n\ndata_baseline = {\n    'Operation': ['fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_token_sort_ratio', 'fuzz_token_set_ratio'],\n    'Library': ['baseline', 'baseline', 'baseline', 'baseline'],\n    'Time (ms) per 10k calls': [15.0, 22.5, 45.0, 52.3]\n}\n\ndf = pd.concat([pd.DataFrame(data), pd.DataFrame(data_baseline)])\n"),
        nbf.v4.new_code_cell("fig = px.bar(\n    df, \n    x='Operation', \n    y='Time (ms) per 10k calls', \n    color='Library', \n    barmode='group',\n    title='Performance Comparison: rustfuzz vs baseline'\n)\nfig.show()")
    ]
    with open("docs/cookbook/03_benchmarks.ipynb", "w") as f:
        nbf.write(nb3, f)
        
if __name__ == '__main__':
    create_notebooks()
