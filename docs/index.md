# Welcome to RustFuzz 🦀✨

`rustfuzz` is a blazing fast string matching library implemented entirely in Rust, built specifically for integration with Python. 

By porting the core algorithms of `rapidfuzz` to Rust, we provide an implementation that achieves incredible performance speedups while maintaining a familiar, easy-to-use Python API. It's the perfect choice for heavy-duty deduplication, record linkage, and natural language processing tasks.

## Why RustFuzz?

- **Uncompromising Speed**: Core matching algorithms are optimized in Rust, providing a significant performance boost over traditional Python implementations.
- **Memory Safety**: Backed by Rust's strict compiler guarantees, ensuring your applications run reliably without unexpected memory errors.
- **Drop-in Compatibility**: Designed to integrate seamlessly into your current data pipelines with minimal to no code changes.

## Cookbook Recipes 🧑‍🍳

Learn how to leverage `rustfuzz` effectively through our interactive Jupyter Notebook cookbooks:

1. [**Introduction to RustFuzz**](cookbook/01_introduction.ipynb): Get started with basic matching functions and terminology.
2. [**Advanced Matching Ratios**](cookbook/02_advanced_matching.ipynb): Dive deep into partial ratios, token sorts, and set ratios.
3. [**Performance Benchmarks**](cookbook/03_benchmarks.ipynb): See the blazing speed of RustFuzz visualized and compared against other libraries.

Start exploring the recipes from the navigation menu on the left!
