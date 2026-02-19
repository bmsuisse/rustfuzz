# rustfuzz

Rust port of the rapidfuzz library, a blazing fast string matching library for Python.

## Installation

```sh
pip install rustfuzz
```

## Usage

```python
import rustfuzz

# Example usage
print(rustfuzz.fuzz_ratio("this is a test", "this is a test!"))
```
