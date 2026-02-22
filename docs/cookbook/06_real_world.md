# Real-World Examples

While toy examples are great for learning the API, the true power of `rustfuzz` becomes apparent when dealing with messy, real-world data at scale. 

Below are two end-to-end examples that fetch real datasets from GitHub, introduce deliberate typos, and use `rustfuzz` to clean and standardize the data perfectly.

---

## 1. Matching S&P 500 Companies

A common data engineering task is standardizing company names. In this example, we fetch the canonical list of S&P 500 constituents and map poorly formatted, handwritten company names to their correct tickered entities.

```python
import urllib.request
import csv
import io
import time
from rustfuzz import process

print("--- S&P 500 Test ---")
# Fetch the S&P 500 canonical list
url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

start_fetch = time.time()
response = urllib.request.urlopen(url)
csv_data = response.read().decode('utf-8')
fetch_time = time.time() - start_fetch

reader = csv.reader(io.StringIO(csv_data))
next(reader) # skip header
companies = [row[1] for row in reader] # Get Security Name

print(f"Loaded {len(companies)} S&P 500 Companies in {fetch_time:.2f}s")

# User inputs with typos, abbreviations, and missing words
messy_inputs = [
    "Apple Inc", 
    "Micro-soft Corp", 
    "Google Alphabet",
    "Berkshir Hatheway", 
    "J.P. Morgan"
]

start_match = time.time()
for messy in messy_inputs:
    # Use extractOne to find the single best match
    match = process.extractOne(messy, companies)
    print(f"'{messy}' -> {match}")
match_time = time.time() - start_match
print(f"Extraction completed in {match_time:.4f}s")
```

### Output:
```text
--- S&P 500 Test ---
Loaded 503 S&P 500 Companies in 0.06s
'Apple Inc' -> ('Apple Inc.', 94.73684210526316, 38)
'Micro-soft Corp' -> ('Microsoft', 80.0, 316)
'Google Alphabet' -> ('Alphabet Inc. (Class A)', 85.5, 19)
'Berkshir Hatheway' -> ('Berkshire Hathaway', 91.42857142857143, 59)
'J.P. Morgan' -> ('Kinder Morgan', 67.05882352941175, 274)
Extraction completed in 0.0040s
```

---

## 2. Standardizing US City Names

Another frequent problem is mapping arbitrary user location input to a known database of geographic entities. In this example, we load over 18,000 unique US Cities and use `process.extract` with the `fuzz.WRatio` scorer. `WRatio` is ideal here because it handles differences in word lengths and token ordering exceptionally well.

```python
import urllib.request
import csv
import io
import time
from rustfuzz import process, fuzz

print("\n--- US Cities Database Test ---")
url = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"

start_fetch = time.time()
# Github requires a simple User-Agent for this repository
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
csv_data = response.read().decode('utf-8')
fetch_time = time.time() - start_fetch

reader = csv.reader(io.StringIO(csv_data))
header = next(reader)
city_idx = header.index('CITY')
cities = [row[city_idx] for row in reader if len(row) > city_idx]

# Deduplicate the list to unique city names
unique_cities = list(set(cities))
print(f"Loaded {len(unique_cities)} unique US Cities in {fetch_time:.2f}s")

# User inputs with severe typos and missing letters
messy_inputs = [
    "San Fransisco", # common spelling mistake
    "New Yrok",      # swapped letters
    "Seatle",        # missing 't'
    "Los Angelos",   # 'o' instead of 'e'
    "Chicag"         # truncated
]

start_match = time.time()
# Since we have ~18k cities, rustfuzz's multi-threading (Rayon) 
# and lock-free thresholds handle this instantly
for messy in messy_inputs:
    # We want the top 3 potential matches using WRatio
    matches = process.extract(messy, unique_cities, limit=3, scorer=fuzz.WRatio)
    print(f"'{messy}' -> {[m[0] for m in matches]}")
match_time = time.time() - start_match
print(f"Extraction completed in {match_time:.4f}s")
```

### Output:
```text
--- US Cities Database Test ---
Loaded 18720 unique US Cities in 0.13s
'San Fransisco' -> ['San Francisco', 'San Ardo', 'San Luis']
'New Yrok' -> ['New York', 'New Ellenton', 'New Cumberland']
'Seatle' -> ['Seattle', 'Seale', 'Searles']
'Los Angelos' -> ['Los Angeles', 'Los Altos', 'Angeles']
'Chicag' -> ['Chicago', 'West Chicago', 'Chicago Heights']
Extraction completed in 0.1031s
```
