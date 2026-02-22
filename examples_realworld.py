import csv
import io
import time
import urllib.request

from rustfuzz import fuzz, process


def test_cities():
    print("\n--- US Cities Test (Top 1000) ---")
    url = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"

    start_fetch = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    csv_data = response.read().decode('utf-8')
    fetch_time = time.time() - start_fetch

    reader = csv.reader(io.StringIO(csv_data))
    header = next(reader)
    # The columns are: ID,CITY,STATE_CODE,STATE_NAME,COUNTY,LATITUDE,LONGITUDE
    # In some versions of this CSV, cities might be index 1 or 2
    # Let's dynamically find the CITY column index
    city_idx = header.index('CITY')
    cities = [row[city_idx] for row in reader if len(row) > city_idx]

    unique_cities = list(set(cities))
    print(f"Loaded {len(unique_cities)} unique US Cities in {fetch_time:.2f}s")

    messy_inputs = [
        "San Fransisco", # common typo
        "New Yrok",
        "Seatle",
        "Los Angelos",
        "Chicag"
    ]

    start_match = time.time()
    for messy in messy_inputs:
        matches = process.extract(messy, unique_cities, limit=3, scorer=fuzz.WRatio)
        print(f"'{messy}' -> {[m[0] for m in matches]}")
    match_time = time.time() - start_match
    print(f"Extraction completed in {match_time:.4f}s")

if __name__ == "__main__":
    test_cities()
