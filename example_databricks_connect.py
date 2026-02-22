import os
import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql.functions import pandas_udf
import pyspark.sql.types as T
from rustfuzz import process, fuzz

def run_example():
    """
    Example of using rustfuzz with PySpark via Databricks Connect.
    This script will execute the DataFrame operations on your remote cluster.
    """
    # 1. Initialize Databricks Session
    # Assumes you have configured your Databricks profile and cluster in ~/.databrickscfg
    # or via environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID)
    print("Connecting to Databricks cluster...")
    builder = DatabricksSession.builder
    profile = os.getenv("DATABRICKS_PROFILE")
    if profile:
        builder = builder.profile(profile)
    spark = builder.getOrCreate()
    print("Connected successfully!")

    # 2. Simulate reading the table from Databricks Unity Catalog:
    # In your real code, this would be:
    # df = spark.sql("SELECT id, name FROM core.dim_article")
    data = [
        (1, "Apple iPhone 14 Pro Max 256GB"),
        (2, "Smsung Glxy S23 Ultra Black"),  # Typos
        (3, "Google Pixel 8 128gb"),
        (4, "Sony Play Station 5 Console"),
        (5, "Unknown Random Item"),
    ]
    df = spark.createDataFrame(data, schema=["id", "name"])

    # 3. Define the reference data we want to fuzzy match against
    CLEAN_CHOICES = [
        "iPhone 14 Pro Max", 
        "Galaxy S23 Ultra", 
        "Pixel 8", 
        "PlayStation 5", 
        "Nintendo Switch OLED"
    ]

    # 4. Define the Pandas UDF for fast vectorised batch processing
    # The return type here is StringType (the matched string).
    # NOTE: rustfuzz must be installed on the Databricks cluster for this to work natively!
    @pandas_udf(T.StringType())
    def fuzzy_match_udf(names: pd.Series) -> pd.Series:
        def get_best_match(name):
            if not isinstance(name, str) or not name:
                return None
            
            # Using RustFuzz native fast-path for String matching
            # Extract one returns: (best_match_string, score, index)
            res = process.extractOne(name, CLEAN_CHOICES, scorer=fuzz.WRatio)
            
            if res and res[1] >= 80.0:  # 80.0 is the minimum match score threshold
                return res[0]
            return None
            
        # Apply the matching logic to the entire Arrow batch (Pandas Series)
        return names.apply(get_best_match)

    # 5. Apply the UDF to our DataFrame
    print("Executing batch fuzzy matching...")
    matched_df = df.withColumn("fuzzy_matched_name", fuzzy_match_udf(df.name))

    print("\n--- Databricks RustFuzz Results ---")
    matched_df.show(truncate=False)

if __name__ == "__main__":
    run_example()
