import pandas as pd
import os

# CSV file paths
CSV_PATH = ['temp/processed_timestamps.csv']

for path in CSV_PATH:
    print(f"\nchecking file: {path}")
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("\n" + "="*60 + "\n")
        continue

    # Load the CSV
    df = pd.read_csv(path)

    print("\n=== 1. Basic Structure ===")
    total_rows = len(df)
    columns = list(df.columns)
    unique_users = df['user_id'].nunique() if 'user_id' in df.columns else 0
    
    print(f"Total rows: {total_rows} -> Expected > 0")
    print(f"Columns: {columns} -> Expected ['user_id', 'timestamp']")
    print(f"Unique users: {unique_users} -> Expected > 0")

    # NaN values can break parsing later. // Expected False for both
    print("\n=== 2. Corruption (NaN / Missing) ===")
    has_nan_user = df['user_id'].isna().any()
    has_nan_time = df['timestamp'].isna().any()
    print(f"Includes NaN in user_id?: {has_nan_user} -> Expected False")
    print(f"Includes NaN in timestamp?: {has_nan_time} -> Expected False")

    # If parsing failed, we might have very few timestamps per user or single dominant user
    print("\n=== 3. Distribution Anomalies ===")
    avg_tweets = total_rows / unique_users if unique_users > 0 else 0
    max_tweets = df['user_id'].value_counts().max() if total_rows > 0 else 0
    print(f"Average timestamps per user: {avg_tweets:.2f}")
    print(f"Max timestamps for a single user: {max_tweets}")

    # Are the records diverse?
    # If the parser duplicated records, sequential rows might be identical
    print("\n=== 4. Diversity / Format test ===")
    if total_rows > 1:
        are_identical = (df.iloc[0]['user_id'] == df.iloc[1]['user_id']) and (df.iloc[0]['timestamp'] == df.iloc[1]['timestamp'])
        print(f"Are first two rows identical?: {are_identical} -> Expected False")
    
    # Verify timestamp format validity on a sample
    try:
        sample_dates = pd.to_datetime(df['timestamp'].dropna().head(1000), format='mixed', utc=True)
        print("Timestamp parsing test: Success")
        print(f"Sample date range: {sample_dates.min()} to {sample_dates.max()}")
    except Exception as e:
        print(f"Timestamp parsing test: Failed ({e})")

    print("\n=== 5. Example ===")
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))

    print("\n" + "="*60 + "\n")
    print("\n" + "="*60 + "\n")