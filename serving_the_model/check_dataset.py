import os
import pandas as pd

FILES = ["../train.csv", "../test.csv"]

def main():
    for f in FILES:
        path = os.path.join(os.path.dirname(__file__), f)
        print(f"Checking: {path}")
        if not os.path.exists(path):
            print("  MISSING")
            continue
        try:
            df = pd.read_csv(path)
            print(f"  Found. Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:10]}")
        except Exception as e:
            print(f"  Error reading: {e}")

if __name__ == '__main__':
    main()
