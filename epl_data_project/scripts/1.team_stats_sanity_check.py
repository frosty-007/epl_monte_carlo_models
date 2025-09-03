import pandas as pd
df = pd.read_parquet("../data/calibrated/team_match_lambdas.parquet")
print("rows:", len(df))
print("have goals %:", df["goals"].notna().mean() if "goals" in df else 0.0)
print(df.head(3))