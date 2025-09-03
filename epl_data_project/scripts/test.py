import pandas as pd
df = pd.read_parquet("../data/calibrated/team_match_lambdas.parquet")
print(df.columns.tolist())
print(df.head(3))