import pandas as pd

df = pd.read_csv("data/responses.csv")

print(df.head())
print(df.columns.tolist())