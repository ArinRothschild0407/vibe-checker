import pandas as pd

df = pd.read_csv("data/responses.csv")

print("Dataset shape:", df.shape)

print("\nData types found:")
print(df.dtypes.value_counts())

for data_type in df.dtypes.unique():
    print(f"\n--- {data_type} columns ---")

    columns = df.columns[df.dtypes == data_type]

    for column in columns:
        print(column)


print("\nNumeric column ranges:")

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    print(
        column,
        "min =", df[column].min(),
        "max =", df[column].max()
    )

missing = df.isnull().sum()

print("\nMissing data:")
print("Total missing answers:", missing.sum())
print("Columns with missing answers:", (missing > 0).sum())
print("Most missing in one column:", missing.max())