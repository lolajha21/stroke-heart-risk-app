import pandas as pd

df = pd.read_csv("data/heart.csv")

print("=== SHAPE ===")
print(df.shape)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== HEAD (5) ===")
print(df.head(5))

print("\n=== FEHLENDE WERTE PRO SPALTE ===")
print(df.isna().sum())

print("\n=== VALUE COUNTS: ca ===")
print(df["ca"].value_counts(dropna=False))

print("\n=== VALUE COUNTS: thal ===")
print(df["thal"].value_counts(dropna=False))

print("\n=== VALUE COUNTS: condition ===")
print(df["condition"].value_counts(dropna=False))

df_stroke = pd.read_csv("data/stroke.csv")

print("\n=== VALUE COUNTS: stroke (stroke.csv) ===")
print(df_stroke["stroke"].value_counts(dropna=False))
