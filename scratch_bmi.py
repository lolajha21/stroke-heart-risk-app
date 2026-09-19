import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/stroke.csv")

print("=== NaN pro Spalte VOR Imputation ===")
print(df.isna().sum())

median_bmi = df["bmi"].median()
print(f"\nMedian von bmi: {median_bmi}")

df["bmi"] = df["bmi"].fillna(median_bmi)

print("\n=== NaN pro Spalte NACH Imputation ===")
print(df.isna().sum())

print("\n=== SHAPE vor One-Hot Encoding ===")
print(df.shape)

kategorische_spalten = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
df_encoded = pd.get_dummies(df, columns=kategorische_spalten)

print("\n=== SPALTENNAMEN NACH One-Hot Encoding ===")
print(list(df_encoded.columns))

print("\n=== SHAPE NACH One-Hot Encoding ===")
print(df_encoded.shape)

df_encoded_drop_first = pd.get_dummies(df, columns=kategorische_spalten, drop_first=True)

print("\n=== SPALTENNAMEN NACH One-Hot Encoding (drop_first=True) ===")
print(list(df_encoded_drop_first.columns))

print("\n=== SHAPE NACH One-Hot Encoding (drop_first=True) ===")
print(df_encoded_drop_first.shape)

df_encoded_drop_first = df_encoded_drop_first.drop(columns=["id"])

print("\n=== FINALE SHAPE (ohne id) ===")
print(df_encoded_drop_first.shape)

print("\n=== DTYPES (final) ===")
print(df_encoded_drop_first.dtypes)

print("\n=== HEAD (final) ===")
print(df_encoded_drop_first.head())

X = df_encoded_drop_first.drop(columns=["stroke"])
y = df_encoded_drop_first["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n=== SHAPES NACH TRAIN-TEST-SPLIT ===")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n=== VERTEILUNG stroke IN y_train (absolut & relativ) ===")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\n=== VERTEILUNG stroke IN y_test (absolut & relativ) ===")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== ACCURACY ===")
print(accuracy_score(y_test, y_pred))

print("\n=== PRECISION ===")
print(precision_score(y_test, y_pred))

print("\n=== RECALL ===")
print(recall_score(y_test, y_pred))

print("\n=== F1-SCORE ===")
print(f1_score(y_test, y_pred))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== RANDOM FOREST: ACCURACY ===")
print(accuracy_score(y_test, y_pred_rf))

print("\n=== RANDOM FOREST: PRECISION ===")
print(precision_score(y_test, y_pred_rf))

print("\n=== RANDOM FOREST: RECALL ===")
print(recall_score(y_test, y_pred_rf))

print("\n=== RANDOM FOREST: F1-SCORE ===")
print(f1_score(y_test, y_pred_rf))

print("\n=== RANDOM FOREST: CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred_rf))
