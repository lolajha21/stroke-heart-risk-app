import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/heart.csv")

print("=== NaN pro Spalte ===")
print(df.isna().sum())

print("\n=== SHAPE vor One-Hot Encoding ===")
print(df.shape)

kategorische_spalten = ["cp", "restecg", "slope", "thal"]
df_encoded = pd.get_dummies(df, columns=kategorische_spalten, drop_first=True)

print("\n=== SPALTENNAMEN NACH One-Hot Encoding (drop_first=True) ===")
print(list(df_encoded.columns))

print("\n=== SHAPE NACH One-Hot Encoding ===")
print(df_encoded.shape)

X = df_encoded.drop(columns=["condition"])
y = df_encoded["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n=== SHAPES NACH TRAIN-TEST-SPLIT ===")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n=== VERTEILUNG condition IN y_train ===")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\n=== VERTEILUNG condition IN y_test ===")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

# LogisticRegression
lr_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n=== LOGISTIC REGRESSION: ACCURACY ===")
print(accuracy_score(y_test, y_pred_lr))
print("\n=== LOGISTIC REGRESSION: PRECISION ===")
print(precision_score(y_test, y_pred_lr))
print("\n=== LOGISTIC REGRESSION: RECALL ===")
print(recall_score(y_test, y_pred_lr))
print("\n=== LOGISTIC REGRESSION: F1-SCORE ===")
print(f1_score(y_test, y_pred_lr))
print("\n=== LOGISTIC REGRESSION: CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred_lr))

# RandomForest
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
