"""
Trainiert die finalen LogisticRegression-Modelle fuer die Schlaganfall- (stroke.csv)
und Herzerkrankungs-Vorhersage (heart.csv) inklusive der jeweiligen Preprocessing-
Schritte und speichert die trainierten Modelle mit joblib im Ordner src/.
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "src")


def train_stroke_model():
    """Laedt stroke.csv, bereitet die Daten auf und trainiert das Stroke-Modell."""
    df = pd.read_csv(os.path.join(DATA_DIR, "stroke.csv"))

    # id-Spalte ist keine Feature-Spalte und wird entfernt
    df = df.drop(columns=["id"])

    # Fehlende bmi-Werte (z.B. "N/A") durch den Median ersetzen
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Kategorische Spalten One-Hot-kodieren
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(X, y)

    return model, list(X.columns)


def train_heart_model():
    """Laedt heart.csv, bereitet die Daten auf und trainiert das Heart-Modell."""
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))

    # Kategorische Spalten One-Hot-kodieren
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["condition"])
    y = df["condition"]

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(X, y)

    return model, list(X.columns)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stroke_model, stroke_columns = train_stroke_model()
    stroke_path = os.path.join(OUTPUT_DIR, "stroke_model.pkl")
    joblib.dump(stroke_model, stroke_path)
    print(f"Stroke-Modell trainiert und gespeichert unter: {stroke_path}")
    print(f"  Feature-Spalten ({len(stroke_columns)}): {stroke_columns}")

    heart_model, heart_columns = train_heart_model()
    heart_path = os.path.join(OUTPUT_DIR, "heart_model.pkl")
    joblib.dump(heart_model, heart_path)
    print(f"Heart-Modell trainiert und gespeichert unter: {heart_path}")
    print(f"  Feature-Spalten ({len(heart_columns)}): {heart_columns}")


if __name__ == "__main__":
    main()
