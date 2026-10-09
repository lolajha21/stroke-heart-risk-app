"""
Trainiert die finalen LogisticRegression-Modelle fuer die Schlaganfall- (stroke.csv)
und Herzerkrankungs-Vorhersage (heart.csv) inklusive der jeweiligen Preprocessing-
Schritte und speichert die trainierten Modelle mit joblib im Ordner src/.

Zusaetzlich werden LogisticRegression, RandomForest und MLPClassifier auf einem
Train/Test-Split verglichen (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
und die trainierten MLP-Modelle ebenfalls im Ordner src/ gespeichert.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "src")


def load_stroke_data():
    """Laedt stroke.csv und bereitet die Daten (Preprocessing) fuer das Training auf."""
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
    return X, y


def load_heart_data():
    """Laedt heart.csv und bereitet die Daten (Preprocessing) fuer das Training auf."""
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))

    # Kategorische Spalten One-Hot-kodieren
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["condition"])
    y = df["condition"]
    return X, y


def train_stroke_model():
    """Trainiert das finale Stroke-Produktionsmodell (LogisticRegression) auf dem gesamten Datensatz."""
    X, y = load_stroke_data()

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(X, y)

    return model, list(X.columns)


def train_heart_model():
    """Trainiert das finale Heart-Produktionsmodell (LogisticRegression) auf dem gesamten Datensatz."""
    X, y = load_heart_data()

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(X, y)

    return model, list(X.columns)


def evaluate_model(model, X_test, y_test):
    """Berechnet Accuracy, Precision, Recall, F1-Score und Confusion Matrix fuer ein Modell."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def compare_models(X, y, dataset_name):
    """Trainiert LogisticRegression, RandomForest und MLPClassifier auf demselben
    X_train/y_train und vergleicht sie anhand von X_test/y_test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=5000
        ),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(50,), random_state=42, max_iter=1000
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    comparison_df = pd.DataFrame(results).T
    print(f"\n=== MODELLVERGLEICH: {dataset_name} ===")
    print(comparison_df.to_string())

    return models


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Produktionsmodelle: LogisticRegression, trainiert auf dem gesamten Datensatz
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

    # Modellvergleich (LogisticRegression, RandomForest, MLPClassifier) auf Train/Test-Split
    X_stroke, y_stroke = load_stroke_data()
    stroke_comparison_models = compare_models(X_stroke, y_stroke, "STROKE")
    stroke_mlp_path = os.path.join(OUTPUT_DIR, "stroke_mlp_model.pkl")
    joblib.dump(stroke_comparison_models["MLPClassifier"], stroke_mlp_path)
    print(f"Stroke-MLP-Modell gespeichert unter: {stroke_mlp_path}")

    X_heart, y_heart = load_heart_data()
    heart_comparison_models = compare_models(X_heart, y_heart, "HEART")
    heart_mlp_path = os.path.join(OUTPUT_DIR, "heart_mlp_model.pkl")
    joblib.dump(heart_comparison_models["MLPClassifier"], heart_mlp_path)
    print(f"Heart-MLP-Modell gespeichert unter: {heart_mlp_path}")


if __name__ == "__main__":
    main()
