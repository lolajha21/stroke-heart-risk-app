import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/stroke.csv")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

kategorische_spalten = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
df_encoded = pd.get_dummies(df, columns=kategorische_spalten, drop_first=True)
df_encoded = df_encoded.drop(columns=["id"])
df_encoded = df_encoded.astype("float64", errors="ignore")
for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype("float64")

X = df_encoded.drop(columns=["stroke"])
y = df_encoded["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_stroke.png", dpi=150)
print("Summary-Plot gespeichert unter shap_summary_stroke.png")
