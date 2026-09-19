import joblib
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

st.title("Stroke & Heart Risk Checker")

tab_stroke, tab_heart = st.tabs(["Stroke-Risiko", "Heart-Risiko"])


@st.cache_data
def load_stroke_background():
    """Hintergrunddaten für shap.LinearExplainer, exakt wie im Trainings-Skript
    (scratch_shap_stroke.py) aufbereitet: gleiches Encoding, gleicher Split."""
    df = pd.read_csv("data/stroke.csv")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    kategorische_spalten = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    df_encoded = pd.get_dummies(df, columns=kategorische_spalten, drop_first=True)
    df_encoded = df_encoded.drop(columns=["id"])
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype("float64")

    X = df_encoded.drop(columns=["stroke"])
    y = df_encoded["stroke"]

    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train


@st.cache_data
def load_heart_background():
    """Hintergrunddaten für shap.LinearExplainer, exakt wie im Trainings-Skript
    (scratch_heart.py) aufbereitet: gleiches Encoding, gleicher Split."""
    df = pd.read_csv("data/heart.csv")

    kategorische_spalten = ["cp", "restecg", "slope", "thal"]
    df_encoded = pd.get_dummies(df, columns=kategorische_spalten, drop_first=True)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype("float64")

    X = df_encoded.drop(columns=["condition"])
    y = df_encoded["condition"]

    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train

with tab_stroke:
    age = st.number_input("Alter", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Bluthochdruck (Hypertonie)", ["Ja", "Nein"])
    heart_disease = st.selectbox("Herzerkrankung", ["Ja", "Nein"])
    avg_glucose_level = st.number_input(
        "Durchschnittlicher Glukosewert",
        min_value=0.0,
        value=100.0,
        help=(
            "Durchschnittlicher Blutzuckerspiegel in mg/dl. Nüchtern gilt ein Wert "
            "unter 100 mg/dl als normal, ab ca. 126 mg/dl spricht man von erhöhten "
            "Werten (Diabetes-Verdacht)."
        ),
    )
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0,
        help=(
            "Body-Mass-Index = Gewicht (kg) ÷ Körpergröße² (m²). "
            "Normalgewicht liegt etwa zwischen 18,5 und 25."
        ),
    )
    gender = st.selectbox("Geschlecht", ["Männlich", "Weiblich", "Divers"])
    ever_married = st.selectbox("Jemals verheiratet gewesen", ["Ja", "Nein"])
    work_type = st.selectbox(
        "Art der Arbeit",
        [
            "Privatwirtschaft",
            "Selbstständig",
            "Öffentlicher Dienst",
            "Kind (noch nicht arbeitstätig)",
            "Nie gearbeitet",
        ],
    )
    Residence_type = st.selectbox("Wohnort", ["Städtisch", "Ländlich"])
    smoking_status = st.selectbox(
        "Raucherstatus",
        ["nie geraucht", "früher geraucht", "raucht", "unbekannt"],
    )

    if st.button("Risiko berechnen", key="stroke_calc"):
        model = joblib.load("src/stroke_model.pkl")

        # Gleiches Encoding wie beim Training: pd.get_dummies(..., drop_first=True)
        # Die Spaltennamen entsprechen den ORIGINAL-Kategorien aus dem Trainings-
        # datensatz; die deutschen Auswahltexte werden daher hier auf die jeweilige
        # Original-Kategorie zurückgeführt.
        # gender: Baseline "Female" (Weiblich) | ever_married: Baseline "No"
        # work_type: Baseline "Govt_job" (Öffentlicher Dienst)
        # Residence_type: Baseline "Rural" (Ländlich)
        # smoking_status: Baseline "Unknown" (unbekannt)
        input_dict = {
            "age": age,
            "hypertension": 1 if hypertension == "Ja" else 0,
            "heart_disease": 1 if heart_disease == "Ja" else 0,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "gender_Male": 1 if gender == "Männlich" else 0,
            "gender_Other": 1 if gender == "Divers" else 0,
            "ever_married_Yes": 1 if ever_married == "Ja" else 0,
            "work_type_Never_worked": 1 if work_type == "Nie gearbeitet" else 0,
            "work_type_Private": 1 if work_type == "Privatwirtschaft" else 0,
            "work_type_Self-employed": 1 if work_type == "Selbstständig" else 0,
            "work_type_children": 1 if work_type == "Kind (noch nicht arbeitstätig)" else 0,
            "Residence_type_Urban": 1 if Residence_type == "Städtisch" else 0,
            "smoking_status_formerly smoked": 1 if smoking_status == "früher geraucht" else 0,
            "smoking_status_never smoked": 1 if smoking_status == "nie geraucht" else 0,
            "smoking_status_smokes": 1 if smoking_status == "raucht" else 0,
        }

        # Spaltenreihenfolge exakt wie beim Training (model.feature_names_in_)
        input_df = pd.DataFrame([input_dict])[model.feature_names_in_]

        # Cast bool-Spalten zu float64, falls nötig (wie im SHAP-Skript)
        for col in input_df.columns:
            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype("float64")

        probability = model.predict_proba(input_df)[0][1]
        st.write(f"**Schlaganfall-Risiko: {probability * 100:.2f} %**")

        # SHAP-Erklärung für genau diese eine Eingabe
        X_background = load_stroke_background()[model.feature_names_in_]
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer(input_df)

        # LinearExplainer erklärt den Logit (linearen Output) der LogisticRegression.
        # Lokale Näherung, um daraus Prozentpunkte der Risiko-Wahrscheinlichkeit zu
        # machen: d(Wahrscheinlichkeit)/d(Logit) = p * (1 - p)
        slope = probability * (1 - probability)
        contributions = shap_values.values[0] * slope * 100

        feature_labels = {
            "age": "Alter",
            "hypertension": "Bluthochdruck",
            "heart_disease": "Herzerkrankung",
            "avg_glucose_level": "Glukosewert",
            "bmi": "BMI",
            "gender_Male": "Geschlecht: männlich",
            "gender_Other": "Geschlecht: divers",
            "ever_married_Yes": "Verheiratet",
            "work_type_Never_worked": "Arbeit: nie gearbeitet",
            "work_type_Private": "Arbeit: Privatwirtschaft",
            "work_type_Self-employed": "Arbeit: selbstständig",
            "work_type_children": "Arbeit: Kind",
            "Residence_type_Urban": "Wohnort: städtisch",
            "smoking_status_formerly smoked": "Raucherstatus: früher geraucht",
            "smoking_status_never smoked": "Raucherstatus: nie geraucht",
            "smoking_status_smokes": "Raucherstatus: raucht",
        }

        contribution_pairs = list(zip(model.feature_names_in_, contributions))
        contribution_pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)

        st.write("**Einflussreichste Merkmale:**")
        for feature, value in contribution_pairs[:5]:
            label = feature_labels.get(feature, feature)
            richtung = "erhöht" if value > 0 else "senkt"
            st.write(f"- {label}: {richtung} das Risiko um {abs(value):.2f} %")

with tab_heart:
    h_age = st.number_input("Alter", min_value=0, max_value=120, value=30, key="h_age")
    h_sex = st.selectbox("Geschlecht", ["Männlich", "Weiblich"], key="h_sex")

    # Reihenfolge entspricht dem numerischen Code im Trainingsdatensatz (0-3)
    cp_options = [
        "belastungsabhängiger Brustschmerz",
        "untypischer Brustschmerz",
        "Brustschmerz ohne Herzursache",
        "kein Brustschmerz",
    ]
    h_cp = st.selectbox(
        "Art der Brustschmerzen",
        cp_options,
        help=(
            "Art der Brustschmerzen: kein Brustschmerz / belastungsabhängiger "
            "Brustschmerz / untypischer Brustschmerz / Brustschmerz ohne Herzursache. "
            "Hinweis: Asymptomatisch bedeutet nicht risikofrei – manche Herzerkrankungen "
            "verlaufen ohne spürbare Symptome (sogenannte stille Ischämie), was die "
            "Diagnose sogar erschweren kann."
        ),
        key="h_cp",
    )

    h_trestbps = st.number_input("Blutdruck (in Ruhe)", min_value=0, value=120, key="h_trestbps")
    h_chol = st.number_input("Cholesterin", min_value=0, value=200, key="h_chol")
    h_fbs = st.selectbox("Blutzucker über 120 mg/dl", ["Ja", "Nein"], key="h_fbs")

    # Reihenfolge entspricht dem numerischen Code im Trainingsdatensatz (0-2)
    restecg_options = [
        "normal",
        "leichte Auffälligkeit (ST-T-Veränderung)",
        "verdickte linke Herzkammer (Herzhypertrophie)",
    ]
    h_restecg = st.selectbox(
        "Ergebnis des Ruhe-EKGs",
        restecg_options,
        help=(
            "Ergebnis des EKGs in Ruhe: normal / leichte Veränderung der ST-T-Strecke "
            "/ Hinweis auf eine verdickte linke Herzkammer"
        ),
        key="h_restecg",
    )

    h_thalach = st.number_input("Maximale Herzfrequenz", min_value=0, value=150, key="h_thalach")
    h_exang = st.selectbox(
        "Brustschmerzen bei Belastung",
        ["Ja", "Nein"],
        help="Treten bei körperlicher Anstrengung (z. B. beim Belastungstest) Brustschmerzen auf?",
        key="h_exang",
    )
    h_oldpeak = st.number_input(
        "ST-Streckensenkung unter Belastung (Oldpeak)",
        value=0.0,
        help=(
            "Absenkung der ST-Strecke im EKG unter Belastung im Vergleich zur Ruhephase – "
            "ein Hinweis auf mögliche Durchblutungsstörungen des Herzens. Typische Werte liegen "
            "zwischen 0 und 6."
        ),
        key="h_oldpeak",
    )

    # Reihenfolge entspricht dem numerischen Code im Trainingsdatensatz (0-2)
    slope_options = ["ansteigend", "flach", "absteigend"]
    h_slope = st.selectbox(
        "Verlauf der ST-Strecke unter Belastung",
        slope_options,
        help=(
            "Verlauf des ST-Segments im Belastungs-EKG: ansteigend (meist unauffällig) / "
            "flach / absteigend (Hinweis auf mögliche Durchblutungsstörung)"
        ),
        key="h_slope",
    )

    h_ca = st.number_input(
        "Anzahl großer Gefäße (0-3)",
        min_value=0,
        max_value=3,
        value=0,
        help=(
            "Anzahl der großen Herzkranzgefäße (0 bis 3), die bei einer Herzkatheter-"
            "Untersuchung (Fluoroskopie) sichtbare Verengungen zeigen."
        ),
        key="h_ca",
    )

    # Reihenfolge entspricht dem numerischen Code im Trainingsdatensatz (0-2)
    thal_options = [
        "normal",
        "dauerhafter Durchblutungsdefekt",
        "vorübergehender Durchblutungsdefekt unter Belastung",
    ]
    h_thal = st.selectbox(
        "Ergebnis des Herzdurchblutungstests (Thallium-Szintigrafie)",
        thal_options,
        help=(
            "Ergebnis eines Herz-Durchblutungstests (Thallium-Szintigrafie). Zeigt, ob der "
            "Herzmuskel überall gut durchblutet ist (normal), dauerhaft schlecht durchblutete "
            "Stellen aufweist, zum Beispiel durch einen früheren Infarkt (fixed defect), oder "
            "nur unter Belastung schlechter durchblutet wird, was auf eine Engstelle im "
            "Herzkranzgefäß hindeutet (reversable defect)."
        ),
        key="h_thal",
    )

    if st.button("Risiko berechnen", key="heart_calc"):
        model = joblib.load("src/heart_model.pkl")

        # Codierung wie im Original-Datensatz (data/heart.csv): cp, restecg,
        # slope, thal sind numerisch 0-basiert codiert; die Position der
        # jeweiligen Auswahl in ihrer Options-Liste entspricht diesem Code.
        # One-Hot-Encoding wie beim Training: pd.get_dummies(..., drop_first=True)
        # -> Baseline ist jeweils der Code 0 (cp_0, restecg_0, slope_0, thal_0).
        cp_code = cp_options.index(h_cp)
        restecg_code = restecg_options.index(h_restecg)
        slope_code = slope_options.index(h_slope)
        thal_code = thal_options.index(h_thal)

        input_dict = {
            "age": h_age,
            "sex": 1 if h_sex == "Männlich" else 0,
            "trestbps": h_trestbps,
            "chol": h_chol,
            "fbs": 1 if h_fbs == "Ja" else 0,
            "thalach": h_thalach,
            "exang": 1 if h_exang == "Ja" else 0,
            "oldpeak": h_oldpeak,
            "ca": h_ca,
            "cp_1": 1 if cp_code == 1 else 0,
            "cp_2": 1 if cp_code == 2 else 0,
            "cp_3": 1 if cp_code == 3 else 0,
            "restecg_1": 1 if restecg_code == 1 else 0,
            "restecg_2": 1 if restecg_code == 2 else 0,
            "slope_1": 1 if slope_code == 1 else 0,
            "slope_2": 1 if slope_code == 2 else 0,
            "thal_1": 1 if thal_code == 1 else 0,
            "thal_2": 1 if thal_code == 2 else 0,
        }

        # Spaltenreihenfolge exakt wie beim Training (model.feature_names_in_)
        input_df = pd.DataFrame([input_dict])[model.feature_names_in_]

        # Cast bool-Spalten zu float64, falls nötig (wie im SHAP-Skript)
        for col in input_df.columns:
            if input_df[col].dtype == bool:
                input_df[col] = input_df[col].astype("float64")

        probability = model.predict_proba(input_df)[0][1]
        st.write(f"**Herzkrankheits-Risiko: {probability * 100:.2f} %**")

        # SHAP-Erklärung für genau diese eine Eingabe
        X_background = load_heart_background()[model.feature_names_in_]
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer(input_df)

        # LinearExplainer erklärt den Logit (linearen Output) der LogisticRegression.
        # Lokale Näherung, um daraus Prozentpunkte der Risiko-Wahrscheinlichkeit zu
        # machen: d(Wahrscheinlichkeit)/d(Logit) = p * (1 - p)
        slope = probability * (1 - probability)
        contributions = shap_values.values[0] * slope * 100

        feature_labels = {
            "age": "Alter",
            "sex": "Geschlecht: männlich",
            "trestbps": "Blutdruck (in Ruhe)",
            "chol": "Cholesterin",
            "fbs": "Blutzucker über 120 mg/dl",
            "thalach": "Maximale Herzfrequenz",
            "exang": "Brustschmerzen bei Belastung",
            "oldpeak": "ST-Streckensenkung (Oldpeak)",
            "ca": "Anzahl großer Gefäße",
            "cp_1": "Brustschmerz: untypisch",
            "cp_2": "Brustschmerz: ohne Herzursache",
            "cp_3": "Brustschmerz: keiner (asymptomatisch)",
            "restecg_1": "Ruhe-EKG: leichte Auffälligkeit",
            "restecg_2": "Ruhe-EKG: verdickte linke Herzkammer",
            "slope_1": "ST-Verlauf unter Belastung: flach",
            "slope_2": "ST-Verlauf unter Belastung: absteigend",
            "thal_1": "Durchblutungstest: dauerhafter Defekt",
            "thal_2": "Durchblutungstest: vorübergehender Defekt",
        }

        contribution_pairs = list(zip(model.feature_names_in_, contributions))
        contribution_pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)

        st.write("**Einflussreichste Merkmale:**")
        for feature, value in contribution_pairs[:5]:
            label = feature_labels.get(feature, feature)
            richtung = "erhöht" if value > 0 else "senkt"
            st.write(f"- {label}: {richtung} das Risiko um {abs(value):.2f} %")
