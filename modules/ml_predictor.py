import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_delay_model(df):
    try:
        data = df.copy()

        data["Departure_Time"] = pd.to_datetime(data["Departure_Time"], errors="coerce")
        data["Departure_Hour"] = data["Departure_Time"].dt.hour

        data["Delayed_Target"] = (data["Delay_Minutes"] > 0).astype(int)

        features = [
            "Airline",
            "Origin",
            "Destination",
            "Gate_Number",
            "Departure_Hour",
            "Cargo_Weight",
            "Baggage_Count",
        ]

        data = data.dropna(subset=features + ["Delayed_Target"])

        X = data[features]
        y = data["Delayed_Target"]

        categorical_cols = ["Airline", "Origin", "Destination", "Gate_Number"]
        numeric_cols = ["Departure_Hour", "Cargo_Weight", "Baggage_Count"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        report_text = classification_report(
            y_test,
            preds,
            target_names=["On Time", "Delayed"],
            zero_division=0,
        )

        classifier = model.named_steps["classifier"]
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": classifier.feature_importances_,
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        return model, accuracy, importance_df, report_text, None

    except Exception as e:
        return None, None, None, None, str(e)