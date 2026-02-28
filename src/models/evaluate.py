import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from src.config import RAW_DIR, MODELS_DIR, METRICS_DIR


def main():
    # Load raw student-level data
    df = pd.read_csv(RAW_DIR / "student_level_math_data.csv")
    X = df.drop(columns=["math_score", "city", "state"]).select_dtypes(include=[np.number])
    y = df["math_score"]

    # Load whichever model was saved by train.py
    # (it saves either linear_regression.pkl or random_forest.pkl)
    candidates = [MODELS_DIR / "linear_regression.pkl", MODELS_DIR / "random_forest.pkl"]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No saved model found. Run: python -m src.models.train")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    model_name = artifact["model_name"]

    # Holdout evaluation (same split seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    preds = model.predict(X_test)

    test_r2 = float(r2_score(y_test, preds))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    cv_mean_r2 = float(cv_scores.mean())

    # If model has feature_importances_ (Random Forest), save them
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = {
            col: float(val)
            for col, val in sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda t: t[1],
                reverse=True,
            )
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = METRICS_DIR / "metrics.json"

    payload = {
        "model_name": model_name,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "cv_scores_r2": [float(x) for x in cv_scores],
        "cv_mean_r2": cv_mean_r2,
        "feature_importance": feature_importance,
    }

    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics to: {outpath}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()