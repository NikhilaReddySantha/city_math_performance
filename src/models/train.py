import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.config import RAW_DIR, MODELS_DIR


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # Load student-level data (local only; ignored by git)
    path = RAW_DIR / "student_level_math_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw data file: {path}\n"
            "Run your notebook to generate it, or run: python -m src.data.make_dataset (if you add that script)."
        )

    df = pd.read_csv(path)

    # Keep numeric features only (drop city/state for now)
    X = df.drop(columns=["math_score", "city", "state"]).select_dtypes(include=[np.number])
    y = df["math_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train two models (Linear + Random Forest)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Quick console report
    print("Holdout Test Metrics")
    print("-" * 40)
    print(f"Linear Regression | R2: {r2_score(y_test, lr_pred):.3f} | RMSE: {rmse(y_test, lr_pred):.3f}")
    print(f"Random Forest     | R2: {r2_score(y_test, rf_pred):.3f} | RMSE: {rmse(y_test, rf_pred):.3f}")

    # Save best model (choose the better R2)
    best_name, best_model, best_pred = (
        ("linear_regression", lr, lr_pred)
        if r2_score(y_test, lr_pred) >= r2_score(y_test, rf_pred)
        else ("random_forest", rf, rf_pred)
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = MODELS_DIR / f"{best_name}.pkl"

    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "columns": list(X.columns),
        },
        outpath,
    )

    print("-" * 40)
    print(f"Saved best model: {best_name} -> {outpath}")


if __name__ == "__main__":
    main()