import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


def train_model(df, feature_cols):
    model = IsolationForest(
        n_estimators=100, contamination=0.01, random_state=42)
    model.fit(df[feature_cols])
    joblib.dump(model, 'fraud_model.pkl')
    return model


if __name__ == "__main__":
    # Create dummy dataset
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(1000, 5), columns=[
                      f'feature_{i}' for i in range(5)])
    feature_cols = df.columns.tolist()

    train_model(df, feature_cols)
    print("✅ Dummy model trained and saved as fraud_model.pkl")
