import numpy as np

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for fraud detection."""
    df['amount_log'] = df['amount'].apply(lambda x: 0 if x <= 0 else np.log(x))
    # Example feature: transaction hour
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    return df
