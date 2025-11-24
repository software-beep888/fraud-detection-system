import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV transactions dataset."""
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove nulls and duplicates."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df
