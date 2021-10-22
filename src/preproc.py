import pandas as pd

from src.transformers import PassThroughMixin


class RawDataCleaner(PassThroughMixin):
    """Removes failed datum and strips desc"""

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X_clean = X.copy()
        X_clean['desc'] = X_clean['description'].str.strip()
        X_clean = X_clean[X_clean['desc'].apply(lambda x: isinstance(x, str))]
        X_clean = X_clean[X_clean['rate'].fillna(0) != 0]
        X_clean['len'] = X_clean['desc'].apply(len)
        X_clean['negative'] = 1 - X_clean['rate'].clip(0)

        X_clean = X_clean[['desc', 'len', 'negative']]
        X_clean = X_clean.reset_index(drop=True)

        return X_clean
