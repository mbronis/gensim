from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PassThroughMixin(ABC, BaseEstimator, TransformerMixin):
    """Mixin implementing basic transform & fit methods"""

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        pass
