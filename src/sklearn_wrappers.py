import pandas as pd
from sklearn.compose import ColumnTransformer

from src.transformers import PassThroughMixin


class CustomColumnTransformer(ColumnTransformer):

    def fit(self, X, y=None):
        self._feature_names_in = list(X.columns)
        return super().fit(X, y=y)

    def transform(self, X):
        column_names_in = list(X.columns)
        print(column_names_in)
        transformed = super().transform(X)
        transformed_df = pd.DataFrame(transformed, index=X.index, columns=list(X.columns))

        return transformed_df

    def fit_transform(self, X, y=None):
        super().fit_transform(X, y=y)
        return self.transform(X)
