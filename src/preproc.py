import re

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


class TextCleaner(PassThroughMixin):
    """Cleans texts of polish language and filters failed datum"""

    def __init__(self, keep_digits: int = 1, keep_punct: int = 0) -> None:
        self.keep_digits = keep_digits
        self.keep_punct = keep_punct

    @staticmethod
    def clean_text(s: str, keep_digits: int = 1, keep_punct: int = 0) -> str:
        """Removes non letter characters from string, optionally keeps digits and punctuation."""
        special = '[\n\r]+'
        
        pattern_flag = tuple([keep_digits, keep_punct])
        pattern_fact = {
            (0, 0): r'[^a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
            (0, 1): r'[^a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s!?\'"\.,\-()]',
            (1, 0): r'[^0-9a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
            (1, 1): r'[^0-9a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s!?\'"\.,\-()]',
        }
        pattern = pattern_fact[pattern_flag]
        
        s = re.sub(special, ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(pattern, '', s)
        s = s.strip()
        
        return s

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X_clean = X.copy()

        cleaner = lambda x: TextCleaner.clean_text(x, keep_digits=self.keep_digits, keep_punct=self.keep_punct)
        X_clean['desc_clean'] = X_clean['desc'].apply(cleaner)
        X_clean['len_clean'] = X_clean['desc_clean'].apply(len)
        X_clean['len_clean_ratio'] = X_clean['len_clean'] / X_clean['len']
        X_clean = X_clean[X_clean['len_clean'] > 1]

        return X_clean
