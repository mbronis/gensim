from abc import abstractmethod
import re
from typing import List

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
    """Cleans texts of polish language and filters out texts with low len after cleaning


    Parameters
    ----------
        keep_digits : bool = True
            flag indicating if digits should be kept
        keep_punct : bool = False
            flag indicating if punctuation characters should be kept
        lower : bool = True
            convert text to lower-case after cleaning
        min_token_len : int = 2
            datum with sentence lenght after cleaning lower than `min_token_len` will be removed
    """

    def __init__(self, keep_digits: bool = True, keep_punct: bool = False, lower: bool = False, min_token_len: int = 2) -> None:
        self.keep_digits = keep_digits
        self.keep_punct = keep_punct
        self.lower = lower
        self.min_token_len = min_token_len

    @staticmethod
    def _base_clean(s: str, keep_digits: bool = True, keep_punct: bool = False) -> str:
        """Removes non letter characters from string, optionally keeps digits and punctuation."""
        special = '[\n\r]+'

        pattern_flag = tuple([keep_digits, keep_punct])
        pattern_fact = {
            (False, False): r'[^a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
            (False, True): r'[^a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s!?\'"\.,\-()]',
            (True, False): r'[^0-9a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
            (True, True): r'[^0-9a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s!?\'"\.,\-()]',
        }
        pattern = pattern_fact[pattern_flag]

        s = re.sub(special, ' ', s)
        s = re.sub(pattern, ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()

        return s

    @staticmethod
    def _collapse_exploded(str, separators: str = ' .-_') -> str:
        """Collapse word expanded with ``separators``.

        Example
        --------
        >>> _collapse_exploded('jesteś b r z y d k i')
        'jesteś brzydki'
        """
        if len(str) < 5:
            return str

        remove = []
        for i, l in enumerate(str[2:-1]):
            if l in separators:
                if (str[i - 2] in separators) & (str[i + 2] in separators):
                    if (str[i - 1].isalpha()) & (str[i + 1].isalpha()):
                        remove.append(i)
                        remove.append(i + 2)

        return ''.join([l for i, l in enumerate(str) if i not in remove])

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X_clean = X.copy()

        X_clean['desc_clean'] = X_clean['desc'].apply(TextCleaner._base_clean, self.keep_digits, self.keep_punct)
        X_clean['desc_clean'] = X_clean['desc_clean'].apply(TextCleaner._collapse_exploded)
        if self.lower:
            X_clean['desc_clean'] = X_clean['desc_clean'].str.lower()

        X_clean['len_clean'] = X_clean['desc_clean'].apply(len)
        X_clean['len_clean_ratio'] = X_clean['len_clean'] / X_clean['len']

        X_clean = X_clean[X_clean['len_clean'] >= self.min_token_len]
        X_clean = X_clean.reset_index(drop=True)

        return X_clean


class BaseTokenizer(PassThroughMixin):
    """Base class for sentence tokenization"""

    @staticmethod
    @abstractmethod
    def tokenize(doc: str) -> List[str]:
        pass

    @classmethod
    def transform(cls, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X_copy = X.copy()

        X_copy['tokens'] = X_copy['desc_clean'].apply(cls.tokenize)
        X_copy['tokens_len'] = X_copy['tokens'].apply(len)

        return X_copy


class SimpleTokenizer(BaseTokenizer):
    """Splits words by space"""

    def tokenize(doc: str) -> List[str]:
        return [tok for tok in doc.split()]
