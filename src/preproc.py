"""Module with classes for raw data preprocessing"""


from abc import abstractmethod
import re
from typing import List, Sequence

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

    def __init__(self, keep_digits: bool = True, keep_punct: bool = False, lower: bool = True, min_token_len: int = 2):
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

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X_clean = X.copy()

        X_clean['desc_clean'] = X_clean['desc'].apply(TextCleaner._base_clean, self.keep_digits, self.keep_punct)
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

    @staticmethod
    def tokenize(doc: str) -> List[str]:
        return [tok for tok in doc.split()]


class ExtractorMixin:
    """Mixin providing method for extracting special items from text, eg: hashtags/emails/links extractors."""

    @staticmethod
    def _extract(docs: Sequence[str], pattern: str) -> Sequence[List[str]]:
        """Return list of items extracted from doc in docs with `pattern`"""

        return docs.apply(lambda doc: re.findall(pattern, doc))


class UserReferenceProcessor(PassThroughMixin, ExtractorMixin):
    """Class for extracting and processing user references found in text"""

    pattern = r'@[^ ]+'

    def transform(self, docs: Sequence[str]) -> pd.DataFrame:
        extracted = self._extract(docs, UserReferenceProcessor.pattern)
        extracted = extracted.apply(lambda x: [user[1:] for user in x])

        df = pd.DataFrame(index=docs.index)
        df['users_mentioned'] = extracted
        df['users_mentioned_count'] = extracted.apply(len)

        return df


class SpecialTokensRemover(PassThroughMixin):
    """Class for removing/replacing special tokens (hashtags, user references, emails, links) from texts


    Attributes
    ----------
        mask : bool (True)
            relace special token with mask, eg. '@john hello there!' -> 'user_reference hello there!'
        user_reference : bool (True)
            process user references
        hashtag : bool (True)
            process hashtags
        link : bool (True)
            process http(s) links
        email : bool (True)
            process emails
        pattern : Dict[str, str]
            dictionary with regex'es matching special tokens
    """

    patterns = {
        'user_reference': r'@[^ ]+',
        'hashtag': r'#[^ ]+',
        'link': r'http[s]*://[^ ]+',
        'email': r'[a-zA-z0-9.-_]+@[a-zA-z0-9.-_]+',
    }

    def __init__(self, mask: bool = True, user_reference: bool = True, hashtag: bool = True, link: bool = True, email: bool = True) -> None:
        """"""
        self.mask = mask
        self.user_reference = user_reference
        self.hashtag = hashtag
        self.link = link
        self.email = email

    def transform(self, docs: Sequence[str], y: Sequence = None) -> pd.DataFrame:
        for special_token, pattern in SpecialTokensRemover.patterns.items():
            if getattr(self, special_token):
                mask = special_token if self.mask else ''
                docs = docs.apply(lambda doc: re.sub(pattern, mask, doc))

        return docs


class Extractor(PassThroughMixin):
    """Base class for creating text extractors, eg. hashtags/emails/links extractors."""

    @staticmethod
    def _extract(doc: str) -> List[str]:
        """extracts all user references from text, returns a list with references found"""

        extracted = re.findall(super().pattern, doc)
        return extracted

    @staticmethod
    def transform(y: Sequence[str]) -> Sequence[List[str]]:
        return y.apply(super()._extract)


class UserReferenceExtractor(Extractor):
    """Returns list of users referenced in tweet texts."""
    pattern = r'@[^ ]+'


class HashtagsExtractor(Extractor):
    """Returns list of hashtags extracted from tweet texts"""
    pattern = r'#[^ ]+'


class LinksExtractor(Extractor):
    """Returns list of links extracted from tweet texts"""
    pattern = r'http[s]*://[^ ]+'
