import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Class for text preprocessing. Cleans text before applying NLP methods.
    
    Cleans user references, emails, urls, punctuation, repeated chars, etc.
    Can replace polish diacritics with latin substitute or convert to lowercase.

    Expects pd.Series with text data as imput.
    
    Attributes
    ----------
        clean_user_ref: bool = True
            Removes user references, eg.: @john_doe
        clean_url: bool = True
            Removes urls from text
        clean_email: bool = True
            Removes emials from text
        clean_hashtag: bool = True
            Removes hashtags from text, eg.: #some_hashtag
        clean_emoji: bool = True
            Removes emojis from text
        clean_non_alpha: bool = True
            Removes all non-alpha chars from text
        clean_non_letter: bool = True
            Removes all non-latin chars (except polish diacritics) from text
        latinize: bool = True
            Replaces polish diacritics with latin substitutes
        drop_repeated: bool = True
            Removes chars repeated more than 2 times, eg.: 'rrr' -> 'rr'
        to_lower: bool = True
            Converts all chars to lowercase
    
    """    
    patterns = {
        'user_ref': r'@[^ ]+',
        'url': r'http[s]*://[^ ]+',
        'email': r'[a-zA-z0-9.-_]+@[a-zA-z0-9.-_]+',
        'hashtag': r'#[^ ]+',
        'digits': r'[0-9]+',
        'punctuation': r'[!?\'"\.,\-()\\\[\]{};„”]+',
        'non_alpha':  r'[^0-9a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
        'non_letter':  r'[^a-ząćęłńóśżźA-ZĄĆĘŁŃÓŚŻŹ\s]',
    }

    @staticmethod
    def _clean_emoji(text: str) -> str:
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    @staticmethod
    def _clean_spaces(text: str) -> str:
        return re.sub(' +', ' ', text).strip()

    @staticmethod
    def _to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def _latinize(text: str) -> str:
        pl = "ĄĆĘŁŃÓŚŻŹąćęłńóśżź"
        latin = "ACELNOSZZacelnoszz"
        trans_table = text.maketrans(pl, latin)
        return text.translate(trans_table)

    @staticmethod
    def _drop_repeated(text: str) -> str:
        return re.sub("(.)\\1{2,}", "\\1", text)

    @staticmethod
    def _drop_rt_tag(text: str) -> str:
        return re.sub(r'^RT ', '', text)
    
    @staticmethod
    def _cleaner_factory(pattern_name: str):
        """Returns function that cleans given pattern from string"""

        if pattern_name not in TextCleaner.patterns:
            msg = f"Pattern {pattern_name} not recognized. "
            msg += f"Use one of: {list(TextCleaner.patterns.keys())}."
            raise AttributeError(msg)
        
        pattern = TextCleaner.patterns[pattern_name]

        return lambda text: re.sub(pattern, '', text)
    
    def __init__(self,
                 clean_user_ref: bool = True,
                 clean_url: bool = True,
                 clean_email: bool = True,
                 clean_hashtag: bool = True,
                 clean_emoji: bool = True,
                 clean_non_alpha: bool = True,
                 clean_non_letter: bool = True,
                 latinize: bool = True,
                 to_lower: bool = True,
                 drop_repeated: bool = True
                ):
        self.clean_user_ref = clean_user_ref
        self.clean_url = clean_url
        self.clean_email = clean_email
        self.clean_hashtag = clean_hashtag
        self.clean_emoji = clean_emoji
        self.clean_non_alpha = clean_non_alpha
        self.clean_non_letter = clean_non_letter
        self.latinize = latinize
        self.to_lower = to_lower
        self.drop_repeated = drop_repeated

        self.functions: list = list()
    
    def fit(self, sentences: pd.Series):
        """Initialize cleaning functions"""
        
        self.functions.append(TextCleaner._drop_rt_tag)

        if self.clean_user_ref: self.functions.append(TextCleaner._cleaner_factory('user_ref'))
        if self.clean_url: self.functions.append(TextCleaner._cleaner_factory('url'))
        if self.clean_email: self.functions.append(TextCleaner._cleaner_factory('email'))
        if self.clean_hashtag: self.functions.append(TextCleaner._cleaner_factory('hashtag'))
        if self.clean_emoji: self.functions.append(TextCleaner._clean_emoji)
        if self.clean_non_alpha: self.functions.append(TextCleaner._cleaner_factory('non_alpha'))
        if self.clean_non_letter: self.functions.append(TextCleaner._cleaner_factory('non_letter'))
        if self.latinize: self.functions.append(TextCleaner._latinize)
        if self.to_lower: self.functions.append(TextCleaner._to_lower)
        if self.drop_repeated: self.functions.append(TextCleaner._drop_repeated)

        self.functions.append(TextCleaner._clean_spaces)        

        return self
    
    def transform(self, sentences: pd.Series) -> pd.Series:
        """Applies all initialized cleaning functions"""

        sentences_clean = sentences.copy()
        if not isinstance(sentences_clean, pd.Series):
            sentences_clean = pd.Series(sentences_clean)

        for fn in self.functions:
            sentences_clean = map(fn, sentences_clean)
    
        sentences_clean = pd.Series(sentences_clean)

        if len(sentences_clean) == len(sentences):
            sentences_clean.index = sentences_clean.index

        return sentences_clean
