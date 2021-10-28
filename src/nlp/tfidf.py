"""Module with classes for tfidf embeddings"""


from typing import Iterable

import scipy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from src.nlp.datatypes import Tokens
from src.exceptions import TfIdfNotFitted
from src.utils import Logger


logger = Logger('tfidf')


class TfIdfTransformer(BaseEstimator, TransformerMixin):
    """Class for training tfidf representation of texts"""

    def __init__(self,
                 normalize: bool = True, smartirs: str = None, slope: float = 0.25,
                 no_below: int = 5, no_above: float = 0.5, keep_n: int = 5000) -> None:
        self.normalize: bool = normalize
        self.smartirs: str = smartirs
        self.slope: float = slope
        self.no_below: int = no_below
        self.no_above: float = no_above
        self.keep_n: int = keep_n

        self.vocab: Dictionary = None
        self.model: TfidfModel = None

    def fit(self, docs: Iterable[Tokens], y: Iterable = None):

        # train dictionary
        logger.log("fitting vocabulary")
        self.vocab = Dictionary(documents=docs)

        # filter vocab entries
        filter_extremes_params = {
            'no_below': self.no_below,
            'no_above': self.no_above,
            'keep_n': self.keep_n,
        }
        filter_extremes_params = {k: v for k, v in filter_extremes_params.items() if v is not None}
        if filter_extremes_params:
            self.vocab.filter_extremes(**filter_extremes_params)
        logger.log(f"vocabulary fitted. unique tokens count: {len(self.vocab)}")

        # train tfifd model
        logger.log("fitting tfidf model")
        model_params = {
            'normalize': self.normalize,
            'smartirs': self.smartirs,
            'slope': self.slope,
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}
        self.model = TfidfModel(dictionary=self.vocab, **model_params)
        logger.log("tfidf model fitted")

        return self

    def transform(self, docs: Iterable[Tokens], y: Iterable = None):
        if not (self.vocab and self.model):
            raise TfIdfNotFitted()

        logger.log("creating corpus")
        corpus = [self.vocab.doc2bow(doc) for doc in docs]
        logger.log(f"corpus created. number of docs: {len(corpus)}")
        logger.log("vectorizing corpus with tfidf model")
        tfidf_vectors = [self.model[bow] for bow in corpus]

        # sparse matrix reprezentation
        logger.log("creating sparse corpus representation")
        i, j, data = zip(*((i, t[0], t[1]) for i, row in enumerate(tfidf_vectors) for t in row))
        tfidf_matrix = scipy.sparse.coo_matrix((data, (i, j)), shape=(len(corpus), len(self.vocab)))

        logger.log("creating sparse df")
        columns = [self.vocab[i] for i in range(len(self.vocab))]
        df_sparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=columns)

        logger.log("creating dense df")
        df_dense = df_sparse.sparse.to_dense()

        return df_dense
