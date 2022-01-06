"""
Word to vec embedder class
"""


from typing import Iterable, List

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


class W2VEmbedder(BaseEstimator, TransformerMixin):
    """
    Word to vec embedder class.
    Allows for vectorization of tokens, which can later be used ad features in other models.

    Sentence is embedded as sum of w2v embeddings of its words

    Wraps gensim.Word2Vec model
    see: https://radimrehurek.com/gensim/models/word2vec.html
    
    
    Params
    ------
        model_name: str = 'w2v'
            Prefix that will be used to name columns in transformed data frame 
        workers: int = 3
            Number of threads while training
        vector_size: int = 100
            Output vector size
        min_count: int = 5
            Ignores tokens with low frequency
        max_final_vocab: int = None
            Max number of tokens to consider.
            Sets `min_count` to match vocab size to this.
        epochs: int = 5
            Number of training epochs
        window: int = 5
            Max distance between words
        alpha: float = 0.025
            Initial learning rate
        min_alpha: float = 0.0001
            Learning rate will decrease lineary to this value while trainig.
        sample: float = 0.001
            Thershold for random downsampling of high frequency tokens (0, 1e-5) 
        sg: int = 0
            Training algo:
            1 - skip-gram algo, 0 - CBOW
        cbow_mean: int = 1
            1 - use mean of context words, 0 - use sum
            Only usable with CBOW
        hs: int = 0
            1 - hierarchical softmax will be used for model training
            0 - if negative is non-zero, negative sampling
        negative: int = 5
            How many "noise words" should be drawn (usually between 5-20)
        ns_exponent: float = 0.75
            The exponent used to shape the negative sampling distribution.
            1.0 samples exactly in proportion to the frequencies
            0.0 samples all words equally
            negative value samples low-frequency words more than high-frequency words
    """
    def __init__(self,
                 model_name: str = 'w2v',
                 workers: int = 8,
                 vector_size: int = 20,
                 min_count: int = 5,
                 max_final_vocab: int = None,
                 epochs: int = 5,
                 window: int = 5,
                 alpha: float = 0.025,
                 sample: float = 0.001,
                 min_alpha: float = 0.0001,
                 sg: int = 0,
                 cbow_mean: int = 1,
                 hs: int = 0,
                 negative: int = 5,
                 ns_exponent: float = 0.75
                 ):
        self.workers=workers
        self.vector_size=vector_size
        self.min_count=min_count
        self.max_final_vocab=max_final_vocab
        self.epochs=epochs
        self.window=window
        self.alpha=alpha
        self.sample=sample
        self.min_alpha=min_alpha
        self.sg=sg
        self.cbow_mean=cbow_mean
        self.hs=hs
        self.negative=negative
        self.ns_exponent=ns_exponent

        self.model: Word2Vec = None
        self.model_name = model_name
        self.default_vector = np.zeros(vector_size)

    def fit(self, sentences: Iterable[List[str]]):
        """Trains w2v model on pd.Series of tokenized sequences"""
        self.model = Word2Vec(
            sentences = sentences,
            workers = self.workers,
            vector_size=self.vector_size,
            min_count=self.min_count,
            max_final_vocab=self.max_final_vocab,
            epochs=self.epochs,
            window=self.window,
            alpha=self.alpha,
            sample=self.sample,
            min_alpha=self.min_alpha,
            sg=self.sg,
            cbow_mean=self.cbow_mean,
            hs=self.hs,
            negative=self.negative,
            ns_exponent=self.ns_exponent)

        return self
    
    def _transform_sequence(self, tokens: List[str]) -> np.ndarray:
        """Embedds single sentence to w2v representation"""
        vectorized = [
            self.model.wv[tok] if self.model.wv.has_index_for(tok) else self.default_vector 
            for tok in tokens]
        
        return sum(vectorized)

    def get_feature_names_out(self) -> List[str]:
        """Return column names of data frame with setntences embeddings"""
        return  [self.model_name + f'_{i}' for i in range(self.model.wv.vector_size)]

    def transform(self, sentences: Iterable[List[str]]) -> pd.DataFrame:
        """
        Return dataframe with sentences embeddings with w2v model.
        Number of columns matches w2v vector size.

        Arguments
        ---------
            sentences: pd.Series[List[str]]
                Series with tokenized sentences to be transformed
        """
        if self.model is None:
            raise AttributeError("Model hasn't been fitted yet. Run `fit` earlier.")
        
        df = pd.DataFrame(map(self._transform_sequence, sentences))
        df.columns = self.get_feature_names_out()

        return df
