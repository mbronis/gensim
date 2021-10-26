"""Module with predefined pipelines.

There are pipelines for raw data preprocessing and for training models.
"""

from sklearn.pipeline import Pipeline

from src.io import CsvDataLoader
from src.preproc import RawDataCleaner, TextCleaner, SimpleTokenizer


data_preproc_pipe = Pipeline([
    ('load_data', CsvDataLoader(file_name='polish_sentiment_dataset.csv')),
    ('basic_cleaning', RawDataCleaner()),
    ('text_cleaning', TextCleaner()),
    ('tokenizer', SimpleTokenizer())
])
