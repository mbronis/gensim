"""A script for running preprocessing on raw data

Main steps are defined as Pipeline allowing for easy parametrization
Preprocessed data is pickled for later usage in models.
"""


from datetime import datetime

from src.git_parser import GitParser
from src.utils import Logger, prep_df_metadata
from src.exceptions import GitIsDirtyException
from src.pipelines import data_preproc_pipe
from src.io import CsvDataLoader, DataPickler


if __name__ == '__main__':
    """Reads raw data, run preprocessing, save the data and meta info"""
    logger = Logger('data_prep')

    logger.log('checking git status')
    git_parser = GitParser()
    if git_parser.is_dirty():
        raise GitIsDirtyException()

    logger.log('load raw data')
    loader = CsvDataLoader(file_name='polish_sentiment_dataset.csv')
    df_raw = loader.transform(None)
    logger.log(f'loaded {len(df_raw)} rows')

    logger.log('processing data')
    pipe = data_preproc_pipe
    df_preproc = pipe.fit_transform(df_raw)
    logger.log(f'rows after preprocessing: {len(df_preproc)}')

    logger.log('saving data')
    curr_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_pickler = DataPickler(save_dir=curr_ts, file_name='df_prepro.pkl')
    data_pickler.save(df_preproc)

    logger.log('saving meta')
    metadata = prep_df_metadata(X=df_preproc, pipe=pipe, git_parser=git_parser, ts=curr_ts)
    metadata_pickler = DataPickler(save_dir=curr_ts, file_name='df_prepro_metadata.pkl')
    metadata_pickler.save(metadata)

    logger.log(f'saved all to: {curr_ts} df_preproc shape: {df_preproc.shape}')
