"""Module with various utility functions and classes"""

import os
import logging
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from src.git_parser import GitParser

logging.basicConfig(
    format='%(levelname)s:%(asctime)s:%(name)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class Logger:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)

    def log(self, message: str) -> None:
        self.logger.info(message)


def get_pipeline_params(pipe: Pipeline) -> Dict:
    """Extracts key params from pipeline"""
    params = {k: v for k, v in pipe.get_params().items() if '__' in k}
    steps = [x[0] for x in pipe.get_params()['steps']]
    transformers = [str(x[1]) for x in pipe.get_params()['steps']]
    params['steps'] = steps
    params['transformers'] = transformers

    return params


def prep_df_metadata(X: pd.DataFrame, pipe: Pipeline, git_parser: GitParser, ts: str):
    """Prepares metadata of pipeline, resulting df and git info"""
    params = get_pipeline_params(pipe)
    working_dir, branch_name, commit_message, commit_sha = git_parser.get_info()
    shape = X.shape
    cols = list(X.columns)

    metadata_data = (ts, params, working_dir, branch_name, commit_message, commit_sha, shape, cols)
    metadata_columns = ['ts', 'params', 'working_dir', 'git_branch', 'git_commit', 'git_sha', 'shape', 'columns']
    metadata_df = pd.Series(metadata_data, index=metadata_columns).to_frame().T

    return metadata_df

def mem_used() -> float:
    """Returns % of memory used."""

    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])

    return round(100 * used_memory / total_memory, 1)
