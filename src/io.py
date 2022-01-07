"""Module with classes for loading and saving data/metadata/models"""


import os
import pickle
from abc import abstractmethod
from typing import List

import toml
import pandas as pd

from src.transformers import PassThroughMixin
from src.exceptions import BadPickleLoaderDataType, BadDatasetName
from src.utils import Logger


config = toml.load('config/settings.toml')
logger = Logger('io')


class DataLoader(PassThroughMixin):
    """Base class for data loaders with interface for Pipeline usage"""

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.load()


class CsvDataLoader(DataLoader):
    """Loading data from local csv"""

    def __init__(self,
                 folder: str = './data/',
                 file_name: str = 'polish_sentiment_dataset.csv',
                 encoding: str = None,
                 names: List[str] = None):
        self.folder = folder
        self.file_name = file_name
        self.encoding = encoding
        self.names = names

    def load(self) -> pd.DataFrame:
        path = os.path.join(self.folder, self.file_name)

        logger.log(f'loading from csv: {path}')
        data = pd.read_csv(path, encoding=self.encoding, names=self.names)
        logger.log(f'loaded {len(data)} rows')

        return data

def csv_loader_factory(dataset_name: str) -> CsvDataLoader:
    if dataset_name not in config['datasets']:
        raise BadDatasetName(dataset_name, list(config['datasets'].keys()))

    folder = config['data']['FOLDER']
    csv_loader = CsvDataLoader(folder=folder, **config['datasets'][dataset_name])

    return csv_loader


class PickleLoader(DataLoader):
    """Loading preprocessed data from pickle"""
    file_name = {
        'data': 'df_prepro.pkl',
        'meta': 'df_prepro_metadata.pkl'
    }
    folder = config['data']['FOLDER']

    def __init__(self, save_dir: str = None, data_type: str = 'data'):
        self.save_dir = save_dir or PickleLoader.get_latest_dir()
        if data_type not in PickleLoader.file_name:
            raise BadPickleLoaderDataType(data_type)
        self.data_type = data_type

    def load(self) -> pd.DataFrame:
        file_name = PickleLoader.file_name[self.data_type]
        path = os.path.join(PickleLoader.folder, self.save_dir, file_name)

        logger.log(f'loading from pickle: {path}')
        data = pd.read_pickle(path)
        logger.log(f'loaded {len(data)} rows')
        return data

    @staticmethod
    def list_dir() -> List[str]:
        """Returns existing save directories"""
        dirs = [os.path.basename(p[0]) for p in os.walk(PickleLoader.folder)][1:]
        return dirs

    @staticmethod
    def get_latest_dir() -> List[str]:
        """Returns the latest save directory"""
        return sorted(PickleLoader.list_dir())[-1]

class DataSaver(PassThroughMixin):
    """Base class for data saver with interface for Pipeline usage"""

    def __init__(self, save_dir: str, file_name: str) -> None:
        folder = config['data']['FOLDER']
        self.save_dir = os.path.join(folder, save_dir)
        self.save_path = os.path.join(self.save_dir, file_name)

    @abstractmethod
    def save(self, save_obj) -> None:
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        save_obj = (X, y) if y else X
        self.save(save_obj)

        return X


class DataPickler(DataSaver):
    """Class for pickling pipeline results"""

    def save(self, save_obj) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(self.save_path, 'wb') as f:
            pickle.dump(save_obj, f, protocol=4)
