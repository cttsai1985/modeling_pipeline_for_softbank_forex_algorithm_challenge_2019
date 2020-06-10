import os
import typing
from typing import Optional

import numpy as np
import pandas as pd
from dask import dataframe as dd

from Decorator import timeit
from IFileHandler import IFileHandler
from LoggerConfigs import initialize_logger
logger = initialize_logger()


class DataCheckHelper:
    def _check_load(self) -> bool:
        if self.data is None:
            err_msg = f"data is not loaded yet"
            raise ValueError(err_msg)

        elif not isinstance(self.data, pd.DataFrame):
            err_msg = f"data is not a pd.DataFrame yet"
            raise ValueError(err_msg)

        return True

    def _display_load_file(self):
        logger.info(f'Load file: {self.file_path}')
        return self

    def _check_file_exist(self) -> bool:
        if not os.path.exists(self.file_path):
            err_msg = f"{self.file_path} does not exist."
            raise ValueError(err_msg)

        return True

    def _update_data(self, data: pd.DataFrame):
        if self.data is None and isinstance(data, pd.DataFrame):
            self.data = data.copy()
            return self

        elif self.data.shape[0] != data.shape[0]:
            err_msg = f"data sets do not match {self.data.shape}, {data.shape}"
            raise ValueError(err_msg)

        self.data = data.copy()
        return self

    def _get_file_path(self, file_path: Optional[str]) -> str:
        if not file_path:
            return self.file_path

        if not isinstance(file_path, str):
            err_msg = f'{file_path} is not str'
            raise TypeError(err_msg)

        return file_path


class CSVDataDaskHandler(IFileHandler, DataCheckHelper):
    def __init__(self, file_path: str, **kwargs):
        """Assumed *.csv are not indexed"""
        self.file_path: str = file_path
        self.data: Optional[pd.DataFrame] = None

    @timeit
    def load(self, **kwargs):
        #self._check_file_exist()
        self._display_load_file()
        self.data = dd.read_csv(self.file_path, **kwargs).compute()
        return self

    def save(self, file_path: Optional[str] = None):
        file_path = self._get_file_path(file_path)
        self.data.to_csv(file_path, index=False)
        logger.info(f"save {self.data.shape} into {file_path}")
        return self

    def get(self):
        self._check_load()
        return self.data

    def update(self, data):
        self._update_data(data)
        return self


class CSVDataHandler(IFileHandler, DataCheckHelper):
    def __init__(self, file_path: str, **kwargs):
        """Assumed *.csv are not indexed"""
        self.file_path: str = file_path
        self.data: Optional[pd.DataFrame] = None

    @timeit
    def load(self, **kwargs):
        self._check_file_exist()
        self._display_load_file()
        self.data = pd.read_csv(self.file_path, **kwargs)
        return self

    def save(self, file_path: Optional[str] = None):
        file_path = self._get_file_path(file_path)
        self.data.to_csv(file_path, index=False)
        logger.info(f"save {self.data.shape} into {file_path}")
        return self

    def get(self):
        self._check_load()
        return self.data

    def update(self, data):
        self._update_data(data)
        return self


class ParquetDataHandler(IFileHandler, DataCheckHelper):
    def __init__(self, file_path: str, **kwargs):
        """Assumed *.parquet are indexed"""
        self.file_path: str = file_path
        self.data: Optional[pd.DataFrame] = None

    def load(self, **kwargs):
        self._check_file_exist()
        self._display_load_file()
        self.data = pd.read_parquet(self.file_path, **kwargs)
        return self

    def save(self, file_path: Optional[str] = None):
        file_path = self._get_file_path(file_path)
        self.data.to_parquet(file_path, index=True)
        logger.info(f"save {self.data.shape} into {file_path}")
        return self

    def get(self):
        self._check_load()
        return self.data

    def update(self, data):
        self._update_data(data)
        return self
