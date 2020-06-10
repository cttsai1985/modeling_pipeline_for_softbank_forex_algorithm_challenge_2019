from multiprocessing import cpu_count
from typing import List, Optional

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask_ml.decomposition import TruncatedSVD

from IFeatureTransformer import IFeatureTransformer


def _make_dask_dataframe(data: pd.DataFrame, npartitions=cpu_count(), chunksize=None):
    return dd.from_pandas(data, npartitions=npartitions, chunksize=chunksize, sort=False, name=None)


def _make_transform_dataframe(data, orig: pd.DataFrame):
    df = pd.DataFrame(data, ).astype(np.float32)
    orig.index = df.index
    df.index.name = orig.index.name
    return df


class DaskTruncatedSVD(IFeatureTransformer):
    def __init__(
            self, columns: Optional[List[str]], target_name: str, n_components: int = 2, n_iter: int = 5,
            random_state: Optional[int] = 42, tol: float = 0.0):
        self.n_components = n_components
        self.algorithm = 'tsqr'
        self.n_iter: int = n_iter
        self.random_state: int = random_state
        self.tol: float = tol

        self.target_name: str = target_name
        self.columns: List[str] = columns
        self.columns_transform: Optional[List[str]] = None
        self.is_fitted: bool = False

        self.transformer = TruncatedSVD(
            n_components=self.n_components, algorithm=self.algorithm, n_iter=self.n_iter,
            random_state=self.random_state, tol=self.tol)

    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        self.fit_transform(x=x, y=y, **kwargs)
        return self

    def fit_transform(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        data = self.transformer.fit_transform(_make_dask_dataframe(x[self.columns]).values).compute()
        df = _make_transform_dataframe(data, x)
        self.is_fitted = True
        self.columns_transform = [f'svd_{self.target_name}_{i:02d}' for i, _ in enumerate(df.columns, 1)]
        df.columns = self.columns_transform
        return df

    def transform(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        if not self.is_fitted:
            raise NotFittedError()

        data = self.transformer.transform(_make_dask_dataframe(x[self.columns]).values).compute()
        df = _make_transform_dataframe(data, x)
        df.columns = self.columns_transform
        return df

    def get_params(self):
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return self.is_fitted
