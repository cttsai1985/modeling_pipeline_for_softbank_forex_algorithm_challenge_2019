import pandas as pd
from typing import Optional, Dict


class IFeatureTransformer:
    def fit(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        raise NotImplementedError()

    def fit_transform(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()

    def transform(self, x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()

    def get_params(self) -> Dict:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

