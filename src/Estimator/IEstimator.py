from typing import Dict, Optional
import pandas as pd


class IEstimator(object):
    """
    """
    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_params(self) -> Dict:
        raise NotImplementedError()

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        raise NotImplementedError()

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()


class ICrossValEstimator(object):
    """
    Estimator w/ CV
    """
    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_params(self) -> Dict:
        raise NotImplementedError()

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        raise NotImplementedError()

    def predict(self, x, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()
