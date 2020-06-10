import logging
from typing import List

import pandas as pd
from .IEstimator import IEstimator, ICrossValEstimator
from dask import dataframe as dd
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor
from xgboost import XGBClassifier

#from ..LoggerConfigs import initialize_logger
#logger = initialize_logger()


def _get_xgb_model_importance(model, name: str = 'feature_importance') -> pd.Series:
    return pd.Series(
        model.feature_importances_, index=model.get_booster().feature_names).to_frame(name)


class XGBProbaEstimator(IEstimator):
    """
    """

    def __init__(self, params):
        self.params = params
        self.model = XGBClassifier(**self.params)
        self.is_fitted: bool = False
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def get_feature_importance(self):
        if not self.is_fitted:
            raise NotFittedError()

        if self.feature_importance.empty:
            self.feature_importance = _get_xgb_model_importance(self.model)

        return self.feature_importance

    def get_params(self):
        return self.params.copy()

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        fit_params = kwargs.get('fit_params', dict())
        eval_metric = fit_params.get('eval_metric', 'logloss')
        eval_set = fit_params.get('eval_set', None)
        early_stopping_rounds = fit_params.get('early_stopping_rounds', None)
        callbacks = fit_params.get('callbacks', None)
        sample_weight = fit_params.get('sample_weight', None)
        sample_weight_eval_set = fit_params.get('sample_weight_eval_set', None)

        self.model.fit(x, y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=early_stopping_rounds, verbose=True, xgb_model=None,
                       sample_weight_eval_set=sample_weight_eval_set, callbacks=callbacks)

        self.is_fitted = True
        return self

    def predict(self, x: pd.DataFrame, **kwargs):
        return pd.Series(
            self.model.predict_proba(x, ntree_limit=self.model.best_ntree_limit)[:, -1], index=x.index).rename('target')


class XGBRegEstimator(IEstimator):
    """
    """

    def __init__(self, params):
        self.params = params
        self.model = XGBRegressor(**self.params)
        self.is_fitted: bool = False
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def get_feature_importance(self):
        if not self.is_fitted:
            raise NotFittedError()

        if self.feature_importance.empty:
            self.feature_importance = _get_xgb_model_importance(self.model)

        return self.feature_importance

    def get_params(self):
        return self.params.copy()

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        fit_params = kwargs.get('fit_params', dict())
        eval_metric = fit_params.get('eval_metric', 'mae')
        eval_set = fit_params.get('eval_set', None)
        early_stopping_rounds = fit_params.get('early_stopping_rounds', 100)
        callbacks = fit_params.get('callbacks', None)
        sample_weight = fit_params.get('sample_weight', None)
        sample_weight_eval_set = fit_params.get('sample_weight_eval_set', None)

        self.model.fit(x, y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=early_stopping_rounds, verbose=True, xgb_model=None,
                       sample_weight_eval_set=sample_weight_eval_set, callbacks=callbacks)

        self.is_fitted = True
        return self

    def predict(self, x: pd.DataFrame, **kwargs):
        return pd.Series(
            self.model.predict(x, ntree_limit=self.model.best_ntree_limit), index=x.index).rename('target')


class XGBRegCVEstimator(ICrossValEstimator):
    def __init__(self, params):
        self.params = params
        self.model_gen: IEstimator = XGBRegEstimator
        self.models: List[IEstimator] = list()

        self.is_fitted: bool = False
        self.feature_importance: pd.DataFrame = pd.DataFrame()

    def get_feature_importance(self):
        if not self.is_fitted:
            raise NotFittedError()

        if self.feature_importance.empty:
            df = pd.concat([model.get_feature_importance() for model in self.models], axis=1)
            self.feature_importance = df.rename(
                columns={col: f"fold_{i:02d}_{col}" for i, col in enumerate(df.columns)})

        return self.feature_importance

    def fit(self, x: pd.DataFrame, y: pd.Series, cv_splits, **kwargs):
        template_fit_params = kwargs.get('fit_params', dict())

        groups = kwargs.get("groups", None)
        for i, (index_train, index_valid) in enumerate(cv_splits.split(x, y=y, groups=groups), 0):
            model = self.model_gen(self.params)

            train_x = x.iloc[index_train]
            train_y = y.iloc[index_train]
            valid_x = x.iloc[index_valid]
            valid_y = y.iloc[index_valid]

            fit_params = template_fit_params.copy()
            fit_params['eval_set'] = [(train_x, train_y), (valid_x, valid_y)]
            # fit_params['sample_weight'] = None
            # fit_params['sample_weight_eval_set'] = None

            model.fit(train_x, train_y, fit_params=fit_params)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, x: pd.DataFrame, **kwargs):
        return pd.concat([model.predict(x) for model in self.models], axis=1).mean(axis=1).rename('target')
