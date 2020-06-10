import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

from sklearn.model_selection import GroupKFold

from DataHandler import CSVDataHandler, ParquetDataHandler
from EstimatorFactory import estimator_factory


def casting_data(data: pd.DataFrame):
    return data.astype({c: np.float32 for c in filter(lambda x: x.startswith('feature'), data.columns)})


def get_train_test(train: pd.DataFrame, test: pd.DataFrame):
    train.set_index("id", inplace=True)
    train_y = train.pop('target')
    train_x = train

    test.set_index("id", inplace=True)
    test_x = test
    return train_x, train_y, test_x


def main():
    folder_path = "../input/softbank_comp1_data/competition_data/"
    train = casting_data(CSVDataHandler(os.path.join(folder_path, "train.csv")).load().get())
    test = casting_data(CSVDataHandler(os.path.join(folder_path, "test.csv")).load().get())
    train_x, train_y, test_x = get_train_test(train, test)
    groups = train_x.index.tolist()

    # modeling params
    model_params = {
        'tree_method': 'gpu_hist', 'colsample_bytree': 0.7, 'gamma': 0.05, 'learning_rate': 0.05, 'max_depth': 3,
        'min_child_weight': 2., 'n_estimators': 250, 'reg_alpha': 0.5, 'reg_lambda': 1., 'subsample': 0.7,
        'silent': 1, 'random_state': 7, 'nthread': -1
    }
    fit_params = {
        'eval_metric': ['mae',],
        'early_stopping_rounds': 25,
    }
    model = estimator_factory("XGBRegCVEstimator", model_params)
    model.fit(train_x, train_y, cv_splits=GroupKFold(n_splits=10), fit_params=fit_params, groups=groups)

    preds = model.predict(test_x[train_x.columns]).to_frame('target').sort_index(ascending=False)
    print(f"describe:\n{preds.describe()}")
    preds.to_csv("../result/submission_baseline.csv")

    # obj_save = ParquetDataHandler(os.path.join(folder_path, "text_features.parquet"))
    # obj_save.update(df).save()
    return


if "__main__"  == __name__:
    main()