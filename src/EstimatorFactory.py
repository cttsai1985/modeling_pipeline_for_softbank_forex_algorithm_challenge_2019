from typing import Dict
from Estimator import XGBProbaEstimator
from Estimator import XGBRegEstimator
from Estimator import XGBRegCVEstimator


def estimator_factory(estimator_gen: str, params: Dict):
    if estimator_gen not in globals().keys():
        err_msg = f"{estimator_gen} is not in transformer factory"
        raise ValueError(err_msg)

    return globals()[estimator_gen](params)
