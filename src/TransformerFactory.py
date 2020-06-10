from typing import Dict
from DaskFeatureTransformer import DaskTruncatedSVD


def transformer_factory(transformer_gen: str, params: Dict):
    if transformer_gen not in globals().keys():
        err_msg = f"{transformer_gen} is not in transformer factory"
        raise ValueError(err_msg)

    return globals()[transformer_gen](**params)
