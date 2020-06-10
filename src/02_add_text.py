import os
from functools import partial
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from dask import dataframe as dd
from cuml import TruncatedSVD, TSNE
from cuml.random_projection import GaussianRandomProjection

from DataHandler import CSVDataHandler, ParquetDataHandler

from TransformerFactory import transformer_factory


def dask_mean_data(data: pd.DataFrame):
    #df_mean = dd.from_pandas(data,  npartitions=cpu_count(), chunksize=None, sort=True)
    #return df_mean.groupby(by=data.index.codes.tolist()).mean().compute()

#    def _do_mean(i):
#        return i.mean()

#    with Pool(processes=cpu_count()) as p:
#        ret = pd.concat(list(p.map(_do_mean, [i for i in data.groupby(level=-1)])))
#        ret.index = data.index
#        p.close()
#        p.join()

    #ret = list((i, df.mean()) for i, df in data.groupby(level=-1))
    #ids, dfs = zip(*ret)
    #ret = pd.concat(dfs, axis=1).T
    #ret.index = ids
    #ret.index.name = data.index.name
    #return ret
    return data.groupby(level=-1).mean()


def main():
    folder_path = "../input/softbank_comp1_data/competition_data/"

    file_path = os.path.join(folder_path, "text_features.parquet")
    data = ParquetDataHandler(file_path).load().get()
    print(data.shape)

    data_base_mean = dask_mean_data(data)
    import pdb;
    pdb.set_trace()

    tf = transformer_factory(
        "DaskTruncatedSVD", {"columns": data.columns.tolist(), "target_name": "text", "n_components": 128})
    data_mean = tf.fit_transform(data_base_mean)

    file_path = os.path.join(folder_path, "text_features_svd_mean.parquet")
    ParquetDataHandler(file_path).update(data_mean).save()

    # df = tf.transform(data)
    # import pdb; pdb.set_trace()

    #df = TruncatedSVD(n_components=100).fit_transform(data)
    #obj_save.update(df).save()

    return


if "__main__" == __name__:
    main()
