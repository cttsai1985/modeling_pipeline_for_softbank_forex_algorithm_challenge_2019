import os
import numpy as np

from DataHandler import CSVDataDaskHandler, ParquetDataHandler


def main():
    configs = {'compression': 'gzip', 'blocksize': None, "dtype": {f"feat_{i}":np.float32 for i in range(301)}}
    folder_path = "../input/softbank_comp1_data/competition_data/"
    obj = CSVDataDaskHandler(os.path.join(folder_path, "text_features/chunk_*.csv.gz"))

    df = obj.load(**configs).get()
    df['id'] = df['id'].astype("category")
    df.set_index("id", inplace=True)

    obj_save = ParquetDataHandler(os.path.join(folder_path, "text_features.parquet"))
    obj_save.update(df).save()

    return


if "__main__"  == __name__:
    main()