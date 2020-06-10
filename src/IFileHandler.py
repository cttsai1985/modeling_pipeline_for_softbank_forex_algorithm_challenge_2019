import pandas as pd


class IFileHandler:
    def load(self, **kwargs):
        raise NotImplementedError()

    def save(self, file_path: str):
        raise NotImplementedError()

    def get(self) -> pd.DataFrame:
        raise NotImplementedError()

    def update(self, data: pd.DataFrame):
        raise NotImplementedError()
