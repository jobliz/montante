import pandas as pd


class DatasourceWrapper:

    def to_df(self) -> pd.DataFrame:
        raise NotImplementedError