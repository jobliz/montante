from typing import Dict, Union, List

import pandas as pd
from rpy2.robjects.vectors import ListVector as RListVector

from ...DatasourceWrapper import DatasourceWrapper
from ...operations.R.caret_wrappers import caret_model_train


def training_operation(source: Union[pd.DataFrame, DatasourceWrapper], payload: Dict) -> Union[RListVector, List]:
    """
    General dispatcher for model training operations.

    Specific operation functions called inside this function must implement the
    following contract:

    1. Receive a pandas dataframe and a Dict payload as input parameters.
    2. Return the created prediction model, or a list of encoutered errors.

    TODO: first input: Union[pd.DataFrame, SQLQuery, etc] or a custom Datasource class?
    TODO: file creation stores file on a /tmp folder AND passes around a file handle?
    TODO:   you give it an uuid, it creates a file in /tmp with that uuid and returns a hadle!
    TODO: Wrapper class for RListVector to explicitly mean that it is a prediction model.
    TODO: Keep in mind that a lot of exceptions can be raised here, a set for each engine in the listing
    TODO: the existence of payload['engine'] should be checked here, or somewhere else?
    TODO: In the future, if models are to be stored somewhere else besides local storage, how?
    Todo: make optional to drop na's; df.dropna(how='any', inplace=True)
    """
    engine = payload['engine']

    if isinstance(source, DatasourceWrapper):
        df = source.to_df()
    else:
        df = source

    if engine == 'caret':
        return training_operation_for_caret(df, payload)
    else:
        raise NotImplementedError


def training_operation_for_caret(df: pd.DataFrame, payload: Dict) -> Union[RListVector, List]:
    """
    TODO: Documentation.
    """
    model_or_error_list = caret_model_train(df, payload)

    if isinstance(model_or_error_list, list):
        return model_or_error_list
    elif isinstance(model_or_error_list, RListVector):
        return model_or_error_list
    else:
        print(type(model_or_error_list))
        print(model_or_error_list)
        raise ValueError('output isnt an error list or a PredictionModel')
