"""
See:
    https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
    https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict
"""

from typing import Tuple, Union

import pandas as pd
from sklearn import datasets
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import DataFrame as RDataFrame


from ..operations.dataframe.functions import pd_sanitize_column_names_for_r


def sklearn_iris() -> Tuple[pd.DataFrame, str, str]:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris['target']
    replacements = dict((k, v) for k, v in enumerate(iris['target_names']))
    df['target'] = df['target'].map(replacements)
    pd_sanitize_column_names_for_r(df)
    return df, 'Iris Dataset from Sklearn examples', iris['DESCR']


def sklearn_wine() -> Tuple[pd.DataFrame, str, str]:
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine['target']
    replacements = dict((k, v) for k, v in enumerate(wine['target_names']))
    df['target'] = df['target'].map(replacements)
    return df, 'Wine dataset from sklearn examples', wine['DESCR']


def r_iris_raw() -> RDataFrame:
    r('data(iris)')
    iris = r('iris')
    return iris


def r_iris() -> Tuple[pd.DataFrame, str, str]:
    """
    See:
        http://www.jamesloach.com/python/pandas2ri.html

    Notes: pandas2ri.activate call seems to be necessary inside function call
    so it gets called from outside. Also, calling r_iris_raw() inside
    pandas2ri.ri2py raises NotImplementedError.
    """
    r('data(iris)')
    iris = r('iris')
    pandas2ri.activate()
    out = pandas2ri.ri2py(iris)
    pandas2ri.deactivate()
    return out, 'Iris dataset from R example data', ''


example_dataset_functions = [
    sklearn_iris,
    sklearn_wine,
    r_iris
]


def data_example_loader(example_name: str) -> Union[None, Tuple[pd.DataFrame, str, str]]:
    """
    Function to load an example dataset by name as a tuple:

    [0]: Pandas dataframe object.
    [1]: Example name
    [2]: Example description. TODO: Is '' if empty.
    """
    for f in example_dataset_functions:
        if f.__name__ == example_name:
            return f()
    return None
