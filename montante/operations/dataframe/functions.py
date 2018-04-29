from typing import Iterator, List, Tuple, Dict

import pandas as pd


def pd_sanitize_column_names_for_r(df: pd.DataFrame):
    """
    Replaces characters in dataframe column names so that R doesn't have problems
    with them.
    """
    df.rename(columns=lambda n: n.replace(' ', '_'), inplace=True)
    df.rename(columns=lambda n: n.replace(')', ''), inplace=True)
    df.rename(columns=lambda n: n.replace('(', ''), inplace=True)


def pd_column_types(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of the dataframe's column dtypes as strings.
    """
    return [str(dtype) for dtype in df.dtypes]


def pd_column_names(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of the dataframe's column names.
    """
    return [str(header) for header in list(df)]


def pd_names_missing(df: pd.DataFrame, col_names: List[str]) -> List[str]:
    """
    Check if the given prediction target and predictors are in the dataframe.

    Returned list is empty when the check passes.
    """
    errors = []

    for name in col_names:
        if name not in list(df):
            errors.append(name)

    return errors


def pd_column_info(df: pd.DataFrame) -> Iterator[Tuple[str, str]]:
    """
    Returns the combined output of pd_column_names and pd_column_types as
    an iterator with tuples.
    """
    return zip(pd_column_names(df), pd_column_types(df))


def pd_column_info_dict(df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns the output of pd_column_info as a dict.
    """
    info = pd_column_info(df)
    d = {}

    for pair in info:
        d[pair[0]] = pair[1]

    return d
