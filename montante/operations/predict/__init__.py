from typing import Dict, Union, List, Tuple

import pandas as pd
from rpy2.robjects.vectors import ListVector as RListVector

from ...operations.R.functions import r_convert_pandas_dataframe, r_predict, r_extract_prediction_pairs


def prediction_operation(model: Union[RListVector], payload: Dict) -> List[Tuple[int, str]]:
    """
    TODO: More specific typing for 'fit'
    """
    if isinstance(model, RListVector):
        return prediction_operation_for_r(model, payload)
    else:
        raise NotImplementedError('fit is not a recognized type')


def prediction_operation_for_r(model: RListVector, payload: Dict) -> List[Tuple[int, str]]:
    """
    TODO: document return format!
    """
    # TODO: different dict formats
    pdf = pd.DataFrame(payload)
    rdf = r_convert_pandas_dataframe(pdf)
    prediction = r_predict(model, rdf, type='raw')  # todo: type config comes from...?
    pairs = r_extract_prediction_pairs(prediction)

    return pairs
