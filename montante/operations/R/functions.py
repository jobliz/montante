"""
See:
    https://machinelearningmastery.com/finalize-machine-learning-models-in-r/
    https://rdrr.io/cran/caret/man/models.html
"""

import os
from typing import Tuple, List, Union, Dict, Any

import pandas
import pandas as pd
import numpy as np

import rpy2.robjects
import rpy2.rinterface
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP

from rpy2.rinterface import NA_Logical as RNA_Logical
from rpy2.robjects import Formula as RFormula
from rpy2.robjects.vectors import Vector as RVector
from rpy2.robjects.vectors import FactorVector as RFactorVector
from rpy2.robjects.vectors import ListVector as RListVector
from rpy2.robjects.vectors import IntVector as RIntVector
from rpy2.robjects.vectors import FloatVector as RFloatVector
from rpy2.robjects.vectors import DataFrame as RDataFrame
from rpy2.robjects.vectors import StrVector as RStrVector

base = importr('base')
rpart = importr('rpart')
stats = importr('stats')
base64enc = importr('base64enc')
C50 = importr('C50')
caret = importr('caret')

# pandas2ri.activate()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts/graphvizC50.R')

with open(path, 'r') as f:
    graphvizC50_string = f.read()

graphvizC50 = STAP(graphvizC50_string, 'graphvizC50')


def r_options(*args, **kwargs):
    return base.options(*args, **kwargs)


def r_c(*args) -> RVector:
    """
    Generic function wrapper around the c function.
    """
    return base.c(*args)


def r_data_frame(*args, **kwargs) -> RDataFrame:
    """
    Access to the data.frame R function.
    """
    return base.data_frame(*args, **kwargs)


def r_dataframe_from_pandas(df: pd.DataFrame) -> RDataFrame:
    """
    Transforms a pandas dataframe into an R dataframe.
    """
    return rpy2.robjects.pandas2ri.py2ri(df)


def r_dataframe_from_dict(d: dict) -> RDataFrame:
    """
    Creates an R dataframe with the passed dict.
    """
    return rpy2.robjects.vectors.DataFrame(d)


def r_dataframe_from_kwargs(**kwargs) -> RDataFrame:
    """
    Transforms **kwargs arguments into a R dataframe
    """
    return r_dataframe_from_dict(**kwargs)


def r_formula(rdf: RDataFrame, target: str, predictors: List[str]) -> RFormula:
    """
    Creates an R modelling formula associated with the given dataframe.

    The produced string formula is 'predictor ~ var1 + var2 + etc...'
    """

    lhs_items = [target, '~']
    rhs_items = []

    for predictor in predictors:
        rhs_items.append(predictor)
        rhs_items.append('+')

    rhs_items = rhs_items[:-1]  # remove the last '+'
    all_items = lhs_items + rhs_items
    formula_string = ' '.join(all_items)

    formula = RFormula(formula_string)

    for predictor in predictors:
        formula.environment[predictor] = rdf.rx(predictor)

    return formula


def r_stats_predict(*args, **kwargs):
    return stats.predict(*args, **kwargs)


def r_summary(*args, **kwargs):
    return r('summary')(*args, **kwargs)


def r_summary_to_pd(summary) -> pd.DataFrame:
    """
    Attempts to transforms a R summary result to a pandas dataframe.

    TODO: Check for error cases. i.e. can summary() output non-tabular data?
    """
    labels = [name.replace(' ', '') for name in list(summary.colnames)]
    column_length = int(len(np.array(summary)) / len(list(summary.colnames)))
    matrix = np.reshape(np.array(summary), (-1, column_length))
    return pd.DataFrame(matrix.T, columns=labels)


def r_dataframe_column_index_from_name(rdf: RDataFrame, name: str) -> Union[None, int]:
    """
    Gets the R index of the given column name if it exists, none otherwise.
    """
    for n, existing_name in enumerate(list(rdf.colnames)):
        if name == existing_name:
            return n + 1

    return None


def r_dataframe_subset_one_element(rdf: RDataFrame, n: int) -> RDataFrame:
    """
    Creates a dataframe with one column from the given dataframe and index.

    See:
        https://github.com/topepo/caret/issues/672
        https://stackoverflow.com/questions/40505994/how-to-apply-preprocessing-in-carets-train-to-only-some-variables
        https://stackoverflow.com/questions/31497479/how-to-select-columns-from-r-dataframe-in-rpy2-in-python
    """
    return r('data.frame')(rdf.rx(RIntVector([n, ])))


def r_rpart(formula: RFormula, data: RDataFrame) -> RListVector:
    """
    Creates a recursive partitioning decision tree.
    TODO: remaining parameters.
    """
    fit = rpart.rpart(formula, data=data)
    return fit


def r_predict(fit: RListVector, data: RDataFrame, type: str) -> Union[RFactorVector]:
    """
    predict(fit, newdata=dataframe, type="class")
    TODO: Rename this function, this is a very specific use case of stats.predict.
    """
    return stats.predict(fit, newdata=data, type=type)


def r_extract_prediction_pairs(p: RFactorVector) -> List[Tuple[int, str]]:
    """
    Extract the numerical value and it's corresponding factor from the predictions
    object. Keep in mind that R indexes start at 1, hence val-1. This means that
    the returned int is the R index of a R vector, NOT a python index.

    TODO: Case when this is regression (Union on the received parameter, isinstance, etc)

    See:
        http://rpy.sourceforge.net/rpy2/doc-2.3/html/vector.html#factorvector
    """
    return [(val, p.levels[val-1]) for index, val in enumerate(list(p))]


def r_predict_index_and_label(prediction) -> str:
    """
    Returns the predicted label/factor in the given prediction.

    Remember that R indexes start at 1, python's at 0, so the index required
    to get the proper prediction from the python structure would be i-1
    """
    index = prediction[0] - 1
    label = prediction.levels[index]
    return label


def r_caret_preprocess(*args, **kwargs):
    return r('preProcess')(*args, **kwargs)


def r_caret_train(*args, **kwargs):
    return r('train')(*args, **kwargs)


def r_caret_train_control(*args, **kwargs) -> RListVector:
    return caret.trainControl(*args, **kwargs)


def r_c50_model_to_dot(model) -> str:
    return base.suppressWarnings(graphvizC50.graphvizC50(model))


def r_c50(rdf: RDataFrame, target: str, predictors: List[str]) -> RListVector:
    """
    Wrapper function around the C5.0 classifier.

    Note: The target column must be a factor vector.
    TODO: Training control and other parameters.
    """
    predictor_slice = rdf.rx(r_c(*predictors))
    target_slice = rdf.rx2(r_c(target))

    return C50.C5_0(predictor_slice, target_slice)


def r_read_rds(path: str):
    return base.readRDS(path)


def r_save_rds(obj: Any, path: str):
    return base.saveRDS(obj, path)


def r_serialize(obj) -> RVector:
    """
    See:
        https://github.com/openml/openml-r/issues/49
    """
    return base.serialize(obj, rpy2.rinterface.NULL)


def r_unserialize(something) -> Any:
    """
    See:
        https://github.com/openml/openml-r/issues/49
    """
    return base.unserialize(something)


def r_base64encode(obj) -> rpy2.robjects.vectors.StrVector:
    return base64enc.base64encode(obj)


def r_base64decode(obj) -> Any:
    return base64enc.base64decode(obj)


def r_cat_to_stdout(obj) -> str:  # rpy2.rinterface.RNULLType
    return str(base.cat(obj, file=base.stdout()))


def r_dataframe_for_stdout(df: pandas.DataFrame) -> str:
    rdf = r_dataframe_from_pandas(df)
    out = r_serialize(rdf)
    out = r_base64encode(out)
    out = r_cat_to_stdout(out)
    return out


def r_dataframe_column_types(rdf: RDataFrame) -> List[str]:
    """
    List the dataframe's column types.

    Types can be: factor, integer, numeric. TODO: dates?

    See:
        https://rpy2.github.io/doc/latest/html/vector.html#dataframe
    """
    return [column.rclass[0] for column in rdf]


def r_dataframe_column_names(rdf: RDataFrame) -> List[str]:
    """
    List the dataframe's column names.
    """
    return list(rdf.colnames)


def r_dataframe_column_to_factor_by_name(rdf: RDataFrame, name: str) -> RDataFrame:
    """
    Transform the column with the given name into a factor vector.

    Note: This modifies the passed dataframe.
    """
    for index, item in enumerate(r_dataframe_column_names(rdf)):
        if item == name:
            rdf[index] = RFactorVector(RFactorVector(rdf.rx2(name)))
            return rdf

    raise ValueError('Given name is not in R dataframe')


def r_convert_pandas_dataframe(df: pd.DataFrame) -> RDataFrame:
    """
    Pandas dataframe to R dataframe conversion.

    See:
        http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/03-data-types-and-format.html

    TODO/FIXME: Error when a dataframe column has NA elements.
    """
    pd_names = [str(header) for header in list(df)]
    pd_types = [str(dtype) for dtype in df.dtypes]
    elements = {}

    for column_name, column_type in zip(pd_names, pd_types):
        if column_type == 'int64':
            elements[column_name] = RIntVector(df[column_name])
        elif column_type == 'float64':
            elements[column_name] = RFloatVector(df[column_name])
        elif column_type == 'object':
            elements[column_name] = RFactorVector(df[column_name])
        elif column_type == 'datetime64' or column_type == 'timedelta[ns]':
            raise NotImplementedError('Date values are not currently implemented')
        else:
            msg = ' '.join(['Given column_type is not recognized', column_type])
            raise TypeError(msg)

    return RDataFrame(elements)


def r_dataframe_names_match_pd(rdf: RDataFrame, pdf: pd.DataFrame) -> bool:
    """
    Checks if the given R and pandas dataframes' names match.
    This is to see if they're compatible in some way, as for conversion.
    """
    r_names = r_dataframe_column_names(rdf)
    p_names = [str(header) for header in list(pdf)]
    return set(r_names).issubset(p_names)
