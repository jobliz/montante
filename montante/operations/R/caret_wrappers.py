from typing import Any, Union, List, Dict

from jsonschema import Draft4Validator
from rpy2.robjects.vectors import ListVector as RListVector

from .functions import *
from ...util import use_validator, new_uuid, local_tmp_fullpath
from ...schemas.train import create_specific_training_schema


def caret_model_train(df: pd.DataFrame, payload: Dict) -> Union[RListVector, List]:
    """
    Trains a caret model.

    On success returns the caret model instance object.
    On error returns the error list produced by the validator.

    TODO: Mind this.
    On payload label mismatch the r_caret_train call can raise an error like:
    rpy2.rinterface.RRuntimeError: Error in `[.data.frame`(list(sepal.length..cm. = c(5.1, 4.9, 4.7, 4.6,  :
    undefined columns selected
    """
    # do validation
    method = payload['engine-parameters']['method']
    schema = create_specific_training_schema('caret', method)
    errors = use_validator(Draft4Validator(schema), payload)

    if len(errors) > 0:
        return errors

    # create model
    rdf = r_convert_pandas_dataframe(df)
    target = payload['target']
    predictors = payload['predictors']
    model_kwargs = caret_model_kwargs_from_payload(rdf, payload)
    model = r_caret_train(r_formula(rdf, target, predictors), **model_kwargs)

    return model


def caret_model_save_to_tmp(model) -> str:
    """
    Saves the passed caret model object to the filesystem.
    """
    uuid = new_uuid()
    path = local_tmp_fullpath(uuid)
    serialized_model = r_base64encode(r_serialize(model))
    r_save_rds(serialized_model, path)
    return path


def caret_model_kwargs_from_payload(rdf: Any, payload: Dict) -> Dict:
    """
    Creates the object that will be passed to the 'train' R function as configuration
    parameters.

    # TODO: preProcess functionality.
    """
    return {
        'data': rdf,
        'trControl': r_caret_train_control(
            method=payload['engine-parameters']['training-control']['method'],
            number=payload['engine-parameters']['training-control']['number'],
            repeats=payload['engine-parameters']['training-control']['repeats']
        ),
        'metric': payload['engine-parameters']['metric'],
        'method': payload['engine-parameters']['method']
    }
