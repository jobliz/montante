from typing import Dict, Union
from .train_caret import caret_train_engine_subschema


def create_base_training_schema() -> Dict:
    """
    Generates the base prediction request schema. These are the attributes that
    are common to every train task.

    The 'engine-parameters' attribute is left empty so that another function
    can fill it with custom data.
    """
    return {
        "type": "object",
        "required": ["target", "predictors", "engine-parameters"],
        "properties": {
            "engine":       {"type": "string"},
            "target":       {"type": "string"},
            "predictors":   {"type": "array", "items": {"type": "string"}},
            "engine-parameters": {}
        }
    }


def create_specific_training_schema(engine: str, method: Union[None, str]) -> Dict:
    """
    Creates a JSONSchema to validate a train petition with the given engine and
    an optional method. This is the function meant to be used in general, in
    testing and in request controllers.

    It appends a dict to the schema on ['properties']['engine-parameters'] that
    is specific to the supplied 'engine' and 'method' values.
    """
    base = create_base_training_schema()

    if engine == 'caret':
        if method is None:
            raise ValueError('method cannot be None for caret')
        base['properties']['engine-parameters'] = caret_train_engine_subschema(method)
        return base
    else:
        # TODO: other engines, specific implementation error
        raise NotImplementedError
