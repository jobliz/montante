from typing import List, Dict


def caret_train_engine_subschema(method: str) -> Dict:
    if method not in _caret_train_available_training_methods():
        raise NotImplementedError

    return {
        "type": "object",
        "required": ["method", "training-control"],
        "properties": {
            "method": {"type": "string", "enum": _caret_train_available_training_methods()},
            "preprocess": {"type": "array", "items": {"type": "string"}},
            "metric": {"type": "string", "enum": _caret_train_metrics()},
            "training-control": {
                "type": "object",
                "required": ["method", "number", "repeats"],
                "properties": {
                    "method": {"type": "string", "enum": _caret_train_control_methods()},
                    "number": {"type": "integer"},
                    "repeats": {"type": "integer"}
                }
            }
        }
    }


def _caret_train_available_training_methods() -> List[str]:
    return ['C5.0']


def _caret_train_control_search() -> List[str]:
    # TODO: not used for now...
    return ["grid", "random"]


def _caret_train_metrics() -> List[str]:
    # TODO: associate metrics with prediction problem type.
    # possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification
    return ["RMSE", "Rsquared", "Accuracy", "Kappa"]


def _caret_train_control_methods() -> List[str]:
    return [
        "boot", "boot632", "optimism_boot", "boot_all", "cv", "repeatedcv",
        "LOOCV", "LGOCV", "none", "oob", "adaptive_cv", "adaptive_boot", "adaptive_LGOCV"
    ]
