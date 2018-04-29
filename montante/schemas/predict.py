from typing import Dict

import jsonschema


"""
General schema for a JSON prediction request.
"""
generic_prediction_schema = {
    "type": "object",
    "required": ["model_uuid", "data"],
    "properties": {
        "model_uuid": {"type": "string"},
        "data": {
            "type": 'object',
        }
    }
}


def create_specific_prediction_schema_validator(column_info: Dict[str, str]) -> jsonschema.Draft4Validator:
    """
    Creates a JSONSchema validator based on the given column info.

    The dict is a mapping of column name to pandas dataframe dtype.

    TODO: Check from old code:
    for some reason the array check gets embedded inside a list with one element...
        for key in schema['properties']:
            schema['properties'][key] = schema['properties'][key][0]
    """
    schema = {
        "type": "object",
        "properties": {
            "model_uuid": {"type": "string"},
            "data": {
                "type": 'object',
                "properties": {},
                "required": []
            }
        }
    }

    for column_name, column_type in column_info:
        if column_type == 'int64':
            schema['properties']['data']['properties'][column_name] = {"type": "integer"}
            schema['properties']['data']['required'].append(column_name)
        elif column_type == 'datetime64' or column_type == 'timedelta[ns]':
            raise NotImplementedError('Date values are not currently implemented')
        elif column_type == 'float64':
            schema['properties']['data']['properties'][column_name] = {"type": "number"}
            schema['properties']['data']['required'].append(column_name)
        elif column_type == 'object':
            schema['properties']['data']['properties'][column_name] = {"type": "string"}
            schema['properties']['data']['required'].append(column_name)
        else:
            msg = ' '.join(['Given column_type is not recognized', column_type])
            raise TypeError(msg)

    return jsonschema.Draft4Validator(schema)
