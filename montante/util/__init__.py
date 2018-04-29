import os
import sys
import uuid
import json
import argparse
from typing import Union, List, Dict

import jsonschema


def filesize(filepath: str) -> int:
    st = os.stat(filepath)
    return st.st_size


def local_file_storage_fullpath(uid: str) -> str:
    """
    Retrieves the local filepath that would be assigned to the given UUID.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'media', uid)


def local_tmp_fullpath(uid: str) -> str:
    return ''.join(['/tmp/', uid])


def new_uuid():
    return str(uuid.uuid4())


def date_parse_expressions() -> List[str]:
    """
    Creates a list of date-parsing strings that can be tested against variable user input.
    """
    return [
        '%m/%d/%Y %H:%M'
    ]


def use_validator(validator: jsonschema.Draft4Validator, payload: Dict) -> List:
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)

    if len(errors) > 0:
        error_list = [(list(e.absolute_schema_path), e.message) for e in errors]
    else:
        error_list = []

    return error_list


def function_argparse_cli(functions):
    """
    Creates and executes a command line parser trying to execute the given functions,
    whose inputs must be integers or strings (because it parses the input as a JSON)
    """
    description = ''.join([
        "Command line execution proxy for:\n",
        "\n".join(f.__name__ for f in functions),
        "\n\nAn example use would be:\n\n",
        "python3 script.py - -function = 'fetch_datasource_by_id' - -args = '[1]'",
        "\n"
    ])

    parser = argparse.ArgumentParser(description)
    parser.add_argument("--function")
    parser.add_argument("--args", default='[]')
    parser.add_argument("--kwargs", default='{}')
    parser.add_argument("--view", default='none')
    args = parser.parse_args()

    for func in functions:
        if args.function == func.__name__:
            f_args = json.loads('{"items": %s }' % args.args)['items']
            f_kwargs = json.loads(args.kwargs)

            if args.view == 'none':
                print(func(*f_args, **f_kwargs))
                sys.exit()
            elif args.view == 'entity_json':
                entity = func(*f_args, **f_kwargs)
                print(json.dumps(entity.to_dictionary()))
                sys.exit()
            elif args.view == 'entity_list_json':
                entity_list = func(*f_args, **f_kwargs)
                print(json.dumps([entity.to_dictionary() for entity in entity_list]))
                sys.exit()
            else:
                print("Unspecified view:")
                print(args.view)
                sys.exit(1)

    print("No function match.")
    sys.exit(1)
