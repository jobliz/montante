import os
import csv
from typing import Union, List


def csv_headers_from_path(path: str) -> Union[None, List[str]]:
    """
    Gets the list of CSV headers name from the file at the given path, or None if they
    are not present.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('CSV file not in media directory')

    with open(path, 'r') as file:
        sniffer = csv.Sniffer()
        # todo: check with different csv files
        has_header = sniffer.has_header(str((file.read(4096))))
        file.seek(0)

        if not has_header:
            return None

        reader = csv.reader(file)
        return next(reader)

