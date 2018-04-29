from typing import Dict

import pandas as pd
import sqlalchemy


def build_sqlalchemy_engine(params: Dict) -> sqlalchemy.engine.Engine:
    """
    # string = 'mysql://root:clave@localhost/online_retail_dataset'
    # engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')
    """
    string = None

    if params['engine'] == 'mysql':
        string = ''.join(['mysql://', params['username'],
                          ':', params['password'],
                          '@', params['host'], '/',
                          params['database']])

    elif params['engine'] == 'postgres':
        string = ''.join(['postgresql://', params['username'],
                          ':', params['password'],
                          '@', params['host'],
                          '/', params['database']])

    if string is None:
        raise TypeError('Unrecognized SQL engine')

    return sqlalchemy.create_engine(string)


def raw_sqlalchemy_query_to_pandas_dataframe(e: sqlalchemy.engine.Engine, sql: str) -> pd.DataFrame:
    connection = e.connect()
    rs = connection.execute(sql)
    data = {}
    keys = rs.keys()

    for key in keys:
        data[key] = []

    for row in rs:
        for key, value in zip(keys, row):
            data[key].append(value)

    return pd.DataFrame(data)
