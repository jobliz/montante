import pandas as pd
import sqlalchemy.ext.declarative


def sqlite_from_dataframe(df: pd.DataFrame, e: sqlalchemy.engine.Engine, uuid: str) -> bool:
    """
    Creates a SQLite file as a side-effect.

    TODO/CHECK: If engine must come from
        engine = sqlalchemy.create_engine(string, encoding='utf-8', convert_unicode=True)

    See:
         https://stackoverflow.com/questions/3033741/sqlalchemy-automatically-converts-str-to-unicode-on-commit
    """
    try:
        # engine.raw_connection().connection.text_factory = str
        connection = e.connect()
        # engine.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        # connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        # engine.raw_connection().connection.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        df.to_sql(name='dataset', con=connection, if_exists='append')
        connection.close()
        return True
    except Exception as e:
        return False
