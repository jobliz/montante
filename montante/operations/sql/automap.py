from typing import Dict, List

import sqlalchemy
from sqlalchemy import orm
from sqlalchemy.ext.automap import automap_base, AutomapBase
import sqlalchemy.ext.declarative
import sqlalchemy.ext.declarative.api


def create_automap_base(e: sqlalchemy.engine.Engine) -> AutomapBase:
    """
    Creates an automapping declarative base for the given engine.
    """
    base = automap_base()
    base.prepare(e, reflect=True)
    return base


def extract_sql_schema(e: sqlalchemy.engine.Engine) -> Dict:
    """
    Extracts JSONable schema data from the given engine.

    Accessing data:
        base.classes['country']
        base.classes['invoice'].__table__.columns['date']
        base.classes['invoice'].__table__.columns['date'].foreign_keys
    """
    base = create_automap_base(e)
    maker = sqlalchemy.orm.sessionmaker(bind=e)
    session = maker()

    structure = {}
    structure['foreign_keys'] = []
    structure['tables'] = {}

    for aclass in base.classes:
        classname = aclass.__name__
        structure['tables'][classname] = {}
        structure['tables'][classname]['columns'] = {}
        structure['tables'][classname]['foreign_keys'] = []
        structure['tables'][classname]['total_rows'] = session.query(aclass).count()

        for column in aclass.__table__.columns:
            name = str(column)
            structure['tables'][classname]['columns'][name] = {}
            structure['tables'][classname]['columns'][name]['type'] = str(column.type)
            structure['tables'][classname]['columns'][name]['unique'] = column.unique
            structure['tables'][classname]['columns'][name]['nullable'] = column.nullable
            structure['tables'][classname]['columns'][name]['primary_key'] = column.primary_key

            for key in column.foreign_keys:
                structure['tables'][classname]['columns'][name]['foreign_keys'] = key.target_fullname
                # TODO: Fix formatting of this last part.
                structure['foreign_keys'].append('.'.join([name, key.target_fullname]))

    return structure
