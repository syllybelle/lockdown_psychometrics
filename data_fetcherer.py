from typing import List, Dict

from mysql.connector import MySQLConnection
from mysql.connector.cursor import MySQLCursor

from db_mx.db_config import DbProperties
from psychometrics_paper.dataclass_definitions import LockdownEntry

covid_db: MySQLConnection() = DbProperties().connect()
SQL_query_default: str = f""" SELECT * FROM covid_data.psychometrics;"""


def filter_data(raw_data: List[LockdownEntry]) -> List[LockdownEntry]:
    filtered_data: List[LockdownEntry] = []
    for entry in raw_data:
        if entry.Progress >= 99 and entry.dem_1_age >= 18 and entry.wm_to_be_excluded != 1:
            filtered_data.append(entry)
        else:
            continue
    return filtered_data


class BasicDataset:
    def __init__(self, SQL_query: str = SQL_query_default) -> None:
        self.SQL_cursor: MySQLCursor = covid_db.cursor(dictionary=True)
        self.SQL_query: str = SQL_query
        self.data: List[LockdownEntry] = []

    def get_data(self) -> List[LockdownEntry]:
        self.SQL_cursor.execute(self.SQL_query)
        raw_table_data: List[LockdownEntry] = [LockdownEntry(**x) for x in self.SQL_cursor.fetchall()]
        self.data = filter_data(raw_table_data)
        return self.data


class DictifiedData:
    def __init__(self, data_object) -> None:
        self.data_object: any = data_object

    def dictify(self) -> Dict[str, any]:
        return dict(
            (name, getattr(self.data_object, name)) for name in dir(self.data_object) if not name.startswith('__'))


data = BasicDataset().get_data()

# print(data)
