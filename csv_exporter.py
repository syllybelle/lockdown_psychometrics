from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

from pandas import DataFrame


class CsvExporter:
    def __init__(self, data: Union[List[any], Dict[any, any], defaultdict], path: Path) -> None:
        self.path = path
        if type(data) is dict:
            self.df: DataFrame = DataFrame.from_dict(data)
        elif type(data) is list:
            self.df: DataFrame = DataFrame(data)
        elif type(data) is defaultdict:
            self.df: DataFrame = DataFrame.from_dict(dict(data))
        else:
            raise Exception("data is not a list or dict")

    def export(self) -> None:
        self.df.to_csv(self.path)
