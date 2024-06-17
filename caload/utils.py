from datetime import date, datetime
from typing import Union


def parse_date(_date: Union[str, datetime, date]):
    if isinstance(_date, str):
        _date = datetime.strptime(_date, '%Y-%m-%d')

    if isinstance(_date, datetime):
        _date = _date.date()

    return _date
