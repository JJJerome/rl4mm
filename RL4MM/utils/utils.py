from datetime import datetime, timedelta
from ray.tune.logger import UnifiedLogger
import tempfile
import os
import pandas_market_calendars as mcal

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from run_populate_database_from_zipped import ticker


def get_date_time(date_string: str):
    return datetime.strptime(date_string, "%Y-%m-%d")


def custom_logger(prefix, custom_path="/home/RL4MM/results"):
    custom_path = os.path.expanduser(custom_path)
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(prefix, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def convert_timedelta_to_freq(delta: timedelta):
    assert sum([delta.seconds > 0, delta.microseconds > 0]) == 1, "Timedelta must be given in seconds or microseconds."
    if delta.seconds > 0:
        return f"{delta.seconds}S"
    else:
        return f"{delta.microseconds}ms"


def save_best_checkpoint_path(path_to_save_dir: str, best_checkpoint_path: str):
    text_file = open(path_to_save_dir + "/best_checkpoint_path.txt", "wt")
    text_file.write(best_checkpoint_path)
    text_file.close()


def get_last_trading_dt(timestamp: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    last_trading_date = ncal.schedule(start_date=timestamp - timedelta(days=4), end_date=timestamp).iloc[-1, 0].date()
    return datetime.combine(last_trading_date, datetime.min.time()) + timedelta(hours=16)


def get_next_trading_dt(timestamp: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    next_trading_date = ncal.schedule(start_date=timestamp, end_date=timestamp + timedelta(days=4)).iloc[0, 0].date()
    return datetime.combine(next_trading_date, datetime.min.time()) + timedelta(hours=9, minutes=30)


def get_trading_datetimes(start_date: datetime, end_date: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    return ncal.schedule(start_date=start_date, end_date=end_date).market_open.index


def daterange_in_db(start: datetime, end: datetime):
    database = HistoricalDatabase()
    bool_1 = database.get_next_snapshot(start, ticker).name - start < timedelta(minutes=1)
    bool_2 = end - database.get_last_snapshot(end, ticker).name < timedelta(minutes=1)
    return bool_1 and bool_2
