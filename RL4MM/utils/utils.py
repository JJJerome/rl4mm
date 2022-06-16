from datetime import datetime, timedelta
from ray.tune.logger import UnifiedLogger
import tempfile
import os


def get_date_time(date_string: str):
    d = [int(x) for x in date_string.split(",")]
    return datetime(d[0], d[1], d[2])


def custom_logger(prefix, custom_path="/home/results"):
    custom_path = os.path.expanduser(custom_path)
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(prefix, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def convert_timedelta_to_freq(delta: timedelta):
    assert sum([delta.seconds > 0, delta.microseconds > 0]) == 1, "Timedelta must be given in seconds or microseconds."
    if delta.seconds > 0:
        return f"{delta.seconds}S"
    else:
        return f"{delta.microseconds}ms"
