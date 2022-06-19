from datetime import datetime, timedelta
from ray.tune.logger import UnifiedLogger
import tempfile
import os


def get_date_time(date_string: str):
    d = [int(x) for x in date_string.split(",")]
    return datetime(d[0], d[1], d[2])


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
