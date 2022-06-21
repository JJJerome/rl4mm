import argparse
import logging
from subprocess import run
import py7zr

from RL4MM.utils.utils import (
    get_date_time,
    get_next_trading_dt,
    get_last_trading_dt,
    get_trading_datetimes,
    daterange_in_db,
)

from RL4MM.database.populate_database import populate_database

parser = argparse.ArgumentParser(description="Populate a postgres database with LOBSTER data")
parser.add_argument("-nl", "--n_levels", action="store", type=int, default=200, help="the number of orderbook levels")
parser.add_argument(
    "-ptld",
    "--path_to_lobster_data",
    action="store",
    type=str,
    default="",
    help="the path to the folder containing the LOBSTER message and book data",
)
parser.add_argument(
    "-bsf",
    "--book_snapshot_freq",
    action="store",
    type=str,
    default="S",
    help="the frequency of book snapshots added to database",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    action="store",
    type=int,
    default=1000000,
    help="the batch size used to populate the db (for reducing memory requirements)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    filenames = run(["ls", args.path_to_lobster_data], capture_output=True).stdout.decode("utf-8").split("\n")
    for filename in filenames[:1]:
        if len(filename) == 0:
            continue
        ticker = filename.split("_")[-4]
        start_date = get_date_time(filename.split("_")[-3])
        end_date = get_date_time(filename.split("_")[-2])
        n_levels = int(filename.split("_")[-1].split(".")[0])
        next_trading_dt = get_next_trading_dt(start_date)
        last_trading_dt = get_last_trading_dt(end_date)
        if daterange_in_db(next_trading_dt, last_trading_dt, ticker):
            logging.info(f"Data for {ticker} between {start_date} and {end_date} already in database so not re-added.")
            continue
        print(filename)
        run(['7z', 'x', args.path_to_lobster_data + filename])
        populate_database(
            tickers=(ticker,),
            trading_datetimes=get_trading_datetimes(start_date, end_date),
            path_to_lobster_data=args.path_to_lobster_data,
            book_snapshot_freq=args.book_snapshot_freq,
            batch_size=args.batch_size,
            n_levels=args.n_levels,
        )
        run('rm', args.path_to_lobster_data + "*.csv")
