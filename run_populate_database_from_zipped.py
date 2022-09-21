import argparse
import logging
from subprocess import run
import glob
import os

from rl4mm.utils.utils import (
    get_date_time,
    get_next_trading_dt,
    get_last_trading_dt,
    get_trading_datetimes,
    daterange_in_db,
)

from rl4mm.database.populate_database import populate_database

parser = argparse.ArgumentParser(description="Populate a postgres database with LOBSTER data")
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


def delete_csvs():
    csv_fpaths = glob.glob(args.path_to_lobster_data + "/*.csv")
    for fp in csv_fpaths:
        run(["rm", fp])


if __name__ == "__main__":
    args = parser.parse_args()
    # filenames = run(["ls", args.path_to_lobster_data], capture_output=True).stdout.decode("utf-8").split("\n")
    # for filename in filenames[:1]:

    ##########################

    fpaths = glob.glob(args.path_to_lobster_data + "/*.7z")
    print("----------")
    print("ORIGINAL ORDER:")
    print("\n".join(fpaths))
    # Need to process data in chronologicaly order!
    # But can't just sort whole filename because e.g.,
    # ORIGINAL ORDER:
    # /home/data/KO/_data_dwn_50_385__KO_2018-04-01_2018-04-30_200.7z
    # /home/data/KO/_data_dwn_50_389__KO_2018-02-01_2018-02-28_200.7z
    # /home/data/KO/_data_dwn_50_386__KO_2018-03-01_2018-03-31_200.7z
    # /home/data/KO/_data_dwn_50_384__KO_2018-05-01_2018-05-31_200.7z
    # ----------
    # ORDER AFTER SORTING:
    # /home/data/KO/_data_dwn_50_384__KO_2018-05-01_2018-05-31_200.7z
    # /home/data/KO/_data_dwn_50_385__KO_2018-04-01_2018-04-30_200.7z
    # /home/data/KO/_data_dwn_50_386__KO_2018-03-01_2018-03-31_200.7z
    # /home/data/KO/_data_dwn_50_389__KO_2018-02-01_2018-02-28_200.7z
    #
    # Need to sort only by dates, and it suffices to just use
    # everything after the __
    #
    fpaths.sort(reverse=False, key=lambda f: f.split("__")[1])
    print("----------")
    print("ORDER AFTER SORTING:")
    print("\n".join(fpaths))
    # sys.exit()

    for fpath in fpaths:
        filename = os.path.basename(fpath)
        print("PROCESSING:", filename)
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
            delete_csvs()
            continue
        print("About to extract:", fpath, "inside", args.path_to_lobster_data)
        run(["7z", "x", fpath, "-o" + args.path_to_lobster_data])
        populate_database(
            tickers=(ticker,),
            trading_datetimes=get_trading_datetimes(start_date, end_date),
            path_to_lobster_data=args.path_to_lobster_data,
            book_snapshot_freq=args.book_snapshot_freq,
            batch_size=args.batch_size,
            n_levels=n_levels,
        )
        delete_csvs()
