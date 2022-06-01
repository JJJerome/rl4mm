import argparse

from RL4MM.database.populate_database import populate_database

parser = argparse.ArgumentParser(description="Populate a postgres database with LOBSTER data")
parser.add_argument("--ticker", action="store", type=str, default="MSFT", help="the ticker to add")
parser.add_argument("--trading_date", action="store", type=str, default="2012-06-21", help="the date to add")
parser.add_argument("--n_levels", action="store", type=int, default=200, help="the number of orderbook levels")
parser.add_argument(
    "--path_to_lobster_data",
    action="store",
    type=str,
    default="",
    help="the path to the folder containing the LOBSTER message and book data",
)
parser.add_argument(
    "--book_snapshot_freq",
    action="store",
    type=str,
    default="S",
    help="the frequency of book snapshots added to database",
)
parser.add_argument(
    "--max_rows",
    action="store",
    type=int,
    default=99999999999,
    help="the frequency of book snapshots added to database",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=100000,
    help="the frequency of book snapshots added to database",
)

if __name__ == "__main__":
    args = parser.parse_args()

    populate_database(
        tickers=(args.ticker,),
        trading_dates=(args.trading_date,),
        n_levels=args.n_levels,
        path_to_lobster_data=args.path_to_lobster_data,
        book_snapshot_freq=args.book_snapshot_freq,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )
