from datetime import datetime
from pathlib import Path
from unittest import TestCase

from sqlalchemy import create_engine

import RL4MM
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.populate_database import populate_database
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator


class TestHistoricalOrderGenerator(TestCase):
    path_to_test_data = str(Path(RL4MM.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012,6,21)
    n_levels = 50
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db, save_messages_locally=False)

    @classmethod
    def setUpClass(cls) -> None:
        populate_database(
            (cls.ticker,),
            (cls.trading_date,),
            database=cls.test_db,
            path_to_lobster_data=cls.path_to_test_data,
            book_snapshot_freq=None,
            max_rows=1000,
            batch_size=1000,
        )

    def test_generate_orders(self):
        start_of_trading = datetime(2012, 6, 21, 9, 30)
        end_of_trading = datetime(2012, 6, 21, 16)
        messages = self.test_db.get_messages(start_of_trading, end_of_trading, self.ticker)
        n_hidden_executions = len(messages[messages.message_type == "market_hidden"])
        orders = self.generator.generate_orders(start_of_trading, end_of_trading)
        self.assertEqual(1000 - n_hidden_executions, len(orders))
