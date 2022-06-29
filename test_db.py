from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.utils.utils import get_date_time

database = HistoricalDatabase()

ticker = "KO"

end = get_date_time("2018-03-01")
last_snapshot = database.get_last_snapshot(end, ticker)
print(last_snapshot)

end = get_date_time("2018-02-28")
last_snapshot = database.get_last_snapshot(end, ticker)
print(last_snapshot)
