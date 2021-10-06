import os
import psycopg2

from RL4MM.helpers.env import setup_environment_variables


def create_book_table():
    setup_environment_variables()
    conn = psycopg2.connect(
        f'host={os.environ["POSTGRES_HOST"]} dbname=lobster user={os.environ["POSTGRES_USER"]} password={os.environ["POSTGRES_PASSWORD"]}'
    )

    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS book_L10(
        id varchar(32) PRIMARY KEY,
        timestamp TIMESTAMP,   
        type VARCHAR(32),
        external_id VARCHAR(32),
        size NUMERIC(64, 32),
        price NUMERIC(64, 32),     
        direction VARCHAR(8),    
        ask_price_0 NUMERIC(64,32),
        ask_size_0 NUMERIC(64,32),
        bid_price_0 NUMERIC(64,32),
        bid_size_0 NUMERIC(64,32),
        ask_price_1 NUMERIC(64,32),
        ask_size_1 NUMERIC(64,32),
        bid_price_1 NUMERIC(64,32),
        bid_size_1 NUMERIC(64,32),
        ask_price_2 NUMERIC(64,32),
        ask_size_2 NUMERIC(64,32),
        bid_price_2 NUMERIC(64,32),
        bid_size_2 NUMERIC(64,32),
        ask_price_3 NUMERIC(64,32),
        ask_size_3 NUMERIC(64,32),
        bid_price_3 NUMERIC(64,32),
        bid_size_3 NUMERIC(64,32),
        ask_price_4 NUMERIC(64,32),
        ask_size_4 NUMERIC(64,32),
        bid_price_4 NUMERIC(64,32),
        bid_size_4 NUMERIC(64,32),
        ask_price_5 NUMERIC(64,32),
        ask_size_5 NUMERIC(64,32),
        bid_price_5 NUMERIC(64,32),
        bid_size_5 NUMERIC(64,32),
        ask_price_6 NUMERIC(64,32),
        ask_size_6 NUMERIC(64,32),
        bid_price_6 NUMERIC(64,32),
        bid_size_6 NUMERIC(64,32),
        ask_price_7 NUMERIC(64,32),
        ask_size_7 NUMERIC(64,32),
        bid_price_7 NUMERIC(64,32),
        bid_size_7 NUMERIC(64,32),
        ask_price_8 NUMERIC(64,32),
        ask_size_8 NUMERIC(64,32),
        bid_price_8 NUMERIC(64,32),
        bid_size_8 NUMERIC(64,32),
        ask_price_9 NUMERIC(64,32),
        ask_size_9 NUMERIC(64,32),
        bid_price_9 NUMERIC(64,32),
        bid_size_9 NUMERIC(64,32),
        exchange VARCHAR(32),
        ticker VARCHAR(32));
    ''')
    conn.commit()
