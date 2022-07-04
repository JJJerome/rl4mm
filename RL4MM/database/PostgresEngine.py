import os

from mypy_extensions import TypedDict
from sqlalchemy import create_engine

from RL4MM.helpers.env import setup_environment_variables


class PostgresConfig(TypedDict):
    host: str
    port: int
    database: str
    user: str
    password: str


class PostgresEngine:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
    ):
        self.config = self.__get_config(host, port, database, user, password)
        self.engine = create_engine(self._construct_url(self.config), pool_size=80, max_overflow=20)

    @staticmethod
    def __get_config(
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
    ) -> PostgresConfig:
        setup_environment_variables()
        return PostgresConfig(
            {
                "host": str(host or os.environ["POSTGRES_HOST"]),
                "port": int(port or os.environ["POSTGRES_PORT"]),
                "database": str(database or os.environ["POSTGRES_DB"]),
                "user": str(user or os.environ["POSTGRES_USER"]),
                "password": str(password or os.environ["POSTGRES_PASSWORD"]),
            }
        )

    @staticmethod
    def _construct_url(config: PostgresConfig) -> str:
        dbapi = "postgresql+psycopg2"
        return f"{dbapi}://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
