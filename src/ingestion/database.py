from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    _instance: Optional["DatabaseManager"] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._create_engine()

    def _create_engine(self) -> None:
        db_url = settings.DB_URL
        logger.info(f"Creating database engine for {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
        self._engine = create_engine(
            db_url,
            echo=False,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "connect_timeout": 30,
                "options": f"-c search_path={settings.POSTGRES_SCHEMA}",
            },
        )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._create_engine()
        return self._engine

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def test_connection(self) -> bool:
        logger.info("Testing database connection...")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 AS status"))
                row = result.fetchone()
                logger.info(f"Database connection OK: {row}")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        if self._session_factory is None:
            self._create_engine()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_sql(self, sql: str, params: Optional[dict] = None):
        with self.get_connection() as conn:
            return conn.execute(text(sql), params or {})

    def execute_script(self, sql_script: str) -> None:
        logger.info("Executing SQL script...")
        with self.engine.connect() as conn:
            conn.execute(text(sql_script))
            conn.commit()
        logger.info("SQL script executed successfully")

    def read_sql_file(self, filepath: str) -> None:
        logger.info(f"Executing SQL file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            sql = f.read()
        self.execute_script(sql)
        logger.info(f"SQL file {filepath} executed successfully")

    def dispose(self) -> None:
        if self._engine is not None:
            logger.info("Disposing database engine...")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


db = DatabaseManager()
