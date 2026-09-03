from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "ecommerce_user"
    POSTGRES_PASSWORD: str = "ecommerce_pass"
    POSTGRES_DB: str = "ecommerce_warehouse"
    POSTGRES_SCHEMA: str = "public"

    DATABASE_URL: Optional[str] = None

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    SEED_RANDOM_STATE: int = 42
    DATA_START_DATE: str = "2022-01-01"
    DATA_END_DATE: str = "2024-12-31"
    ANALYSIS_AS_OF_DATE: str = "2024-12-31"
    NUM_CUSTOMERS: int = 100000
    NUM_PRODUCTS: int = 5000
    NUM_ORDERS: int = 200000

    STREAMLIT_SERVER_PORT: int = 8501
    STREAMLIT_SERVER_ADDRESS: str = "0.0.0.0"

    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    FASTAPI_RELOAD: bool = True

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def STAGING_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "staging"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def MODELS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "models"

    @property
    def SQL_DIR(self) -> Path:
        return self.PROJECT_ROOT / "sql"

    @property
    def DB_URL(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg2://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.RAW_DATA_DIR,
            self.STAGING_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
