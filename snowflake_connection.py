from contextlib import contextmanager

from pydantic import BaseSettings, Field, SecretStr


class WareHouseConfig(BaseSettings):
    """
    Configuration class for Snowflake warehouse settings.
    Reads the configuration from a .env file.
    """

    sf_username: str = Field(..., env="SF_USERNAME")
    sf_password: str = Field(..., env="SF_PASSWORD")
    sf_account: str = Field(..., env="SF_ACCOUNT")
    sf_region: str = Field(..., env="SF_REGION")
    sf_warehouse: str = Field(..., env="SF_WAREHOUSE")
    sf_role: str = Field(..., env="SF_ROLE")
    sf_database: str = Field(..., env="SF_DATABASE")
    sf_schema: str = Field(..., env="SF_SCHEMA")

    @property
    def config(self) -> dict:
        """
        Constructs and returns the configuration dictionary for Snowflake connection.

        :return: dict - The configuration dictionary.
        """
        return dict(
            user=self.sf_username,
            password=self.sf_password,
            account=self.sf_account,
            region=self.sf_region,
            warehouse=self.sf_warehouse,
            role=self.sf_role,
            database=self.sf_database,
            schema=self.sf_schema,
        )

    class Config:
        env_prefix = ""
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


@contextmanager
def snowflake_connection():
    """
    Context manager to establish a connection to Snowflake.
    Ensures the connection is closed after use.
    """
    from snowflake import connector

    sf_config = WareHouseConfig()

    connection = connector.connect(**sf_config.config)
    try:
        yield connection
    finally:
        connection.close()
