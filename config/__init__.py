import datetime

from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """
    Configuration settings for the application, sourced from environment variables.
    """

    # General configuration
    debug_mode: bool = Field(..., env="DEBUG_MODE")
    deployed_at: str = str(datetime.datetime.now())
    version: str = Field(..., env="VERSION")

    # Product Recommendation configuration
    pdm_styles_snowflake_table: str = Field(..., env="PDM_STYLES_SNOWFLAKE_TABLE")
    product_ranking_snowflake_table: str = Field(..., env="PRODUCT_RANKING_AGE_SNOWFLAKE_TABLE")

    # Recommended For You configuration
    top_n_consequents: int = Field(default=3, env="TOP_N_CONSEQUENTS")
    sample_from_top_n_products: int = Field(default=20, env="SAMPLE_FROM_TOP_N_PRODUCTS")
    max_products_recommended: int = Field(default=10, env="MAX_PRODUCTS_RECOMMENDED")
    algorithm_name: str = Field(default="rec4you_v1", env="ALGORITHM_NAME")
    batch_size: int = Field(default=2**22, env="BATCH_SIZE")

    # Snowflake table used for modeling
    demographics_snowflake_table: str = Field(..., env="DEMOGRAPHICS_SNOWFLAKE_TABLE")
    brand_code_snowflake_table: str = Field(..., env="BRAND_CODE_SNOWFLAKE_TABLE")
    emonster_copy_snowflake_table: str = Field(..., env="EMONSTER_COPY_SNOWFLAKE_TABLE")
    cdm_products_viewed_not_ordered_snowflake_table: str = Field(
        ..., env="CDM_PRODUCTS_VIEWED_NOT_ORDERED_SNOWFLAKE_TABLE"
    )
    order_item_snowflake_table: str = Field(..., env="ORDER_ITEM_SNOWFLAKE_TABLE")

    # Snowflake tables to write results
    recommended_for_you_snowflake_table: str = Field(..., env="RECOMMENDED_FOR_YOU_SNOWFLAKE_TABLE")
    general_recs_snowflake_table: str = Field(..., env="ALGORITHMS_SNOWFLAKE_TABLE")
    rec4you_snowflake_table: str = Field(..., env="REC4YOU_INITIAL_SNOWFLAKE_TABLE")

    class Config:
        """
        Pydantic meta configuration for the main Config class.
        Sets case sensitivity, environment file location, and its encoding.
        """

        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_config():
    """
    Generator function that yields the configuration object.
    """
    confing = Config()
    yield confing
