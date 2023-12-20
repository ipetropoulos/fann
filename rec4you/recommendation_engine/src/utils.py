import json
import logging
from datetime import datetime
from itertools import chain, combinations
from logging.config import dictConfig
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from tqdm import tqdm

from config import Config
from snowflake_connection import snowflake_connection

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

config = Config()

# Path to the log file
logging_config_fp = Path(__file__).parent.parent.parent.parent / "config/logging/model_log.json"

# Set up logging
logger = logging.getLogger(__name__)
try:
    with open(logging_config_fp) as fr:
        dictConfig(json.load(fr))
except Exception as e:
    logger.error(f"Error reading the logging configuration: {e}")
    raise e

# Get dict BRAND_ID_TO_NAME_MAPPING
BRAND_ID_TO_NAME_MAPPING = {
    0: "WW",
    5: "RM",
    11: "KS",
    15: "BH",
    23: "JL",
    24: "SA",
    25: "EL",
    26: "CA",
    27: "ZQ",
    28: "CP",
    74: "IA",
    75: "SH",
    76: "AA",
    77: "OS",
    78: "FO",
    79: "JV",
}


def check_if_active_9m_is_empty(brand_id: int) -> bool:
    """
    Checks if the ACTIVE_FILE9M table for a given brand is empty.

    Args:
    brand_id (int): The identifier for the brand.

    Returns:
    bool: True if the ACTIVE_FILE9M table is empty, False otherwise.
    """
    # Establish a connection to the Snowflake database
    with snowflake_connection() as conn:
        cursor = conn.cursor()
        # Execute a query to count the number of emails in the ACTIVE_FILE9M table for the specified brand
        active_9m_emails = cursor.execute(
            f"select count(*) as NUM_EMAILS from EPS_{BRAND_ID_TO_NAME_MAPPING[brand_id]}_ACTIVE_FILE9M limit 1"
        ).fetch_pandas_all()

    # Check if the number of emails is zero (i.e., the table is empty)
    is_empty = active_9m_emails["NUM_EMAILS"] == 0
    # Return the result as a boolean value
    return is_empty.values[0]


def check_if_oih_is_empty() -> bool:
    """
    Checks if the ORDER_ITEM_HIST table for a given brand is empty.

    Args:
    brand_id (int): The identifier for the brand.

    Returns:
    bool: True if the ORDER_ITEM_HIST table is empty, False otherwise.
    """
    # Establish a connection to the Snowflake database
    with snowflake_connection() as conn:
        cursor = conn.cursor()
        # Execute a query to count the number of emails in the ACTIVE_FILE9M table for the specified brand
        oih = cursor.execute(
            f"select count(*) as NUM_MASTER_IDS from {config.order_item_snowflake_table} limit 1"
        ).fetch_pandas_all()

    # Check if the number of emails is zero (i.e., the table is empty)
    is_empty = oih["NUM_MASTER_IDS"] == 0
    # Return the result as a boolean value
    return is_empty.values[0]


def get_rest_of_master_ids(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Gets master IDs from df1 that are not present in df2.

    Args:
    df1 (pd.DataFrame): The first DataFrame containing master IDs. Requires column 'MASTER_ID'.
    df2 (pd.DataFrame): The second DataFrame to compare against. Requires column 'MASTER_ID'.

    Returns:
    pd.DataFrame: A DataFrame containing the unique master IDs from df1 that are not contained in df2.
    """
    # If df2 is empty, return df1 as is
    if df2.empty:
        return df1

    # Return the subset of df1 where 'MASTER_ID' values are not in df2
    return df1[~df1["MASTER_ID"].isin(df2["MASTER_ID"].drop_duplicates().tolist())].reset_index(
        drop=True
    )  # Reset index for the resulting DataFrame


def get_top_n_consequents(orders: pd.DataFrame, rules: pd.DataFrame, sort_cols: list, ascending: list) -> pd.DataFrame:
    """
    Gets the top N consequents based on defined sorting criteria.

    Args:
    orders (pd.DataFrame): The orders DataFrame. Requires columns 'ORDER_LIST', 'MASTER_ID', and 'ITEM_ORDER_DEMAND_DATE'.
    rules (pd.DataFrame): The rules DataFrame. Requires columns 'antecedents', 'consequents', 'confidence', and 'lift'.
    sort_cols (list): List of columns to sort by.
    ascending (list): List indicating sorting order for each column in sort_cols.

    Returns:
    pd.DataFrame: A DataFrame with the top N consequents.
    """
    # Merge orders with rules on matching product lists and antecedents
    results = orders.merge(rules, left_on="ORDER_LIST", right_on="antecedents")

    # Convert MASTER_ID to integer type if it's not already
    if results["MASTER_ID"].dtype != "int":
        results["MASTER_ID"] = results["MASTER_ID"].astype(int)

    # Convert ITEM_ORDER_DEMAND_DATE to datetime type if it's not already
    if not pd.api.types.is_datetime64_any_dtype(results["ITEM_ORDER_DEMAND_DATE"]):
        results["ITEM_ORDER_DEMAND_DATE"] = pd.to_datetime(results["ITEM_ORDER_DEMAND_DATE"])

    # Sort the results based on specified columns and order
    results = results.sort_values(by=sort_cols, ascending=ascending)

    # Create a cumulative count column for deduplication purposes
    results["cumcount"] = results.groupby(["MASTER_ID", "ITEM_ORDER_DEMAND_DATE", "consequents"]).cumcount()

    # Filter out duplicates, keeping only the first occurrence
    results = results[~(results["cumcount"] != 0)]

    # Group by MASTER_ID and ITEM_ORDER_DEMAND_DATE, keeping only the top N consequents as per configuration
    results = (
        results.groupby(["MASTER_ID", "ITEM_ORDER_DEMAND_DATE"]).head(config.top_n_consequents).reset_index(drop=True)
    )

    # Drop the cumcount column as it's no longer needed
    return results.drop("cumcount", axis=1).reset_index(drop=True)


def generate_power_set(m_id: int, p_list: list, order_date: str, max_length: int) -> pd.DataFrame:
    """
    Generates a power set of product combinations.

    Args:
    m_id (int): Master ID.
    p_list (list): List of products.
    order_date (str): The date of the order.
    max_length (int): Maximum length of the product combinations.

    Returns:
    pd.DataFrame: A DataFrame with all combinations of products in the 'PRODUCT_LIST' column, along with 'ITEM_ORDER_DEMAND_DATE' and 'MASTER_ID'.
    """
    # If max_length is not specified, use the length of the product list
    if max_length is None:
        max_length = len(p_list)

    # Generate all possible combinations (power set) of the product list
    # considering combinations of lengths from 1 to max_length
    power_set = chain.from_iterable(combinations(p_list, r) for r in range(1, max_length + 1))

    # Create a DataFrame from the power set with corresponding 'ITEM_ORDER_DEMAND_DATE' and 'MASTER_ID'
    return pd.DataFrame(
        {
            "ORDER_LIST": power_set,
            "ITEM_ORDER_DEMAND_DATE": order_date,
            "MASTER_ID": [m_id] * sum(1 for r in range(1, max_length + 1) for _ in combinations(p_list, r)),
        }
    )


def get_powerset_consequents(df: pd.DataFrame, rules: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
    """
    Generates consequents for each product combination in the power set.

    Args:
    df (pd.DataFrame): The DataFrame containing orders. Requires columns 'MASTER_ID', 'PRODUCT_LIST', and 'ITEM_ORDER_DEMAND_DATE'.
    rules (pd.DataFrame): The DataFrame containing association rules. Requires columns 'antecedents' and 'consequents'.

    Returns:
    pd.DataFrame: A DataFrame with consequents for each product combination.
    """
    # Determine the maximum length of tuples in the 'consequents' column
    max_tuple = len(max(rules["consequents"], key=len))

    # Get unique master IDs from the DataFrame
    unique_master_ids = df["MASTER_ID"].unique().tolist()

    # Calculate the number of batches to process
    num_batches = int(np.ceil(len(unique_master_ids) / batch_size))

    # Split the master IDs into batches for processing
    master_id_batches = np.array_split(unique_master_ids, num_batches)

    batches = []
    for batch in tqdm(master_id_batches):
        # Filter the DataFrame for the current batch of master IDs
        batch_df = df[df["MASTER_ID"].isin(batch)]

        # Generate power sets for each row in the batch DataFrame
        expanded_df = pd.concat(
            (
                generate_power_set(
                    row["MASTER_ID"], row["ORDER_LIST"], row["ITEM_ORDER_DEMAND_DATE"], max_length=max_tuple
                )
                for _, row in batch_df.iterrows()
            ),
            ignore_index=True,
        )

        # Add a column indicating the length of each product list
        expanded_df["LEN"] = expanded_df["ORDER_LIST"].apply(lambda x: len(x))

        # Sort the product list in each row for consistency
        expanded_df["ORDER_LIST"] = expanded_df["ORDER_LIST"].apply(lambda x: tuple(sorted(x)))
        rules["antecedents"] = rules["antecedents"].apply(lambda x: tuple(sorted(x)))

        # Get the top N consequents for each combination
        results = get_top_n_consequents(
            orders=expanded_df,
            rules=rules,
            sort_cols=["MASTER_ID", "ITEM_ORDER_DEMAND_DATE", "LEN", "confidence", "lift"],
            ascending=[True, True, False, False, False],
        )
        # Add the batch results to the list
        batches.append(results)

    # Combine all batches into a single DataFrame
    batches = pd.concat(batches).reset_index(drop=True)
    # Drop the 'LEN' column as it's no longer needed
    batches.drop("LEN", axis=1, inplace=True)

    return batches


def calculate_remaining_products(df_need_filling: pd.DataFrame, group_col: str, product_col: str) -> pd.DataFrame:
    """
    Calculates the remaining number of products needed for each group in the DataFrame.

    Args:
    df_need_filling (pd.DataFrame): DataFrame needing product calculations. Requires the specified 'group_col' and 'product_col'.
    group_col (str): Column name to group by.
    product_col (str): Column name of the product.

    Returns:
    pd.DataFrame: Updated DataFrame with remaining product counts in a new column 'REMAINING'.
    """
    # Calculate the count of products per group
    count_per_group = df_need_filling.groupby(group_col)[product_col].transform("size")

    # Subtract the count from the maximum number of products recommended
    # to find out how many more products are needed for each group
    df_need_filling["REMAINING"] = config.max_products_recommended - count_per_group

    # Ensure that the remaining count does not fall below zero
    df_need_filling["REMAINING"] = df_need_filling["REMAINING"].clip(lower=0)

    # Return the updated DataFrame with the new 'REMAINING' column
    return df_need_filling


def get_most_n_viewed_categories(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Gets the top N viewed categories based on 'SIZE' for each group in the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame with category data. Requires columns 'MASTER_ID', 'REMAINING', 'RANK', and 'SIZE'.

    Returns:
    pd.DataFrame: A DataFrame with the top N viewed categories.
    """
    # Rank the categories within each group by SIZE
    df["SIZE_rank"] = df.groupby(["MASTER_ID", "REMAINING", "RANK"])["SIZE"].rank(method="first", ascending=False)

    # Filter to keep only the top N ranks
    top_n_df = df[df["SIZE_rank"] <= n]

    return top_n_df.drop(columns="SIZE_rank").reset_index(drop=True)


def sample_products(recommendations: pd.DataFrame, sample_function: Callable) -> pd.DataFrame:
    """
    Applies a sampling function to recommendations dataframe and select final columns

    Args:
    recommendations (pd.DataFrame): DataFrame containing recommendations. Requires columns 'MASTER_ID', 'COMBINE', 'RANK', 'SITE_ID', 'PRODUCT', and 'AGE_RANK'.
    sample_function (Callable): Function to sample the recommendations.

    Returns:
    pd.DataFrame: The merged and sampled DataFrame.
    """
    # Apply the sampling function to each group of recommendations identified by 'MASTER_ID'
    sampled_dfs = [sample_function(group) for _, group in recommendations.groupby("MASTER_ID", group_keys=False)]
    recommendations = pd.concat(sampled_dfs).reset_index(drop=True)

    # Convert the 'PRODUCT' column to integer type for consistency
    recommendations["PRODUCT"] = recommendations["PRODUCT"].astype("Int64")

    # Ensure 'MASTER_ID' is also in integer format
    recommendations["MASTER_ID"] = recommendations["MASTER_ID"].astype("Int64")

    # Return the DataFrame with only the specified columns
    return recommendations[["MASTER_ID", "COMBINE", "RANK", "SITE_ID", "PRODUCT", "AGE_RANK"]]


def rows_exist(table: str, brand_code: str, df: Optional[pd.DataFrame] = None) -> bool:
    """
    Checks for the existence of specific rows in a Snowflake table.

    This function verifies the presence of rows in a Snowflake table that match a given brand code,
    the current date, and a specific model or algorithm name. The function adapts to different table
    structures based on the input table name.

    Args:
    - df (pd.DataFrame): Dataframe containing the recommendations.
    - table (str): Name of the Snowflake table to be queried.
    - brand_code (str): Brand code to match in the query (e.g., 'WW').

    Returns:
    - bool: True if at least one matching row is found in the Snowflake table, False otherwise.
    """
    # Retrieve the column names of the Snowflake table
    if table == config.rec4you_snowflake_table:
        site_column = "BRAND"
        model_name_column = "MODEL_NAME"
        if not df["CUSTOMERKEY"].isna().all():  # Check if there's any non-NA value in CUSTOMERKEY
            customer_key = df["CUSTOMERKEY"].dropna().iloc[0]
            extra_text = f"and CUSTOMERKEY = '{customer_key}'"
        elif not df["MASTER_ID"].isna().all():  # Check if there's any non-NA value in MASTER_ID
            # Filter for MASTER_ID with more than 5 digits
            master_ids_with_length = df["MASTER_ID"].dropna().astype(str)
            valid_master_ids = master_ids_with_length[master_ids_with_length.apply(lambda x: len(x) > 5)]

            if not valid_master_ids.empty:
                master_id = valid_master_ids.iloc[0]
                extra_text = f"and MASTER_ID = '{master_id}'"
        else:
            extra_text = ""
    if table == config.general_recs_snowflake_table:
        site_column = "SITE"
        model_name_column = "ALGORITHM"
        extra_text = ""
    try:
        # Establish a connection to the Snowflake database
        with snowflake_connection() as con:
            # Prepare a SQL query to check for existing rows with the specified parameters and today's date
            sql = f"""
            select * from {table}
            where {site_column} = '{brand_code}'
                and CREATED_AT = '{datetime.now().strftime("%Y-%m-%d")}'
                and {model_name_column} = '{config.algorithm_name}'
                {extra_text}
            limit 1
            """
            # Execute the query and store the result
            rows_found = pd.read_sql_query(sql, con)
    except Exception as e:
        # Log an error message in case of an exception
        logger.error(f"Error checking if rows exist in snowflake: {e}")
        raise e

    # Check if the result is empty. Return False if empty (no rows exist), True otherwise
    if rows_found.empty:
        return False

    return True


def write_results(df: pd.DataFrame, table: str, brand_code: str) -> None:
    """
    Writes the results contained in the provided dataframe to the Snowflake table specified in the configuration.

    Args:
    df (pd.DataFrame): Dataframe containing the results to be written.
    table (str): Name of the snowflake table to write results
    brand_code (str): Brand code of the df (e.g 'WW')

    Returns:
    None
    """
    # Check if rows for this brand ID already exist in the Snowflake table
    if (table != config.rec4you_snowflake_table) & (rows_exist(df=df, table=table, brand_code=brand_code)):
        # Log a warning if rows already exist
        logger.warning(
            f"Recommendations for brand {brand_code} and date {df['CREATED_AT'].iloc[0]} already exist in snowflake"
        )
    else:
        # If no rows exist, establish a connection to the Snowflake database
        with snowflake_connection() as con:
            # Write the dataframe to the Snowflake table
            success, nchunks, nrows, _ = write_pandas(con, df, table, quote_identifiers=False)
        logger.info(f"Total rows added to snowflake table: {len(df)}")
