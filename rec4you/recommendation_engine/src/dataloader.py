import datetime
import json

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from rec4you.recommendation_engine.src.utils import (
    BRAND_ID_TO_NAME_MAPPING,
    config,
    get_rest_of_master_ids,
)
from snowflake_connection import snowflake_connection


class Dataloader:
    """
    A class to load and transform data from a Snowflake database for a specific brand.

    Attributes:
        brand_id (int): The identifier for a specific brand.

    Methods:
        get_customer_data_query(): Retrieves the last order for the specified brand.
        get_orders_last_n_months(n_months): Retrieves orders for the specified brand for the last n months.
        get_products_ranking_by_age(): Fetches product ranking by age.
        _transform_order_item_history(order_item_history_df): Static method to transform order item history data.
    """

    def __init__(self, run: str, brand_id: int = None) -> None:
        """
        Initializes the Dataloader with a specific brand ID.

        Args:
            brand_id (int): The identifier for a specific brand.
            run (str): Running mode train or inference)
        """
        self.brand_id = brand_id
        if run == "inference":
            self.ranked_by_age = self.get_products_ranking_by_age()

    def set_brand_id(self, brand_id: int) -> None:
        """
        Sets the brand ID for the Dataloader instance.

        Args:
            brand_id (int): The identifier for a specific brand.
        """
        self.brand_id = brand_id

    def get_customer_data_query(self) -> pd.DataFrame:
        """
        Retrieves the last order of each customer from the database for the current brand ID.

        Returns:
            pd.DataFrame: A DataFrame containing the last order data.
        """
        customer_data_query = f"""
            with effort_brand_code_cte as (
                select
                    DISTINCT
                    OH.MASTER_ID,
                    OH.ITEM_ORDER_DEMAND_DATE,
                    case
                        when OH.ADDCART_BRAND = '01' THEN 0
                        when OH.ADDCART_BRAND = '07' THEN 5
                        when OH.ADDCART_BRAND = '14' THEN 77
                        when OH.ADDCART_BRAND IS NULL OR TRIM(OH.ADDCART_BRAND) = '' THEN
                                case
                                    when OH.ORDERED_EFFORT = 1 THEN 0
                                    when OH.ORDERED_EFFORT = 7 THEN 5
                                    when OH.ORDERED_EFFORT = 14 THEN 77
                                    else TO_NUMBER(OH.ORDERED_EFFORT)
                                end
                        else TO_NUMBER(OH.ADDCART_BRAND)
                    end as EFFORT_BRAND_CODE
                from
                    {config.order_item_snowflake_table} OH
                where EFFORT_BRAND_CODE = {self.brand_id} and ITEM_ORDER_DEMAND_DATE >= dateadd(years, -2, current_date())
            ),
            order_history_max_date as (
                select MASTER_ID, EFFORT_BRAND_CODE, max(ITEM_ORDER_DEMAND_DATE) as max_order_date
                from effort_brand_code_cte
                group by MASTER_ID, EFFORT_BRAND_CODE
            ),

            order_history_n_months AS (
                select
                    order_history.MASTER_ID,
                    order_history.ITEM_ORDER_DEMAND_DATE,
                    order_history.INVOICE_NO,
                    order_history.ORDERED_EFFORT,
                    order_history.DEPARTMENT,
                    order_history.ORDERED_STYLE,
                    order_history.ITEM_SEQ,
                    order_history.MERCH_ITEM_NUM,
                    order_history.ITEM_CONFIRM_NUM,
                    order_history.EFFORT_BRAND_CODE,
                    pdm_styles.SF_PRODUCT_ID,
                    pdm_styles.COMBINE
                from (
                    select
                        distinct
                        OH.*,
                        case
                            when OH.ADDCART_BRAND = '01' THEN 0
                            when OH.ADDCART_BRAND = '07' THEN 5
                            when OH.ADDCART_BRAND = '14' THEN 77
                            when OH.ADDCART_BRAND IS NULL OR TRIM(OH.ADDCART_BRAND) = '' THEN
                                case
                                    when OH.ORDERED_EFFORT = 1 THEN 0
                                    when OH.ORDERED_EFFORT = 7 THEN 5
                                    when OH.ORDERED_EFFORT = 14 THEN 77
                                    else TO_NUMBER(OH.ORDERED_EFFORT)
                                end
                            else TO_NUMBER(OH.ADDCART_BRAND)
                        end as EFFORT_BRAND_CODE
                    from
                        {config.order_item_snowflake_table} OH
                    where
                        EFFORT_BRAND_CODE = {self.brand_id} and
                        exists (
                            select 1
                            from order_history_max_date
                            where MASTER_ID = OH.MASTER_ID
                                AND max_order_date = OH.ITEM_ORDER_DEMAND_DATE
                        )
                ) order_history
                inner join (
                select
                    CASE
                        WHEN BRAND_ID = '00' THEN 1
                        WHEN BRAND_ID = '05' THEN 7
                        ELSE CAST(BRAND_ID AS INT)
                    END as TRANSFORMED_BRAND_ID,
                    SELLING_DEPT,
                    MF_ITEM_ID,
                    MF_STYLE_ID,
                    SF_PRODUCT_ID,
                    nullif(rtrim(coalesce(trim(division), '') || '/' || coalesce(trim(category), '') || '/' || coalesce(trim(subcategory), ''), '//'), '') as COMBINE
                from {config.pdm_styles_snowflake_table}
            ) pdm_styles
            on TO_NUMBER(trim(order_history.ORDERED_EFFORT)) = TO_NUMBER(trim(pdm_styles.TRANSFORMED_BRAND_ID))
            and TO_NUMBER(trim(order_history.DEPARTMENT)) = TO_NUMBER(trim(pdm_styles.SELLING_DEPT))
            and TO_NUMBER(trim(order_history.MERCH_ITEM_NUM)) = TO_NUMBER(trim(pdm_styles.MF_ITEM_ID))
            and TO_NUMBER(trim(order_history.ORDERED_STYLE)) = TO_NUMBER(trim(pdm_styles.MF_STYLE_ID))
            ),

            promotable_customers as ("""

        # Conditional logic for promotable_customers
        if self.brand_id not in [27, 28]:
            customer_data_query += f"""
                WITH prioritized_customers AS (
                            SELECT
                                distinct
                                MASTER_ID,
                                UPPER(trim(EMAILADDRESS)) as EMAIL_ADDRESS,
                                FIRST_VALUE(TYPE) OVER (
                                    PARTITION BY MASTER_ID
                                    ORDER BY
                                        CASE TYPE
                                            WHEN 'active' THEN 1
                                            WHEN 'active_lc' THEN 2
                                            WHEN 'lapsed' THEN 3
                                            WHEN 'inactive' THEN 4
                                            ELSE 5
                                        END
                                ) AS prioritized_type
                            FROM
                                EPS_{BRAND_ID_TO_NAME_MAPPING[self.brand_id]}_ACTIVE_FILE9M
                            WHERE
                                to_date(INSERTED_AT) = (SELECT to_date(max(INSERTED_AT)) FROM EPS_{BRAND_ID_TO_NAME_MAPPING[self.brand_id]}_ACTIVE_FILE9M)
                        )

                        select distinct MASTER_ID, UPPER(trim(EMAIL_ADDRESS)) as EMAIL_ADDRESS, MAX(TYPE) as TYPE
                        from (
                                (
                                    SELECT DISTINCT
                                        MASTER_ID,
                                        UPPER(trim(EMAIL_ADDRESS)) as EMAIL_ADDRESS,
                                        NULL as TYPE
                                    FROM {config.emonster_copy_snowflake_table}
                                    WHERE
                                        CASE
                                            WHEN EFFORT_CD = 1 THEN 0
                                            WHEN EFFORT_CD = 7 THEN 5
                                            WHEN EFFORT_CD = 14 THEN 77
                                            ELSE EFFORT_CD
                                        END = {self.brand_id}
                                )
                                UNION
                                (   SELECT DISTINCT
                                        MASTER_ID,
                                        EMAIL_ADDRESS,
                                        prioritized_type AS TYPE
                                    FROM
                                        prioritized_customers
                                )
                            )
                        GROUP BY MASTER_ID, EMAIL_ADDRESS),"""
        else:
            customer_data_query += """
                SELECT CAST(NULL AS VARCHAR) AS MASTER_ID, CAST(NULL AS VARCHAR) AS TYPE, CAST(NULL AS VARCHAR) AS EMAIL_ADDRESS
                WHERE FALSE),
            """

        customer_data_query += f"""
            demographics as (
                select
                    distinct
                    MASTER_ID,
                    MIN(
                        CASE
                            WHEN TRY_TO_DATE(DATE_OF_BIRTH) IS NOT NULL THEN
                                FLOOR(DATEDIFF(day, TRY_TO_DATE(DATE_OF_BIRTH), CURRENT_DATE()) / 365)
                        END
                    ) as AGE
                from {config.demographics_snowflake_table}
                group by MASTER_ID
            ),

            views as (
                SELECT
                    MASTER_ID,
                    ARRAY_AGG(VIEWED_NOT_ORDERED) AS VIEWS_LIST
                FROM
                    {config.cdm_products_viewed_not_ordered_snowflake_table}
                WHERE
                    LATEST_SESSION >= dateadd(year, -2, current_date())
                GROUP BY
                    MASTER_ID
                    ),

            concatenated_master_ids AS (
                with master_ids as(
                    SELECT distinct MASTER_ID FROM order_history_n_months
                    UNION
                    SELECT distinct MASTER_ID FROM promotable_customers
                    UNION
                    SELECT distinct MASTER_ID FROM views
                ),

                date_agg AS (
                    SELECT
                        distinct
                            MASTER_ID,
                            ITEM_ORDER_DEMAND_DATE
                    FROM
                        order_history_n_months
                ),

                type_agg AS (
                    SELECT
                        distinct
                            MASTER_ID,
                            TYPE,
                            EMAIL_ADDRESS
                    FROM
                        promotable_customers
                )

                SELECT
                    m.MASTER_ID,
                    d.ITEM_ORDER_DEMAND_DATE,
                    t.TYPE,
                    t.EMAIL_ADDRESS
                FROM
                    master_ids m
                LEFT JOIN
                    date_agg d ON m.MASTER_ID = d.MASTER_ID
                LEFT JOIN
                    type_agg t ON m.MASTER_ID = t.MASTER_ID)

            SELECT
                cm.MASTER_ID,
                ARRAY_AGG(
                    OBJECT_CONSTRUCT(
                        'email', IFF(cm.EMAIL_ADDRESS IS NULL, 'NULL', cm.EMAIL_ADDRESS),
                        'type', IFF(cm.TYPE IS NULL, 'NULL', cm.TYPE)
                    )
                ) AS EMAIL_TYPE_LIST,
                cm.ITEM_ORDER_DEMAND_DATE,
                (SELECT LISTAGG(trim(ohnm.COMBINE), ', ') WITHIN GROUP (ORDER BY ohnm.COMBINE)
                    FROM (
                        SELECT DISTINCT MASTER_ID, COMBINE
                        FROM order_history_n_months ohnm
                        WHERE ohnm.MASTER_ID = cm.MASTER_ID
                    ) ohnm) AS ORDER_LIST,
                v.VIEWS_LIST,
                MIN(d.AGE) AS AGE
            FROM
                concatenated_master_ids cm
            LEFT JOIN
                order_history_n_months ohnm ON
                cm.MASTER_ID = ohnm.MASTER_ID
            LEFT JOIN
                views v ON cm.MASTER_ID = v.MASTER_ID
            LEFT JOIN
                demographics d ON cm.MASTER_ID = d.MASTER_ID
            GROUP BY
                cm.MASTER_ID, cm.ITEM_ORDER_DEMAND_DATE, v.VIEWS_LIST;
            """
        return customer_data_query

    def get_orders_last_n_months(self, n_months: int) -> pd.DataFrame:
        """
        Retrieves orders from the last n months for the current brand ID.

        Args:
            n_months (int): The number of months to look back for orders.

        Returns:
            pd.DataFrame: A DataFrame containing order data from the last n months.
        """
        with snowflake_connection() as conn:
            last_orders = (
                conn.cursor()
                .execute(
                    f"""
                with order_history_n_months as (
                    select
                        order_history.MASTER_ID,
                        order_history.ITEM_ORDER_DEMAND_DATE,
                        order_history.INVOICE_NO,
                        order_history.ITEM_SEQ,
                        order_history.ORDERED_EFFORT,
                        order_history.DEPARTMENT,
                        order_history.ORDERED_STYLE,
                        order_history.MERCH_ITEM_NUM,
                        order_history.ITEM_CONFIRM_NUM,
                        order_history.EFFORT_BRAND_CODE,
                        pdm_styles.SF_PRODUCT_ID,
                        pdm_styles.COMBINE
                    from (
                        select
                        *,
                        case
                            when ADDCART_BRAND = '01' then 0
                            when ADDCART_BRAND = '07' then 5
                            when ADDCART_BRAND = '14' then 77
                            when ADDCART_BRAND is null or trim(ADDCART_BRAND) = '' then
                                case
                                        when ORDERED_EFFORT = '1' THEN 0
                                        when ORDERED_EFFORT = '7' THEN 5
                                        when ORDERED_EFFORT = '14' THEN 77
                                        else TO_NUMBER(ORDERED_EFFORT)
                                    end
                            else to_number(ADDCART_BRAND)
                        end as EFFORT_BRAND_CODE
                        from {config.order_item_snowflake_table}
                        where ITEM_ORDER_DEMAND_DATE >= dateadd(months, -{n_months}, current_date())
                    ) order_history
                    inner join (
                    select
                        CASE
                            WHEN BRAND_ID = '00' THEN 1
                            WHEN BRAND_ID = '05' THEN 7
                            ELSE CAST(BRAND_ID AS INT)
                        END as TRANSFORMED_BRAND_ID,
                        SELLING_DEPT,
                        MF_ITEM_ID,
                        MF_STYLE_ID,
                        SF_PRODUCT_ID,
                        nullif(rtrim(coalesce(trim(division), '') || '/' || coalesce(trim(category), '') || '/' || coalesce(trim(subcategory), ''), '//'), '') as COMBINE
                    from {config.pdm_styles_snowflake_table}
                    ) pdm_styles
                    on TO_NUMBER(trim(order_history.ORDERED_EFFORT)) = TO_NUMBER(trim(pdm_styles.TRANSFORMED_BRAND_ID))
                    and TO_NUMBER(trim(order_history.DEPARTMENT)) = TO_NUMBER(trim(pdm_styles.SELLING_DEPT))
                    and TO_NUMBER(trim(order_history.MERCH_ITEM_NUM)) = TO_NUMBER(trim(pdm_styles.MF_ITEM_ID))
                    and TO_NUMBER(trim(order_history.ORDERED_STYLE)) = TO_NUMBER(trim(pdm_styles.MF_STYLE_ID))
                        where order_history.EFFORT_BRAND_CODE = {self.brand_id}
                    )

                select * from order_history_n_months
            """
                )
                .fetch_pandas_all()
            )
        return self._transform_order_item_history(last_orders)

    def get_products_ranking_by_age(self) -> pd.DataFrame:
        """
        Fetches product ranking by age from the database.

        Returns:
            pd.DataFrame: A DataFrame containing product ranking by age.
        """
        previous_working_day = (datetime.datetime.now() - BDay(1)).strftime("%Y-%m-%d")
        with snowflake_connection() as conn:
            ranked_by_age = (
                conn.cursor()
                .execute(
                    f"""
                with pdm_products as (
                    select distinct SF_PRODUCT_ID,
                        nullif(rtrim(coalesce(trim(division), '') || '/' || coalesce(trim(category), '') || '/' || coalesce(trim(subcategory), ''), '//'), '') as COMBINE
                    from {config.pdm_styles_snowflake_table}
                ),
                rank as(
                    select * from {config.product_ranking_snowflake_table}
                    where DATE_TIME = '{previous_working_day}'
                )

                select * from rank
                left join pdm_products
                on pdm_products.SF_PRODUCT_ID = rank.PRODUCT
            """
                )
                .fetch_pandas_all()
            )

        ranked_by_age = ranked_by_age.drop(["DATE_TIME", "SF_PRODUCT_ID"], axis=1)
        # Filter out rows with invalid product codes (containing '_' or '-')
        ranked_by_age = ranked_by_age[~ranked_by_age["PRODUCT"].str.contains("_|-")].drop_duplicates()
        ranked_by_age["PRODUCT"] = ranked_by_age["PRODUCT"].astype("Int64")

        # Reshape the DataFrame to a long format, where each row represents a single age rank for a product
        ranked_by_age = pd.melt(
            ranked_by_age,
            id_vars=["SITE_ID", "PRODUCT", "COMBINE"],
            value_vars=[
                "RANKING_AGE_0_35",
                "RANKING_AGE_36_49",
                "RANKING_AGE_50_60",
                "RANKING_AGE_61_1000",
                "RANKING_AGE_0_1000",
            ],
            var_name="RANK",
            value_name="AGE_RANK",
        )

        ranked_by_age["AGE_RANK"] = ranked_by_age["AGE_RANK"].astype("Int64")
        return ranked_by_age

    @staticmethod
    def _transform_order_item_history(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the transactional data by adding column BRANDCODE
        and check column COMBINE

        Args:
            df (pd.DataFrame): A DataFrame containing transactional data.

        Returns:
            pd.DataFrame: A transformed DataFrame.
        """
        with snowflake_connection() as conn:
            brand_id_df = (
                conn.cursor()
                .execute(
                    f"""
                select BRANDCODE,
                CASE
                    WHEN BRANDID = '00' THEN 1
                    WHEN BRANDID = '05' THEN 7
                    ELSE CAST(BRANDID AS INT)
                END as ORDERED_EFFORT from {config.brand_code_snowflake_table}
                """
                )
                .fetch_pandas_all()
            )

        # Merge the initial DataFrame with the brand information based on the 'ORDERED_EFFORT' column
        order_item_history_df = df.merge(brand_id_df, on="ORDERED_EFFORT", how="left")
        # Replace empty strings in 'COMBINE' column with None for consistency
        order_item_history_df["COMBINE"].replace("", None, inplace=True)
        # Convert 'ITEM_ORDER_DEMAND_DATE' column to datetime objects and extract only the date part
        order_item_history_df["ITEM_ORDER_DEMAND_DATE"] = pd.to_datetime(
            order_item_history_df["ITEM_ORDER_DEMAND_DATE"]
        ).dt.date

        # If the ordered effort is not 15, drop rows where 'COMBINE' is null
        if order_item_history_df["ORDERED_EFFORT"].iloc[0] != 15:
            order_item_history_df.dropna(subset=["COMBINE"], inplace=True)

        # Group by 'MASTER_ID' and 'ITEM_ORDER_DEMAND_DATE', then aggregate 'COMBINE' values into a list
        # Only keep unique values in each list and drop rows where the list is None
        order_item_history_df = (
            order_item_history_df.groupby(["MASTER_ID", "ITEM_ORDER_DEMAND_DATE", "ITEM_CONFIRM_NUM"])["COMBINE"]
            .apply(lambda x: list(set(x)) if len(set(x)) >= 1 else None)
            .dropna()
            .rename("PRODUCT_LIST")
            .reset_index(drop=False)
        )
        # Convert each 'PRODUCT_LIST' to a sorted tuple for uniformity
        order_item_history_df["PRODUCT_LIST"] = [tuple(sorted(x)) for x in order_item_history_df["PRODUCT_LIST"]]

        return order_item_history_df

    def get_products_per_brand(self) -> pd.DataFrame:
        """
        Retrieves products for a specific brand.

        Returns:
            pd.DataFrame: DataFrame containing products for the specified brand.
        """
        brand_code = BRAND_ID_TO_NAME_MAPPING[self.brand_id]
        selected_products = self.ranked_by_age[self.ranked_by_age["SITE_ID"] == brand_code]
        selected_products = selected_products.replace(
            "Sleep & Lounge/Pajamas/Pajama Bottoms", "Sleep & Lounge/Pajamas/Pajama Pants"
        )
        return selected_products

    @staticmethod
    def get_top_n_products_per_age_group_and_category(
        products: pd.DataFrame, n: int = config.sample_from_top_n_products
    ) -> pd.DataFrame:
        """
        Retrieves the top N products per age group and Category

        Args:
            products (pd.DataFrame): DataFrame containing product data.
            n (int): Number of top products to retrieve per age group.

        Returns:
            pd.DataFrame: DataFrame containing top N products per age group.
        """
        selected_top_n_products = (
            products.sort_values("AGE_RANK").groupby(["COMBINE", "RANK"]).head(n).reset_index(drop=True)
        )
        return selected_top_n_products

    @staticmethod
    def get_top_n_products_per_age_group(
        products: pd.DataFrame, n: int = config.sample_from_top_n_products
    ) -> pd.DataFrame:
        """
        Retrieves the top N products per age group

        Args:
            products (pd.DataFrame): DataFrame containing product data.
            n (int): Number of top products to retrieve per age group.

        Returns:
            pd.DataFrame: DataFrame containing top N products per age group.
        """
        selected_top_n_products = products.sort_values("AGE_RANK").groupby("RANK").head(n).reset_index(drop=True)
        return selected_top_n_products

    def prepare_df(self, df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and transforms the DataFrame by adding the 'RANK' column and dropping unnecessary columns.
        Also, replaces 'Sleep & Lounge/Pajamas/Pajama Bottoms' with 'Sleep & Lounge/Pajamas/Pajama Pants'.

        Args:
            df (pd.DataFrame): The DataFrame to be prepared.
            products (pd.DataFrame): Product data for rank adjustment.

        Returns:
            pd.DataFrame: The prepared DataFrame.
        """
        # Get top N products per age group
        top_n_products = self.get_top_n_products_per_age_group(products)
        ranking_columns = top_n_products["RANK"].unique()

        # Efficiently compute rank based on age
        def get_rank(age):
            for col in ranking_columns:
                age_range = [int(x) for x in col.split("_")[-2:]]
                if age_range[0] <= age <= age_range[1]:
                    return col
            return "RANKING_AGE_0_1000"

        df["RANK"] = df["AGE"].apply(get_rank)

        # Conditional operations based on 'consequents' column
        if "consequents" in df.columns:
            df["consequents"].replace(
                "Sleep & Lounge/Pajamas/Pajama Bottoms", "Sleep & Lounge/Pajamas/Pajama Pants", inplace=True
            )
            df = df.explode("consequents")
            df["VIEWS_LIST"] = df["VIEWS_LIST"].replace({np.nan: None})
            df = df.rename(columns={"consequents": "COMBINE"})
            return df[["MASTER_ID", "ITEM_ORDER_DEMAND_DATE", "COMBINE", "RANK", "VIEWS_LIST"]]

        df["VIEWS_LIST"] = df["VIEWS_LIST"].replace({np.nan: None})
        return df[["MASTER_ID", "RANK", "VIEWS_LIST"]].drop_duplicates()

    def get_non_rule_master_ids(
        self,
        customer_data: pd.DataFrame,
        rule_based_recommended_products: pd.DataFrame,
        website: bool,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Retrieves MASTER_IDs that do not have rule-based recommendations.

        Args:
            customer_data (pd.DataFrame): DataFrame containing customer data.
            rule_based_recommended_products (pd.DataFrame): DataFrame containing rule-based recommended products.
            promotable_customers (pd.DataFrame): DataFrame containing promotable customer information.
            website (bool): Flag indicating if the operation is for a website.
            products (pd.DataFrame): DataFrame containing product information.

        Returns:
            pd.DataFrame: A DataFrame containing MASTER_IDs without rule-based recommendations.
        """
        rest_master_ids = get_rest_of_master_ids(customer_data, rule_based_recommended_products)

        if website:
            # Combine and drop duplicates
            return self.prepare_df(rest_master_ids, products)
        else:
            # Merge, filter and drop duplicates
            return self.prepare_df(rest_master_ids[~rest_master_ids["TYPE"].isna()].reset_index(drop=True), products)
