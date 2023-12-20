import json
from datetime import datetime, timedelta
from typing import Any, List

import numpy as np
import pandas as pd

from rec4you.recommendation_engine.src.dataloader import Dataloader
from rec4you.recommendation_engine.src.model import Rules
from rec4you.recommendation_engine.src.utils import (
    BRAND_ID_TO_NAME_MAPPING,
    calculate_remaining_products,
    config,
    get_most_n_viewed_categories,
    get_powerset_consequents,
    get_rest_of_master_ids,
    get_top_n_consequents,
    logger,
    sample_products,
    write_results,
)
from snowflake_connection import snowflake_connection

pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning


class Recommender:
    """
    A class for generating product recommendations.

    Attributes:
        dataloader (Dataloader): An instance of the Dataloader class to load and transform data.
        rules (Rules): An instance of the Rules class for market basket analysis and rule-based recommendations.
    """

    def __init__(self, dataloader: Dataloader, rules: Rules) -> None:
        """
        Initializes the Recommender with instances of Dataloader and Rules.

        Args:
            dataloader (Dataloader): An instance of the Dataloader class.
            rules (Rules): An instance of the Rules class.
        """
        self.dataloader = dataloader
        self.rules = rules

    def inference(self, website: bool = False) -> pd.DataFrame:
        """
        Generates product recommendations.

        Args:
            website (bool): Flag to determine if the recommendation is for a website.

        Returns:
            pd.DataFrame: DataFrame containing the final product recommendations.
        """
        # Retrieve the brand ID and product details
        brand_id = self.dataloader.brand_id
        products = self.dataloader.get_products_per_brand()
        customer_data_query = self.dataloader.get_customer_data_query()

        def convert_to_tuples(email_type_list_str):
            try:
                # Parse the JSON string
                email_type_list = json.loads(email_type_list_str)

                # Convert each dictionary to a tuple, replace 'NULL' with None
                return [
                    (
                        item.get("email") if item.get("email") != "NULL" else None,
                        item.get("type") if item.get("type") != "NULL" else None,
                    )
                    for item in email_type_list
                ]
            except json.JSONDecodeError:
                return None

        with snowflake_connection() as con:
            for i, customer_data in enumerate(pd.read_sql_query(customer_data_query, con, chunksize=config.batch_size)):
                customer_data["EMAIL_TYPE_LIST"] = customer_data["EMAIL_TYPE_LIST"].apply(convert_to_tuples)

                customer_data["ORDER_LIST"].replace("", None, inplace=True)
                customer_data["VIEWS_LIST"].replace("", None, inplace=True)

                customer_data["VIEWS_LIST"] = customer_data["VIEWS_LIST"].apply(
                    lambda x: [int(item) for sublist in json.loads(x) for item in sublist] if x is not None else None
                )

                customer_data["ORDER_LIST"] = customer_data["ORDER_LIST"].apply(
                    lambda x: x.split(", ") if x is not None else None
                )

                customer_data["ORDER_LIST"] = [
                    tuple(sorted(x)) if x is not None else None for x in customer_data["ORDER_LIST"]
                ]

                customer_data["VIEWS_LIST"] = [
                    tuple(sorted(x)) if x is not None else None for x in customer_data["VIEWS_LIST"]
                ]

                # Filter for active promotable customers if not for website
                if not website:
                    customer_data = customer_data[
                        ~customer_data["EMAIL_TYPE_LIST"].apply(
                            lambda lst: any(item[1] is None for item in lst if item is not None)
                        )
                    ]
                    logger.info(
                        f"{len(customer_data)} promotable customers for brand {BRAND_ID_TO_NAME_MAPPING[brand_id]}..."
                    )
                else:
                    # Filter for customers with at least one view or order
                    customer_data = customer_data[
                        ~(customer_data["VIEWS_LIST"].isna() & customer_data["ORDER_LIST"].isna())
                    ]

                # Load the set of rules for recommendations
                rules = self.rules.load_rules()
                rules_matching_ids = pd.DataFrame()
                rule_based_recommended_products = pd.DataFrame()

                # Process rule-based recommendations if rules are available
                if not rules.empty:
                    # Get the top consequents based on the loaded rules
                    top_consequents = get_top_n_consequents(
                        orders=customer_data[~customer_data["ORDER_LIST"].isna()]
                        .drop(columns=["EMAIL_TYPE_LIST"], axis=1)
                        .drop_duplicates(),
                        rules=rules,
                        sort_cols=["MASTER_ID", "ITEM_ORDER_DEMAND_DATE", "confidence", "lift"],
                        ascending=[True, True, False, False],
                    )
                    rest_master_ids = get_rest_of_master_ids(
                        customer_data[~customer_data["ORDER_LIST"].isna()]
                        .drop(columns=["EMAIL_TYPE_LIST"], axis=1)
                        .drop_duplicates(),
                        top_consequents,
                    )
                    logger.info("Extracting powerset consequents...")
                    powerset_consequents = pd.DataFrame()
                    if not rest_master_ids.empty:
                        powerset_consequents = get_powerset_consequents(df=rest_master_ids, rules=rules)

                    # Combine the results for further processing
                    rules_matching_ids = pd.concat([top_consequents, powerset_consequents], axis=0)
                    rules_matching_ids = self.dataloader.prepare_df(rules_matching_ids, products)

                    logger.info("Extracting rule-based recommendations...")
                    rule_based_recommended_products = self.extract_product_recommendations(
                        df=rules_matching_ids, products=products, rule_based=True
                    )

                # Get the master IDs for customers not covered by rule-based recommendations
                # (less than max_products recommended or other customers with no transactional data)
                rest_master_ids = self.dataloader.get_non_rule_master_ids(
                    customer_data=customer_data.drop(columns=["EMAIL_TYPE_LIST"], axis=1),
                    rule_based_recommended_products=rule_based_recommended_products,
                    website=website,
                    products=products,
                )

                logger.info("Extracting non-rule based recommendations...")
                non_rule_based_recommended_products = self.extract_product_recommendations(
                    df=rest_master_ids, products=products, rule_based=False
                )

                # Combine rule-based and non-rule-based recommendations
                final_recommendations = pd.concat(
                    [rule_based_recommended_products, non_rule_based_recommended_products], axis=0
                )

                final_recommendations = self.get_final_recommendations_format(
                    recommendations=final_recommendations[["MASTER_ID", "SITE_ID", "PRODUCT", "RECO_SRC"]],
                    email_data=customer_data[["MASTER_ID", "EMAIL_TYPE_LIST"]],
                    brand_id=brand_id,
                )

                logger.info(
                    f"Product recommendations for brand {brand_id} and batch number {i} are generated successfully! "
                    f"Total unique recommendations generated: {len(final_recommendations[['MASTER_ID']].drop_duplicates())}"
                )
                # Load the personalized recommendations to Snowflake
                logger.info(
                    f"Load the personalized recommendations for brand {brand_id} and batch number {i} to Snowflake!"
                )
                write_results(
                    df=final_recommendations,
                    table=config.rec4you_snowflake_table,
                    brand_code=BRAND_ID_TO_NAME_MAPPING[brand_id],
                )
                logger.info(
                    f"The personalized recommendations for brand {brand_id} and batch number {i} are completed!"
                )
        logger.info(f"Personalized Product recommendations for brand {brand_id} are generated successfully! ")

    def extract_product_recommendations(
        self, df: pd.DataFrame, products: pd.DataFrame, rule_based: bool = False
    ) -> pd.DataFrame:
        """
        Extracts product recommendations.

        Args:
            df (pd.DataFrame): DataFrame containing the data for recommendation extraction. Requires 'MASTER_ID', 'ITEM_ORDER_DEMAND_DATE', 'PRODUCT_LIST' columns.
            products (pd.DataFrame): DataFrame containing product data. Requires 'PRODUCT', 'RANK', 'COMBINE' columns.
            rule_based (bool): Flag to determine if the recommendation should be rule-based.

        Returns:
            pd.DataFrame: DataFrame containing product recommendations.
        """
        completed_recommendations = pd.DataFrame()
        recommendations_rules = pd.DataFrame()

        if rule_based:
            recommendations_rules = self._recommend_based_on_rules(df=df, products=products)

            # Calculate the completed set of recommendations from the concatenated batches
            completed_recommendations = self.__calculate_completed_recommendations(recommendations_rules)

            # Assign a source indicator (RECO_SRC) for these rule-based recommendations
            completed_recommendations["RECO_SRC"] = 1

        if (not recommendations_rules.empty) & rule_based:
            # Handle incomplete recommendations
            logger.info(
                f"Fill product recommendations for customers with less than {config.max_products_recommended} products recommended..."
            )
            logger.info("Start filling based on views...")
            customers_less_than_10 = recommendations_rules[
                ~recommendations_rules["MASTER_ID"].isin(completed_recommendations["MASTER_ID"].unique().tolist())
            ].reset_index(drop=True)
            reco_src = 1
        else:
            # For non-rule based recommendations start filling based on views
            logger.info("Start recommendations based on views...")
            rule_based = False
            customers_less_than_10 = df.drop_duplicates()
            reco_src = 2

        # Start extracting or filling recommendations based on views first and then age
        if not customers_less_than_10.empty:
            # Generate recommendations based on product views
            recommendations_views = self._recommend_based_on_views(
                df=customers_less_than_10, products=products, rule_based=rule_based
            )

            # Calculate the complete set of recommendations, ensuring each customer has the required number of recommendations
            completed_recommendations_views = self.__calculate_completed_recommendations(recommendations_views)

            # Assign a source indicator (RECO_SRC) to these recommendations for tracking
            completed_recommendations_views["RECO_SRC"] = reco_src

            # If RECO_SRC column is not present in the existing completed recommendations, add it
            if "RECO_SRC" not in completed_recommendations.columns:
                completed_recommendations["RECO_SRC"] = reco_src

            # Combine the new set of recommendations with the previously completed ones
            completed_recommendations = pd.concat([completed_recommendations, completed_recommendations_views], axis=0)

            logger.info(
                f"Fill product recommendations for customers with less than {config.max_products_recommended} products recommended..."
            )
            logger.info("Start filling based on best sellers (by age if available)...")

            # Identify customers who still have less than the required number of recommendations after the view-based filling
            customers_less_than_10 = recommendations_views[
                ~recommendations_views["MASTER_ID"].isin(completed_recommendations_views["MASTER_ID"].unique().tolist())
            ]

            # Fill remaining recommendations based on best sellers per age group
            if not customers_less_than_10.empty:
                # Generate recommendations based on best-selling products for each age group
                # 'fill=True' indicates that this step is to fill up any remaining recommendations needed
                recommendations_age = self._recommend_based_on_best_sellers_per_age(
                    df_need_filling=customers_less_than_10, products=products, fill=True
                ).drop_duplicates()

                # Calculate the complete set of age-based recommendations, ensuring each customer has the required number of recommendations
                completed_recommendations_age = self.__calculate_completed_recommendations(recommendations_age)

                # Assign a source indicator (RECO_SRC) for these recommendations
                # This helps in tracking the origin of the recommendations
                completed_recommendations_age["RECO_SRC"] = reco_src

                # If RECO_SRC column is not present in the existing completed recommendations, add it
                if "RECO_SRC" not in completed_recommendations.columns:
                    completed_recommendations["RECO_SRC"] = reco_src

                # Combine the age-based recommendations with the existing completed recommendations
                completed_recommendations = pd.concat(
                    [completed_recommendations, completed_recommendations_age], axis=0
                )

            # For non-rule based recommendations fill only with best sellers per age group
            if not rule_based:
                # Remove duplicate entries to ensure unique recommendations
                customers_less_than_10 = df.drop_duplicates()

                # Filter out customers who already have the required number of recommendations
                customers_less_than_10 = customers_less_than_10[
                    ~customers_less_than_10["MASTER_ID"].isin(completed_recommendations["MASTER_ID"].unique().tolist())
                ]

                # Log the start of the process for generating recommendations based solely on best sellers
                logger.info("Start recommendations based only on best sellers (by age if available)...")

                # Generate recommendations for the remaining customers based on best-selling products per age group
                # 'fill=False' indicates that this step does not need to fill up any remaining recommendations
                recommendations_age = self._recommend_based_on_best_sellers_per_age(
                    customers_less_than_10, products, fill=False
                ).drop_duplicates()

                # Calculate the complete set of recommendations from these best sellers
                completed_recommendations_age = self.__calculate_completed_recommendations(recommendations_age)

                # Assign a unique source indicator (RECO_SRC) for these non-rule-based recommendations
                completed_recommendations_age["RECO_SRC"] = 3

                # Add the RECO_SRC column if it doesn't exist in the completed recommendations
                if "RECO_SRC" not in completed_recommendations.columns:
                    completed_recommendations["RECO_SRC"] = 3

                # Merge the age-based recommendations with the existing completed recommendations
                completed_recommendations = pd.concat(
                    [completed_recommendations, completed_recommendations_age], axis=0
                )

        return completed_recommendations.reset_index(drop=True)

    def _recommend_based_on_rules(self, df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """
        Generates product recommendations based on rules (previous purchases)

        Args:
            df (pd.DataFrame): DataFrame containing customer data. Requires 'MASTER_ID', 'ITEM_ORDER_DEMAND_DATE', 'COMBINE', 'RANK' columns.
            products (pd.DataFrame): DataFrame containing product information. Requires 'PRODUCT', 'RANK', 'COMBINE', 'AGE_RANK' columns.
        Returns:
            pd.DataFrame: DataFrame containing recommendations.
        """
        # Retrieve the top N products per age group
        top_n_products = self.dataloader.get_top_n_products_per_age_group_and_category(products)
        # Merge with top N products data based on rank and combine criteria
        df = df.merge(top_n_products, on=["RANK", "COMBINE"])
        # Identify new buyers based on their recent order date
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        new_buyers = df[df["ITEM_ORDER_DEMAND_DATE"] == yesterday].reset_index(drop=True)
        # Sample top products for new buyers, limiting to the configured maximum number
        new_buyers = self._sample_from_top_products(new_buyers, sample_n=config.max_products_recommended)

        # Identify old buyers (not part of the new buyers group)
        old_buyers = df[~df["MASTER_ID"].isin(new_buyers["MASTER_ID"].unique().tolist())]
        # Sample top products for old buyers, based on the top products configuration
        old_buyers = self._sample_from_top_products(old_buyers, sample_n=config.sample_from_top_n_products)

        # Combine the new and old buyers into a single DataFrame
        df = pd.concat([new_buyers, old_buyers], axis=0).reset_index(drop=True)
        # Remove the 'ITEM_ORDER_DEMAND_DATE' column as it is no longer needed after separation
        df = df.drop("ITEM_ORDER_DEMAND_DATE", axis=1).reset_index(drop=True).drop_duplicates()

        return df

    def _recommend_based_on_views(
        self, df: pd.DataFrame, products: pd.DataFrame, rule_based: bool = False
    ) -> pd.DataFrame:
        """
        Generates product recommendations based on product views.

        Args:
            df (pd.DataFrame): DataFrame containing customer data. Requires 'MASTER_ID', 'PRODUCT' columns.
            products (pd.DataFrame): DataFrame containing product information. Requires 'PRODUCT', 'RANK', 'COMBINE', 'AGE_RANK' columns.
            rule_based (bool): Flag to determine if the recommendation should be rule-based.

        Returns:
            pd.DataFrame: DataFrame containing recommendations.
        """
        df_need_filling = df[~df["VIEWS_LIST"].isna()]
        # Calculate the remaining number of products needed for each customer if the process is rule-based
        if rule_based:
            df_need_filling = calculate_remaining_products(df_need_filling, "MASTER_ID", "PRODUCT")
        else:
            # Otherwise, set a uniform number equal to max_products_recommended remaining products for each customer
            df_need_filling["REMAINING"] = config.max_products_recommended

        df_short = df_need_filling[["MASTER_ID", "RANK", "REMAINING", "VIEWS_LIST"]].drop_duplicates()
        df_short["VIEWS_LIST"] = df_short["VIEWS_LIST"].apply(list)
        df_short = df_short.explode("VIEWS_LIST").rename(columns={"VIEWS_LIST": "PRODUCT"})

        # Merge the DataFrame with view data and product data based on master ID and rank
        df_short = df_short.merge(products, on=["PRODUCT", "RANK"])

        # Calculate the size (number of views) for each product within each combine category
        df_short["SIZE"] = df_short.groupby(["MASTER_ID", "COMBINE"])["PRODUCT"].transform("size")

        # Narrow down the data to relevant columns and remove duplicates
        df_short = df_short[["MASTER_ID", "COMBINE", "REMAINING", "RANK", "SIZE"]].drop_duplicates()

        # Identify the most viewed categories for each customer
        most_viewed_categories = get_most_n_viewed_categories(df_short)

        # Merge the most viewed categories with product (top products) data to form the recommendations
        top_n_products = self.dataloader.get_top_n_products_per_age_group_and_category(
            products, n=config.max_products_recommended
        )
        recommendations = most_viewed_categories.merge(top_n_products, on=["COMBINE", "RANK"])

        # For rule-based recommendations, filter out products already selected
        if rule_based:
            existing_products = df_need_filling[["MASTER_ID", "PRODUCT"]].drop_duplicates()
            recommendations = recommendations.merge(
                existing_products, on=["MASTER_ID", "PRODUCT"], how="left", indicator=True
            )
            recommendations = recommendations[recommendations["_merge"] == "left_only"].drop(columns="_merge")

        # Pre-calculate weights for each COMBINE category viewed by MASTER_ID based on frequency
        recommendations["WEIGHTS"] = recommendations.groupby(["MASTER_ID", "COMBINE"])["SIZE"].transform(
            "mean"
        ) / recommendations.groupby("MASTER_ID")["SIZE"].transform("sum")

        if not recommendations.empty:
            filled_with_views = sample_products(recommendations, self.__select_top_products_from_views)
        else:
            filled_with_views = recommendations[["MASTER_ID", "COMBINE", "RANK", "SITE_ID", "PRODUCT", "AGE_RANK"]]

        if rule_based:
            # For rule-based recommendations, combine the new recommendations with the existing dataset and remove duplicates
            return pd.concat([df, filled_with_views], axis=0).drop_duplicates().reset_index(drop=True)
        else:
            # For non-rule-based recommendations, use the newly filled recommendations directly
            return filled_with_views

    def _recommend_based_on_best_sellers_per_age(
        self, df_need_filling: pd.DataFrame, products: pd.DataFrame, fill: bool = False
    ) -> pd.DataFrame:
        """
        Generates product recommendations based on best sellers per age group.

        Args:
            df_need_filling (pd.DataFrame): DataFrame containing customer data. Requires 'MASTER_ID', 'PRODUCT' columns.
            products (pd.DataFrame): DataFrame containing product information. Requires 'PRODUCT', 'RANK', 'AGE_RANK' columns.
            fill (bool): Flag to determine if the recommendations should be filled with additional products.

        Returns:
            pd.DataFrame: DataFrame containing recommendations.
        """
        # Calculate remaining products needed if fill is true, otherwise set a uniform number
        # equal to max_products_recommended for all customers
        if fill:
            df_need_filling = calculate_remaining_products(df_need_filling, "MASTER_ID", "PRODUCT")
        else:
            df_need_filling["REMAINING"] = config.max_products_recommended

        # Filter necessary columns and remove duplicates
        df_short = df_need_filling[["MASTER_ID", "RANK", "REMAINING"]].drop_duplicates()

        # Select top products per age group and sample based on configuration
        top_products_per_age = (
            products.sort_values(by="AGE_RANK")
            .groupby("RANK")
            .head(config.sample_from_top_n_products)
            .groupby("RANK")
            .sample(config.max_products_recommended)
            .reset_index(drop=True)
        )

        # Merge customer data with top products based on RANK
        df_short.set_index("RANK", inplace=True)
        top_products_per_age.set_index("RANK", inplace=True)
        recommendations = df_short.merge(top_products_per_age, left_index=True, right_index=True).reset_index()

        # If fill is true, filter out products already in df_need_filling
        if fill:
            existing_products = df_need_filling[["MASTER_ID", "PRODUCT"]].drop_duplicates()
            recommendations = recommendations.merge(
                existing_products, on=["MASTER_ID", "PRODUCT"], how="left", indicator=True
            )
            recommendations = recommendations[recommendations["_merge"] == "left_only"].drop(columns="_merge")

            # Apply a custom sampling function
            filled_with_age = sample_products(recommendations, self.__select_top_products_from_age)
            recommendations_age = pd.concat([df_need_filling.drop("REMAINING", axis=1), filled_with_age], axis=0)
        else:
            # If not filling, just drop the 'REMAINING' column
            recommendations_age = recommendations.drop("REMAINING", axis=1)

        return recommendations_age

    @staticmethod
    def __select_top_products_from_views(group: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weighted sampling to select products for each MASTER_ID based on product views.

        Args:
            group (pd.DataFrame): DataFrame group containing products sorted by views. Requires 'REMAINING', 'WEIGHTS' columns.

        Returns:
            pd.DataFrame: DataFrame containing the selected products
        """
        # Ensure not to sample more items than exist in the group
        total_remaining = min(
            group["REMAINING"].iloc[0], len(group)
        )  # the number of products still needed for each customer
        # Sample indices based on weights
        sampled_indices = np.random.choice(group.index, size=total_remaining, p=group["WEIGHTS"], replace=False)
        return group.loc[sampled_indices]

    @staticmethod
    def __select_top_products_from_age(group: pd.DataFrame) -> pd.DataFrame:
        """
        Selects top products from a group based on age rank.

        Args:
            group (pd.DataFrame): DataFrame group containing products sorted by age rank. Requires 'REMAINING', 'AGE_RANK' columns.

        Returns:
            pd.DataFrame: DataFrame containing selected top products.
        """
        # Retrieve the total number of products that still need to be recommended for each customer
        remaining = group["REMAINING"].iloc[0]

        # Sort the group of products by their age rank in ascending order
        # Age rank is a measure that may reflect the product's suitability or popularity among different age groups
        # Select the top 'remaining' number of products after sorting
        # This way, products that are more relevant or popular for certain age groups are prioritized in the recommendations
        return group.sort_values(by="AGE_RANK").head(remaining)

    @staticmethod
    def __calculate_completed_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the completed product recommendations.

        Args:
            recommendations (pd.DataFrame): DataFrame containing initial recommendations. Requires 'MASTER_ID', 'PRODUCT' columns.

        Returns:
            pd.DataFrame: DataFrame containing completed recommendations with max_products_recommended
        """
        # Remove duplicate entries to ensure each product recommendation is unique for a customer
        recommendations = recommendations.drop_duplicates(subset=["MASTER_ID", "PRODUCT"])

        # Group the recommendations by 'MASTER_ID' and count the number of products recommended for each customer
        group_sizes = recommendations.groupby("MASTER_ID").size()

        # Identify the MASTER_IDs which have reached the desired number of recommendations
        # This is defined by 'config.max_products_recommended'
        # The goal is to filter out customers who have not yet reached the maximum number of product recommendations
        master_ids_with_10 = group_sizes[group_sizes == config.max_products_recommended].index

        # Filter the recommendations to only include those customers who have the complete set of recommendations
        # This ensures that the returned DataFrame only contains customers with exactly 'max_products_recommended' products
        completed_recommendations = recommendations[recommendations["MASTER_ID"].isin(master_ids_with_10)]

        return completed_recommendations

    @staticmethod
    def _sample_from_top_products(df: pd.DataFrame, sample_n: int = config.sample_from_top_n_products) -> pd.DataFrame:
        """
        Samples top products from the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing product data. Requires 'MASTER_ID', 'COMBINE', 'AGE_RANK' columns.
            sample_n (int): Number of top products to sample.

        Returns:
            pd.DataFrame: DataFrame containing sampled top products.
        """
        # Sort the DataFrame by 'MASTER_ID', 'COMBINE', and 'AGE_RANK' to prepare for sampling
        df_sorted = df.sort_values(by=["MASTER_ID", "COMBINE", "AGE_RANK"], ascending=[True, True, True])

        # Select the top 'sample_n' rows for each combination of 'MASTER_ID' and 'COMBINE'
        # This ensures that the most relevant products are considered for each customer
        df_top20 = df_sorted.groupby(["MASTER_ID", "COMBINE"]).head(sample_n).copy()

        # Assign a random value to each row, which will be used for random sampling
        df_top20["random_sample"] = np.random.rand(len(df_top20))

        # Sort the data within each group by the random sample value
        # This randomization step is crucial for unbiased sampling of products
        df_top20_sorted = df_top20.sort_values(by=["MASTER_ID", "random_sample"])

        # From the randomly sorted rows, select the top rows equal to 'max_products_recommended' for each 'MASTER_ID'
        # This selection provides a diversified set of recommendations for each customer
        df_sampled = df_top20_sorted.groupby(["MASTER_ID"]).head(config.max_products_recommended)

        # Convert 'PRODUCT' and 'AGE_RANK' columns to numeric, handling any non-numeric values as missing
        df_sampled["PRODUCT"] = pd.to_numeric(df_sampled["PRODUCT"], errors="coerce")
        df_sampled["AGE_RANK"] = pd.to_numeric(df_sampled["AGE_RANK"], errors="coerce")

        # Drop the 'random_sample' column as it's no longer needed and reset the DataFrame index
        return df_sampled.drop(columns=["random_sample"]).reset_index(drop=True)

    def get_final_recommendations_format(
        self, recommendations: pd.DataFrame, email_data: pd.DataFrame, brand_id: int
    ) -> pd.DataFrame:
        """
        Formats the final recommendations for presentation or export.

        Args:
            recommendations (pd.DataFrame): DataFrame containing recommendations. Requires various columns including 'MASTER_ID', 'SITE_ID', 'PRODUCT', 'RECO_SRC'.
            email_data (pd.DataFrame): DataFrame containing email data. Requires 'MASTER_ID', 'EMAIL_TYPE_LIST' columns.
            brand_id (int): Brand identifier for which recommendations are made.
            promotable (pd.DataFrame): DataFrame containing promotable customer information. Requires 'MASTER_ID', 'EMAIL_ADDRESS' columns.

        Returns:
            pd.DataFrame: Formatted DataFrame containing final recommendations.
        """
        logger.info("Start reformatting recs...")

        # Add sequential numbering for recommendations within each group
        recommendations["RECO_NUM"] = "RECO" + (
            recommendations.groupby(["MASTER_ID", "SITE_ID"]).cumcount() + 1
        ).astype(str)

        # Optimize memory usage by categorizing string columns
        recommendations["RECO_NUM"] = recommendations["RECO_NUM"].astype("category")

        # Pivot the DataFrame with efficient data type
        recommendations = recommendations.pivot(
            index=["MASTER_ID", "SITE_ID", "RECO_SRC"], columns="RECO_NUM", values="PRODUCT"
        ).reset_index()

        # Convert SITE_ID to a category for memory efficiency
        recommendations["SITE_ID"] = recommendations["SITE_ID"].astype("category")

        # Rename columns for clarity
        recommendations.rename(columns={"SITE_ID": "BRAND"}, inplace=True)
        recommendations["EFFORT_CD"] = brand_id

        # This block sorts the recommendation columns ('RECO_NUM') to ensure they are in the correct order
        # It excludes the 'RECO_SRC' column from this sorting process
        # The sorting is based on the numerical part of the 'RECO' column names, ensuring that recommendations are ordered sequentially
        reco_cols = sorted(
            [col for col in recommendations.columns if col.startswith("RECO") and col != "RECO_SRC"],
            key=lambda x: int(x[4:]),
        )

        # Concatenate brand name with product codes for a consistent format
        brand_lower = recommendations["BRAND"].astype(str).str.lower()
        for column in reco_cols:
            product_col = recommendations[column].fillna(-1).astype(int).astype(str)
            recommendations[column] = brand_lower + "_" + product_col

        # Set DataFrame column names and additional attributes
        recommendations.columns.name = None
        recommendations["MODEL_NAME"] = config.algorithm_name
        recommendations["CREATED_AT"] = datetime.now().strftime("%Y-%m-%d")
        recommendations["SEND_TO_EPSILON"] = 0

        # Add email data to the recommendations
        email_data = email_data.explode("EMAIL_TYPE_LIST")
        email_data[["EMAIL_ADDRESS", "TYPE"]] = pd.DataFrame(
            email_data["EMAIL_TYPE_LIST"].apply(lambda x: list(x) if x is not None else [None, None]).tolist(),
            index=email_data.index,
        )
        email_data.drop(columns=["EMAIL_TYPE_LIST"], inplace=True)
        email_data.drop_duplicates(inplace=True)
        email_data["PROMO"] = np.where(email_data["TYPE"].isna(), 0, 1)
        email_data["EMAIL_ADDRESS"] = email_data["EMAIL_ADDRESS"].str.lower()
        email_data["CUSTOMERKEY"] = email_data["EMAIL_ADDRESS"].str.upper()

        # Keep customers with less than 10 emails (avoiding bots)
        email_counts = email_data.groupby("MASTER_ID")["EMAIL_ADDRESS"].transform("count")
        email_data = email_data[email_counts <= 10]  # keep customers with less than 10 emails

        recommendations["MASTER_ID"] = recommendations["MASTER_ID"].astype(int)
        email_data["MASTER_ID"] = email_data["MASTER_ID"].astype(int)

        # Merge the recommendations with the email data
        recommendations = pd.merge(recommendations, email_data, on="MASTER_ID")

        # Fill any missing promotional flags with zero and cast to integer type
        recommendations["PROMO"] = recommendations["PROMO"].fillna(0).astype(int)
        logger.info("Reformatting done!...")

        # Ensure the final DataFrame has the desired column order for presentation or export
        columns_order = (
            ["MASTER_ID", "EFFORT_CD", "CUSTOMERKEY", "EMAIL_ADDRESS", "BRAND"]
            + reco_cols
            + ["RECO_SRC", "CREATED_AT", "TYPE", "MODEL_NAME", "SEND_TO_EPSILON", "PROMO"]
        )

        return recommendations[columns_order]

    def get_general_recommendations(self) -> pd.DataFrame:
        """
        Generates a pd.DataFrame with top recommended products for different age groups.
        This is for backfall scenarios where MASTER_ID has no personalized recs

        This function performs several steps to arrive at the final recommendations:
        1. Retrieves products by brand and their top rankings per age group using dataloader methods.
        2. Samples these products based on a predefined configuration.
        3. Creates a pivot table to structure the recommendations for each site and rank.
        4. Formats the final DataFrame with additional algorithm and timestamp information.

        Returns:
            pd.DataFrame: A DataFrame containing the recommended products for each site,
                        categorized by age groups, along with the algorithm used and the creation date.
        """
        products_by_age = self.dataloader.get_products_per_brand()
        top_products_by_age = self.dataloader.get_top_n_products_per_age_group(products_by_age)
        top_products_by_age = top_products_by_age.groupby("RANK").sample(config.max_products_recommended)
        top_products_by_age = top_products_by_age.rename(columns={"SITE_ID": "SITE"})
        top_products_by_age["RANK"] = top_products_by_age["RANK"].replace("RANKING_", "PRODUCTS_", regex=True)
        top_products_by_age["PRODUCT"] = top_products_by_age["PRODUCT"].astype(str)

        def unique_list(series: pd.Series) -> List[Any]:
            """
            Takes a pandas Series and returns a list containing the unique elements of the series.

            The order of the elements in the resulting list is not guaranteed as `set` does not preserve order.

            :param series: A pandas Series from which unique elements are to be extracted.
            :return: A list containing the unique elements of the input series.
            """
            return list(set(series))

        final_products = top_products_by_age.pivot_table(
            index="SITE", columns="RANK", values="PRODUCT", aggfunc=unique_list
        ).reset_index()

        final_products.columns.name = None

        final_products["ALGORITHM"] = config.algorithm_name
        final_products["CREATED_AT"] = datetime.now().strftime("%Y-%m-%d")

        return final_products[
            [
                "SITE",
                "PRODUCTS_AGE_0_35",
                "PRODUCTS_AGE_36_49",
                "PRODUCTS_AGE_50_60",
                "PRODUCTS_AGE_61_1000",
                "PRODUCTS_AGE_0_1000",
                "ALGORITHM",
                "CREATED_AT",
            ]
        ]
