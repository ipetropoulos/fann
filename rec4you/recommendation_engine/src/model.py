import os
import pickle

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from rec4you.recommendation_engine.src.dataloader import Dataloader
from rec4you.recommendation_engine.src.utils import logger


class Rules:
    """
    A class for performing market basket analysis on transaction data.

    Attributes:
        dataloader (Dataloader): An instance of the Dataloader class to load and transform data.
        min_support (float): The minimum support value for the market basket analysis.
        min_threshold (float): The minimum threshold value for the market basket analysis.

    Methods:
        _create_sparse_df(df): Creates a sparse DataFrame required for market basket analysis.
        _perform_market_basket_analysis(df): Performs market basket analysis on the given DataFrame.
        create_rules(n_months): Creates and saves association rules based on transactions of the last n months.
    """

    def __init__(self, dataloader: Dataloader, min_support: float = 0.0001, min_threshold: float = 1) -> None:
        """
        Constructs all the necessary attributes for the Rules object.

        Args:
            dataloader (Dataloader): An instance of the Dataloader class.
            min_support (float): Minimum support for the frequent itemsets.
            min_threshold (float): Minimum threshold for the association rules.
        """
        self.dataloader = dataloader
        self.min_support = min_support
        self.min_threshold = min_threshold

    @staticmethod
    def _create_sparse_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a DataFrame into a sparse DataFrame required for efficient market basket analysis.

        Args:
            df (pd.DataFrame): A DataFrame containing transaction data.

        Returns:
            pd.DataFrame: A sparse DataFrame for market basket analysis.
        """
        transaction_data = df["PRODUCT_LIST"].tolist()

        te = TransactionEncoder()
        oht_ary = te.fit(transaction_data).transform(transaction_data, sparse=True)
        sparse_df = pd.DataFrame.sparse.from_spmatrix(oht_ary, columns=te.columns_)
        return sparse_df

    def _perform_market_basket_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs market basket analysis on the given DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with transaction data.

        Returns:
            pd.DataFrame: A DataFrame containing the association rules.
        """
        sparse_df = self._create_sparse_df(df)
        logger.info("Sparse matrix has been created!")

        frequent_items = (
            fpgrowth(sparse_df, min_support=self.min_support, use_colnames=True)
            .sort_values("support", ascending=False)
            .reset_index(drop=True)
        )

        rules = association_rules(frequent_items, metric="lift", min_threshold=self.min_threshold).reset_index(
            drop=True
        )

        return rules

    def create_rules(self, n_months: int = 48) -> None:
        """
        Loads transactions for the last n months, performs market basket analysis,
        and saves the generated rules to a file.

        Args:
            n_months (int): The number of months to look back for transactions.

        Raises:
            IOError: If there is an error in writing the pickle file.
            pickle.PickleError: If there is an error in pickling the data.
        """
        logger.info("Loading transactions of the last 4 years.This will take a couple of minutes....")
        order_last4_years = self.dataloader.get_orders_last_n_months(n_months)
        logger.info("Dataset has been succesfully loaded!")

        logger.info("Performing market basket analysis...")
        rules = self._perform_market_basket_analysis(df=order_last4_years)
        try:
            os.makedirs("rec4you/RULES/", exist_ok=True)
            with open(f"rec4you/RULES/rules_{self.dataloader.brand_id}.pickle", "wb") as file:
                pickle.dump(rules, file)
            logger.info(f"Successfully saved rules for brand {self.dataloader.brand_id} in pickle file.")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error saving rules for brand {self.dataloader.brand_id}: {e}")

    def load_rules(self) -> pd.DataFrame:
        """
        Loads the saved association rules from a pickle file for the current brand.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded association rules, or an empty DataFrame if brand ID is 15 or no rules are saved.
        """
        if self.dataloader.brand_id != 15:
            with open(f"rec4you/RULES/rules_{self.dataloader.brand_id}.pickle", "rb") as file:
                rules = pickle.load(file)

            rules = rules[(rules["lift"] > 1) & (rules["confidence"] >= 0.05)]
            rules = rules[["antecedents", "consequents", "confidence", "lift"]]
            rules["antecedents"] = [tuple(sorted(x)) for x in rules["antecedents"]]
            rules["consequents"] = [tuple(sorted(x)) for x in rules["consequents"]]
            return rules
        return pd.DataFrame()
