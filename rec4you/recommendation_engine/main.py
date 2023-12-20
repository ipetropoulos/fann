from datetime import datetime

from rec4you.recommendation_engine.src.argparser import parser
from rec4you.recommendation_engine.src.dataloader import Dataloader
from rec4you.recommendation_engine.src.model import Rules
from rec4you.recommendation_engine.src.recommender import Recommender
from rec4you.recommendation_engine.src.utils import (
    BRAND_ID_TO_NAME_MAPPING,
    check_if_active_9m_is_empty,
    check_if_oih_is_empty,
    config,
    logger,
    rows_exist,
    write_results,
)


def main():
    """
    Main function to execute training or inference based on provided arguments.

    Args:
        args: Command-line arguments passed to the script.

    Returns:
        None
    """
    args = parser.parse_args()

    # Parse brand IDs from command-line arguments
    brand_ids = [int(item) for item in args.brand_ids.split(",")]

    # Run training if the 'train' argument is specified
    if args.run.lower() == "train":
        dataloader = Dataloader(run=args.run)
        for brand_id in brand_ids:
            if check_if_oih_is_empty():
                raise Exception("ORDER_ITEM_HIST table is empty!")

            if brand_id == 15:
                continue  # Skip brand ID 15
            dataloader.set_brand_id(brand_id=brand_id)
            logger.info(f"Creating model for brand {brand_id}...")
            model = Rules(dataloader)
            logger.info("Begin training...")
            model.create_rules(args.n_months)
            logger.info(f"Training for brand {brand_id} is completed!")

    # Run inference if the 'inference' argument is specified
    elif args.run.lower() == "inference":
        dataloader = Dataloader(run=args.run)
        for brand_id in brand_ids:
            logger.info(f"Preparing to generate recommendations for brand {BRAND_ID_TO_NAME_MAPPING[brand_id]}...")
            # Set the brand ID in the dataloader and initialize the Rules and Recommender
            dataloader.set_brand_id(brand_id=brand_id)
            rules = Rules(dataloader)
            recommender = Recommender(dataloader, rules)

            if check_if_oih_is_empty():
                raise Exception("ORDER_ITEM_HIST table is empty!")

            if args.type_of_recs in ["personal", "both"]:
                # Skip brand IDs 27 and 28, check if ACTIVE_FILE9M table is populated for other brands
                if brand_id not in [27, 28]:
                    logger.info(
                        f"Check if EPS_{BRAND_ID_TO_NAME_MAPPING[brand_id]}_ACTIVE_FILE9M table is populated..."
                    )
                    if check_if_active_9m_is_empty(brand_id):
                        raise Exception(f"EPS_{BRAND_ID_TO_NAME_MAPPING[brand_id]}_ACTIVE_FILE9M table is empty!")

                # Generate recommendations and write them to snowflake (in batches)
                recommender.inference(website=args.website)

            if args.type_of_recs in ["general", "both"]:
                # Check if general recommendations for the specific brand already exist in Snowflake for the current date
                if rows_exist(table=config.general_recs_snowflake_table, brand_code=BRAND_ID_TO_NAME_MAPPING[brand_id]):
                    logger.warning(
                        f'General recommendations for brand {brand_id} and date {datetime.now().strftime("%Y-%m-%d")} already exist in snowflake'
                    )
                else:
                    logger.info(
                        f"Preparing to generate general recommendations for brand {BRAND_ID_TO_NAME_MAPPING[brand_id]}..."
                    )
                    # Run script for backfall scenario (new customers on website)
                    general_recs = recommender.get_general_recommendations()

                    # Load the general recommendations to Snowflake
                    logger.info(f"Load the general recommendations for brand {brand_id} to Snowflake!")
                    write_results(
                        general_recs,
                        table=config.general_recs_snowflake_table,
                        brand_code=BRAND_ID_TO_NAME_MAPPING[brand_id],
                    )
                    logger.info(f"The general recommendations for brand {brand_id} are sent to Snowflake!")


if __name__ == "__main__":
    main()
