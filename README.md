# Recommended For You

Personalized product recommendations.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installing Dependencies](#installing-dependencies)
- [Configuration](#configuration)
- [Usage](#usage)

## About

Recommended4you is a Python project for generating personalized product recommendations. It is designed to work with multiple brands and provides both training and inference capabilities.

## Features

- Train recommendation models for different website brands.
- Generate product recommendations for FBB website or email campaigns.
- Flexible configuration via environment variables.
- Integration with Snowflake for data storage.

## Prerequisites

- Python version 3.10 or above, but below 3.12.

## Installing Dependencies
Built using `poetry 1.6.1`

To install poetry:

Python 3.10:

`curl -sSL https://install.python-poetry.org | python3.10 -`

or 

Python 3.11:

`curl -sSL https://install.python-poetry.org | python3.11 -`

For Production (Main Dependencies Only)

- To install dependencies:
`poetry install --only main`

For Development (Including Development Dependencies)

- To install dependencies:
`poetry install`

To activate the virtual environment created by poetry, run:

`poetry shell`

Faster way to run a scipt through poetry environment:

`poetry run python script.py`

## Configuration

Before executing the main script, it is essential to set up the configuration parameters in an `.env` file. This configuration includes details for Snowflake database credentials, Snowflake database table names and other application-specific settings.

The `.env` file should contain the following variables:

**Snowflake Configuration**

* `SF_USERNAME`: Username for Snowflake access.

* `SF_PASSWORD`: Password for Snowflake access.

* `SF_ACCOUNT`: Snowflake account identifier (e.g., `fbb`).

* `SF_REGION`: Region for the Snowflake account (e.g., `us-east-1`).

* `SF_WAREHOUSE`: Snowflake warehouse to be used (e.g., `DS_ANALYTICS`).

* `SF_ROLE`: User role in Snowflake (e.g., `DS_ANALYTICS_DEV`).

* `SF_DATABASE`: Database name in Snowflake (e.g., `DS_PROJECTS`).

* `SF_SCHEMA`: Schema name in Snowflake for the connection.

**Snowflake Table References**

* `REC4YOU_INITIAL_SNOWFLAKE_TABLE`: Name of the destination table in Snowflake where the results will be stored (e.g., `RECOMMENDED_FOR_YOU_PRODUCT_RECS_DEV`).

* `PDM_STYLES_SNOWFLAKE_TABLE`: Reference to product styles table in Snowflake (e.g `PDM_PRODUCT_STYLES_VW`).

* `PRODUCT_RANKING_AGE_SNOWFLAKE_TABLE`: Reference to product ranking by age table in Snowflake (e.g., `PRODUCT_RANKING_AGE_ML_VW`).

* `BRAND_CODE_SNOWFLAKE_TABLE`: Reference to brand code table in Snowflake (e.g., `BRAND_VW`).

* `DEMOGRAPHICS_SNOWFLAKE_TABLE`: Reference to demographics table in Snowflake (e.g., `DEMOGRAPHIC_VW`).

* `ORDER_ITEM_SNOWFLAKE_TABLE`: Reference to order item history table in Snowflake (e.g., `ORDER_ITEM_HIST_MKTG_VW`).

* `EMONSTER_COPY_SNOWFLAKE_TABLE`: Reference to emonster table (that contains customer emails) in Snowflake (e.g., `EMONSTER_COPY`).

* `CDM_PRODUCTS_VIEWED_NOT_ORDERED_SNOWFLAKE_TABLE`: Reference to CDM_PRODUCTS_VIEWED_NOT_ORDERED table (that contains customer products that were viewed but not ordered) in Snowflake (e.g., `CDM_PRODUCTS_VIEWED_NOT_ORDERED_VW`).

**Recommendation Algorithm Configuration**

* `TOP_N_CONSEQUENTS`: The number of consequents to consider in the recommendation (e.g., `3`).

* `SAMPLE_FROM_TOP_N_PRODUCTS`: The top number of products to sample from (e.g., `20`).

* `MAX_PRODUCTS_RECOMMENDED`: The maximum number of products to recommend (e.g., `10`).

* `ALGORITHM_NAME`: The name of the recommendation model (e.g., `rec4you_v1`).

Ensure to replace the example values with your actual Snowflake configuration details and other application-specific settings. Rename `.env_example` to `.env` after updating it with your details.

## Usage

Recommended4you is designed to be flexible and user-friendly. Here's how you can utilize its capabilities:

### Running the Application

To execute the main script, navigate to your project directory and use the command format:

```bash
rec4you --run [mode] --options
```

### Command-Line Arguments

#### `--run`
**Description:** Specifies the operating mode of the application.

**Choices:**
- `train`: For training the recommendation model.
- `inference`: For generating recommendations.

**Type:** String
**Default:** None

#### `--brand_ids`
**Description:** Comma-separated list of brand IDs to process.

**Type:** String
**Default:** 0,5,11,15,23,24,25,26,27,28,74,75,76,78,79

#### `--n_months`
**Description:** Defines the timeframe in months for the training dataset.

**Type:** Integer
**Default:** 48

#### `--website`
**Description:** Indicates whether to generate recommendations for the website (True) or only email recs (False).

**Type:** Boolean (true/false)
**Default:** True

#### `--type_of_recs`
**Description:** Indicates the type of recommendations to run. Available options: 'personal', 'general', 'both'

**Type:** String
**Default:** both

### Usage Examples:

- Training mode:

    ```bash
    rec4you --run train
    ```

    ```bash
    rec4you --run train --n_months 36 --brand_ids 0,5,79
    ```

- Inference mode:

    ```bash
    rec4you --run inference --type_of_recs both
    ```

    ```bash
    rec4you --run inference --type_of_recs personal
    ```
    ```bash
    rec4you --run inference --type_of_recs general
    ```

    ```bash
    rec4you --run inference --brand_ids 11,23
    ```

    ```bash
    rec4you --run inference --brand_ids 24 --website True
    ```

 