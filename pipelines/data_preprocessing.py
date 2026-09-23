# %%
"""Preprocessing pipeline which turns raw CSV to train/test Delta tables in Unity Catalog."""

import os

import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from restaurant_forecasting.config import ProjectConfig
from restaurant_forecasting.data_processor import DataProcessor

# %%
# Load environment + config
load_dotenv()
profile = os.environ["PROFILE"]

#creating a serverless spark session using databricks connect
spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()

config = ProjectConfig.from_yaml("../project_config.yml", env="dev")

# %%
# Read raw data
df = pd.read_csv("../data/data.csv")
print(f"Loaded {len(df)} rows")

# %%
# Preprocess + feature engineering
processor = DataProcessor(df, config, spark)
processor.preprocess()
print(f"After preprocessing: {len(processor.df)} rows")

# %%
# Time-based split
train, test = processor.split_data()
print(f"Train: {train.shape}, Test: {test.shape}")

# %%
# Write to Unity Catalog as Delta tables
processor.save_to_catalog(train, test)
print(f"Tables written to {config.catalog_name}.{config.schema_name}")
