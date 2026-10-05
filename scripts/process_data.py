"""Preprocessing script: raw CSV → train/test Delta tables in Unity Catalog."""

import argparse

import pandas as pd
from pyspark.sql import SparkSession

from restaurant_forecasting.config import ProjectConfig
from restaurant_forecasting.data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="dev")
parser.add_argument("--root_path", default="..")
args = parser.parse_args()

config = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/project_config.yml", env=args.env
)
spark = SparkSession.builder.getOrCreate()

df = pd.read_csv(f"{args.root_path}/data/data.csv")
print(f"Loaded {len(df)} rows")

processor = DataProcessor(df, config, spark)
processor.preprocess()
train, test = processor.split_data()
print(f"Train: {train.shape}, Test: {test.shape}")

processor.save_to_catalog(train, test)
print(f"Tables written to {config.catalog_name}.{config.schema_name}")