"""Training script: tune, evaluate, log, and register the forecasting model."""

import argparse

import mlflow
from pyspark.sql import SparkSession

from restaurant_forecasting.config import ProjectConfig, Tags
from restaurant_forecasting.basic_model import BasicModel

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="dev")
parser.add_argument("--root_path", default="..")
parser.add_argument("--git_sha", default="abcd12345")
parser.add_argument("--branch", default="main")
args = parser.parse_args()

config = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/project_config.yml", env=args.env
)
spark = SparkSession.builder.getOrCreate()
tags = Tags(git_sha=args.git_sha, branch=args.branch)

model = BasicModel(config=config, tags=tags, spark=spark)

model.load_data()
model.prepare_features()
model.tune(n_iter=10, n_splits=4)
model.evaluate()
model.log_model()
model.register_model()

print("Training complete and model registered.")