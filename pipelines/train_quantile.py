# %%
"""Quantile training pipeline: train P10/P50/P90 models, evaluate calibration, and log."""

import os

import mlflow
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from restaurant_forecasting.config import ProjectConfig, Tags
from restaurant_forecasting.quantile_model import QuantileModel

# %%
# Setup
load_dotenv()
profile = os.environ["PROFILE"]

spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()

mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml("../project_config.yml", env="dev")
tags = Tags(git_sha="abcd12345", branch="main")

# %%
# Create quantile model
model = QuantileModel(config=config, tags=tags, spark=spark)

# %%
# Load data
model.load_data()

# %%
# Prepare features
model.prepare_features()

# %%
# Train P10/P50/P90
model.train()

# %%
# Evaluate (pinball loss + coverage)
model.evaluate()

# %%
# Log models and metrics to MLflow
model.log_model()
# %%
