# %%
"""Training pipeline for basic model: tune, evaluate, log, and register the forecasting model."""

import os

import mlflow
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from restaurant_forecasting.config import ProjectConfig, Tags
from restaurant_forecasting.basic_model import BasicModel

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
# Create model
model = BasicModel(config=config, tags=tags, spark=spark)

# %%
# Load data from catalog
model.load_data()

# %%
# Prepare features (align categoricals)
model.prepare_features()

# %%
# Tune hyperparameters (time-series CV on train) — produces the final model
model.tune(n_iter=20, n_splits=4)

# %%
# Evaluate on held-out test set + baseline
model.evaluate()

# %%
# Log to MLflow
model.log_model()

# %%
# Register in Unity Catalog
model.register_model()
# %%
