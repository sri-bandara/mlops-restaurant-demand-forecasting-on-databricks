# %%
"""Deploy the registered forecasting model as a serving endpoint."""

import os

import mlflow
from dotenv import load_dotenv

from restaurant_forecasting.config import ProjectConfig
from restaurant_forecasting.model_serving import ModelServing

# %%
# Setup
load_dotenv()
profile = os.environ["PROFILE"]

mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml("../project_config.yml", env="dev")

model_name = f"{config.catalog_name}.{config.schema_name}.restaurant_demand_model"

# %%
# Create the serving manager
model_serving = ModelServing(
    model_name=model_name,
    endpoint_name="restaurant-demand-serving",
)

# %%
# Deploy (or update) the endpoint
model_serving.deploy_or_update_serving_endpoint()
# %%
