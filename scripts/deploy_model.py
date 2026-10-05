"""Deploy script: create or update the model serving endpoint."""

import argparse

from restaurant_forecasting.config import ProjectConfig
from restaurant_forecasting.model_serving import ModelServing

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="dev")
parser.add_argument("--root_path", default="..")
args = parser.parse_args()

config = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/project_config.yml", env=args.env
)

model_name = f"{config.catalog_name}.{config.schema_name}.restaurant_demand_model"

model_serving = ModelServing(
    model_name=model_name,
    endpoint_name="restaurant-demand-serving",
)

model_serving.deploy_or_update_serving_endpoint()

print("Serving endpoint deployed/updated.")