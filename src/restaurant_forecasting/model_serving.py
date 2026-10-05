"""Model serving module for restaurant demand forecasting."""

import mlflow
from databricks.sdk import WorkspaceClient #to control databricks workspace programmatically
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)


class ModelServing:
    """Manages the model serving endpoint in Databricks."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Get the latest registered model version via its alias."""
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(
            self.model_name, alias="latest-model"
        ).version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Create the endpoint if it doesn't exist, otherwise update it."""
        endpoint_exists = any(
            item.name == self.endpoint_name
            for item in self.workspace.serving_endpoints.list()
        )
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    name=self.endpoint_name,
                    served_entities=served_entities,),
            )
            print(f"Created endpoint: {self.endpoint_name}")
        else:
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name, 
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
            print(f"Updated endpoint: {self.endpoint_name}")