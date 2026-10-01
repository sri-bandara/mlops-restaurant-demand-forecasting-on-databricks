"""Basic LightGBM model for restaurant demand forecasting."""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature #builds the input/output signature for the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from restaurant_forecasting.config import ProjectConfig, Tags


class BasicModel:
    """Trains, evaluates, logs, and registers a LightGBM forecasting model."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark) -> None:
        self.config = config
        self.tags = tags
        self.spark = spark

        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.restaurant_demand_model"

    def load_data(self) -> None:
        """Load train and test sets from Unity Catalog."""
        self.train_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        ).toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()

        #excludes visitors, store_id and date
        features = self.num_features + self.cat_features

        self.X_train = self.train_set[features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[features]
        self.y_test = self.test_set[self.target]

    def prepare_features(self) -> None:
        """Set up the LightGBM model with categorical features."""
        # Ensure categoricals are the category dtype for LightGBM
        for col in self.cat_features:
            self.X_train[col] = self.X_train[col].astype("category")
            self.X_test[col] = self.X_test[col].astype("category")

        self.model = LGBMRegressor(**self.parameters)

    def train(self) -> None:
        """Fit the model on the training set."""
        self.model.fit(
            self.X_train,
            self.y_train,
            categorical_feature=self.cat_features,
        )

    def evaluate(self) -> dict:
        """Evaluate the model and compare against the last-day baseline.

        Metrics are computed on the ORIGINAL visitor scale (undoing log1p).
        """
        # Model predictions (in log space) -> back to real visitor counts
        preds_log = self.model.predict(self.X_test)
        preds = np.expm1(preds_log)
        actual = np.expm1(self.y_test)

        model_rmse = np.sqrt(mean_squared_error(actual, preds))
        model_mae = mean_absolute_error(actual, preds)

        # Baseline: predict "same as last open day" (visitors_last_day)
        baseline_preds = np.expm1(self.X_test["visitors_last_day"])
        baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_preds))
        baseline_mae = mean_absolute_error(actual, baseline_preds)

        self.metrics = {
            "rmse": model_rmse,
            "mae": model_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }

        print(f"Model    RMSE: {model_rmse:.2f}  MAE: {model_mae:.2f}")
        print(f"Baseline RMSE: {baseline_rmse:.2f}  MAE: {baseline_mae:.2f}")
        return self.metrics

    def log_model(self) -> None:
        """Log the model, metrics, and metadata to MLflow."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags.dict()) as run:
            # Save the run ID for later reference
            self.run_id = run.info.run_id

            # Log hyperparameters and metrics
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(self.metrics)

            # Model signature (inputs/outputs)
            signature = infer_signature(
                self.X_train, self.model.predict(self.X_train)
            )

            # Log the model
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="lightgbm-model",
                signature=signature,
                input_example=self.X_train.iloc[:5],
            )

    def register_model(self) -> None:
        """Register the logged model in Unity Catalog with an alias."""
        registered = mlflow.register_model(
            model_uri=self.model_info.model_uri,
            name=self.model_name,
            tags=self.tags.dict(),
        )

        # Set an alias so we can always fetch the latest
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=registered.version,
        )
        print(f"Registered {self.model_name} version {registered.version}")