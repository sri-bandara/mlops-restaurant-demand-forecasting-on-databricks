"""Basic LightGBM model for restaurant demand forecasting."""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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

        self.X_train = self.train_set[features].copy()
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[features].copy()
        self.y_test = self.test_set[self.target]

    def prepare_features(self) -> None:
        """Set up the LightGBM model with categorical features."""

        # Ensure categoricals are the category dtype for LightGBM
        for col in self.cat_features:
            self.X_train[col] = self.X_train[col].astype("category")
            self.X_test[col] = pd.Categorical(self.X_test[col],categories=self.X_train[col].cat.categories)

    def tune(self, n_iter: int = 20, n_splits: int = 4) -> None:
        """Tune hyperparameters with time-series CV on the TRAIN set."""

        # Hyperparameter search space
        param_distributions = {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 500, 800, 1000],
            "max_depth": [4, 6, 8, 10],
            "num_leaves": [15, 31, 63],
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)

        search = RandomizedSearchCV(
            estimator=LGBMRegressor(verbose=-1),
            param_distributions=param_distributions,
            n_iter=n_iter,                     
            cv=tscv,                           
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )

        # Fit the search (trains across folds, picks best, retrains best on full train)
        search.fit(self.X_train, self.y_train)

        self.model = search.best_estimator_        # the final tuned model
        self.best_params = search.best_params_
        self.cv_best_score = -search.best_score_    

        print(f"Best parameters: {self.best_params}")
        print(f"Best TSCV RMSE (log scale): {self.cv_best_score:.4f}")

    def evaluate(self) -> dict:
        """Evaluate the model and compare against the TSCV score and the last-day baseline score."""

        #log-scale metrics for comparison with tscv
        preds_log = self.model.predict(self.X_test)
        actual_log = self.y_test
        rmse_log = np.sqrt(mean_squared_error(actual_log, preds_log))

        #real scale metrics
        preds = np.expm1(preds_log)
        actual = np.expm1(actual_log)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)

        #baseline scores (same as last open day)
        baseline_preds = np.expm1(self.X_test["visitors_last_day"])
        baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_preds))
        baseline_mae = mean_absolute_error(actual, baseline_preds)

        self.metrics = {
            "rmse_log": rmse_log,
            "rmse": rmse,
            "mae": mae,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }

        print(f"Model    RMSE (log scale): {rmse_log:.4f}")
        print(f"Model    RMSE: {rmse:.2f}  MAE: {mae:.2f}")
        print(f"Baseline RMSE: {baseline_rmse:.2f}  MAE: {baseline_mae:.2f}")
        return self.metrics

    def log_model(self) -> None:
        """Log the model, metrics, and metadata to MLflow."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags.dict()) as run:
            # Save the run ID for later reference
            self.run_id = run.info.run_id

            # Log hyperparameters and metrics
            mlflow.log_params(self.best_params)
            mlflow.log_metrics(self.metrics)

            # Model signature (inputs/outputs)
            signature = infer_signature(
                self.X_train, self.model.predict(self.X_train)
            )

            # Log the model
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                name="lightgbm-model",
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