"""Quantile LightGBM models for restaurant demand forecasting (prediction intervals)."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss

from restaurant_forecasting.config import ProjectConfig, Tags


class QuantileModel:
    """Trains P10/P50/P90 quantile models and evaluates calibration."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark) -> None:
        self.config = config
        self.tags = tags
        self.spark = spark

        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        self.parameters = config.quantile_parameters
        self.alphas = config.quantile_alphas
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name

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
        """Align categorical dtypes for train and test."""
        for col in self.cat_features:
            self.X_train[col] = self.X_train[col].astype("category")
            self.X_test[col] = pd.Categorical(
                self.X_test[col], categories=self.X_train[col].cat.categories
            )

    def train(self) -> None:
        """Train one LightGBM model per quantile (P10, P50, P90)."""
        self.models = {}
        for alpha in self.alphas:
            model = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                verbose=-1,
                **self.parameters,
            )
            model.fit(self.X_train, self.y_train, categorical_feature=self.cat_features)
            self.models[alpha] = model
            print(f"Trained quantile model for alpha={alpha}")

    def evaluate(self) -> dict:
        """Evaluate pinball loss per quantile and coverage of the interval."""
        actual_log = self.y_test
        actual = np.expm1(actual_log)

        self.predictions = {}
        pinball_losses = {}

        # Predict each quantile, compute pinball loss (on log scale, as trained)
        for alpha in self.alphas:
            preds_log = self.models[alpha].predict(self.X_test)
            self.predictions[alpha] = np.expm1(preds_log)  #reverting to original scale
            pinball_losses[alpha] = mean_pinball_loss(actual_log, preds_log, alpha=alpha)

        # Coverage: fraction of actuals between the lowest and highest quantile
        low_alpha = min(self.alphas) #p10
        high_alpha = max(self.alphas) #p90
        low_preds = self.predictions[low_alpha]
        high_preds = self.predictions[high_alpha]

        within = (actual >= low_preds) & (actual <= high_preds)
        coverage = within.mean()
        expected_coverage = high_alpha - low_alpha 

        self.metrics = {
            **{f"pinball_loss_{a}": pinball_losses[a] for a in self.alphas},
            "coverage": coverage,
            "expected_coverage": expected_coverage,
        }

        print("\n--- Quantile Model Evaluation ---")
        for a in self.alphas:
            print(f"Pinball loss (alpha={a}): {pinball_losses[a]:.4f}")
        print(f"\nInterval [{low_alpha}, {high_alpha}] coverage: {coverage:.1%} "
              f"(expected ~{expected_coverage:.0%})")

        return self.metrics