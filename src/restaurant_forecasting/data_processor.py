""" Data preprocessing module for restaurant forecasting. """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from restaurant_forecasting.config import ProjectConfig


class DataProcessor:
    """Handles preprocessing, feature engineering, and splitting."""

    def __init__(self, df: pd.DataFrame, config: ProjectConfig, spark) -> None:
        self.df = df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Clean data, engineer features, and log-transform the target."""
        df = self.df

        # 1. Rename columns
        df = df.rename(
            columns={
                "air_store_id": "store_id",
                "visit_date": "date",
                "day_of_week": "dow",
                "air_genre_name": "store_genre",
                "air_area_name": "area",
            }
        )

        # 2. Date to datetime
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        # 3. Time features
        df["month"] = df["date"].dt.month
        df["weekend_flg"] = df["dow"].isin(["Saturday", "Sunday"]).astype(int)

        # 4. Sort for feature building
        df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

        # 5. Lag-1 (date-based)
        temp = df[["store_id", "date", "visitors"]].copy()
        temp["date"] = temp["date"] + pd.Timedelta(days=1)
        temp = temp.rename(columns={"visitors": "visitors_lag_1"})
        df = pd.merge(df, temp, on=["store_id", "date"], how="left")

        # 6. Rolling 7-day mean (date-based, excludes today)
        roll = (
            df.set_index("date")
            .groupby("store_id")["visitors"]
            .rolling("7D", closed="left")
            .mean()
            .reset_index()
            .rename(columns={"visitors": "visitors_mean_roll_7"})
        )
        df = pd.merge(df, roll, on=["store_id", "date"], how="left")

        # 7. Drop start-of-series nulls
        df = df.dropna(subset=["visitors_lag_1", "visitors_mean_roll_7"])

        # 8. Log-transform the target
        df["visitors"] = np.log1p(df["visitors"])

        # 9. Ensure correct dtypes for categorical features
        for col in self.config.cat_features:
            df[col] = df[col].astype("category")

        #saving the processed dataframe back to the instance variable
        self.df = df 

    #time based split to avoid data leakage
    def split_data(self, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based split: earlier dates go to train table, later dates go to test table."""
        df = self.df.sort_values("date")
        cutoff_idx = int(len(df) * (1 - test_ratio))
        cutoff_date = df.iloc[cutoff_idx]["date"]

        train = df[df["date"] < cutoff_date]
        test = df[df["date"] >= cutoff_date]
        return train, test

    def save_to_catalog(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Write train/test tables to Delta tables in Unity Catalog."""
        catalog = self.config.catalog_name
        schema = self.config.schema_name

        train_spark = self.spark.createDataFrame(train)
        test_spark = self.spark.createDataFrame(test)

        train_spark.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.train_set")
        test_spark.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.test_set")