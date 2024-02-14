# Last updated February 6, 2024
# Version 0.1.0

import pandas as pd
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from sklearn.pipeline import Pipeline


class SqueezeDF(BaseEstimator, TransformerMixin):
    """Transformer class to squeeze single-column dataframe to series."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.Series:
        x_new = X.squeeze()
        return x_new

    def get_feature_names_out(self, input_features=None):
        pass


class TextCleaner(BaseEstimator, TransformerMixin):
    """Text cleaning transformer class."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.Series:
        x_new = X.squeeze()
        x_new = x_new.str.replace("_", " ").str.lower().str.strip(".!? \n\t")
        x_new = x_new.fillna("")
        return x_new

    def get_feature_names_out(self, input_features=None):
        pass


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Feature dropper transformer class."""

    def __init__(self, drop_features: List[str] = None):
        self.drop_features = drop_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_dropped = X.drop(columns=self.drop_features)
        self._feature_names = X_dropped.columns
        return X_dropped

    def get_feature_names_out(self, input_features=None):
        return self._feature_names


def X_y_spilt(df: pd.DataFrame, target: str = "target") -> Tuple:
    """Splitting Pandas dataframe to features and target."""
    y = df[target]
    X = df.drop(columns=target)
    return X, y


def stratified_sample(
    df: pd.DataFrame, frac: float, col: str = "target"
) -> pd.DataFrame:
    """Subsampling from Pandas dataframe while preserving stratification."""
    df_sub = df.groupby(col, observed=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    )
    df_sub.index = df_sub.index.droplevel(0)
    return df_sub


def all_combinations(items: List, constants: List = None) -> List[List]:
    """Function to get all combinations of items plus optional constant items."""
    subset_list = list()
    for L in range(len(items) + 1):
        for subset in combinations(items, L):
            if constants:
                subset_list.append(list(subset) + constants)
            else:
                subset_list.append(list(subset))
    return subset_list


def prepare_data_evaluation(
    model: Pipeline,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    drop_cols: List[str] = None,
) -> tuple:
    """Function to prepare validation and testing data for a pretrained model
    evaluation. Returns tuple of true validation target, predicted validation
    target, true test target and predicted test target"""
    X_val, target_val = df_val.drop(columns=drop_cols).pipe(
        X_y_spilt, target=target
    )
    target_val_pred = model.predict(X_val)
    X_test, target_test = df_test.drop(columns=drop_cols).pipe(
        X_y_spilt, target=target
    )
    target_test_pred = model.predict(X_test)
    return target_val, target_val_pred, target_test, target_test_pred
