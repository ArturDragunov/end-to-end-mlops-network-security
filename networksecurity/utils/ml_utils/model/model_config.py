# networksecurity/utils/ml_utils/model/model_config.py

"""
Model configurations for classification tasks.
Each configuration contains:
- Model name
- Instantiated model object
- Parameter search space for hyperparameter tuning
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class ModelConfig:
    """Stores a model definition and its parameter search space."""
    name: str
    model: Any
    param_grid: Dict[str, List[Any]]


MODEL_CONFIGS: List[ModelConfig] = [

    # ModelConfig(
    #     name="Decision Tree",
    #     model=DecisionTreeClassifier(),
    #     param_grid={
    #         "criterion": ["gini", "entropy", "log_loss"],
    #         "max_depth": [3, 5, 7, None],
    #         "min_samples_split": [2, 5, 10],
    #     },
    # ),

    # ModelConfig(
    #     name="Random Forest",
    #     model=RandomForestClassifier(),
    #     param_grid={
    #         "n_estimators": [50, 100, 200],
    #         "max_depth": [5, 10, 15, None],
    #         "min_samples_split": [2, 5, 10],
    #         "bootstrap": [True, False],
    #     },
    # ),

    # ModelConfig(
    #     name="Gradient Boosting",
    #     model=GradientBoostingClassifier(),
    #     param_grid={
    #         "n_estimators": [50, 100, 200],
    #         "learning_rate": [0.01, 0.05, 0.1],
    #         "subsample": [0.6, 0.8, 1.0],
    #         "max_depth": [3, 5, 7],
    #     },
    # ),

    # ModelConfig(
    #     name="Logistic Regression",
    #     model=LogisticRegression(max_iter=1000),
    #     param_grid={
    #         "C": [0.1, 1.0, 10.0],
    #         "solver": ["lbfgs", "liblinear"],
    #     },
    # ),

    ModelConfig(
        name="K-Nearest Neighbors",
        model=KNeighborsClassifier(),
        param_grid={
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    ),

    ModelConfig(
        name="XGBoost",
        model=XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
    ),

    # ModelConfig(
    #     name="CatBoost",
    #     model=CatBoostClassifier(verbose=False),
    #     param_grid={
    #         "depth": [4, 6, 8, 10],
    #         "learning_rate": [0.01, 0.05, 0.1],
    #         "iterations": [50, 100, 200],
    #     },
    # ),

    # ModelConfig(
    #     name="AdaBoost",
    #     model=AdaBoostClassifier(),
    #     param_grid={
    #         "n_estimators": [50, 100, 200],
    #         "learning_rate": [0.01, 0.05, 0.1, 1.0],
    #     },
    # ),
]
