import os
import sys
import optuna
from typing import Any, Dict

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logger.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.model_config import MODEL_CONFIGS
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object, load_object, write_yaml_file, load_numpy_array_data
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.model_selection import cross_val_score


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def perform_hyperparameter_tuning(self, x_train, y_train, model_class, param_grid: Dict[str, Any], n_trials: int = 10):
        """
        Runs Optuna tuning based on the parameter grid of the model.
        Only includes hyperparameters available in model_config.
        """
        def objective(trial):
            params = {}
            for param_name, param_values in param_grid.items():
                first_value = param_values[0]
                if isinstance(first_value, float):
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                elif isinstance(first_value, int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                elif isinstance(first_value, str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

            model = model_class(**params)
            score = cross_val_score(model, x_train, y_train, cv=3, scoring="f1").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value


    def train_model(self, x_train, y_train, model_class, best_params):
        """Train model with best parameters."""
        model = model_class(**best_params)
        model.fit(x_train, y_train)
        return model


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed data arrays...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            best_overall_model = None
            best_overall_score = -1
            best_overall_params = {}
            best_overall_model_name = ""

            for config in MODEL_CONFIGS:
                logging.info(f"🔍 Tuning hyperparameters for model: {config.name}")

                try:
                    best_params, best_score = self.perform_hyperparameter_tuning(
                        x_train, y_train, config.model.__class__, config.param_grid
                    )

                    logging.info(f"✅ {config.name} best score: {best_score:.4f} with params: {best_params}")

                    if best_score > best_overall_score:
                        best_overall_score = best_score
                        best_overall_model = config.model.__class__
                        best_overall_params = best_params
                        best_overall_model_name = config.name

                except Exception as e:
                    logging.warning(f"⚠️ Skipping {config.name} due to error: {e}")

            if best_overall_model is None:
                raise Exception("No valid model found during tuning!")

            logging.info(f"🏆 Best Model: {best_overall_model_name} | Score: {best_overall_score:.4f}")

            model = self.train_model(x_train, y_train, best_overall_model, best_overall_params)

            # evaluate performance
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # basic validation
            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                logging.warning("Trained model did not meet expected accuracy threshold.")

            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning("Potential overfitting/underfitting detected.")

            # combine preprocessor + model
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)

            # save best params to YAML
            write_yaml_file(self.model_trainer_config.best_hyperparameters_file_path, {
                "best_model": best_overall_model_name,
                "best_score": best_overall_score,
                "best_params": best_overall_params,
            })

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
                best_model_name=best_overall_model_name,
                best_hyperparameters=best_overall_params,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
