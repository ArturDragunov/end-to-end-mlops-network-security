from networksecurity.constant.training_pipeline.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logger.logger import logging 
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file, read_csv_data
from scipy.stats import ks_2samp # Performs the two-sample Kolmogorov-Smirnov test for goodness of fit
import pandas as pd
import os,sys
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    
    def validate_column_existence(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that all expected columns exist in the DataFrame,
        and there are no unexpected extra columns.
        """
        try:
            validation_status = True
            expected_schema = self._schema_config.columns

            missing_cols = [col for col in expected_schema.keys() if col not in dataframe.columns]
            extra_cols = [col for col in dataframe.columns if col not in expected_schema.keys()]

            if missing_cols:
                logging.error(f"Missing columns: {missing_cols}")
                validation_status = False

            if extra_cols:
                logging.warning(f"Unexpected columns found: {extra_cols}")

            return validation_status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_column_types(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that each column has the expected data type.
        """
        try:
            validation_status = True
            expected_schema = self._schema_config.columns

            for col, expected_dtype in expected_schema.items():
                if col in dataframe.columns:
                    actual_dtype = str(dataframe[col].dtype)
                    if actual_dtype != expected_dtype:
                        logging.error(
                            f"Column '{col}' has wrong dtype. "
                            f"Expected {expected_dtype}, got {actual_dtype}"
                        )
                        validation_status = False

            return validation_status

        except Exception as e:
            raise NetworkSecurityException(e, sys)
            
    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config.numerical_columns
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present=False
                    missing_numerical_columns.append(num_column)
            
            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_column_present
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        """H0: d1 == d2; H1: d1 != d2"""
        try:
            status=True
            report ={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2  = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False # we couldn't reject H0. No data drift found
                else:
                    is_found = True 
                    status=False
                # we save columns with data drift to report
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            
            data_drift_report_file_path = self.data_validation_config.data_drift_report_file_path
            
            #Create directory
            dir_path = os.path.dirname(data_drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=data_drift_report_file_path,content=report)
            return status
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def detect_concept_drift(self, base_df, current_df, threshold=0.10) -> bool:
        """
        Detects concept drift by comparing target prediction performance 
        between base and current datasets.
        H0: relationship between X and TARGET_COLUMN is the same.
        H1: relationship changed (concept drift detected).

        Base_df is your train_df, and current_df is your test_df
        """
        try:
            status = True
            report = {}

            # Separate features and target
            X_base = base_df.drop(columns=[TARGET_COLUMN])
            y_base = base_df[TARGET_COLUMN]

            X_current = current_df.drop(columns=[TARGET_COLUMN])
            y_current = current_df[TARGET_COLUMN]

            # Split base into train/validation for stability
            X_train, X_val, y_train, y_val = train_test_split(
                X_base, y_base, test_size=0.3, random_state=42, stratify=y_base
            )

            # Train on base dataset
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate on both base validation and current data
            base_preds = model.predict(X_val)
            current_preds = model.predict(X_current)

            base_acc = accuracy_score(y_val, base_preds)
            current_acc = accuracy_score(y_current, current_preds)
            base_f1 = f1_score(y_val, base_preds, average="weighted")
            current_f1 = f1_score(y_current, current_preds, average="weighted")

            # Compare metrics
            acc_diff = abs(base_acc - current_acc)
            f1_diff = abs(base_f1 - current_f1)

            if acc_diff > threshold or f1_diff > threshold:
                drift_found = True
                status = False
            else:
                drift_found = False

            report = {
                "base_accuracy": float(base_acc),
                "current_accuracy": float(current_acc),
                "accuracy_diff": float(acc_diff),
                "base_f1": float(base_f1),
                "current_f1": float(current_f1),
                "f1_diff": float(f1_diff),
                "concept_drift_detected": drift_found
            }

            # Save report
            concept_drift_report_file_path = self.data_validation_config.concept_drift_report_file_path
            os.makedirs(os.path.dirname(concept_drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=concept_drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            #Reading data from train and test file location
            train_dataframe = read_csv_data(train_file_path)
            test_dataframe = read_csv_data(test_file_path)
            
            error_message = ""
            #Validate columns
            status = self.validate_column_existence(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all columns.\n"
            status = self.validate_column_existence(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Test dataframe does not contain all columns.\n"    
            
            status = self.validate_column_types(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe columns don't match data type schema.\n"    

            status = self.validate_column_types(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Test dataframe columns don't match data type schema.\n"    

            #Check data drift -> In real life you should have a dataframe set aside.
            # and you compare old and new data. not train and test data
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            if not status:
                error_message=f"{error_message}Data drift is detected.\n"

            #Check concept drift
            status = self.detect_concept_drift(base_df=train_dataframe,current_df=test_dataframe)
            if not status:
                error_message=f"{error_message}Concept drift is detected.\n"
            
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Error messages during data validation: \n{error_message}")
            if not status:
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path, index=False, header=True
                )
                test_dataframe.to_csv( 
                    self.data_validation_config.invalid_test_file_path, index=False, header=True
                )
            else:
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path, index=False, header=True
                )
                test_dataframe.to_csv( 
                    self.data_validation_config.valid_test_file_path, index=False, header=True
                )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                data_drift_report_file_path=self.data_validation_config.data_drift_report_file_path,
                concept_drift_report_file_path=self.data_validation_config.concept_drift_report_file_path,
                error_message=error_message
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)