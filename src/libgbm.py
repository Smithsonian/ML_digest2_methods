import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from typing import Dict, Tuple, Optional, List, Union
import logging

class GBMClassifier:
    """
    A wrapper class for XGBoost classifier with additional functionality for
    training and evaluation.
    """

    def __init__(self,
                 features: List[str],
                 params: Optional[Dict] = None):
        """
        Initialize the GBM classifier with feature list and parameters.

        Args:
            features (List[str]): List of feature names to use for training
            params (dict, optional): Custom parameters for XGBoost classifier
        """
        self.features = features
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42,
            'n_estimators': 100,
            'subsample': 0.8,
            'learning_rate': 0.1,
            'max_depth': 5,
            'colsample_bytree': 0.8
        }

        if params:
            self.default_params.update(params)

        self.model = xgb.XGBClassifier(**self.default_params)

    def fit(self,
            X_train: Union[pd.DataFrame, np.ndarray],
            y_train: pd.Series,
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[pd.Series] = None,
            verbose: bool = True) -> None:
        """
        Train the model with optional validation set.

        Args:
            X_train: Training features (DataFrame or numpy array)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
        """
        # Handle numpy arrays
        if isinstance(X_train, np.ndarray):
            if len(X_train.shape) == 1:
                X_train = X_train.reshape(-1, 1)
            if X_val is not None and len(X_val.shape) == 1:
                X_val = X_val.reshape(-1, 1)
        # Handle DataFrames
        elif isinstance(X_train, pd.DataFrame) and self.features is not None:
            X_train = X_train[self.features]
            if X_val is not None:
                X_val = X_val[self.features]

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features (DataFrame or numpy array) to predict on

        Returns:
            numpy.ndarray: Predicted labels
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame) and self.features is not None:
            X = X[self.features]

        # Handle numpy array input
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions on new data.

        Args:
            X: Features (DataFrame or numpy array) to predict on

        Returns:
            numpy.ndarray: Predicted probabilities
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame) and self.features is not None:
            X = X[self.features]

        # Handle numpy array input
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

        # Print shape for debugging
        print(f"Shape of input to GBM predict_proba: {X.shape}")

        return self.model.predict_proba(X)

    def evaluate(self,
                 y_true: pd.Series,
                 y_pred: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        Evaluate model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            tuple: (accuracy, classification_report, confusion_matrix)
        """
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        return accuracy, report, conf_matrix

    def save_model(self, save_path: str) -> None:
        """
        Save the model to disk.

        Args:
            save_path: Path where the model should be saved
        """
        save_dict = {
            'model': self.model,
            'features': self.features
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(save_dict, save_path)
        logging.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str) -> 'GBMClassifier':
        """
        Load a saved model from disk.

        Args:
            load_path: Path to the saved model

        Returns:
            GBMClassifier: Loaded model
        """
        save_dict = joblib.load(load_path)
        instance = cls(features=save_dict['features'])
        instance.model = save_dict['model']
        return instance