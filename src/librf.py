import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Optional, List, Union
import logging


class RFClassifier:
    """
    A wrapper class for Random Forest classifier with additional functionality for
    training and evaluation.
    """

    def __init__(self,
                 features: List[str],
                 params: Optional[Dict] = None):
        """
        Initialize the RF classifier with feature list and parameters.

        Args:
            features (List[str]): List of feature names to use for training
            params (dict, optional): Custom parameters for Random Forest classifier
        """
        self.features = features
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }

        if params:
            self.default_params.update(params)

        self.model = RandomForestClassifier(**self.default_params)

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            verbose: bool = True) -> None:
        """
        Train the model with optional validation set.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            X_val: Validation features DataFrame (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
        """
        # Select only the specified features
        X_train = X_train[self.features]

        if verbose:
            print("Training Random Forest model...")

        self.model.fit(X_train, y_train)

        if verbose and X_val is not None and y_val is not None:
            X_val = X_val[self.features]
            val_score = self.model.score(X_val, y_val)
            print(f"Validation score: {val_score:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Predicted labels
        """
        X = X[self.features]  # Select only the specified features
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions on new data.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Predicted probabilities
        """
        X = X[self.features]  # Select only the specified features
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
    def load_model(cls, load_path: str) -> 'RFClassifier':
        """
        Load a saved model from disk.

        Args:
            load_path: Path to the saved model

        Returns:
            RFClassifier: Loaded model
        """
        save_dict = joblib.load(load_path)
        instance = cls(features=save_dict['features'])
        instance.model = save_dict['model']
        return instance