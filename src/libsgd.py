import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List, Union
import logging

class SGDClassifierWrapper:
    """
    A wrapper class for SGD classifier with additional functionality for
    training, evaluation, and automatic feature scaling.
    """

    def __init__(self,
                 features: List[str],
                 params: Optional[Dict] = None):
        """
        Initialize the SGD classifier with feature list and parameters.

        Args:
            features (List[str]): List of feature names to use for training
            params (dict, optional): Custom parameters for SGD classifier
        """
        self.features = features
        self.default_params = {
            'loss': 'log_loss',  # SVM loss
            'penalty': 'l2',
            'alpha': 1e-4,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42,
            'learning_rate': 'optimal',
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 5,
            'shuffle': True
        }

        if params:
            self.default_params.update(params)

        self.model = SGDClassifier(**self.default_params)
        self.scaler = StandardScaler()

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            verbose: bool = True) -> None:
        """
        Train the model with optional validation set.
        Scale features using StandardScaler.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            X_val: Validation features DataFrame (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
        """
        # Select only the specified features
        X_train = X_train[self.features]

        # Fit and transform the scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)

        if verbose:
            print("Training SGD model...")

        # If validation data is provided and early stopping is enabled
        if X_val is not None and y_val is not None and self.default_params.get('early_stopping', True):
            X_val = X_val[self.features]
            X_val_scaled = self.scaler.transform(X_val)

            # Combine training and validation data
            X_combined = np.vstack((X_train_scaled, X_val_scaled))
            y_combined = np.concatenate((y_train, y_val))

            self.model.fit(X_combined, y_combined)
        else:
            self.model.fit(X_train_scaled, y_train)

        if verbose:
            print(f"Number of iterations: {self.model.n_iter_}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Predicted labels
        """
        X = X[self.features]  # Select only the specified features
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions on new data.
        Note: Only works if loss='log_loss' was set in parameters.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Predicted probabilities
        """
        if self.default_params.get('loss') != 'log_loss':
            raise ValueError("Probability prediction is only available for loss='log_loss'")

        X = X[self.features]  # Select only the specified features
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

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
        Save the model and scaler to disk.

        Args:
            save_path: Path where the model should be saved
        """
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(save_dict, save_path)
        logging.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str) -> 'SGDClassifierWrapper':
        """
        Load a saved model from disk.

        Args:
            load_path: Path to the saved model

        Returns:
            SGDClassifierWrapper: Loaded model with scaler
        """
        save_dict = joblib.load(load_path)
        instance = cls(features=save_dict['features'])
        instance.model = save_dict['model']
        instance.scaler = save_dict['scaler']
        return instance