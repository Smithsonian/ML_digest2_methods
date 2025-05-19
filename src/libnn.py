import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List, Union
import logging

class NeuralNetwork:
    """Neural Network architecture using TensorFlow"""
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        self.model = Sequential()

        # First hidden layer
        self.model.add(Dense(hidden_sizes[0],
                             input_dim=input_size,
                             activation='relu'))

        # Additional hidden layers
        for hidden_size in hidden_sizes[1:]:
            self.model.add(Dense(hidden_size, activation='relu'))

        # Output layer - single node with sigmoid activation for binary classification
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

class NNClassifier:
    """
    A wrapper class for Neural Network classifier with additional functionality for
    training, evaluation, and automatic feature scaling.
    """

    def __init__(self,
                 features: List[str],
                 params: Optional[Dict] = None):
        """
        Initialize the Neural Network classifier with feature list and parameters.

        Args:
            features (List[str]): List of feature names to use for training
            params (dict, optional): Custom parameters for Neural Network
        """
        self.features = features

        self.default_params = {
            'hidden_sizes': [64, 32],  # sizes of hidden layers
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'random_state': 42
        }

        if params:
            self.default_params.update(params)

        # Set random seeds for reproducibility
        tf.random.set_seed(self.default_params['random_state'])
        np.random.seed(self.default_params['random_state'])

        self.scaler = StandardScaler()
        self.model = None  # Will be initialized in fit

    def _initialize_model(self, input_size: int):
        """Initialize the neural network model"""
        self.model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=self.default_params['hidden_sizes']
        ).model

    def get_intermediate_features(self, X: pd.DataFrame, layer_index: Optional[int] = -2) -> np.ndarray:
        """
        Extract features from an intermediate layer of the neural network.

        Args:
            X: Input data for feature extraction.
            layer_index: Index of the layer from which to extract features. Defaults to -2 (second-to-last layer).

        Returns:
            numpy.ndarray: Features extracted from the specified layer.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please train the model first.")

        # Scale input features
        X = X[self.features]
        X_scaled = self.scaler.transform(X)

        # Create a new model that outputs the specified layer
        intermediate_layer = self.model.layers[layer_index]
        intermediate_output = intermediate_layer.output
        intermediate_model = tf.keras.Model(inputs=self.model.layers[0].input,
                                            outputs=intermediate_output)

        # Predict using the intermediate model to extract features
        intermediate_features = intermediate_model.predict(X_scaled,
                                                           batch_size=self.default_params['batch_size'])
        return intermediate_features

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            verbose: bool = True) -> None:
        """
        Train the neural network with optional validation set.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            X_val: Validation features DataFrame (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
        """
        # Select features and scale data
        X_train = X_train[self.features]
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None:
            X_val = X_val[self.features]
            X_val_scaled = self.scaler.transform(X_val)

        # Initialize model if not already initialized
        if self.model is None:
            self._initialize_model(input_size=len(self.features))

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.default_params['early_stopping_patience'],
                restore_best_weights=True,
                mode='min'
            )
        ]

        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=self.default_params['epochs'],
            batch_size=self.default_params['batch_size'],
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

        if verbose:
            # Print final metrics
            final_epoch = len(history.history['loss'])
            print(f"\nTraining completed after {final_epoch} epochs")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            if X_val is not None:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Predicted labels (0 or 1)
        """
        X = X[self.features]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions on new data.

        Args:
            X: Features DataFrame to predict on

        Returns:
            numpy.ndarray: Two columns of probabilities [P(class 0), P(class 1)]
        """
        X = X[self.features]
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        # Convert to two-column format [P(class 0), P(class 1)]
        return np.column_stack([1 - probs, probs])

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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Ensure the path has .h5 extension
        if not save_path.endswith('.h5'):
            save_path = f"{os.path.splitext(save_path)[0]}.h5"

        # Save the model
        self.model.save(save_path)

        # Save other components
        components_path = f"{os.path.splitext(save_path)[0]}_components.joblib"
        save_dict = {
            'scaler': self.scaler,
            'features': self.features,
            'params': self.default_params
        }
        joblib.dump(save_dict, components_path)
        logging.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str) -> 'NNClassifier':
        """
        Load a saved model from disk.

        Args:
            load_path: Path to the saved model

        Returns:
            NNClassifier: Loaded model with scaler
        """
        # Ensure the path has .h5 extension
        if not load_path.endswith('.h5'):
            load_path = f"{os.path.splitext(load_path)[0]}.h5"

        # Load components
        components_path = f"{os.path.splitext(load_path)[0]}_components.joblib"
        save_dict = joblib.load(components_path)

        # Create instance with saved parameters
        instance = cls(
            features=save_dict['features'],
            params=save_dict['params']
        )

        # Load the TensorFlow model
        instance.model = tf.keras.models.load_model(load_path)
        instance.scaler = save_dict['scaler']

        return instance