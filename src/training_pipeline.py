import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from libgbm import GBMClassifier
from librf import RFClassifier
from libsgd import SGDClassifierWrapper
from libnn import NNClassifier


def prepare_validation_data(training_path, evaluation_path=None, test_size=0.2):
    df_train = pd.read_csv(training_path)
    X_train = df_train.drop('orbtype', axis=1)
    y_train = df_train['orbtype']

    if evaluation_path and os.path.exists(evaluation_path):
        print("Using separate evaluation dataset...")
        df_eval = pd.read_csv(evaluation_path)
        X_val = df_eval.drop('orbtype', axis=1)
        y_val = df_eval['orbtype']
    else:
        print("Splitting training data for validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=test_size,
            random_state=42,
            stratify=y_train
        )

    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models using NEOCP data.")
    parser.add_argument('--train_csv', required=True, help='Path to training data CSV')
    parser.add_argument('--eval_csv', default=None, help='Optional path to evaluation CSV')
    parser.add_argument('--model_save_dir', required=True, help='Directory to save trained models')
    parser.add_argument('--features_file', help='Optional path to text file listing features to use')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size when splitting training data')

    args = parser.parse_args()

    # Prepare data
    X_train, X_val, y_train, y_val = prepare_validation_data(
        training_path=args.train_csv,
        evaluation_path=args.eval_csv,
        test_size=args.test_size
    )

    # Load features
    if args.features_file and os.path.exists(args.features_file):
        with open(args.features_file, 'r') as f:
            unified_features = f.read().splitlines()
    else:
        unified_features = ['Int1', 'Int2', 'Neo1', 'Neo2', 'MC1', 'MC2', 'Hun1', 'Hun2', 'Pho1', 'Pho2',
                            'MB1_1', 'MB1_2', 'Pal1', 'Pal2', 'Han1', 'Han2', 'MB2_1', 'MB2_2',
                            'MB3_1', 'MB3_2', 'Hil1', 'Hil2', 'JTr1', 'JTr2', 'JFC1', 'JFC2']

    # === GBM ===
    print("\n=== GBM Classifier ===")
    gbm_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'n_estimators': 100,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'max_depth': 5,
        'colsample_bytree': 0.8
    }

    gbm = GBMClassifier(features=unified_features, params=gbm_params)
    gbm.fit(X_train, y_train, X_val, y_val)

    y_val_pred = gbm.predict(X_val)
    acc, report, conf = gbm.evaluate(y_val, y_val_pred)

    print("\nValidation Set Evaluation:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", report)

    gbm.save_model(os.path.join(args.model_save_dir, 'gbm_model.joblib'))

    # === RF ===
    print("\n=== Random Forest Classifier ===")
    rf = RFClassifier(features=unified_features)
    rf.fit(X_train, y_train)

    y_val_pred = rf.predict(X_val)
    acc, report, conf = rf.evaluate(y_val, y_val_pred)

    print("\nValidation Set Evaluation:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", report)

    rf.save_model(os.path.join(args.model_save_dir, 'rf_model.joblib'))

    # === SGD ===
    print("\n=== SGD Classifier ===")
    sgd = SGDClassifierWrapper(features=unified_features)
    sgd.fit(X_train, y_train)

    y_val_pred = sgd.predict(X_val)
    acc, report, conf = sgd.evaluate(y_val, y_val_pred)

    print("\nValidation Set Evaluation:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", report)

    sgd.save_model(os.path.join(args.model_save_dir, 'sgd_model.joblib'))

    # === NN ===
    print("\n=== NN Classifier ===")
    nn_params = {
        'n_classes': 1,
        'neurons_per_layer': 64,
        'n_hidden_layers': 2,
        'activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'n_epochs': 20,
        'batch_size': 32
    }

    nn = NNClassifier(features=unified_features, params=nn_params)
    nn.fit(X_train, y_train, X_val, y_val)

    y_val_pred = nn.predict(X_val)
    acc, report, conf = nn.evaluate(y_val, y_val_pred)

    print("\nValidation Set Evaluation:")
    print("Accuracy:", acc)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf)

    nn.save_model(os.path.join(args.model_save_dir, 'nn_model.h5'))


if __name__ == '__main__':
    main()


# python training_pipeline.py \
#        --train_csv /path/to/neocp.csv \
#        --eval_csv /path/to/neocp_2024.csv \
#        --model_save_dir /path/to/save/models \
#        --features_file /path/to/cols_of_interest.txt \
#        --test_size 0.25
