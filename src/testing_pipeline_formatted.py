import argparse
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from MPC_NEOCP.code.models.libgbm import GBMClassifier
from MPC_NEOCP.code.models.librf import RFClassifier
from MPC_NEOCP.code.models.libsgd import SGDClassifierWrapper
from MPC_NEOCP.code.models.libnn import NNClassifier


def load_and_evaluate_model(model_name, model_path, X_test, y_test, trksub, ground_truth, neo2, save_dir, save_csv=True, return_probs=True):
    print(f"\n=== Evaluating {model_name} ===")
    try:
        # Load the model
        if model_name == 'GBM':
            model = GBMClassifier.load_model(model_path)
        elif model_name == 'RF':
            model = RFClassifier.load_model(model_path)
        elif model_name == 'SGD':
            model = SGDClassifierWrapper.load_model(model_path)
        elif model_name == 'NN':
            if not model_path.endswith('.h5'):
                model_path = f"{os.path.splitext(model_path)[0]}.h5"
            model = NNClassifier.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        y_pred = model.predict(X_test)

        if return_probs:
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                print(f"Warning: Probability prediction not available for {model_name}")
                y_prob = None
        else:
            y_prob = None

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_mat)

        if save_csv:
            output_csv_path = os.path.join(save_dir, f"{model_name}_predictions.csv")
            results_df = pd.DataFrame({
                'trksub': trksub,
                'prediction': y_pred,
                'ground_truth': ground_truth,
                'neo2': neo2
            })

            if y_prob is not None:
                prob_df = pd.DataFrame(y_prob, columns=[f"prob_class_{i}" for i in range(y_prob.shape[1])])
                results_df = pd.concat([results_df, prob_df], axis=1)

            results_df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")

        return {
            'name': model_name,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob,
            'confusion_matrix': conf_mat,
            'report': report
        }

    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None


def plot_confusion_matrices(results, save_path=None):
    results = [r for r in results if r is not None]
    if not results:
        print("No results to plot")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{result['name']} Confusion Matrix")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple ML models on test data")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to directory containing saved models')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test dataset CSV')
    parser.add_argument('--save_results', action='store_true', help='Save CSV and evaluation results')
    args = parser.parse_args()

    print("Loading test data...")
    df_test = pd.read_csv(args.test_data)
    trksub = df_test['trksub']
    neo2 = df_test['Neo2']
    df_test = df_test.drop('trksub', axis=1)
    X_test = df_test.drop('orbtype', axis=1)
    y_test = df_test['orbtype']

    # Auto-discover model files based on known naming
    model_configs = []
    model_map = {
        'GBM': 'gbm_model.joblib',
        'RF': 'rf_model.joblib',
        'SGD': 'sgd_model.joblib',
        'NN': 'nn_model.h5'
    }

    for name, filename in model_map.items():
        path = os.path.join(args.model_dir, filename)
        if os.path.exists(path):
            model_configs.append({'name': name, 'path': path})
        else:
            print(f"Warning: {name} model file not found at {path}")

    results = []
    for config in model_configs:
        result = load_and_evaluate_model(
            model_name=config['name'],
            model_path=config['path'],
            X_test=X_test,
            y_test=y_test,
            trksub=trksub,
            ground_truth=y_test,
            neo2=neo2,
            save_dir=args.model_dir,
            save_csv=args.save_results
        )
        if result:
            results.append(result)

    if results:
        print("\n=== Model Comparison ===")
        comparison_df = pd.DataFrame([{'Model': r['name'], 'Accuracy': r['accuracy']} for r in results])
        print(comparison_df.to_string(index=False))

        plot_confusion_matrices(results, save_path=os.path.join(args.model_dir, 'confusion_matrices.png'))

        if args.save_results:
            output_txt = os.path.join(args.model_dir, 'model_evaluation_results.txt')
            with open(output_txt, 'w') as f:
                for r in results:
                    f.write(f"{r['name']} Results:\n")
                    f.write(f"Accuracy: {r['accuracy']:.4f}\n")
                    f.write(f"{r['report']}\n")
                    f.write(str(r['confusion_matrix']))
                    f.write("\n" + "="*50 + "\n")
            print(f"\nDetailed results saved to {output_txt}")
    else:
        print("No models were successfully evaluated.")


if __name__ == '__main__':
    main()

# python testing_pipeline.py \
#        --model_dir /path/to/models/dir \
#        --test_data /path/to/testdata/neocp_2024.csv \
#        --save_results