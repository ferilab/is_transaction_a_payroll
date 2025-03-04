import os
import sys
from sklearn.metrics import roc_auc_score

    # First set the absolute path to the project root (for proper imports)
sys.path.append(os.path.abspath(".."))
    # All utility functions are in utils
import src.utils as utils

def print_res(data) -> None:
        # Lets plot the ROC of the model and calculate its AUC.
    utils.plot_roc_curve(data['y_test'], data['y_test_probabilities'][:, 1])
    print(f'\033[1m Model AUC score for the test data: {roc_auc_score(data['y_test'], data['y_test_probabilities'][:, 1]):3f}') 

        # 2.6 The most important features
    utils.feature_importance(data['best_model_encoded_descrip'], data['X_train'])