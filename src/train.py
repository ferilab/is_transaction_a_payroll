import pandas as pd
import os
import sys

import joblib

 ### Necessary preprocessings and machine learning models
from sklearn.model_selection import train_test_split
import xgboost as xgb

    # First set the absolute path to the project root (for proper imports)
sys.path.append(os.path.abspath(".."))
    # All utility functions are in utils
import src.utils as utils


def train_eval(file_name):

            # Also, get the project root path (one level up from dev_and_run folder) to navigate through other folders like data
    project_root = os.path.abspath("..")

    file_path = os.path.join(project_root, "data/")
    preprocessed_df = pd.read_csv(file_path + file_name)
        # Train the model
        # List of The features that are already replaced with more descriptive variables or are not right inputs for the model
    todrop = ['user_id', 'date', 'description', 'debit', 'credit', 'description_trimmed', 'is_payroll']

        # 2.1 Extracting valid features (X) and the target variable (y)
    X = preprocessed_df.drop(todrop, axis=1)  # Drop unnecessary columns from the feature matrix
    y = preprocessed_df['is_payroll']  # Extract the target column
        
        # 2.2 Splitting the encoded dataset into training and testing sets. As we also have a hold out test data, for alidation will only use 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

        # 2.3 Fit and tune the XGBoost with simply encoded description
        # First initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

        # 2.4 Get and save the best model
    best_model_encoded_descrip= utils.fit_xgb(xgb_model, X_train, y_train)

        # Save the best model
    joblib.dump(best_model_encoded_descrip, file_path + 'best_xgb_model_encoded_descrip.pkl')

        # 2.5 Evaluate the performance of the model
    y_test_probabilities, prob_table = utils.model_performance(best_model_encoded_descrip, X_train, y_train, X_test, y_test)
    var_dict = {name: value for name, value in globals().items() if name in \
                ["y_test", "y_test_probabilities", "best_model_encoded_descrip", "X_train"]}

    return var_dict
