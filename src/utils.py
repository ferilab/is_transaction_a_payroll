from typing import Tuple, List
import pandas as pd
import numpy as np
import os
import sys

    ### For working with strings, timing, file saving/loading
import re
import time
import pickle
import joblib

    ### For sentiment analysis
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_curve
    ### Necessary preprocessings and machine learning models
from sklearn.model_selection import GridSearchCV
    ### For feature reduction
from sklearn.decomposition import PCA
    ### For visualizations
import matplotlib.pyplot as plt

    # First set the absolute path to the project root (for proper imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def remove_wrongly_labelled_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    '''
    Function to remove the wrongly labled records (a transaction can't be a debit and
    a credit at the same time) from a dataframe.
    '''
    wrongly_labelled = df[(df.debit > 0) & (df.credit > 0)].index.tolist()
    df.drop(wrongly_labelled, inplace = True)
    df.reset_index(inplace = True, drop = True)

    return df, wrongly_labelled


def add_date_info(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function to convert 'date' to datetime format and extract day of week and month, week
    and month from it.
    '''
        # First we need to convert 'date' column to datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek + 1   # The method will set Monday=0 and Sunday=6, so we added 1 to make it 1-7
    df['week_of_month'] = df['date'].apply(lambda d: (d.day - 1) // 7 + 1)  # First week=1, etc.
    df['month_of_year'] = df['date'].dt.month

    return df


def get_embeddings(text_list, tokenizer, model) -> np.array:
    '''
    Function that takes a list of descriptions, tokenizes them and produces embeddings.
    '''
        # Tokenize the input text and move it to PyTorch tensors
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    
        # Pass through BERT model to get embeddings
    with torch.no_grad():  # No gradient needed for inference
        model_output = model(**inputs)
    
        # It averages the token embeddings to get a fixed-size vector for each sentence
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()


    ### We have to do it on batches otherwise we'll get insufficient memory issue. 
    # It will take about 90 min.
def batch_embedding(df, tokenizer, model) -> List:
    '''
    Function to generate embeddings for entire dataframe (in batches)
    '''
    batch_size = 5000 
    embeddings = []
    for i in range(0, len(df), batch_size):
        batch = df['description_trimmed'].iloc[i:i + batch_size].tolist()
        batch_embeddings = get_embeddings(batch)  # Run your embedding function on the batch
        embeddings.extend(batch_embeddings)       # Add batch results to a list
        print(f"\rUp to row number {i} ({i/batch_size}th batch) is done!", end="")
        
    return embeddings


def pca_transform(n_components, embedding_columns, df):
    ### Function to get desired number of PCAs from embeddings

    pca = PCA(n_components = n_components)
    
        # Then, we fit PCA on embeddings and do transformation
    reduced_embeddings = pca.fit_transform(embedding_columns)
    
        # Now, we should convert the result back to a DataFrame and name columns as 'pca_1', 'pca_2', etc.
    reduced_embeddings_df = pd.DataFrame(reduced_embeddings, columns=[f'pca_{i+1}' for i in range(n_components)])
    
        # Finally, let's concatenate reduced embeddings with the original DataFrame, dropping the original embedding columns
    pca_df = pd.concat([df.drop(columns=embedding_columns.columns), reduced_embeddings_df], axis=1)
    
    print("PCA transformation complete. The data now contains reduced embeddings.")
    return pca_df


def fit_xgb(model, X_train, y_train):
    ### Function to fit and tune a model
    start = time.time()
    
    ### Model tuning - XGBoost
    
        # First let's define the parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300]
    }
    
        # Then, we apply the grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    end = time.time()
    
        ### Get and save the best model as we might need to repeat our experiments with the same model
        # Get the best model
    best_model = grid_search.best_estimator_    
    print(f"The model is tuned and trained in {end - start:.2f}s.")
    
    return best_model
    

def model_performance(best_model, X_train, y_train, X_test, y_test):
    ### Function to measure the performance metrics

        # These are prediction indicators on training set
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
        # And prediction indicators on testing set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print("Training", "\n")
    print("Accuracy:", train_accuracy)
    print("Precision:", train_precision)
    print("F1 Score:", train_f1)
    print('-' * 70, "\n")
    print("Testing", "\n")
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("F1 Score:", test_f1)

    ### -------------------------------------------------------------------------------
          ### However, class accuracies are more important. So, let's begin with a look at the confusion matrix.
    print('-' * 70, "\n")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    print("\nConfusion Matrix:", "\n")
    print(conf_matrix, '\n')
    
    ### -------------------------------------------------------------------------------
    ### We also need to generate and evaluate the probabilities vector of the test data
    
        # The probabilities vector will have the likelihood for every transaction to be a payroll (payroll probability)
    y_test_probabilities = best_model.predict_proba(X_test)
    
        # Calculating accuracy separately for each class (False and True)
    accuracy_class_0 = accuracy_score(y_test[y_test == 0], y_test_pred[y_test == 0])
    accuracy_class_1 = accuracy_score(y_test[y_test == 1], y_test_pred[y_test == 1])
    
        # Make the probability table
    prob_table = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, \
                               'Probability_0': y_test_probabilities[:, 0], \
                               'Probability_1': y_test_probabilities[:, 1]})
    
    print('-' * 70, "\n")
    print("Accuracy for class 0 (validation data):", round(accuracy_class_0, 3))
    print("Accuracy for class 1 (validation data):", round(accuracy_class_1, 3))
    
    print('-' * 70, "\n")
    print("\nProbability Table:")
    print(prob_table.head(10))

    return  y_test_probabilities, prob_table
    

def plot_roc_curve(true_y, y_prob: Tuple[np.ndarray, np.ndarray]) -> None:
    '''
    The function plots the ROC curve based of the arrays of true target values 
    and probabilities given to calss 1 by the pedictive model.
    '''

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 


def feature_importance(best_model, X_train) -> None:
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop = True)
    print('The most important features of the data (sorted descendingly):\n')
    print(feature_importance_df)


def evaluate(model, X, y) -> np.array:

        # And prediction indicators - overall
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    ### -------------------------------------------------------------------------------
          # However, class accuracies are more important. So, let's begin with having a look at the confusion matrix.
    print('-' * 70, "\n")
    conf_matrix = confusion_matrix(y, y_pred)
    
        # Calculating accuracy separately for each class (False and True)
    accuracy_class_0 = accuracy_score(y[y == 0], y_pred[y == 0])
    accuracy_class_1 = accuracy_score(y[y == 1], y_pred[y == 1])

            # The probabilities vector will have the likelihood for every transaction to be a payroll (payroll probability)
    y_probabilities = model.predict_proba(X)

    print("Overall", "\n")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("F1 Score:", f1)
    print('-' * 70, "\n")
    print("\nConfusion Matrix:", "\n")
    print(conf_matrix, '\n')
    print('-' * 70, "\n")
    print("Accuracy - Class 0:", accuracy_class_0)
    print("Accuracy - Class 1:", accuracy_class_1)
    
    return y_probabilities


def fit_lr(model, X_train, y_train):
    ### Function to fit and tune a logistic regression model
    start = time.time()
    
    ### Model tuning - Logistic Regression
    
        # First let's define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1.0],   # Regularization strength
        'penalty': ['l1', 'l2'],  # Regularization type (L1 or L2)
        'solver': ['liblinear', 'saga']  # Solvers that support both L1 and L2 penalties
    }
    
        # Then, we apply the grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    end = time.time()
    
        ### Get and save the best model as we might need to repeat our experiments with the same model
        # Get the best model
    best_model = grid_search.best_estimator_    
    print(f"The model is tuned and trained in {end - start:.2f}s.")
    
    return best_model