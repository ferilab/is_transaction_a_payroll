# is_transaction_a_payroll
Using a ML model we will predict if a given user’s transaction is payroll or not.

Detail:

The dataset (split into train.csv and test.csv) contains the transaction history of some users. The variables are:
user_id: unique user id
date: the transaction date, 
description: the raw description of the transaction,
debit: amount withdrawn from account
credit: amount entered into account
is_payroll: the target variable (Boolean)

The goal is to use this dataset to make a ML model predicting if a given user’s transaction is payroll or not. The approach includes:

- Data Exploration (done in payroll.ipynb)
- Data Preprocessing and Cleaning (done by scripts in src folder)
- Feature Engineering/Selection (done by scripts in src folder)
- Model Building and Evaluation (done by scripts in src folder)

Folders:
# data
raw data (train_csv, test_csv)
train.csv is used to train and evaluate the model.
test.csv will be used for testing purpose with unseen data.
The 
results (embeddings, trained models, ...)

# src
utils.py that contains all utility functions

# dev_and_run
development and tests in a single notebook.

# doc
a discussion on findings

# dags
predict_payroll if the script for CI/CD

# .git (might be hidden
deploy.yml in airflow folder triggers the code in GCP and runs the schedule (reruns the code as scheduled).