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

The goal is to use this dataset to make a ML model predicting if a given user’s transaction is
payroll or not. The approach includes:

- Data Exploration
- Data Preprocessing and Cleaning
- Feature Engineering/Selection
- Model Building and Evaluation

Folders:
# data
raw data (train_csv, test_csv)
results (embeddings, trained models, ...)

# src
utils.py that contains all utility functions

# dev_and_run
development and tests in a single notebook

# doc
a discussion on findings

# dags
is considered for CI/CD: emply for now

# .git (might be hidden
is considered for cloud run: empty for now