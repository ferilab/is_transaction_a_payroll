from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import pandas as pd

    # Set the absolute path to the project's root to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Set the path to save/load files (from local or cloud)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import preprocess_data
from src.train import train_eval
from src.report import print_res


    # Load preprocessed sales data
def preprocessing(**kwargs):
        # Preoare the data and save it in the data folder
    file_name = preprocess_data()
        # Push the file name to XCom for the next task
    kwargs['ti'].xcom_push(key='data_name', value=file_name)
    return file_name

def training(**kwargs):
        # Pull the path to data from XCom and load the file
    file_name = kwargs['ti'].xcom_pull(task_ids='preprocess', key='data_name')
        # The result has these keys: y_test, y_test_probabilities, best_model_encoded_descrip, X_train
    result = train_eval(file_name)
        # Push the result to XCom for the next task
    kwargs['ti'].xcom_push(key='train_result', value=result)
    return result

def report_result(**kwargs):
        # Pull the result from XCom
    result = kwargs['ti'].xcom_pull(task_ids='train', key='train_result')
    print_res(result)


default_args = {
    'owner': 'Fereidoun',
    'start_date': datetime(2025, 3, 4),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG('predict_payroll',
         default_args = default_args,
         schedule_interval='*/60 * * * *',
         catchup=False
         ) as dag:
    
    preprocess = PythonOperator(task_id = 'preprocess',
                               python_callable = preprocessing,
                               provide_context=True)
    train = PythonOperator(task_id = 'train',
                                python_callable = training,
                                provide_context=True)
    print_result = PythonOperator(task_id = 'print_result',
                                python_callable = report_result,
                                provide_context=True)

preprocess_data >> train >> print_result