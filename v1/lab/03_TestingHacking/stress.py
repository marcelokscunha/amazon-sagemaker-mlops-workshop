import threading
import boto3
import numpy as np
import time
import math

from multiprocessing.pool import ThreadPool
from sklearn import datasets

from sagemaker.predictor import csv_serializer,csv_deserializer

import ipywidgets as widgets

def predict(payload):
    global sm
    global endpoint_name
    global env
    
    payload = payload
    X = payload[1:]
    y = payload[0]
    
    elapsed_time = time.time()
    resp = sm.invoke_endpoint(
        EndpointName=endpoint_name_mask % env,
        ContentType='text/csv',
        Accept='text/csv',
        Body=csv_serializer(X)
    )
    elapsed_time = time.time() - elapsed_time
    resp = float(resp['Body'].read().decode('utf-8').strip())
    return (resp == y, elapsed_time)

def run_test(max_threads, max_requests, dataset):
    num_batches = math.ceil(max_requests / len(dataset))
    requests = []
    for i in range(num_batches):
        batch = dataset.copy()
        np.random.shuffle(batch)
        requests += batch.tolist()
    len(requests)

    pool = ThreadPool(max_threads)
    result = pool.map(predict, requests)
    pool.close()
    pool.join()
    
    correct_random_forest=0
    elapsedtime_random_forest=0
    for i in result:
        correct_random_forest += i[0]
        elapsedtime_random_forest += i[1]
    print("Score random forest: {}".format(correct_random_forest/len(result)))

    print("Elapsed time random forest: {}s".format(elapsedtime_random_forest))

def run_stress_test(_):
    iris = datasets.load_iris()
    dataset = np.insert(iris.data, 0, iris.target,axis=1)

    print("Starting test 1")
    run_test(10, 1000, dataset)

    print("Starting test 2")
    run_test(100, 10000, dataset)

    print("Starting test 3")
    run_test(150, 100000, dataset)

sm = boto3.client("sagemaker-runtime")
endpoint_name_mask='iris-model-%s'
env='production'

run_test_btn = widgets.Button(description="Run stress test", button_style='success', icon='check')
run_test_btn.on_click(run_stress_test)
stress_button = widgets.HBox([run_test_btn])