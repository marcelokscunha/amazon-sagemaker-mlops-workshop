import threading
import boto3
import json
import numpy as np
import time
import math

from multiprocessing.pool import ThreadPool
from sklearn import datasets

import ipywidgets as widgets

    
def predict(payload):
    global sm
    global endpoint_name
    
    payload = payload
    X = [ payload[1:] ]
    y = payload[0]
    response = []
    elapsed_time = time.time()
    resp = sm.invoke_endpoint(
        EndpointName=endpoint_name,
        CustomAttributes='logistic',
        Body=json.dumps(X)
    )
    elapsed_time = time.time() - elapsed_time
    resp = json.loads(resp['Body'].read())
    response.append((resp['iris_id'][0] == y, elapsed_time))

    elapsed_time = time.time()
    resp = sm.invoke_endpoint(
        EndpointName=endpoint_name,
        CustomAttributes='random_forest',
        Body=json.dumps(X)
    )
    elapsed_time = time.time() - elapsed_time
    resp = json.loads(resp['Body'].read())
    response.append((resp['iris_id'][0] == y, elapsed_time))

    return response
    
def run_test(max_threads, max_requests, dataset):  
      
    num_batches = math.ceil(max_requests / len(dataset))
    requests = []
    for i in range(num_batches):
        batch = dataset.copy()
        np.random.shuffle(batch)
        requests += batch.tolist()

    pool = ThreadPool(max_threads)
    result = pool.map(predict, requests)
    pool.close()
    pool.join()

    correct_logistic=0
    correct_random_forest=0
    elapsedtime_logistic=0
    elapsedtime_random_forest=0
    for i in result:
        correct_logistic += i[0][0]
        correct_random_forest += i[1][0]

        elapsedtime_logistic += i[0][1]
        elapsedtime_random_forest += i[1][1]
    print("Score logistic: {}".format(correct_logistic/len(result)))
    print("Score random forest: {}".format(correct_random_forest/len(result)))

    print("Elapsed time logistic: {}s".format(elapsedtime_logistic))
    print("Elapsed time random forest: {}s".format(elapsedtime_random_forest))
        
        

def run_stress_test(b):
    iris = datasets.load_iris()
    dataset = np.insert(iris.data, 0, iris.target,axis=1)

    print("Starting test 1")
    run_test(10, 1000, dataset)

    print("Starting test 2")
    run_test(100, 10000, dataset)

    print("Starting test 3")
    run_test(150, 100000, dataset)
    
    
sm = boto3.client("sagemaker-runtime")
endpoint_name='mlops-iris-model-prd'

run_test_btn = widgets.Button(description="Run stress test", button_style='success', icon='check')
run_test_btn.on_click(run_stress_test)
stress_button = widgets.HBox([run_test_btn])