import boto3
import ipywidgets as widgets
from threading import Thread
from time import sleep
import time

runtime_client = boto3.client('runtime.sagemaker')

def invoke_endpoint(ep_name, file_name, runtime_client):
    with open(file_name, 'r') as f:
        for row in f:
            payload = row.rstrip('\n')
            response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                          ContentType='text/csv', 
                                          Accept='text/csv',
                                          Body=payload)
            time.sleep(1)
            
def invoke_endpoint_forever():
    global endpoint_name
    while True:
        invoke_endpoint(endpoint_name, '../0_custom_train/lab/data/test_no_label/iris_test_no_label.csv', runtime_client)

def run_stress_test(_):
    print('Sending some artificial traffic...\nGo check endpoint metrics and S3 bucket!')
    thread = Thread(target = invoke_endpoint_forever)
    thread.start()

endpoint_name = 'IrisMLPipeline'
run_test_btn = widgets.Button(description="Run stress test", button_style='success', icon='check')
run_test_btn.on_click(run_stress_test)
stress_button = widgets.HBox([run_test_btn])