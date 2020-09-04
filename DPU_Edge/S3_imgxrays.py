import json
import urllib.parse
import boto3

print('INFO: Loading gg trigger function')
iotclient = boto3.client('iot-data')

def function_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the bucket and file names from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print("Bucket: " + bucket + " Filename: " + key + " has been uploaded")
    
    # send this command to greengrass by publishing to the topic gg/command
    command = json.dumps({"Action" : "StartInference", "Bucket" : bucket, "Filename" : key})
    response = iotclient.publish(topic='gg/imgxray', qos=0, payload=command)
    
    return response
    
