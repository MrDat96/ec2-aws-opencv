import boto3
import json
import sys
import logging
import datetime

sys.path.append('../StitchingMaker')

from stitching import stitching 

QUEUE_URL = 'https://sqs.ap-southeast-1.amazonaws.com/498107424281/DatntQueue'
DIR_DATA = '../Data'
BUCKET_NAME = 'iot-centre-projects'

#download a file from s3
def download_file(bucket_name, key, store_path):
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).download_file(key, store_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(key + " : the object does not exist.")
        else:
            raise

#polling for a new message
def sqs_polling(queue_url):
    # Create SQS client
    sqs = boto3.client('sqs')

    # Receive message from SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        AttributeNames=[
            'SentTimestamp'
        ],
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=20
    )

    if ('Messages' not in response):
        print('Nothing in SQS')
        sys.exit()

    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']
    try:
        index = 1
        body = json.loads(message['Body'])
        print('Message json found) Received message: %s' % body)
        try:
            download_file(BUCKET_NAME, body['cam01_key'], DIR_DATA + '/Cam01/' + 'cam01_{}.jpg'.format(index))
            download_file(BUCKET_NAME, body['cam02_key'], DIR_DATA + '/Cam02/' + 'cam02_{}.jpg'.format(index))
            download_file(BUCKET_NAME, body['cam03_key'], DIR_DATA + '/Cam03/' + 'cam03_{}.jpg'.format(index))
            download_file(BUCKET_NAME, body['cam04_key'], DIR_DATA + '/Cam04/' + 'cam04_{}.jpg'.format(index))
            print("Download finished")
        except:
            print("Download errored")
    except:
        #print("Exception occur")
        print('No message json found instead text: %s' % message['Body'])
   
    # Delete received message from queue
    '''sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )
    print('Delete message')'''

if __name__ == '__main__':
    sqs_polling(QUEUE_URL)
    stitching()  
