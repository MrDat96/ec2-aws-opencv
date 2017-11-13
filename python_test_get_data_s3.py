import boto3
import botocore

BUCKET_NAME = 'iot-centre-projects' # replace with your bucket name
KEY_DIR = '360-degree-camera/001' # replace with your object key
client = boto3.client('s3')

list = client.list_objects(Bucket=BUCKET_NAME)['Contents']

for s3_key in list:
    s3_obj = s3_key['Key']
    if not s3_obj.endswith("/"):
        print(s3_obj)
