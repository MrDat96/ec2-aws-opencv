import boto3
import botocore

def donwload_file(bucket_name, key, store_path):
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).download_file(key, store_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(key + " : the object does not exist.")
        else:
            raise
