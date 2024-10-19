import boto3
from botocore.exceptions import NoCredentialsError
import time

# Initialize an S3 client
s3 = boto3.client('s3')

def upload_image_to_s3(file_name, upload_bucket, object_name=None):
    """Upload an image to an S3 bucket and trigger Lambda.
    
    :param file_name: File to upload (local file path)
    :param upload_bucket: S3 bucket name to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: The object key if file was uploaded, else None
    """
    
    # If S3 object_name was not specified, use the file_name
    if object_name is None:
        object_name = file_name

    try:
        # Upload the file to S3
        s3.upload_file(file_name, upload_bucket, object_name)
        print(f"File uploaded successfully to https://{upload_bucket}.s3.amazonaws.com/{object_name}")
        return object_name
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

def download_processed_image_from_s3(processed_bucket, processed_key, download_path):
    """Download the processed image from the second S3 bucket.
    
    :param processed_bucket: S3 bucket name where processed image is stored
    :param processed_key: Key of the processed image in the second bucket
    :param download_path: Local path to save the downloaded image
    """
    try:
        # Download the processed image from S3
        s3.download_file(processed_bucket, processed_key, download_path)
        print(f"Processed image downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading the processed image: {e}")

def wait_for_processed_image(processed_bucket, processed_key, timeout=300, interval=10):
    """Wait for the processed image to appear in S3 by polling.
    
    :param processed_bucket: S3 bucket name where processed image is stored
    :param processed_key: Key of the processed image in the second bucket
    :param timeout: Maximum time to wait for the image to appear
    :param interval: Time between polls
    :return: True if file found, else False
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if the object exists in S3
            s3.head_object(Bucket=processed_bucket, Key=processed_key)
            print(f"Processed image {processed_key} found in {processed_bucket}")
            return True
        except s3.exceptions.NoSuchKey:
            print(f"Processed image {processed_key} not yet available, waiting {interval} seconds...")
            time.sleep(interval)
    
    print(f"Timeout reached. Processed image {processed_key} not found.")
    return False

def list_objects_in_s3_bucket(bucket_name):
    """List objects in a specific S3 bucket to see if the file is present."""
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            print("Files in processed bucket:")
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found in the bucket.")
    except Exception as e:
        print(f"Error listing objects in S3 bucket: {e}")

def main():
    # Input parameters
    file_name = 'images/tosvg3/converted.png'  # Local image file to upload
    upload_bucket = 'testbucket1178'  # S3 bucket to upload image (triggers Lambda)
    processed_bucket = 'testbucket1178output'  # S3 bucket where processed image is stored
    download_path = 'downloaded_processed_image.jpg'  # Local path to save the downloaded image

    # Upload the image to the first bucket
    object_key = upload_image_to_s3(file_name, upload_bucket)

    if object_key:
        # Construct the processed key (ensure the logic matches the Lambda function's output)
        processed_key = f'processed_{object_key}'

        # Introduce an initial delay before checking the bucket
        print("Giving the Lambda function some time to process the image...")
        time.sleep(15)  # Initial delay to give Lambda time to process before polling
        
        # List objects in the processed bucket to debug (optional)
        list_objects_in_s3_bucket(processed_bucket)

        # Wait for the Lambda function to process the image and upload it to the processed bucket
        if wait_for_processed_image(processed_bucket, processed_key, timeout=300, interval=10):
            # Download the processed image from the second bucket
            download_processed_image_from_s3(processed_bucket, processed_key, download_path)
        else:
            print("Processed image not found within the timeout period.")

if __name__ == "__main__":
    main()
