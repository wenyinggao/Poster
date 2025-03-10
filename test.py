import requests
import concurrent.futures
import base64
import json
import os
import time

# Lambda function URL or API Gateway endpoint
# LAMBDA_URL = "https://7ix45cy5qixqg6yzlvsyvhbwwu0tsmql.lambda-url.us-east-2.on.aws/ "
LAMBDA_URL = "https://rvnmc0njj7.execute-api.us-east-2.amazonaws.com/ocr/image_url/search_prediction"

# Sample image file path and search text
IMAGE_PATH = "images/testocr.png"
SEARCH_TEXT = "dog"

# Directory to store response files
OUTPUT_DIR = "lambda_responses"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to send a single request to the Lambda function
def call_lambda(image_path, search_text, request_number):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # Encode image to base64 to simulate form-data file upload
    files = {
        "image": (IMAGE_PATH, image_data, "image/jpeg"),
        "search_text": (None, search_text),
    }

    try:
        response = requests.post(LAMBDA_URL, files=files)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        result = {"error": str(e)}
    
    # Save the result to a file
    result_filename = os.path.join(OUTPUT_DIR, f"response_{request_number}.json")
    with open(result_filename, "w") as result_file:
        json.dump(result, result_file, indent=4)
    
    print(f"Request {request_number} result saved to {result_filename}")
    return result

# Main function to send multiple concurrent requests
def test_lambda_concurrency(num_requests=10):
    search_text = SEARCH_TEXT
    image_path = IMAGE_PATH

    # Record the start time
    start_time = time.time()

    # Use ThreadPoolExecutor to send requests simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submit requests to the executor with a unique request number
        futures = [
            executor.submit(call_lambda, image_path, search_text, i + 1)
            for i in range(num_requests)
        ]
        
        # Wait for all futures to complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()  # Result is already saved in call_lambda

    # Record the end time
    end_time = time.time()

    # Calculate and print the total elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time taken for {num_requests} requests: {elapsed_time:.2f} seconds")

# Run the test
if __name__ == "__main__":
    # Number of simultaneous requests
    test_lambda_concurrency(num_requests=30)