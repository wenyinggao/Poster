import requests
import time

# Define the API URL from your deployed API Gateway
# api_url = 'https://9he6fa65wj.execute-api.us-east-2.amazonaws.com/image_compare_stage/lambda_test'
api_url = 'https://sx5hioawbdnyfy6plcznxicy7a0ylvwd.lambda-url.us-east-2.on.aws/'

# Paths to the two images you want to upload
image1_path = 'images/tosvg3/converted.png'
image2_path = 'images/tosvg3/imageWithWhiteBg.png'

# Open the two image files in binary mode
with open(image1_path, 'rb') as image1, open(image2_path, 'rb') as image2:
    # Prepare the multipart form-data payload with file data
    files = {
        'image1': ('image1.png', image1, 'image/png'),  # Replace 'image1.png' with your file name if necessary
        'image2': ('image2.png', image2, 'image/png')   # Replace 'image2.png' with your file name if necessary
    }
    
    # Start the timer
    start_time = time.time()

    # Send the POST request to the API Gateway
    response = requests.post(api_url, files=files)
  
    # End the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time

    # Print the response from the Lambda function and the time taken
    if response.status_code == 200:
        print("Response from Lambda:", response.json())
    else:
        print("Error:", response.status_code, response.text)

    # Print the time taken for the request
    print(f"Time taken for the request: {time_taken:.2f} seconds")