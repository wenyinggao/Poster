import json
import base64
import re

def lambda_handler(event, context):
    # Get the raw body of the request
    body = event['body']
    
    # Check if the body is base64 encoded and decode it if necessary
    if event.get("isBase64Encoded", False):
        body = base64.b64decode(body)

    # Extract the boundary from the content-type header
    content_type = event['headers']['content-type']
    boundary = re.search(r'boundary=(.*)', content_type).group(1)

    # Split the body based on the boundary to get multipart parts
    parts = body.split(('--' + boundary).encode('utf-8'))

    # Initialize variables to hold the images' content
    image1_content = None
    image2_content = None

    # Iterate over each part to find image1 and image2
    for part in parts:
        if b'name="image1"' in part:
            image1_content = part.split(b'\r\n\r\n')[1].split(b'\r\n--')[0]
        elif b'name="image2"' in part:
            image2_content = part.split(b'\r\n\r\n')[1].split(b'\r\n--')[0]

    # Calculate sizes of the two images
    image1_size = len(image1_content) if image1_content else 0
    image2_size = len(image2_content) if image2_content else 0

    # Prepare the response with the sizes of both images
    result = {
        "image1_size": image1_size,
        "image2_size": image2_size
    }

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
