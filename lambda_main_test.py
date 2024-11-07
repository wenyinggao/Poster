import os
import sys

# Add local_package to the system path
sys.path.insert(0, '/local_package')

import json
import requests
import regex as re
import base64
import numpy as np
import cv2
from typing import List
from paddleocr import PaddleOCR
from requests_toolbelt.multipart import decoder

# Configuration for PaddleOCR model directory in /tmp for AWS Lambda
custom_model_dir = "/tmp/.paddleocr"
os.environ['PADDLEOCR_MODEL_DIR'] = custom_model_dir

# Initialize PaddleOCR with custom directory
ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                det_model_dir=f"{custom_model_dir}/det", 
                rec_model_dir=f"{custom_model_dir}/rec", 
                cls_model_dir=f"{custom_model_dir}/cls")


def decode_image(image_data: bytes):
    """Convert raw image bytes into OpenCV image format."""
    try:
        np_arr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Failed to decode image. Data type: {type(image_data)}. Error: {str(e)}")


def remove_invisible_chars(text):
    """Remove invisible characters and spaces from text."""
    try:
        return re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text).replace(" ", "")
    except Exception as e:
        raise ValueError(f"Failed to clean text. Data type: {type(text)}. Error: {str(e)}")


def ocr_prediction(image_data: bytes, search_text: str) -> dict:
    """Perform OCR on the image and filter results based on search text."""
    try:
        image = decode_image(image_data)
        result = ocr.ocr(image, cls=True)

        # Filter results based on search_text if provided
        search_text_lower = search_text.lower() if search_text else None
        output: List[dict] = []

        for line in result:
            for res in line:
                text, confidence, bbox = res[1][0], res[1][1], res[0]
                cleaned_text = remove_invisible_chars(text)
                
                if not search_text or (search_text_lower in cleaned_text.lower()):
                    output.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })

        return {'err_no': 0, 'err_msg': 'Success', 'data': output}
    except Exception as e:
        return {'err_no': 2, 'err_msg': f"OCR Prediction Error. Data type: {type(image_data)}. Error: {str(e)}"}


def parse_json_body(event_body):
    """Parse JSON body to extract image_url and search_text."""
    try:
        body = json.loads(event_body) if isinstance(event_body, str) else event_body
        return body.get('image_url'), body.get('search_text', '')
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON body. Data type: {type(event_body)}. Error: {str(e)}")


def parse_multipart_body(event, content_type):
    """Parse multipart/form-data to extract image data, image_url, and search_text."""
    try:
        image_data, image_url, search_text = None, None, ""
        body = event.get("body", "")
        decoded_body = base64.b64decode(body) if event.get("isBase64Encoded") else body
        multipart_data = decoder.MultipartDecoder(decoded_body, content_type)

        for part in multipart_data.parts:
            disposition = part.headers.get(b"Content-Disposition", b"").decode()
            if 'name="image"' in disposition:
                image_data = part.content
            elif 'name="image_url"' in disposition:
                image_url = part.text.strip('"')  # Strip quotes if present
            elif 'name="search_text"' in disposition:
                search_text = part.text.strip('"')  # Strip quotes if present

        return image_data, image_url, search_text
    except Exception as e:
        raise ValueError(f"Failed to parse multipart body. Data type: {type(event)}. Error: {str(e)}")


def fetch_image_from_url(image_url):
    """Fetch image content from a URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL. Data type: {type(image_url)}. Error: {str(e)}")


def lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:
        # Default values
        image_url, search_text = None, ""
        image_data = None

        # Detect content type and parse body
        content_type = event.get("headers", {}).get("Content-Type", "").lower()
        
        # Check if content is multipart/form-data before trying JSON
        if "multipart/form-data" in content_type:
            image_data, image_url, search_text = parse_multipart_body(event, content_type)
        elif event.get('body'):
            # Parse as JSON if not multipart
            try:
                image_url, search_text = parse_json_body(event['body'])
            except ValueError as e:
                return {'statusCode': 400, 'body': json.dumps({'err_no': 1, 'err_msg': str(e)})}

        # Fetch image data if image_url is provided and image data is not in form
        if image_url and not image_data:
            try:
                image_data = fetch_image_from_url(image_url)
            except ValueError as e:
                return {'statusCode': 400, 'body': json.dumps({'err_no': 1, 'err_msg': str(e)})}

        # Validate presence of image data
        if not image_data:
            return {'statusCode': 400, 'body': json.dumps({'err_no': 1, 'err_msg': 'image_url or image (file) is required'})}

        # Perform OCR and respond
        result = ocr_prediction(image_data, search_text)
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        # Error handling
        return {'statusCode': 500, 'body': json.dumps({'err_no': 2, 'err_msg': f"Unexpected Error. Data type: {type(event)}. Error: {str(e)}"})}
