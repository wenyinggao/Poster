import os
import json
import requests
import regex as re
import base64
import numpy as np
import cv2
import argparse
from typing import List
from paddleocr import PaddleOCR

# Set environment variable for PaddleOCR model directory
os.environ['PADDLEOCR_MODEL_DIR'] = '/var/task/local_package/.paddleocr'
os.environ['PADDLEOCR_BASE_URL'] = '/var/task/local_package/.paddleocr'

print("PaddleOCR model directory:", os.getenv('PADDLEOCR_MODEL_DIR'))
# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir='~/projects/OCR/.paddleocr/det',
    rec_model_dir='~/projects/OCR/.paddleocr/rec',
    cls_model_dir='~/projects/OCR/.paddleocr/cls',
    enable_mkldnn=False
)

def decode_image(image_path_or_url: str):
    """Load image from file or URL."""
    try:
        if image_path_or_url.startswith("http"):  # URL case
            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image_data = response.content
        else:  # Local file case
            with open(image_path_or_url, "rb") as f:
                image_data = f.read()
        
        np_arr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path_or_url}. Error: {str(e)}")

def remove_invisible_chars(text):
    """Remove invisible characters from text."""
    try:
        return re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
    except Exception as e:
        raise ValueError(f"Failed to clean text. Error: {str(e)}")

def ocr_prediction(image_path_or_url: str) -> dict:
    """Perform OCR on the image and return all detected text."""
    try:
        image = decode_image(image_path_or_url)
        result = ocr.ocr(image, cls=True)

        output: List[dict] = []
        for line in result:
            for res in line:
                text, confidence, bbox = res[1][0], res[1][1], res[0]
                cleaned_text = remove_invisible_chars(text)
                output.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox
                })

        return {'err_no': 0, 'err_msg': 'Success', 'data': output}
    except Exception as e:
        return {'err_no': 2, 'err_msg': f"OCR Prediction Error. Error: {str(e)}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR using PaddleOCR")
    parser.add_argument("--image", required=True, help="Path to image file or URL")
    args = parser.parse_args()
    
    result = ocr_prediction(args.image)
    print(json.dumps(result, indent=4))
