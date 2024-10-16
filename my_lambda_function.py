import os
import cv2
import numpy as np
import base64
import logging
import time
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from zipfile import ZipFile
import json
import concurrent.futures

# Explicitly set the device to CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)

def load_image_from_bytes(file_content):
    """Load and return the image as a NumPy array from bytes."""
    img = Image.open(BytesIO(file_content)).convert("RGB")
    return np.array(img)

def calculate_ssim_batch(img1_batch, img2_batch):
    """Calculate SSIM between batches of images using PyTorch."""
    ssim_values = F.mse_loss(img1_batch, img2_batch)
    return 100 - (ssim_values.item() * 100)  # Convert to a percentage scale

def process_images_batch(file1, file2_batch, structure_weight, color_weight, aliasing_weight):
    try:
        # Load the reference image and batch of images to compare
        img1 = load_image_from_bytes(file1)
        img2_list = [load_image_from_bytes(f) for f in file2_batch]

        # Resize all images to 512x512
        img1 = cv2.resize(img1, (512, 512))
        img2_list = [cv2.resize(img, (512, 512)) for img in img2_list]

        # Convert images to tensors for batch processing
        transform = T.ToTensor()

        img1_tensor = transform(img1).unsqueeze(0).to(device)  # Add batch dimension and move to CPU
        img2_tensor_batch = torch.stack([transform(img).to(device) for img in img2_list])

        # Calculate SSIM for the batch
        struct_scores = []
        for img2_tensor in img2_tensor_batch:
            img2_tensor = img2_tensor.unsqueeze(0)  # Add batch dimension
            struct_score = calculate_ssim_batch(img1_tensor, img2_tensor)
            struct_scores.append(struct_score)

        # Calculate color and aliasing scores for each image in the batch
        results = []
        for img2, struct_score in zip(img2_list, struct_scores):
            color_diff = np.mean((img1 - img2) ** 2)
            color_score = max(0, 100 - (color_diff / (255 ** 2)) * 100)

            gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            sobelx1 = cv2.Sobel(gray_img1, cv2.CV_64F, 1, 0, ksize=5)
            sobelx2 = cv2.Sobel(gray_img2, cv2.CV_64F, 1, 0, ksize=5)
            sobely1 = cv2.Sobel(gray_img1, cv2.CV_64F, 0, 1, ksize=5)
            sobely2 = cv2.Sobel(gray_img2, cv2.CV_64F, 0, 1, ksize=5)

            aliasing_diff_x = np.abs(sobelx1 - sobelx2)
            aliasing_diff_y = np.abs(sobely1 - sobely2)
            aliasing_diff_map = (aliasing_diff_x + aliasing_diff_y) / 2
            aliasing_diff_map = aliasing_diff_map / (np.max(aliasing_diff_map) + 1e-8)

            aliasing_score = 100 - (np.mean(aliasing_diff_map) * 100)
            aliasing_score = max(0, min(100, aliasing_score))

            total_weight = structure_weight + color_weight + aliasing_weight
            weighted_score = (
                (struct_score * structure_weight) +
                (color_score * color_weight) +
                (aliasing_score * aliasing_weight)
            ) / total_weight

            results.append({
                "structure_score": struct_score,
                "color_score": color_score,
                "aliasing_score": aliasing_score,
                "weighted_score": weighted_score
            })

        return results

    finally:
        pass

def lambda_handler(event, context):
    try:
        # Decode base64 files
        reference_image_content = base64.b64decode(event['body']['reference_image'])
        zip_file_content = base64.b64decode(event['body']['zip_file'])

        # Retrieve weights from the event, with defaults
        structure_weight = event['body'].get('structure_weight', 1.4)
        color_weight = event['body'].get('color_weight', 1.2)
        aliasing_weight = event['body'].get('aliasing_weight', 1.0)

        # Open the ZIP file directly without extracting
        with ZipFile(BytesIO(zip_file_content)) as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]

            file2_contents = []

            # Read all images into memory
            for image_file in image_files:
                with zip_ref.open(image_file) as img:
                    file2_contents.append(img.read())

            # Process images in batches using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(
                    lambda f: process_images_batch(reference_image_content, [f], structure_weight, color_weight, aliasing_weight),
                    file2_contents
                ))

        # Prepare the response
        response = []
        for image_file, result in zip(image_files, batch_results):
            response.append({
                "image": image_file,
                **result[0]  # Since we processed one image at a time in the batch
            })

        return {
            "statusCode": 200,
            "body": json.dumps(response),
            "headers": {
                "Content-Type": "application/json"
            }
        }

    except Exception as e:
        logging.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }

if __name__ == "__main__":
    print("test")
