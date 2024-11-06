from paddleocr import PaddleOCR

def download_models():
    # Set your desired language here, e.g., 'en'
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("PaddleOCR models downloaded successfully.")

if __name__ == "__main__":
    download_models()