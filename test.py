import json
import paddleocr
import paddlepaddle as paddle

def lambda_handler(event, context):
    """
    Lambda function handler to test the availability of paddleocr and paddlepaddle.
    This function attempts to import these libraries and prints out their versions.
    """

    try:
        # Instantiate OCR to confirm paddleocr availability
        ocr = paddleocr.OCR()
        paddle_version = paddle.__version__
        ocr_version = paddleocr.__version__
        
        # Return success response with library versions
        result = {
            "status": "success",
            "paddle_version": paddle_version,
            "paddleocr_version": ocr_version,
            "message": "Libraries are available and imported successfully."
        }
        print("Lambda Test Result:", json.dumps(result))
        return result

    except Exception as e:
        # Return error response in case of failure
        result = {
            "status": "error",
            "message": str(e)
        }
        print("Lambda Test Result:", json.dumps(result))
        return result
