import json
import base64

def encode_image(image_path):
    """Read and encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_lambda_handler():
    from lambda_main import lambda_handler  # Replace with actual module name
    
    test_image_path = "images/testocr.png"  # Replace with an actual image path
    
    base64_image = encode_image(test_image_path)
    
    # Test JSON input with base64 image
    event_json_base64 = {
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "image": base64_image,
            "search_text": "the"
        }),
        "isBase64Encoded": False
    }
    
    # Run test
    print("Testing JSON with Base64 Image...")
    response = lambda_handler(event_json_base64, None)
    print(response)
    
if __name__ == "__main__":
    test_lambda_handler()
