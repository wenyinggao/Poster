def lambda_handler(event, context):
    """
    your codes
    """
    return {
            "statusCode": 200,
            "body": "test",
            "headers": {
                "Content-Type": "application/json"
            }
    }