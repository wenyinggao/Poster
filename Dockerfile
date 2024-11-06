# Use a Python base image compatible with AWS Lambda
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies, including libgomp
RUN yum install -y \
    gcc \
    libgomp \
    libxml2 \
    libxslt \
    tesseract \
    tesseract-langpack-eng \
    opencv \
    && yum clean all

# Create a working directory
WORKDIR /var/task

# Copy downloaded packages for local installation into the Docker image
COPY packages /packages
COPY requirements.txt ./

# Install specific libraries (paddleocr, paddlepaddle) from local directory
RUN pip install --find-links=/packages paddleocr paddlepaddle

# Install remaining libraries from requirements.txt
RUN pip install -r requirements.txt

# Copy the Lambda handler or test script into the working directory
COPY test.py ./

# Specify the Lambda function handler
CMD ["test.lambda_handler"]
