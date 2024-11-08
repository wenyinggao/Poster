# Use the AWS Lambda Python 3.10 runtime as the base image
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies needed for PaddleOCR, Tesseract, and OpenCV
RUN yum install -y \
    gcc \
    libgomp \
    libxml2 \
    libxslt \
    tesseract \
    tesseract-langpack-eng \
    opencv \
    && yum clean all

# Set environment variable for PaddleOCR model directory to the local_package path
ENV PADDLEOCR_MODEL_DIR=/var/task/local_package/.paddleocr

# Create and set the working directory
WORKDIR /var/task

# Copy the local Python packages (wheels) and requirements.txt file, including the PaddleOCR models
COPY local_package /var/task/local_package
COPY requirements.txt /var/task/requirements.txt

# Install Python dependencies from local_package
RUN pip install --no-index --find-links=/var/task/local_package -r /var/task/requirements.txt

# Copy the Lambda handler and any additional scripts to the working directory
COPY lambda_main.py /var/task/lambda_main.py

# Define the Lambda function handler
CMD ["lambda_main.lambda_handler"]
