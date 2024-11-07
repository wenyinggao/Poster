# Use the AWS Lambda Python 3.10 runtime as the base image
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies needed for PaddleOCR, Tesseract, OpenCV, and tar
RUN yum install -y \
    gcc \
    libgomp \
    libxml2 \
    libxslt \
    tesseract \
    tesseract-langpack-eng \
    opencv \
    tar \
    && yum clean all

# Set environment variable for PaddleOCR model directory
ENV PADDLEOCR_MODEL_DIR=/tmp/.paddleocr

# Pre-download the PaddleOCR models to /tmp/.paddleocr
RUN mkdir -p /tmp/.paddleocr/det /tmp/.paddleocr/rec /tmp/.paddleocr/cls \
    && curl -o /tmp/.paddleocr/det/en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar \
    && curl -o /tmp/.paddleocr/rec/en_PP-OCRv3_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar \
    && curl -o /tmp/.paddleocr/cls/ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar \
    && tar -xf /tmp/.paddleocr/det/en_PP-OCRv3_det_infer.tar -C /tmp/.paddleocr/det \
    && tar -xf /tmp/.paddleocr/rec/en_PP-OCRv3_rec_infer.tar -C /tmp/.paddleocr/rec \
    && tar -xf /tmp/.paddleocr/cls/ch_ppocr_mobile_v2.0_cls_infer.tar -C /tmp/.paddleocr/cls

# Create and set the working directory
WORKDIR /var/task

# Copy the local Python packages (wheels) and requirements.txt file
COPY local_package /var/task/local_package
COPY requirements.txt /var/task/requirements.txt

# Install Python dependencies from local_package, using --no-index to avoid PyPI
RUN pip install --no-index --find-links=/var/task/local_package -r /var/task/requirements.txt

# Copy the Lambda handler and any additional scripts to the working directory
COPY lambda_main.py /var/task/lambda_main.py

# Define the Lambda function handler
CMD ["lambda_main.lambda_handler"]
