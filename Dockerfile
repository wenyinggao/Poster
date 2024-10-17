FROM public.ecr.aws/lambda/python:3.10
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY lambda_main.py ./
CMD ["lambda_main.lambda_handler"]