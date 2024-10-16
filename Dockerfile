FROM public.ecr.aws/lambda/python:3.10
RUN yum install -y mesa-libGL mesa-libEGL
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY my_lambda_function.py ./
CMD ["my_lambda_function.lambda_handler"]