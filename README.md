# aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 533267421480.dkr.ecr.us-east-2.amazonaws.com
# docker build -t ocr .
# docker tag ocr:latest 533267421480.dkr.ecr.us-east-2.amazonaws.com/ocr:latest
# docker push 533267421480.dkr.ecr.us-east-2.amazonaws.com/ocr:latest
#
#
#
# pip download -r requirements.txt -d local_package
# pip install --no-index --find-links=local_package -r requirements.txt
#