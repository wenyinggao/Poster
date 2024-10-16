# devaida-Artisk_lambda_docker_template

## 1.Environment

### 1.1 Install docker
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### 1.2 Install AWS CLI
```bash
sudo apt update
sudo apt install awscli
```

### 1.3 Configure AWS CLI
```bash
aws configure
```
- AWS Access Key ID: Obtain from your AWS IAM user.
- AWS Secret Access Key: Obtain from your AWS IAM user.
- Default Region Name: e.g., us-east-1.
- Default Output Format: e.g., json.

## 2. Write lambda function
- Install relative packages for your lambda function
- fill my_lambda_function.lambda_handler. This is the function to receive input and return output from AWS lambda
- export your environment
```bash
pip freeze > requirements.txt
```

## 3. Modify Dockerfile
- You can modify the file path, upload multiple scripts, install system package by modifying Dockerfile

## 4. Create an ECR Repository
```bash
aws ecr create-repository --repository-name my-lambda-repo --region <your-region>
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com
```

## 5. Build and Push Docker Image to ECR
```bash
docker build -t my-lambda-function .
docker tag my-lambda-function:latest <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest
docker push <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest
```

## 6. Create Lambda Function
- Log in to AWS console -> Lambda -> Functions
- Click Create function on top right, choose container image, browse the image just uploaded and modify the settings

## 7. Test Lambda Function
- You can write test case in the function page or setup an URL in lambda configuration and test with it

## 8. Function URL
- In Lambda -> Functions -> <your_function> -> Configuration -> Function URL, you can create a URL. If you choose AWS_IAM, Only authenticated IAM users and roles can make requests to the function URL. If you choose NONE for Auth type. It will be a public URL which everyone is able to send and receive data.
