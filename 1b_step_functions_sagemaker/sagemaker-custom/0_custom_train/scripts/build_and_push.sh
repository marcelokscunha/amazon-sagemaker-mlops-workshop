ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

# 1. Create Python package and copy it to ../docker/code/
echo "Creating the Python package..."
cd ../package/ && python setup.py sdist && cp dist/custom_lightgbm_framework-1.0.0.tar.gz ../docker/code/

# 2. Build the Docker image and tag it
echo "Building docker image..."
docker build -f ../docker/Dockerfile -t $REPO_NAME ../docker
echo "Tagging docker image..."
docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# 3. Log in to ECR and create Docker image repository if there's none with that name
$(aws ecr get-login --no-include-email --registry-ids $ACCOUNT_ID)
aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

# 4. Push the image to our ECR repository
echo "Pushing docker image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
