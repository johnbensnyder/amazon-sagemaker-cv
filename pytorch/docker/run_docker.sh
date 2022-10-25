CONTAINER_NAME=mlperf
REPO=${1-ecr-pt-repo}
TAG=pytorch-mlperf
REGION=${2-us-east-1}
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`
IMAGE_NAME=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}

docker run -d -it --rm --gpus all --name ${CONTAINER_NAME} \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    -w /opt/ml/code \
    -v /home/ubuntu/data:/opt/ml/input/data \
    -v /home/ubuntu/amazon-sagemaker-cv/pytorch/sm_src:/opt/ml/code \
    ${IMAGE_NAME}
