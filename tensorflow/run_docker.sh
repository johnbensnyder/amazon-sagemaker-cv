REGION=us-east-1
AWS_DLC_ACCOUNT=763104351884
DLC_TYPE=tensorflow-training
TAG=2.10.0-gpu-py39-cu112-ubuntu20.04-ec2
CONTAINER_NAME=mlperf
IMAGE_NAME=${AWS_DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${DLC_TYPE}:${TAG}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker run -d -it --rm --gpus all --name ${CONTAINER_NAME} \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    -w /opt/ml/code/tensorflow \
    -v /home/ubuntu/data:/opt/ml/input/data \
    -v /home/ubuntu/amazon-sagemaker-cv:/opt/ml/code \
    ${IMAGE_NAME}