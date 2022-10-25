CONTAINER_NAME=mlperf
IMAGE_NAME=920076894685.dkr.ecr.us-east-1.amazonaws.com/jbsnyder:pytorch-mlperf
docker run -d -it --rm --gpus all --name ${CONTAINER_NAME} \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    -w /opt/ml/code \
    -v /home/ubuntu/data:/opt/ml/input/data \
    -v /home/ubuntu/amazon-sagemaker-cv/pytorch/sm_src:/opt/ml/code \
    ${IMAGE_NAME}
