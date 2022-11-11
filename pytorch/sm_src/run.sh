# DATA_DIR=/opt/ml/input/data/
# 
# mkdir -p ${DATA_DIR}
# 
# aws s3 cp s3://jbsnyder-sagemaker-us-east/data/coco/archive/coco.tar ${DATA_DIR}/coco.tar
# 
# tar -xf ${DATA_DIR}/coco.tar -C ${DATA_DIR}

NUM_GPUS=8

torchrun \
    --nproc_per_node $NUM_GPUS \
    train.py \
    --config-file "e2e_mask_rcnn_R_50_FPN_1x_local.yaml"
