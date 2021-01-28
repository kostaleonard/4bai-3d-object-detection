KITTI_EXEC_DIR="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/models/research"
PIPELINE_CONFIG_PATH="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/Exp1/kitti500v2_aws_ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config"
KITTI_EVAL_SCRIPT="object_detection/model_main_tf2.py"
MODEL_DIR="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/Exp1/kitti500_training"
CHECKPOINT_DIR="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/Exp1/kitti500_training"

cd ${KITTI_EXEC_DIR}

python ${KITTI_EVAL_SCRIPT} \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --model_dir="${MODEL_DIR}" \
    --checkpoint_dir="${CHECKPOINT_DIR}" \
    --alsologtostderr

