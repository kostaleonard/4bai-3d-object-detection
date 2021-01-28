KITTI_EXEC_DIR="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/models/research"
KITTI_RECORD_SCRIPT="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/models/research/object_detection/dataset_tools/create_kitti_tf_record.py"
DATA_DIR="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/kitti_data_500"
OUTPUT_PATH="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection"
LABEL_MAP_PATH="/home/ubuntu/4bai/4bai-3d-object-detection/2d-object-detection/Exp1/kitti_label_map.pbtxt"
VALIDATION_SIZE=30

cd ${KITTI_EXEC_DIR}

python "${KITTI_RECORD_SCRIPT}" \
--data_dir="${DATA_DIR}" \
--output_path="${OUTPUT_PATH}/kitti500.record" \
--label_map_path="${LABEL_MAP_PATH}" \
--classes_to_use='car,pedestrian,dontcare' \
--validation_set_size=$VALIDATION_SIZE


