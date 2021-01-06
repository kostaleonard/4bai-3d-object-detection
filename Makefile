train_image_path=data/KITTI/data_object_image_2/training/image_2/
train_label_path=data/KITTI/training/label_2/
train_2d_boxes_path=unused
test_image_path=data/KITTI/data_object_image_2/testing/image_2/
model_path=model/model-1
train_output_file_path=data/predictions/3d_boxes/train/
val_output_file_path=data/predictions/3d_boxes/val/
test_output_file_path=data/predictions/3d_boxes/test/
yolo_train_predictions_path=data/predictions/2d_boxes/train/
yolo_val_predictions_path=data/predictions/2d_boxes/val/
yolo_test_predictions_path=data/predictions/2d_boxes/test/

all: train

vgg:
	wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	tar zxvf vgg_16_2016_08_28.tar.gz

train:
	python src/main/main.py --mode train --gpu -1 --image $(train_image_path) --label $(train_label_path) --box2d $(train_2d_boxes_path)

predict_on_train:
	python src/main/main.py --mode test --gpu -1 --image $(train_image_path) --box2d $(train_label_path) --model $(model_path) --output $(train_output_file_path)

predict_on_yolo_train:
	python src/main/main.py --mode test --gpu -1 --image $(train_image_path) --box2d $(yolo_train_predictions_path) --model $(model_path) --output $(train_output_file_path)

predict_on_yolo_val:
	python src/main/main.py --mode test --gpu -1 --image $(train_image_path) --box2d $(yolo_val_predictions_path) --model $(model_path) --output $(val_output_file_path)

predict_on_yolo_test:
	python src/main/main.py --mode test --gpu -1 --image $(train_image_path) --box2d $(yolo_test_predictions_path) --model $(model_path) --output $(test_output_file_path)

eval_train_predictions:
	g++ -O3 -DNDEBUG -o ./kitti_eval/evaluate_object_3d_offline ./kitti_eval/evaluate_object_3d_offline.cpp
	./kitti_eval/evaluate_object_3d_offline $(train_label_path) $(train_output_file_path)

eval_val_predictions:
	g++ -O3 -DNDEBUG -o ./kitti_eval/evaluate_object_3d_offline ./kitti_eval/evaluate_object_3d_offline.cpp
	./kitti_eval/evaluate_object_3d_offline $(train_label_path) $(val_output_file_path)

eval_test_predictions:
	g++ -O3 -DNDEBUG -o ./kitti_eval/evaluate_object_3d_offline ./kitti_eval/evaluate_object_3d_offline.cpp
	./kitti_eval/evaluate_object_3d_offline $(train_label_path) $(test_output_file_path)
