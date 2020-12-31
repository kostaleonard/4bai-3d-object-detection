train_image_path=data/KITTI/data_object_image_2/training/image_2/
train_label_path=data/KITTI/training/label_2/
train_2d_boxes_path=unused
test_image_path=data/KITTI/data_object_image_2/testing/image_2/
model_path=model/model-1
train_output_file_path=predictions_train/
val_output_file_path=predictions_val/

all: train

vgg:
	wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	tar zxvf vgg_16_2016_08_28.tar.gz

train:
	python src/main/main.py --mode train --gpu -1 --image $(train_image_path) --label $(train_label_path) --box2d $(train_2d_boxes_path)

predict_on_train:
	python src/main/main.py --mode test --gpu -1 --image $(train_image_path) --box2d $(train_label_path) --model $(model_path) --output $(output_file_path)
