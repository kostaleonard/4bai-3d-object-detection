train_image_path=data/KITTI/data_object_image_2/training/image_2/
train_label_path=data/KITTI/training/label_2/
train_2d_boxes_path=unused
test_image_path=data/KITTI/data_object_image_2/testing/image_2/
test_2d_boxes_path=data/KITTI/testing/label_2/
model_path=model/model
output_file_path=predictions/

all: train

vgg:
	wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	tar zxvf vgg_16_2016_08_28.tar.gz

train:
	python src/main/main.py --mode train --gpu -1 --image $(train_image_path) --label $(train_label_path) --box2d $(train_2d_boxes_path)

test:
	python src/main/main.py --mode test --gpu -1 --image $(test_image_path) --box2d $(test_2d_boxes_path) --model $(model_path) --output $(output_file_path)
