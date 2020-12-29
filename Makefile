train_image_path=data/KITTI/data_object_image_2/training/image_2/
train_label_path=data/KITTI/training/label_2/
train_2d_boxes=unused

all: train

vgg:
	wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	tar zxvf vgg_16_2016_08_28.tar.gz

train:
	python src/main/main.py --mode train --gpu -1 --image $(train_image_path) --label $(train_label_path) --box2d $(train_2d_boxes)
