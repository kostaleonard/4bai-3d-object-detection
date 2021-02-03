import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
import numpy as np
import cv2
from .dataset import display_image_3d_boxes


def orientation_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """Returns the orientation loss from the predictions.
    :param y_true: The true orientation, a tensor of shape (None, 2, 2).
    :param y_pred: The predicted orientation of the same shape.
    :return: The orientation loss as a tensor.
    """
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    loss = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * \
            y_pred[:, :, 1])
    loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors
    return tf.reduce_mean(loss)


def get_model_3d():
    return tf.keras.models.load_model(MODEL_FILE, custom_objects={'orientation_loss': orientation_loss})


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(CURR_DIR, 'static/model/model_30_epochs_31JAN.h5')
MODEL_3D = get_model_3d()
BIN = 2
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']
dims_avg = {'Cyclist': np.array([1.73532436, 0.58028152, 1.77413709]),
            'Van': np.array([2.18928571, 1.90979592, 5.07087755]),
            'Tram': np.array([3.56092896, 2.39601093, 18.34125683]),
            'Car': np.array([1.52159147, 1.64443089, 3.85813679]),
            'Pedestrian': np.array([1.75554637, 0.66860882, 0.87623049]),
            'Truck': np.array([3.07392252, 2.63079903, 11.2190799])}


def read_image(image_file):
    return mpimg.imread(image_file)


def predict_on_image_3d(image_file, cam_config_file):
    npimg = np.fromfile(image_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_save_path = os.path.join(CURR_DIR, 'static/images/input.png')
    prediction_save_path = os.path.join(CURR_DIR, 'static/images/output.png')
    cam_config_outfile = os.path.join(CURR_DIR, 'static/config/cam.txt')
    box3d_file = os.path.join(CURR_DIR, 'static/box3d/box3d.txt')
    # TODO run the model on the image.
    #img = cv2.imread(image_file)
    img = img.astype(np.float32, copy=False) / 255.
    with open(cam_config_outfile, 'wb') as outfile:
        outfile.write(cam_config_file.read())
    with open(cam_config_outfile, 'r') as infile:
        config = infile.readlines()
    print(config)
    print('truncating')
    config = config[7:]
    print(config)
    with open(box3d_file, 'w') as box3d:
        for line in config:
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            obj = {'xmin': int(float(line[4])),
                   'ymin': int(float(line[5])),
                   'xmax': int(float(line[6])),
                   'ymax': int(float(line[7])),
                   }

            patch = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            patch = cv2.resize(patch, (NORM_H, NORM_W))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)
            prediction = MODEL_3D.predict(patch)
            # Transform regressed angle
            max_anc = np.argmax(prediction[2][0])
            anchors = prediction[1][0][max_anc]

            if anchors[1] > 0:
                angle_offset = np.arccos(anchors[0])
            else:
                angle_offset = -np.arccos(anchors[0])

            wedge = 2. * np.pi / BIN
            angle_offset = angle_offset + max_anc * wedge
            angle_offset = angle_offset % (2. * np.pi)

            angle_offset = angle_offset - np.pi / 2
            if angle_offset > np.pi:
                angle_offset = angle_offset - (2. * np.pi)

            line[3] = str(angle_offset)

            line[14] = angle_offset + np.arctan(
                float(line[11]) / float(line[13]))

            # Transform regressed dimension
            if line[0] in VEHICLES:
                dims = dims_avg[line[0]] + prediction[0][0]
            else:
                dims = dims_avg['Car'] + prediction[0][0]

            line = line[:8] + list(dims) + line[11:]

            # Write regressed 3D dim and oritent to file
            line = ' '.join([str(item) for item in line]) + ' ' + str(
                np.max(prediction[2][0])) + '\n'
            box3d.write(line)
    plt.imsave(image_save_path, img)
    with open(box3d_file, 'r') as infile:
        predicted_labels = [line.strip() for line in infile.readlines()]
    with open(cam_config_outfile, 'r') as infile:
        config = [line.strip() for line in infile.readlines()]
    display_image_3d_boxes(image_save_path,
                           predicted_labels,
                           config,
                           force_box_color='orange')
    plt.savefig(prediction_save_path)
