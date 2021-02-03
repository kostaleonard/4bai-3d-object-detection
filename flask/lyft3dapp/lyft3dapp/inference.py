import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#model_3d = tf.keras.models.load_model(model, custom_objects={'orientation_loss': orientation_loss})


def read_image(image_file):
    return mpimg.imread(image_file)


def predict_on_image_3d(image_file, cam_config_file):
    image = read_image(image_file)
    print(type(image))
    print(image.shape)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root_dir + '/static/images/output.png')
    plt.imsave(path, image)
