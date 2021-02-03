import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_image(image_file):
    return mpimg.imread(image_file)


def predict_on_image(image_file):
    image = read_image(image_file)
    print(type(image))
    print(image.shape)
