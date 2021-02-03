from flask import request, render_template
from flask import current_app as app

from .inference import predict_on_image_3d


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print('TODO inference')
        if 'image_file' not in request.files:
            print('Image file not uploaded')
            return
        if 'cam_config_file' not in request.files:
            print('Camera configuration file not uploaded')
            return
        image_file = request.files['image_file']
        config_file = request.files['cam_config_file']
        predict_on_image_3d(image_file, config_file)
        return render_template('result.html')
