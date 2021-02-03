from flask import request, render_template
from flask import current_app as app

from .inference import predict_on_image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print('TODO inference')
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        file = request.files['file']
        print(type(file))
        predict_on_image(file)
        return render_template('index.html')
