from flask import request, render_template
from flask import current_app as app

#from .inference import get_category, plot_category


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print('TODO inference')
        # TODO inference
        return render_template('index.html')
