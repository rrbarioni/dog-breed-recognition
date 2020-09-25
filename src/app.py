import io
import os

from PIL import Image

from flask import Flask, jsonify, request, render_template, redirect, url_for
app = Flask(__name__)

from enroll import Enroller

enroller = Enroller(
    initial_enroll_path=os.path.join('..', 'models', 'initial_enroll.pkl'),
    model_ckpt_path=os.path.join('..', 'models', 'embedder.pth'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_dog_breed_classification():
    if request.method == 'POST':
        if request.form['action'] == 'Classify':
            img_file = request.files['file']
            img = Image.open(io.BytesIO(img_file.read()))
            dog_breed = enroller.get_classification(img)

            return render_template('index_dog_breed_classify.html',
                data={ 'dog_breed': dog_breed })

        elif request.form['action'] == 'Enroll':
            dog_breed = request.form['fname']

            imgs_files = request.files.getlist('files')
            imgs = [Image.open(io.BytesIO(img_file.read()))
                for img_file in imgs_file]

            

            return render_template('index_new_dog_breed_add.html',
                data={ 'dog_breed': dog_breed })

if __name__ == "__main__":
    app.run()
