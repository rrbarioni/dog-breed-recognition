import io
import os
import argparse

from PIL import Image

from flask import Flask, jsonify, request, render_template, redirect, url_for
app = Flask(__name__)

from enroll import Enroller

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_dog_breed_classification():
    if request.method == 'POST':
        if request.form['action'] == 'Classify':
            img_file = request.files['file']
            img = Image.open(io.BytesIO(img_file.read()))
            dog_breed = enroller.get_classification(img=img)

            return render_template('index_dog_breed_classify.html',
                data={ 'dog_breed': dog_breed })

        elif request.form['action'] == 'Enroll':
            dog_breed = request.form['fname']

            imgs_files = request.files.getlist('files')
            imgs = [Image.open(io.BytesIO(img_file.read()))
                for img_file in imgs_files]

            enroller.enroll_new_class_from_imgs_in_batches(
                imgs=imgs, class_name=dog_breed)

            return render_template('index_new_dog_breed_add.html',
                data={ 'dog_breed': dog_breed })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True,
        help='Batch size for enrolling new dog breeds')

    args = parser.parse_args()
    batch_size = args.batch_size

    enroller = Enroller(
        model_ckpt_path=os.path.join('..', 'models', 'embedder.pth'),
        initial_enroll_path=os.path.join('..', 'models', 'initial_enroll.pkl'),
        batch_size=batch_size)

    app.run()
