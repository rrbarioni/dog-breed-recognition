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
        # Check if the user uploaded a dog image for classifying its dog breed
        if request.form['action'] == 'Classify':

            # Get uploaded image file
            img_file = request.files['file']

            # Convert file bytes to a image
            img = Image.open(io.BytesIO(img_file.read()))

            # Predict dog breed
            dog_breed = enroller.get_classification(img=img)

            # Redirect to the page showing which dog breed it is
            return render_template('index_dog_breed_classify.html',
                data={ 'dog_breed': dog_breed })

        # Check if the user wants to enroll a new dog breed by uploading a list
        #   of images and a dog breed name
        elif request.form['action'] == 'Enroll':

            # Get uploaded dog breed name
            dog_breed = request.form['fname']

            # Get uploaded list of dog image files
            imgs_files = request.files.getlist('files')

            # Convert file bytes to a list of images
            imgs = [Image.open(io.BytesIO(img_file.read()))
                for img_file in imgs_files]

            # Enroll new class
            enroller.enroll_new_class_from_imgs_in_batches(
                imgs=imgs, class_name=dog_breed)

            # Redirect to the page showing that a new dog breed was added
            return render_template('index_new_dog_breed_add.html',
                data={ 'dog_breed': dog_breed })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True,
        help='Batch size for enrolling new dog breeds')

    args = parser.parse_args()
    batch_size = args.batch_size

    # When the service is on, instantiate the dog breed classifier/enroller
    #   object
    enroller = Enroller(
        model_ckpt_path=os.path.join('..', 'models', 'embedder.pth'),
        initial_enroll_path=os.path.join('..', 'models', 'initial_enroll.pkl'),
        batch_size=batch_size)

    app.run()
