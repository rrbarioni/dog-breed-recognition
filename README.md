# dog-breed-recognition
An algorithm for the recognition of dog breeds from RGB images.

## Dog Breed Classification
This is a **ResNet50-based Convolutional Neural Network (CNN)** to classify dogs among 100 breeds. Given an image of a dog, the algorithm is responsible for predicting its breed.

### Instructions

#### Training the classifier

It is important to mention that, as I do not have a machine with GPU, I have used Google Colab for this purpose. As so, here is what I did:
 
- Created a directory in Google Drive (let's call it *dog-breed-recognition/*);
- Created a directory inside *dog-breed-recognition/* called *src/* (so it should be *dog-breed-recognition/src/*), in order to place the training code there;
- Inside *dog-breed-recognition/src/*, I have placed my training classifier notebook code (*src/classifier.ipynb*) there;
- Created a directory inside *dog-breed-recognition/* called *models/* (so it should be *dog-breed-recognition/models/*), in order to place the trained model there;
- Downloaded the dog breed dataset [here](https://drive.google.com/file/d/1DAyRYzZ9B-Nz5hLL9XIm3S3kDI5FBJH0/view);
- Unzipped the dataset (now locally having a directory called *dogs/*);
- Uploaded the dataset *dogs/train* into Google Drive (at directory *dog-breed-recognition/dogs/train*).
- Now, *dog-breed-recognition/* at Google Drive should be like this:
````
|-- dog-breed-recognition/
    |-- dogs/
        |-- train/
            |-- n02085620-Chihuahua/
            |-- n02085782-Japanese_spaniel/
                        ⋮
            |-- n02115913-dhole/
    |-- src/
        |-- classifier.ipynb
    |-- models/
````
- Now, follow the instructions at *dog-breed-recognition/src/classifier.ipynb* to generate the classifier, which will be stored at *dog-breed-recognition/models/classifier.pth*.

#### Using the classifier

- After cloning the repository locally (let's call it *root/*), create a directory called *models/* inside of it (so it should be *root/models/*) and place the generated classifier from Google Colab (*dog-breed-recognition/models/classifier.pth*) there.

For usage on a input image, you may refer to *root/src/classification.py*; from an image of a dog, the classifier outputs a list of n-mostlike dog breeds, as well as their confidence values.

![chihuahua](doc/n02085620_242_pred.jpg)