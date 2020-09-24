# dog-breed-recognition
An algorithm for the recognition of dog breeds from RGB images.

## Dog Embeddings Extraction and Breed Enrolling
This is a two-step algorithm for extract embeddings from images and dynamically add new unknown dog breeds for recognition. Given an image of a dog, the algorithm is responsible for extracting embeddings by using a **ResNet50-based Convolutional Neural Network (CNN)**. Also, a *K-Nearest Neighbors (KNN)* classifier is applied to separate different dog breeds by their embeddings.

### Instructions

#### Training the embedding extractor

It is important to mention that, as I do not have a machine with GPU, I have used Google Colab for this purpose. As so, here is what I did:
 
- Created a directory in Google Drive (let's call it *dog-breed-recognition/*);
- Created a directory inside *dog-breed-recognition/* called *src/* (so it should be *dog-breed-recognition/src/*), in order to place the training code there;
- Inside *dog-breed-recognition/src/*, I have placed my training embedding extractor notebook code (*src/embedder.ipynb*) there;
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
        |-- embedder.ipynb
    |-- models/
````
- Now, follow the instructions at *dog-breed-recognition/src/embedder.ipynb* to generate the embeddings extractor, which will be stored at *dog-breed-recognition/models/embedder.pth*.

#### Generating embeddings from a initial set of dog breeds

This step is also performed at Google Colab, once it aims to extract embeddings of dog images of all dog breed from the training set. Here is what I did:

- Similarly to the embedding extractor training, I have placed my initial embedding extraction notebook code (*src/initial_enroller.ipynb*) in Google Drive, at *dog-breed-recognition/src/*;
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
        |-- embedder.ipynb
        |-- initial_enroller.ipynb
    |-- models/
````
- Now, follow the instructions at *dog-breed-recognition/src/initial_enroller.ipynb* to generate embeddings from the training set, which will be stored at *dog-breed-recognition/models/initial_enroll.pkl*.

#### Using the dog breed enroller

- After cloning the repository locally (let's call it *root/*), create a directory called *models/* inside of it (so it should be *root/models/*) and place the generated embedder from Google Colab (*dog-breed-recognition/models/embedder.pth*) there. Also, place the generated training set embeddings from Google Colab (*dog-breed-recognition/models/initial_enroll.pkl*) there.

For usage on an input new dog breed, you may refer to *root/src/enroll.py*; from a directory containing images from the same dog breed, the enroller dynamically adds this dog breed to a classifier, in order to be recognized when inputting new dog image samples of that dog breed.