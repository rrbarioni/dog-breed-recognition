import os
import tqdm
from six.moves import cPickle as pickle

from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from embeddings import EmbeddingsExtractor

class Enroller:
    """
    A class to address new dog breeds dynamically.
    
    Attributes
    ----------
    embeddings_extractor : embeddings.EmbeddingsExtractor
        object for extracting embeddings from RGB images of dogs
    embeddings list<np.array>
        list of current extracted embeddings
    labels : list<int>
        list of current extracted corresponding dog breed indexes
    classes : list<str>
        list of current dog breeds
    curr_class_index : int
        index to which address the next dog breed to be added
    batch_size : int
        defines the number of images to be fed simultaneously to the model at
            the enrolling step
    classifier : sklearn.KNeighborsClassifier
        a classifier for predicting dog breeds from embeddings
    
    Methods
    -------
    create_classifier()
        Creates a new dog breed classifier from the current existing embeddings
          and corresponding labels.
    
    enroll_new_class_from_imgs(imgs, class_name)
        Dynamically adds a new dog breed to be applied to the classifier from a 
          list of dog images.
          
    enroll_new_class_from_path(new_class_path, class_name=None)
        Dynamically adds a new dog breed to be applied to the classifier from a
          path containing the dog images.
    
    get_classification(img)
        Predicts the dog breed from an image of a dog, considering the current
          dog breeds existent on the classifier.
    """
    
    def __init__(self, model_ckpt_path, initial_enroll_path=None, batch_size=1):
        '''
        Constructs all the attributes for the enroller object.

        Parameters
        ----------
        model_ckpt_path : str
            path to the file containing the results of the trained model
        initial_enroll_path : str
            path to the file containing the embeddings from the initial set of
              dog images
        batch_size : int
            defines the number of images to be fed simultaneously to the model
              at the enrolling step
        '''
        
        # Instantiate embeddings extractor
        self.embeddings_extractor = EmbeddingsExtractor(model_ckpt_path)
        
        # Initialize existing embeddings, labels and classes
        self.embeddings, self.labels, self.classes = [], [], []
        
        # If provided, get the initial embeddings, dog breed indexes and dog
        #   breeds names
        if initial_enroll_path is not None:
            with open(initial_enroll_path, 'rb') as f:
                initial_enroll = pickle.load(f)
                self.embeddings += initial_enroll['embeddings']
                self.labels += initial_enroll['labels']
                self.classes += initial_enroll['classes']
        
        # Instantiate the index for when enrolling a new dog breed
        self.curr_class_index = len(self.classes)
        
        self.batch_size = batch_size
        
        # Instantiate dog breed classifier from initial embeddings and dog
        #   breed indexes
        self.create_classifier()
        
    def create_classifier(self):
        '''
        Creates a new dog breed classifier from the current existing embeddings
          and corresponding labels.
        '''
        
        # Instantiate a KNN classifier. Also considering the distance criteria
        #   for prediction
        self.classifier = KNeighborsClassifier(n_neighbors=10,
            weights='distance')
        
        # "Training" the classifier from the current embeddings and dog breed
        #   indexes
        self.classifier.fit(self.embeddings, self.labels)
        
    def enroll_new_class_from_imgs_in_batches(self, imgs, class_name):
        '''
        Dynamically adds a new dog breed to be applied to the classifier from a 
          list of dog images.

        Parameters
        ----------
        imgs : list<PIL.Image>
            List of dog images of the current dog breed to be enrolled
        class_name : str
            Name of the dog breed
        '''
        
        # Add new dog breed to the list of dog breed names
        self.classes.append(class_name)
        
        # Divide list of dog images into batches (`imgs_batches`) in order to
        #   be more efficiently consumed by the embeddings extractor
        imgs_batches = [imgs[i:i + self.batch_size]
            for i in range(0, len(imgs), self.batch_size)]
        
        # Using tqdm to iteratively keep track on the number of iterated
        #   samples on the console
        imgs_batches = tqdm.tqdm(imgs_batches, position=0, leave=True)
        
        # Iterate through all batches of images of the current dog breed to be
        #   enrolled
        for imgs in imgs_batches:
            
            # Extract embeddings from the image batch (`curr_embeddings_list`)
            curr_embeddings_list = \
                self.embeddings_extractor.get_embeddings_batch(imgs)
            
            # Attach current extracted embeddings list to the list of all
            #   embeddings
            self.embeddings += list(curr_embeddings_list)
            
            # Instantiate the list of dog breed indexes for each sample of the
            #   current batch
            curr_class_index_list = [self.curr_class_index
                for i in range(len(imgs))]
            
            # Attach the dog breed index of the current instance to the list of
            #   all extracted dog breed indexes of the current batch
            self.labels += curr_class_index_list
            
        # After adding all instances of the current dog breed, increment the
        #   dog breed index for when enrolling another new dog breed
        self.curr_class_index += 1
        
        # Instantiate dog breed classifier from the currentembeddings and dog
        #   breed indexes
        self.create_classifier()
        
    def enroll_new_class_from_imgs(self, imgs, class_name):
        '''
        Dynamically adds a new dog breed to be applied to the classifier from a 
          list of dog images.

        Parameters
        ----------
        imgs : list<PIL.Image>
            List of dog images of the current dog breed to be enrolled
        class_name : str
            Name of the dog breed
        '''
        
        # Add new dog breed to the list of dog breed names
        self.classes.append(class_name)
        
        # Using tqdm to iteratively keep track on the number of iterated
        #   samples on the console
        imgs = tqdm.tqdm(imgs, position=0, leave=True)
        
        # Iterate through all images of the current dog breed to be enrolled
        for img in imgs:
            
            # Extract embeddings from the image (`curr_embeddings`)
            curr_embeddings = self.embeddings_extractor.get_embeddings(img)
            
            # Attach current extracted embeddings to the list of all embeddings
            self.embeddings.append(curr_embeddings)
            
            # Attach the dog breed index of the current instance to the list of
            #   all extracted dog breed indexes
            self.labels.append(self.curr_class_index)
            
        # After adding all instances of the current dog breed, increment the
        #   dog breed index for when enrolling another new dog breed
        self.curr_class_index += 1
        
        # Instantiate dog breed classifier from the currentembeddings and dog
        #   breed indexes
        self.create_classifier()
        
    def enroll_new_class_from_path(self, new_class_path, class_name=None):
        '''
        Dynamically adds a new dog breed to be applied to the classifier from a
          path containing the dog images.

        Parameters
        ----------
        new_class_path : str
            path containing the set of images corresponding to the new dog
              breed
        class_name : str
            name of the dog breed
        '''
        
        # If the name of the dog breed is not provided, derive it from
        #   `new_class_path`
        if class_name is None:
            class_name = new_class_path.split('\\')[-1]
        
        # Add new dog breed to the list of dog breed names
        self.classes.append(class_name)
        
        # `instances` contain the list of all filenames of dog image instances
        #   of the dog breed to be added
        instances = sorted(os.listdir(new_class_path))
        
        # Using tqdm to iteratively keep track on the number of iterated
        #   samples on the console
        instances = tqdm.tqdm(instances, position=0, leave=True)
        
        # Iterate through all instances of the class (`curr_instance`)
        for curr_instance in instances:
            
            # `instance_path` refers to the filepath of the current instance
            #   image
            instance_path = os.path.join(new_class_path, curr_instance)
            
            # Read image from the filepath (`img`)
            img = Image.open(instance_path)
            
            # Extract embeddings from the image (`curr_embeddings`)
            curr_embeddings = self.embeddings_extractor.get_embeddings(img)
            
            # Attach current extracted embeddings to the list of all embeddings
            self.embeddings.append(curr_embeddings)
            
            # Attach the dog breed index of the current instance to the list of
            #   all extracted dog breed indexes
            self.labels.append(self.curr_class_index)
        
        # After adding all instances of the current dog breed, increment the
        #   dog breed index for when enrolling another new dog breed
        self.curr_class_index += 1
        
        # Instantiate dog breed classifier from the currentembeddings and dog
        #   breed indexes
        self.create_classifier()
        
    def get_classification(self, img):
        '''
        Predicts the dog breed from an image of a dog, considering the current
          dog breeds existing on the classifier.

        Parameters
        ----------
        img : PIL.Image
            image of a dog

        Return
        ------
        dog_breed : str
            predicted dog breed
        '''
        
        # Extract embeddings from the image (`img_embeddings`)
        img_embeddings = self.embeddings_extractor.get_embeddings(img)
        
        # Predict the dog breed index (`label`) from the embeddings by using
        #   the KNN classifier (as the KNN prediction receives a batch of
        #   samples, `img_embeddings` is attached to a list and the first
        #   result of the batch is extracted)
        label = self.classifier.predict([img_embeddings])[0]
        
        # Get the dog breed name by its index (`dog_breed`)
        dog_breed = self.classes[label]
        
        return dog_breed
        
if __name__ == '__main__':
    model_ckpt_path = os.path.join('..', 'models', 'embedder.pth')
    initial_enroll_path = os.path.join('..', 'models', 'initial_enroll.pkl')
    
    enroller = Enroller(model_ckpt_path, initial_enroll_path)
    
    new_class_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02087394-Rhodesian_ridgeback')
    enroller.enroll_new_class_from_path(new_class_path)
    
    test_img_path = os.path.join('..', 'dogs', 'recognition', 'test',
        'n02087394-Rhodesian_ridgeback', 'n02087394_381.jpg')
    test_img_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02087394-Rhodesian_ridgeback', 'n02087394_9855.jpg')
    test_img_path = os.path.join('..', 'dogs', 'train',
        'n02088364-beagle', 'n02088364_2360.jpg')
    test_img_path = os.path.join('..', 'dogs', 'train',
        'n02091635-otterhound', 'n02091635_1043.jpg')
    
    test_img = Image.open(test_img_path)
    c = enroller.get_classification(test_img)
