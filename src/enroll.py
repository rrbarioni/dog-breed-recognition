import os
import tqdm
from six.moves import cPickle as pickle

from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from embeddings import EmbeddingsExtractor

class Enroller:
    def __init__(self, initial_enroll_path, model_ckpt_path):
        self.embeddings_extractor = EmbeddingsExtractor(model_ckpt_path)
        
        with open(initial_enroll_path, 'rb') as f:
            initial_enroll = pickle.load(f)
        self.embeddings = initial_enroll['embeddings']
        self.labels = initial_enroll['labels']
        self.classes = initial_enroll['classes']
        
        self.next_class_index = len(self.classes)
        
        self.classifier = self.create_classifier()
        
    def create_classifier(self):
        classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
        classifier.fit(self.embeddings, self.labels)
        
        return classifier
        
    def enroll_new_class(self, new_class_path, class_name=None):
        if class_name is None:
            class_name = new_class_path.split('\\')[-1]
        
        self.classes.append(class_name)
        
        instances = sorted(os.listdir(new_class_path))
        instances = tqdm.tqdm(instances, position=0, leave=True)
        for curr_instance in instances:
            instance_path = os.path.join(new_class_path, curr_instance)
            img = Image.open(instance_path)
            
            embeddings = self.embeddings_extractor.get_embeddings(img)
            
            self.embeddings.append(embeddings)
            self.labels.append(self.next_class_index)
        
        self.next_class_index += 1
        
        self.classifier = self.create_classifier()
        
    def get_classification(self, img):
        embeddings = self.embeddings_extractor.get_embeddings(img)
        label = self.classifier.predict([embeddings])[0]
        dog_breed = self.classes[label]
        
        return dog_breed
        
if __name__ == '__main__':
    initial_enroll_path = os.path.join('..', 'models', 'initial_enroll.pkl')
    model_ckpt_path = os.path.join('..', 'models', 'embedder.pth')
    
    enroller = Enroller(initial_enroll_path, model_ckpt_path)
    
    new_class_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02087394-Rhodesian_ridgeback')
    enroller.enroll_new_class(new_class_path)
    
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
    c