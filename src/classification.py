import os

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms

class Classifier:
    """
    A class to perform dog breed classification from RGB images.
    
    Attributes
    ----------
    device : torch.device
        object for applying inputs, outputs and models to GPU or CPU
    classes: list<str>
        existing classes (dog breeds) on the dataset
    model : torch.nn
        model for making the dog breed predictions
    transform : torch.transforms
        input preprocessing pipeline

    Methods
    -------
    classifier_model(n_classes)
        Generates a CNN ResNet50-based model.
    
    get_classification(img)
        Predicts the breed of a dog in an image.
    """
    
    def __init__(self, model_ckpt_path):
        '''
        Constructs all the attributes for the classifier object.
        
        Parameters
        ----------
        model_ckpt_path : str
            path to the file containing the result of the training model
        '''
        
        # Initialize device (GPU or CPU)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint of the trained model (`model_ckpt`)
        model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        
        # Get the list of dog breeds from the checkpoint
        self.classes = np.array(model_ckpt['classes'])
        
        # Get the trained model weights from the checkpoint
        state_dict = model_ckpt['state_dict']
        
        # Get the number of dog breeds
        n_classes = len(self.classes)
        
        # Initialize the model archictecture (`model`)
        self.model = self.classifier_model(n_classes)
        
        # Load the weights into the model
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = torch.jit.script(self.model).to(self.device)
        
        # Initialize preprocessing input pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def classifier_model(self, n_classes):
        '''
        Generates a CNN ResNet50-based model.
        
        Parameters
        ----------
        n_classes
            number of classes (dog breeds) to be outputted
        
        Returns
        -------
        x : torch.nn
            the model
        '''
        
        # First, `x` is a new ResNet50 CNN model
        x = torchvision.models.resnet50(pretrained=False)
        
        # Change the final fully connected layer so that the output size
        #   matches the desired `n_classes` size. Also, apply sigmoid function
        x.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, n_classes),
            torch.nn.Sigmoid())
    
        return x
        
    def get_classification(self, img, n_best=5):
        '''
        Predicts the breed of a dog in an image.
        
        Parameters
        ----------
        img : PIL.Image
            image containing a dog
        n_best : int
            determines the classification to return the n-most confident breeds

        Returns
        -------
        n_best_class_conf : list<tuple<str,float>>
            list of n-most confident breed types, along their confidence values
        '''
        
        # Read image (`img`) and convert to red-green-blue channels (RGB),
        #   ensuring the input will have 3 channels
        img = img.convert('RGB')
        
        # `x` refers to the image when the preprocessing pipeline
        #   (`self.transform`) is applied to the image (`img`)
        x = self.transform(img)
        
        # Add a new dimension to the tensor (simulate a 1-batch size)
        x.unsqueeze_(0)
        
        # Pass the input tensor to the used device (GPU or CPU)
        x = x.to(self.device)
        
        # Calculate the prediction (`y`) of the current model according to the
        #   input `x`
        y = self.model(x)
        
        # Remove extra dimension 0
        y.squeeze_(0)
        
        # Convert from torch.Tensor to np.array
        y = y.detach().numpy()
        
        # `n_best_class_indexes` encodes the index of the n-most confident dog
        #   breeds from the input image
        n_best_class_indexes = np.argsort(y)[::-1][:n_best]
        
        # `n_best_conf` encodes the confidence values of these n-best
        n_best_conf = y[n_best_class_indexes]
        
        # `best_n_class` encodes the dog breeds names of these n-best
        n_best_class = self.classes[n_best_class_indexes]
        
        # `n_best_class_conf` aggregates the dog breeds names and confidence
        #   values in a tuple fashion
        n_best_class_conf = list(zip(n_best_class, n_best_conf))
        
        return n_best_class_conf
        
if __name__ == '__main__':
    model_ckpt_path = os.path.join('..', 'models', 'classifier.pth')
    classifier = Classifier(model_ckpt_path)
    
    img_path = os.path.join('..', 'dogs', 'train',
        'n02085620-Chihuahua', 'n02085620_242.jpg')
    img = Image.open(img_path)
    
    classification = classifier.get_classification(img, n_best=10)
    