import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

class EmbeddingsExtractor:
    """
    A class to extract embeddings from RGB images of dogs.
    
    Attributes
    ----------
    device : torch.device
        object for applying inputs, outputs and models to GPU or CPU
    model : torch.nn
        model for making the dog embeddings prediction
    transform : torch.transforms
        input preprocessing pipeline
    
    Methods
    -------
    embedder_model(n_embeddings)
        Generates a CNN ResNet50-based embedder
        
    get_embeddings_batch(img)
        Predicts the embeddings from a batch of dog images..
        
    get_embeddings(img)
        Predicts the embeddings of a dog image.
    
    distance_two_embeddings(embeddings_a, embeddings_b)
        Computes the difference between two embeddings.
    """
    
    def __init__(self, model_ckpt_path):
        '''
        Constructs all the attributes for the embedder object.
        
        Parameters
        ----------
        model_ckpt_path : str
            path to the file containing the result of the trained model
        '''
        
        # Initialize device (GPU or CPU)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint of the trained model (`model_ckpt`)
        model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        
        # Get the number of embeddings from the trained model
        n_embeddings = model_ckpt['n_embeddings']
        
        # Get the trained model weights from the checkpoint
        state_dict = model_ckpt['state_dict']
        
        # Initialize the model architecture (`model`)
        self.model = self.embedder_model(n_embeddings)
        
        # Load the weights into the model
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = torch.jit.script(self.model).to(self.device)
        
        # Do first predict, which is always the slowest
        self.model(torch.rand(1, 3, 224, 224))
        
        # Initialize preprocessing input pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def embedder_model(self, n_embeddings):
        '''
        Generates a CNN ResNet50-based embedder.
        
        Parameters
        ----------
        n_embeddings
            number of embeddings to be outputted
        
        Returns
        -------
        x : torch.nn
            the model
        '''
        
        # First, `x` is a new ResNet50 CNN model
        x = torchvision.models.resnet50(pretrained=False)
        
        # Change the final fully connected layer so that the output size
        #   matches the desired `n_embeddings` size. Also, apply sigmoid
        #   function
        x.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, n_embeddings),
            torch.nn.Sigmoid())
    
        return x
    
    def get_embeddings_batch(self, imgs):
        '''
        Predicts the embeddings from a batch of dog images.
        
        Parameters
        ----------
        imgs : list<PIL.Immage>
            batch of dog images
        
        Returns
        -------
        y : np.array
            embeddings from each dog image
        '''
        
        # Convert each dog image (`img`) to red-green-blue channels (RGB),
        #   ensuring the input will have 3 channels
        imgs = [img.convert('RGB') for img in imgs]
        
        # Apply preprocessing pipeline to each input (`xs`)        
        xs = [self.transform(img) for img in imgs]
        
        # Concatenate the list of preprocessed input tensors to a single one
        x = torch.stack(xs)

        # Pass the input tensor to the used device (GPU or CPU)
        x = x.to(self.device)
        
        # Calculate the list of embeddings (`y`) from the input `x` according
        #   to `model`
        y = self.model(x)
        
        # Convert from torch.Tensor to np.array
        y = y.detach().numpy()
        
        return y
        
    def get_embeddings(self, img):
        '''
        Predicts the embeddings of a dog image.
        
        Parameters
        ----------
        img : PIL.Image
            image containing a dog

        Returns
        -------
        y : np.array
            array of embeddings from the input image
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
        
        # Calculate the embeddings (`y`) from the input `x` according to `model`
        y = self.model(x)
        
        # Remove extra dimension 0
        y.squeeze_(0)
        
        # Convert from torch.Tensor to np.array
        y = y.detach().numpy()
        
        return y
    
    def distance_two_embeddings(self, embeddings_a, embeddings_b):
        '''
        Computes the difference between two embeddings.
        
        Parameters
        ----------
        embeddings_a : np.array
            embeddings from the first image
        embeddings_b : np.array
            embeddings from the second image
        '''
        
        # `distance` computes the euclidean distance between the embeddings
        distance = np.linalg.norm(embeddings_a - embeddings_b)
        
        return distance
    
if __name__ == '__main__':
    # model_ckpt_path = os.path.join('..', 'models', 'embedder.pth')
    model_ckpt_path = os.path.join('..', 'models', 'embedder_lossless.pth')
    embeddings_extractor = EmbeddingsExtractor(model_ckpt_path)
    '''
    img1_path = os.path.join('..', 'dogs', 'train',
        'n02085620-Chihuahua', 'n02085620_199.jpg')
    img2_path = os.path.join('..', 'dogs', 'train',
        'n02085620-Chihuahua', 'n02085620_242.jpg')
    img3_path = os.path.join('..', 'dogs', 'train',
        'n02085936-Maltese_dog', 'n02085936_352.jpg')
    '''
    '''
    img1_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02090379-redbone', 'n02090379_91.jpg')
    img2_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02090379-redbone', 'n02090379_223.jpg')
    img3_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02087394-Rhodesian_ridgeback', 'n02087394_889.jpg')
    '''
    
    img1_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02105056-groenendael', 'n02105056_143.jpg')
    img2_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02105056-groenendael', 'n02105056_1960.jpg')
    img3_path = os.path.join('..', 'dogs', 'recognition', 'enroll',
        'n02099429-curly-coated_retriever', 'n02099429_227.jpg')
    
    
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)
    
    e1 = embeddings_extractor.get_embeddings(img1)
    e2 = embeddings_extractor.get_embeddings(img2)
    e3 = embeddings_extractor.get_embeddings(img3)
    
    print(embeddings_extractor.distance_two_embeddings(e1, e2))
    print(embeddings_extractor.distance_two_embeddings(e1, e3))
    print(embeddings_extractor.distance_two_embeddings(e2, e3))
