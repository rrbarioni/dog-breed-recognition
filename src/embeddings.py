import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

class EmbeddingsExtractor:
    def get_architecture(self):
        x = torchvision.models.resnet50(pretrained=False)
        x.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, self.n_embeddings),
            torch.nn.Sigmoid())
    
        return x
    
    def __init__(self, model_ckpt_path):
        self.model_ckpt_path = model_ckpt_path
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_ckpt = torch.load(
            self.model_ckpt_path, map_location=self.device)
        self.n_embeddings = self.model_ckpt['n_embeddings']
        self.state_dict = self.model_ckpt['state_dict']
        
        self.model = self.get_architecture()
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.model = torch.jit.script(self.model).to(self.device)
        
        self.input_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def get_embeddings(self, img):
        rgb_img = img.convert('RGB')
        
        x = self.input_transform(rgb_img)
        x.unsqueeze_(0)
        x = x.to(self.device)
        
        y = self.model(x)
        y.squeeze_(0)
        y = y.detach().numpy()
        
        return y
    
    def distance_two_embeddings(self, e1, e2):
        distance = np.sum((e1 - e2) ** 2)
        
        return distance
    
if __name__ == '__main__':
    model_ckpt_path = os.path.join('..', 'models', 'embedder.pth')
    ee = EmbeddingsExtractor(model_ckpt_path)
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
    
    e1 = ee.get_embeddings(img1)
    e2 = ee.get_embeddings(img2)
    e3 = ee.get_embeddings(img3)
    
    print(ee.distance_two_embeddings(e1, e2))
    print(ee.distance_two_embeddings(e1, e3))
    print(ee.distance_two_embeddings(e2, e3))
