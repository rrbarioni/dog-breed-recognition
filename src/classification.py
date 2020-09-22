import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

class Classifier:
    def get_architecture(self):
        x = torchvision.models.resnet50(pretrained=True)
        x.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, self.n_classes),
            torch.nn.Sigmoid())
    
        return x
    
    def __init__(self, model_ckpt_path):
        self.model_ckpt_path = model_ckpt_path
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_ckpt = torch.load(
            self.model_ckpt_path, map_location=self.device)
        self.n_classes = self.model_ckpt['n_classes']
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
        
    def get_classification(self, img):
        rgb_img = img.convert('RGB')
        
        x = self.input_transform(rgb_img)
        x.unsqueeze_(0)
        x = x.to(self.device)
        
        y = self.model(x)
        y.squeeze_(0)
        y = y.detach().numpy()
        
        return y
        
if __name__ == '__main__':
    model_ckpt_path = os.path.join('..', 'models', 'classifier.pth')
    c = Classifier(model_ckpt_path)
    
    img_path = os.path.join('..', 'dogs', 'train',
        'n02085620-Chihuahua', 'n02085620_242.jpg')
    img = Image.open(img_path)
    
    y = c.get_classification(img)
    
    dog_breeds_list = os.listdir(os.path.join('..', 'dogs', 'train'))
    dog_breeds_list[y.argmax()]
    