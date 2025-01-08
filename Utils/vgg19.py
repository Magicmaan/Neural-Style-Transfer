#importing the required libraries
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# vgg class to store model and features
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        # convolution layers to be used
        self.req_features= ['0','5','10','19','28'] 
        self.model=models.vgg19(models.VGG19_Weights.DEFAULT).features[:29]
    

    def forward(self, image):
        features=[]
        # extract layers from model
        for i, layer in enumerate(self.model):
            #activation of the layer will stored in x
            image = layer(image)
            #appending the activation of the selected layers and return the feature array
            if (str(i) in self.req_features):
                features.append(image)
                
        return features
