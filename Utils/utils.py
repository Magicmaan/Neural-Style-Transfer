import numpy as np
import os

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from PIL import Image


# takes in string path, and outputs the tensor of the image
def loadImage(path, device):
    image = Image.open(path)
    # defining image transformer to resize to 512, and convert to tensor
    loader = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    
    return image.to(device,torch.float)

def getContentLoss(generatedFeatures, originalFeatures) -> torch.Tensor:
    # Calculate the content loss as the Mean Squared Error (MSE) between the generated and original features
    contentLoss = torch.mean((generatedFeatures - originalFeatures) ** 2)
    return contentLoss

def getStyleLoss(generatedFeatures, styleFeatures) -> torch.Tensor:
    # Calculate the Gram matrix for the style and the generated image
    batchSize, channel, height, width = generatedFeatures.shape

    G = torch.mm(generatedFeatures.view(channel, height * width), generatedFeatures.view(channel, height * width).t())
    A = torch.mm(styleFeatures.view(channel, height * width), styleFeatures.view(channel, height * width).t())
        
    # Calculate the style loss as the MSE between the Gram matrices of the style and generated images
    styleLoss = torch.mean((G - A) ** 2)
    return styleLoss

def getTotalLoss(generatedFeatures, originalFeatures, styleFeatures, alpha, beta) -> torch.Tensor:
    styleLoss = 0
    contentLoss = 0
    # Iterate over the activations of each layer and calculate the loss, adding it to the total content and style loss
    for genFeat, origFeat, styleFeat in zip(generatedFeatures, originalFeatures, styleFeatures):
        contentLoss += getContentLoss(genFeat, origFeat)
        styleLoss += getStyleLoss(genFeat, styleFeat)
    
    # Calculate the total loss for the current epoch
    totalLoss = alpha * contentLoss + beta * styleLoss 
    return totalLoss

def extractFeatures( model, generatedImage, layers ):
    # https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
    # heavily inspired by the above link
    
    # Extract convolution layers from the model
    children = list(model.children())
    convolutionLayers: list[torch.nn.Conv2d] = []
    for layer in layers:
        convolutionLayers.append(children[0][layer])
    
    featureMaps: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    
    # Create image from tensor and add to images list
    tempImage = generatedImage
    for layer in convolutionLayers:
        tempImage = layer(tempImage)
        featureMaps.append(tempImage)
    
    # turn tensor into grayscale image
    for fMap in featureMaps:
        fMap = fMap.squeeze(0)
        gray_scale = torch.sum(fMap, 0)
        gray_scale = gray_scale / fMap.shape[0]
        images.append(gray_scale.data.cpu().numpy())
    
    return images