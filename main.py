from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision import models
from Utils import VGG, loadImage, getTotalLoss, extractFeatures

# Load VGG19 model
model = models.vgg19(models.VGG19_Weights.DEFAULT).features
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define paths
sourcePath: Path = Path.cwd()
dataPath: Path = sourcePath.joinpath("data")
inputPath: Path = sourcePath.joinpath("data/input")
outputPath: Path = sourcePath.joinpath("data/output")
stylePath: Path = sourcePath.joinpath("data/style")

# simple list of images in input and styles
inputList: list[str] = list(inputPath.glob('**/*.[jp][pn]g'))
styleList: list[str] = list(stylePath.glob('**/*.[jp][pn]g'))

# Loading the original and the style image
contentImage = loadImage(inputList[6], device)
styleImage = loadImage(styleList[1], device)
outputImage = contentImage.clone().requires_grad_(True)

def neuralTransfer():
    # Model hyperparameters
    iterations = 1000
    # learning rate, how similar the content and style should be. 
    # higher = more style, lower = more content
    lr = 0.05
    alpha = 8
    beta = 70
    convLayers = [0, 2, 5, 7, 10, 14, 16, 19, 21, 23, 25, 28]
    
    # Load the model to the GPU
    model = VGG().to(device).eval()
    optimizer = optim.Adam([outputImage], lr=lr)
    
    # Iterating for 1000 times
    for e in range(iterations):
        # extract features
        generatedFeatures = model(outputImage)
        originalFeatures = model(contentImage)
        styleFeatures = model(styleImage)
        
        # get the total loss of model
        totalLoss = getTotalLoss(generatedFeatures, originalFeatures, styleFeatures, alpha, beta)

        print(f"I: {e} Loss: {totalLoss}")
        
        # Optimize the image + back propogation
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()
        
        
        # Extract feature maps
        if e == 1:
            print("Extracting Feature Maps")
            images = extractFeatures(model, outputImage, convLayers)
            fig = plt.figure(figsize=(30, 50))
            for i in range(len(images)):
                a = fig.add_subplot(5, 4, i + 1)
                imgplot = plt.imshow(images[i])
                a.axis("off")
                a.set_title(f"Layer {convLayers[i]}")
            plt.savefig(outputPath.joinpath("feature_map.png"), bbox_inches='tight')
        
        # occassionally save the image
        if e % 50 == 0:
            print("Saving Image")
            save_image(outputImage, outputPath.joinpath(f"gen_{e}.png"))


if __name__ == "__main__":
    neuralTransfer()

