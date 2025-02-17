from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision import models
from Utils import VGG, loadImage, getTotalLoss, extractFeatures
import tkinter as tk
from tkinter import filedialog
import threading
import os
import subprocess

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
contentImage = loadImage(inputList[2], device)
styleImage = loadImage(styleList[2], device)
outputImage = contentImage.clone().requires_grad_(True)

def neuralTransfer(contentImage, styleImage, outputImage):
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
            save_image(outputImage, outputPath.joinpath("gen_latest.png"))
        
        # occassionally save the image
        if e % 50 == 0:
            print("Saving Image")
            save_image(outputImage, outputPath.joinpath(f"gen_{e}.png"))
            
            # save as latest image
            save_image(outputImage, outputPath.joinpath("gen_latest.png"))

def loadImageToLabel(entry, canvas, callback):
    image_path = filedialog.askopenfilename()
    if image_path:
        entry.delete(0, tk.END)
        entry.insert(0, image_path)
        image = Image.open(image_path)
        image.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        callback()



def setupUI(root: tk.Tk):
    menu = tk.Menu(root)
    root.config(menu=menu)
    
    # Create frames
    inputFrame = tk.Frame(root)
    inputFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    styleFrame = tk.Frame(root)
    styleFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    displayFrame = tk.Frame(root)
    displayFrame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
    
    outputFrame = tk.Frame(root)
    outputFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    # Display input and style images
    inputImageCanvas = tk.Canvas(displayFrame, width=250, height=250)
    inputImageCanvas.pack(side=tk.LEFT, padx=10)
    inputImageLabel = tk.Label(displayFrame, text="Input Image")
    inputImageLabel.pack(side=tk.LEFT, padx=10)

    styleImageCanvas = tk.Canvas(displayFrame, width=250, height=250)
    styleImageCanvas.pack(side=tk.LEFT, padx=10)
    styleImageLabel = tk.Label(displayFrame, text="Style Image")
    styleImageLabel.pack(side=tk.LEFT, padx=10)
    
    
    
    def run(input, style):
        print("Both images have been selected. Performing neural Transfer...")
        print(f"Input: {input}")
        print(f"Style: {style}")
        contentImage = loadImage(input, device)
        styleImage = loadImage(style, device)
        outputImage = contentImage.clone().requires_grad_(True)
        
        
        
        # Run neuralTransfer in a separate thread
        transfer_thread = threading.Thread(target=neuralTransfer, args=(contentImage, styleImage, outputImage))
        transfer_thread.start()
        
        
    
    def onInputAndStyle():
        if inputEntry.get() and styleEntry.get():
            run(inputEntry.get(), styleEntry.get())
    
    # Input file selection
    tk.Label(inputFrame, text="Input Image:").pack(side=tk.LEFT)
    inputEntry = tk.Entry(inputFrame, width=50)
    inputEntry.pack(side=tk.LEFT, padx=5)
    tk.Button(inputFrame, text="Browse", command=lambda: loadImageToLabel(inputEntry, inputImageCanvas, onInputAndStyle)).pack(side=tk.LEFT)
    
    # Style file selection
    tk.Label(styleFrame, text="Style Image:").pack(side=tk.LEFT)
    styleEntry = tk.Entry(styleFrame, width=50)
    styleEntry.pack(side=tk.LEFT, padx=5)
    tk.Button(styleFrame, text="Browse", command=lambda: loadImageToLabel(styleEntry, styleImageCanvas, onInputAndStyle)).pack(side=tk.LEFT)
    
    
    # Display output images in a grid with 4 per row
    outputImageLabels = []
    outputImageCanvases = []
    for i in range(12):
        outputImageLabel = tk.Label(outputFrame, text=f"Output Image {i+1}")
        outputImageLabel.grid(row=i//4*2, column=i%4, padx=10, pady=5)
        outputImageCanvas = tk.Canvas(outputFrame, width=50, height=50)
        outputImageCanvas.grid(row=i//4*2+1, column=i%4, padx=10, pady=5)
        
        # Load and display the image
        image_path = sourcePath.joinpath("data/example/gen_0.png")
        image = Image.open(image_path)
        image.thumbnail((50, 50))
        photo = ImageTk.PhotoImage(image)
        outputImageCanvas.create_image(0, 0, anchor=tk.NW, image=photo)
        outputImageCanvas.image = photo
        
        outputImageLabels.append(outputImageLabel)
        outputImageCanvases.append(outputImageCanvas)
    
    
    # Display large output image
    largeOutputImageLabel = tk.Label(outputFrame, text="Large Output Image")
    largeOutputImageLabel.grid(row=6, column=0, columnspan=4, padx=10, pady=5)
    largeOutputImageCanvas = tk.Canvas(outputFrame, width=250, height=250)
    largeOutputImageCanvas.grid(row=7, column=0, columnspan=4, padx=10, pady=5)
    
    def updateImages():
        image_path = sourcePath.joinpath("data/output/gen_latest.png")
        if image_path.exists():
            image = Image.open(image_path)
            image.thumbnail((250, 250))
            photo = ImageTk.PhotoImage(image)
            largeOutputImageCanvas.create_image(0, 0, anchor=tk.NW, image=photo)
            largeOutputImageCanvas.image = photo
            
            
            for i in range(len(outputImageCanvases)):
                image_path = sourcePath.joinpath(f"data/output/gen_{i*50}.png")
                if image_path.exists():
                    image = Image.open(image_path)
                    image.thumbnail((50, 50))
                    photo = ImageTk.PhotoImage(image)
                    outputImageCanvases[i].create_image(0, 0, anchor=tk.NW, image=photo)
                    outputImageCanvases[i].image = photo
        
        
            
        root.after(10000, updateImages)  # run every 10 seconds
    
    updateImages()  # Initial call to start the loop
    
    def openOutputFolder():
        subprocess.Popen(r'explorer /select,' + str(outputPath.joinpath("gen_latest.png")))
    
    # Open output button
    openOutputButton = tk.Button(root, text="Open Output", command=openOutputFolder)
    openOutputButton.pack(side=tk.TOP, pady=10)
    
    return inputEntry, styleEntry, outputImage, outputImageCanvases

def main():
    root = tk.Tk()
    setupUI(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
    # neuralTransfer()

