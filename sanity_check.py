import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def process_image(image):
    ''' 
    This function takes in the path of an image, converts it into a PIL image,
    preprocess it with transforms and converts it to a tensor ready for input to pytorch
    '''
    image = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    img = image_transform(image)
    numpy_image = np.array(img)
    
    torch_image = torch.tensor(numpy_image).permute(0, 2, 1)

    return torch_image

def sanity_check(image, model, label_dir, topk, device):
    """A function that takes in an image, a model, and a dictionary containing the classes based on the index in the datasets
    """
    with open(label_dir, 'r') as f:
        label_dict = json.load(f)
    
    with torch.no_grad():
        top_prob, top_class = predict(image, model,device, topk)
        
    prob = [prob.item for prob in top_prob[0]]
    labels = [label_dict.get(str(idx), "Unknown") for idx in top_class[0]]
    
#     x = np.arange(len(label))

#     # Create the bar chart
#     plt.bar(x, prob)

#     # Set the x-axis labels
#     plt.xticks(x, label, rotation=15)  

#     # Set the title and labels
#     plt.title("Top Probabilities")
#     plt.xlabel("Labels")
#     plt.ylabel("Probability")
    

#     # Show the plot
#     plt.tight_layout()  # Ensures labels are not cut off
#     plt.show()
    return prob, labels

def predict(image_path, model, device, topk=5, ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    # Add a batch dimension to the image
    image = image.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        image = image.to(device)
        logits = model.forward(image)
        prob = torch.exp(logits)
        top_prob, top_classes = prob.topk(topk, dim = 1)
    return top_prob, top_classes


