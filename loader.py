import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_image(data_dir):
    """ Takes the directory containing the images for the training, validation, 
    and testing as an input and returns the data loaders
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining the structure of the images and converting them to tensors
    # Working on the validation set
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])])
    #Working on the training set
    #More transformations goes into the training set, to make the model more robust
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.Resize(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])])
    #Working onthe testing set
    test_transforms = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],
                                                        [0.229,0.224,0.225])])
    
    #Loading the images using ImageFolder and converting to dataloaders that can be put into the model; 
    #For the Validation set
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 30, shuffle = True)
    
    #For the training set
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 30, shuffle = True)
    
    #For the testing set
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 30, shuffle = True)
    
    return validloaders, trainloaders, testloaders, train_datasets



def load_model(model_name, lr):
    # Initializing models and setting the classifier
    if model_name in models.__dict__:
        model_class = models.__dict__[model_name]
        model = model_class(pretrained=True)
        # You can then work with model_instance as your initialized model
    else:
        print("Model not found in the models module")
    
    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, 4096)),
        ("dropout_1", nn.Dropout(p=0.2)),
        ("relu_1", nn.ReLU()),
        ("fc2", nn.Linear(4096, 2000)),
        ("dropout_2", nn.Dropout(p=0.2)),
        ("relu_2", nn.ReLU()),
        ("fc3", nn.Linear(2000, 102)),
        ("output", nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion
    
    
    
    
    
    
    