import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
def validation(validloaders, model, criterion, device):
    """ A function that takes in a model,criterion, and a dataloader for validation and runs a validation iteration using the model.
    It returns the loss and accuracy of the model"""
    model.eval()
    validation_loss = 0
    accuracy = 0
            
    for images, labels in validloaders:
        images = images.view(images.shape[0], 3, 224, 224)
        images, labels = images.to(device), labels.to(device)
                
        logits = model.forward(images)
        validation_loss += criterion(logits, labels).item()
        prob = torch.exp(logits)
                
        top_prob, top_class = prob.topk(1, dim = 1)
        comparison = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(comparison.type(torch.FloatTensor))
                
    return accuracy, validation_loss
                
def test_model(testloaders, model, criterion, device):
    """A function that takes in a model,criterion, and the dataset for testing the model and goes on to test the model
    """
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloaders:
            print("New Batch")
            images = images.view(images.shape[0], 3, 224, 224)
            images, labels = images.to(device), labels.to(device)
            logits = model.forward(images)
            test_loss += criterion(logits, labels)
            prob = torch.exp(logits)

            top_prob, top_class = prob.topk(1, dim =1)
            compare = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(compare.type(torch.FloatTensor))
    return accuracy, test_loss


def train(model, trainloaders, optimizer, criterion, device):
    """ A function that takes in a model architecture, criterion, optimizer, training and validation data, the learning rate, hidden units, and the system device. It uses these inputs to train the given model, while simultaneously printing the running loss, the validation loss and the accuracy of the model.It iterates through the data and returns the model """
    running_loss = 0
    for images, labels in trainloaders:
        images = images.view(images.shape[0], 3,224,224)
        images, labels = images.to(device), labels.to(device)
            
        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
            
        optimizer.step()
        running_loss += loss.item()
    return running_loss


