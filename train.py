import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
    
    
from run_model import train
from loader import load_image, load_model
from check_point import check_point
from run_model import validation


def main():
    parser = argparse.ArgumentParser(description = "Inputs for training and saving a model")
    parser.add_argument("--arch", default = "vgg11", help = "The model architecture")
    parser.add_argument("--lr", type = float, help = "The learning rate")
    parser.add_argument("--epochs", type = int, help = "Epochs for training")
    parser.add_argument("--gpu", action = "store_true", default = True, help = "Gpu for training")
    parser.add_argument("--data_dir", help = "Directory containing image data")
    parser.add_argument("--dir", help = "Directory to save checkpoint")
    args = parser.parse_args()
    
    if args.gpu:
        device = torch.device("cuda")
        validloaders, trainloaders, testloaders, train_datasets = load_image(args.data_dir)
        model, optimizer, criterion = load_model(args.arch, args.lr)
        model.to(device)
#         optimizer.to(device)
#         criterion.to(device)
        for epoch in range(args.epochs):
            running_loss = train(model, trainloaders, optimizer, criterion, device)
        else:
            accuracy, validation_loss = validation(validloaders, model, criterion, device) 
            print(f"Epoch {epoch + 1}/{args.epochs}"
                      "Training loss: {:.3f}".format(running_loss/len(trainloaders)),
                      "Test loss: {:.3f}".format(validation_loss/len(validloaders)),
                      "Accuracy: {:.3f}".format(accuracy/len(validloaders)))
            model.train()
        check_point(model, optimizer, args.dir, train_datasets)
    else:
        print("Please turn on gpu")
if __name__ == "__main__":
    main()
 
#I had to install torch version 1.10.2 and also install a later version of twine before it worked
#python3 train.py --arch vgg11 --lr 0.03 --epochs 1 --gpu --data_dir "/data/flowers" --dir "/home/workspace/ImageClassifier"