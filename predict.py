import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pickle

from check_point import load_check_point
from sanity_check import predict, process_image, sanity_check
# def persistent_load(saved_id):
#     # Implement your logic to load the object associated with the persistent ID
#     # For simplicity, let's assume we have a dictionary called 'persistent_objects'
#     return persistent_objects.get(saved_id)

# # Set the persistent_load function in the pickle module
#     pickle._getattribute__ = persistent_load

def main():
    parser = argparse.ArgumentParser(description = "Predicting image")
    parser.add_argument("--image", help = "Image dir")
    parser.add_argument("--topk", type = int, help = "Topk")
    parser.add_argument("--gpu", action = "store_true", default = True, help = "Gpu for training")
    parser.add_argument("--checkpoint", help = "Model checkpoint")
    parser.add_argument("--label_dir", help = "Label directory")
    
    
    args = parser.parse_args()
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, optimizer, epochs, criterion = load_check_point(args.checkpoint)
    
    probabilities, labels = sanity_check(args.image, model, args.label_dir, args.topk, device)
    print(probabilities)
    
if __name__ == "__main__":
    main()
    
#I had to install torch version 1.10.2 and also install a later version of twine before it worked 
#python3 predict.py  --image "/data/flowers/valid/99/image_07860.jpg" --topk 3 --gpu --checkpoint checkpoint.pth --label_dir cat_to_name.json

#"/workspace/ImageClassifier/flowers/valid/99/image_07860.jpg