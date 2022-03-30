import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import pandas as pd
import time
import pickle
import shutil
import argparse
import numpy as np
import glob
import torch.nn as nn
import torch
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import os
import time
import pickle

import shutil
import argparse
import numpy as np

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder

from sklearn.metrics import precision_score

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (index,) + (path,))
        return tuple_with_path

def load_images(directory):
    from torchvision import transforms
    transforms = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
    data = ImageFolderWithPaths(root=directory, transform=transforms)
#    test_data_path = "/content/drive/My Drive/Colab Notebooks/Kaggle/DogCat/sample/test/"
#    test_data = ImageFolderWithIDs(root=test_data_path, transform=transforms)
    return data
batch_size = 1
val_dataset   = load_images('/home/huong_n_pham01/data/retinal/train')
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
device = "cuda" if torch.cuda.is_available() else "cpu"
#modelPath = './probsThresthold_0.89_batch_size_16_batchNumber_1_epochs_3000_ver_0_model'
#modelPath = './probsThresthold_0.40_batch_size_16_batchNumber_1_epochs_1000_weightIncrement_100.00_ver_0_model'
#modelPath = './best_model_state'
#modelPath = "./model_2998"
modelPath = "probsThresthold_0.97_batch_size_16_batchNumber_1_epochs_3000_ver_0_model"
modelNoWeight = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
modelNoWeight.fc = nn.Sequential(nn.Linear(modelNoWeight.fc.in_features,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,3))
model_temp = torch.load(modelPath)
#name = 'best_model_state'
name = 'model_state1_dict'
#name = "model_2998"
modelNoWeight.to(device)
modelNoWeight.load_state_dict(model_temp[name])
modelNoWeight.eval()

num_correct = 0
num_examples = 0
match = []
targetList = []
predList   = []
probList   = []
pathList   = []

for index, data in enumerate(val_loader):
    input, target, id, path = data
    input = input.to(device)
    output = modelNoWeight(input)
    target = target.to(device)
    pred   = torch.max(F.softmax(output, dim = 1), dim=1)[1]
    prob = (torch.max(F.softmax(output, dim = 1), dim=1)[0])
    print("%s | %s | %s | %0.2f | %s | %s"%(((target == pred).item(), target.item(), pred.item(), prob.item(), path[0], id.item())))
    correct = torch.eq(torch.max(F.softmax(output, dim = 1), dim=1)[1],
    target).view(-1)
    num_correct += torch.sum(correct).item()
    num_examples += correct.shape[0]
    match.append((target == pred).item())
    targetList.append(target.item())
    predList.append(pred.item())
    probList.append(prob.item())
    pathList.append(path[0])
print("Accuracy: {}".format(num_correct/num_examples))
print("Precision: {}".format(precision_score(targetList, predList, average='micro')))
