import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import cv2
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random

SIZE = 70000

def read_img(path):
    return Image.fromarray(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB))

def shuffle(X,y):

    new_train=[]
    for m,n in zip(X,y):
        new_train.append([m,n])
    random.shuffle(new_train)

    X,y=[],[]
    for x in new_train:
        X.append(x[0])
        y.append(x[1])
    return X,y

def load_labels(PATH):
    
    data = pd.read_csv(PATH)
    people = data['gender']              #male is labeled 1 and female is labeled 0
    labels = []

    count = 0
    for person in people:
        if person == 'male':
            labels.append(1)
        else:
            labels.append(0)

        count = count + 1
        if(count>SIZE):
            break

    return labels

def get_random_sampling():

    ffhq = os.listdir('thumbnails128x128/')
    X = []

    count = 0
    ffhq = sorted(ffhq)

    for file in tqdm(ffhq):
        im = read_img(f'thumbnails128x128/{file}')

        X.append(im)

        count = count + 1
        if (count>SIZE):
            break

    y = load_labels('ffhq_labels.csv')
    X,y = shuffle(X,y)

    return X[0:int(0.9*SIZE)],y[0:int(0.9*SIZE)],X[int(0.9*SIZE):int(1*SIZE)],y[int(0.9*SIZE):int(1*SIZE)]

class ImageDataset(Dataset):
    def __init__(self, X, y, training=True, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]

        if self.transform is not None:
            img = self.transform((img))
        
        label = self.y[idx]

        tensor_img = (img).float()
        tensor_label = torch.tensor(label)

        return (tensor_img,tensor_label)

def data_loader(transform = None, batch_size = 32,num_workers = 0,size = 1000):

    SIZE = size

    input_size = 224

    if transform == None:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        valid_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        train_transform = transform
        valid_transform = transform

    train_data,train_label,val_data,val_label = get_random_sampling()

    train_dataset   =  ImageDataset(train_data, train_label,True, transform=train_transform)
    val_dataset     =  ImageDataset(val_data, val_label,False, transform=valid_transform)

    """
    nrow, ncol = 5, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        image, label = train_dataset[i]
        image = np.transpose(image.detach().numpy(), (1, 2, 0))
        ax.imshow(image)
        ax.set_title(f'label: {label}')

    plt.show()
    """

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader,val_loader

if __name__ == '__main__':
    train_loader,test_loader = data_loader()