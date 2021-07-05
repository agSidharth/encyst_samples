import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from trojanvision.utils import summary
import argparse
import trojanvision
import trojanzoo



if __name__ == '__main__':

    """
    train_data = datasets.MNIST(root='./data', train=True, download=True)
    train_data_loader= DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    model = BadNet(train_data_loader.dataset.channels,train_data_loader.dataset.class_num);
    torch.save(model,'classifers/badnet_architecture.pth')
    
    """
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    print(model.state_dict())

    for name,param in model.named_parameters():
        print(name)
        print(param.shape)
    #torch.save(model,'classifers/net_architecture.pth')
    #"""
