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
import lpips


def read_img(path):
    return Image.fromarray(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB))

img_size = 224

transforms_1 = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

loss_fn_alex = lpips.LPIPS(net='alex')
sensitive = os.listdir('VGGFace-Clean/')
images = []

print('Reading directory..')

for file in sensitive:
	if '.png' in file or '.jpg' in file:
		im = read_img(f'VGGFace-Clean/{file}')
		img = transforms_1(im)
		tensor_img = img.float()
		images.append(tensor_img)


suma = 0
count = 0

random.shuffle(images)
print('Calculating loss..')
reference_img = images[0]
max_value = 0

number_of_samples = 20

for i in range(number_of_samples-1):
	for j in range(i+1,number_of_samples):
		first_img =  images[i]
		second_img = images[j]
		this_loss = loss_fn_alex(first_img, second_img)[0][0][0][0].detach().numpy()
		max_value = max(max_value,this_loss)
		suma += this_loss
		count += 1

threshold = suma/count
print(threshold)
print(max_value)

testing = os.listdir('generate/')
passed_images = 0
total_images = 0

for file in testing:
	if '.png' in file or '.jpg' in file:
		im = read_img(f'generate/{file}')
		img = transforms_1(im)
		tensor_img = img.float()
		total_images += 1

		this_loss = 0
		for base_img in images[0:number_of_samples]:
			this_loss = loss_fn_alex(base_img,tensor_img)[0][0][0][0].detach().numpy()
			if(this_loss>threshold):
				break
		if(this_loss<=threshold):
			passed_images+=1

print(passed_images)
print(total_images)

