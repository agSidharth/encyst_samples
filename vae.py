#!/usr/bin/env python3

# CUDA_VISIBLE_DEVICES=0 python examples/vae.py --dataset mnist --pretrain --model net
# import sys
# from path import Path
# folder = Path(__file__).abspath()
# sys.path.append(folder.parent.parent)
import trojanvision
from trojanvision.utils import summary
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.test import FactorVAE
import torch
import numpy as np
import matplotlib.pyplot as plt


RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
import matplotlib.gridspec as gridspec
# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True
load_ck=True
def defence_vae_change_plot(input,label,limit=2):

    reimage64 = input

    zo = Enc.encode(reimage64)

    z = zo[:, :, 0, 0].to(DEVICE)

    outimage_all_all = []

    label_all_all = []


    for row in potentialSet_short:

        outimage_all = []
        label_all = []

        outimage_all.append(reimage64.cpu().numpy()[0])

        label_all.append(label)

        add_val=np.linspace(-1*limit,limit,n_samples)

        for step in add_val:

            z_copy=z.clone()
            z_copy[:, row] = z[:, row] + step

            z_tilde = z_copy.clone()
            xxx = Enc.decode(z_tilde.unsqueeze(-1).unsqueeze(-1), toArray=False)
            outimage = torch.sigmoid(xxx).data  # 1*1*64*64

            # print(outimage.cpu().numpy()[0][0])
            outimage_all.append(outimage.cpu().numpy()[0][0].tolist())
            c_out=model(xxx)

            y_tilde = torch.argmax(c_out, dim=1).to(DEVICE)
            label_all.append(y_tilde.cpu().numpy()[0].tolist())
        outimage_all_all.append(outimage_all)
        label_all_all.append(label_all)

    return label_all_all,outimage_all_all

def plot_img(name,label_numpy,img_numpy):
    gs = gridspec.GridSpec(4, 21)
    plt.figure()
    for i in range(4):
        for j in range(21):
            plt.subplot2grid((4, 21), (i, j))
            # plt.subplot(4, 21, (i+1)*(j+1))
            plt.title(str(label_numpy[i][j]))
            # print(label_numpy[i][j])
            # print(img_numpy[i][j])

            plt.imshow(np.array(img_numpy[i][j]) / 255.)  # division by 255 to convert [0, 255] to [0, 1]
            plt.axis('off')
            # plt.show()
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model)

    loss, acc1 = model._validate()


    Enc = FactorVAE('mnist')
    potentialSet = [1, 2, 3, 4, 5, 9, 12, 13, 16, 17]
    potentialSet_short = [9, 13, 16, 17]
    n_samples = 20
    normalrange = [[-3.39912, 2.83359], [-2.23347, 1.76427], [-3.33213952, 2.4100915200000004],
                   [-3.1712631040000003, 2.2483367040000006], [-3.38752704, 3.9973510400000007],
                   [-4.353424, 3.9273840000000004], [-3.5737352000000002, 3.6870752],
                   [-3.9186784000000006, 4.770078400000001], [-4.384398400000001, 4.134798400000001],
                   [-3.1040488, 3.0058888]]

    classifier=model.model.eval().to(DEVICE)
    mnistdataset=dataset.get_org_dataset(mode="train")
    test_loader=dataset.get_dataloader(mode="train",dataset=mnistdataset,batch_size=BATCH_SIZE)
    exit_flag = False
    jishu = 0

    for j, (images, labels) in enumerate(test_loader):

        if exit_flag:
            break

        for i in range(images.size()[0]):
            if jishu > 20:
                exit_flag = True
                break
            img, label = images[i].to(DEVICE), labels[i].to(DEVICE)
            reimage64=img

            img_numpy = img.cpu().numpy()
            label_numpy = label.cpu().numpy()
            la, oima = defence_vae_change_plot(reimage64, label_numpy.tolist())
            Fimage_out = oima
            Flabelo_out = la
            print(jishu)
            jishu += 1
            plot_img('outimage/ori_' + str(jishu) + '.png', Flabelo_out, Fimage_out)





