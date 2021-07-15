import os, sys, time, pdb
import torch
from models_32 import *
from tqdm import tqdm
from imageio import imsave
from cifar_misc import *
import numpy as np
from imageio import imsave
import torchvision.transforms as transforms
import torchvision

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

zdim=128

encoder = torch.nn.DataParallel(Encoder(zdim))
encoder.load_state_dict(torch.load("cifar_vae/580_encoder.sd",map_location = 'cpu'))
decoder = torch.nn.DataParallel(Decoder(zdim))
decoder.load_state_dict(torch.load("cifar_vae/580_decoder.sd",map_location = 'cpu'))

decoder.eval()
encoder.eval()

num_eval_imgs = 50000
limit=2
row_of_z=0

ds = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                    (0.5, 0.5, 0.5), 
                                    (0.5, 0.5, 0.5)),
                           ]))
dl = torch.utils.data.DataLoader(ds, batch_size=1, 
                        shuffle=True, pin_memory=True, drop_last=True,
                        num_workers=4)

for iter_idx, (img, _) in enumerate(tqdm(dl)):

    #print(img.shape)
    bs, imsize = img.shape[0], img.shape[2]
    mu_logvar = encoder(img).view(bs, -1)
    mu = mu_logvar[:, 0:zdim]
    logvar = mu_logvar[:, zdim:]

    z0 = reparameterize(mu, logvar)
    print(z0.shape)
    z = z0[:, :].to(device)
    #print(z.shape)

    add_val = np.linspace(-1 * limit, limit, 20)

    for step in add_val:

        z_copy = z.clone()
        z_copy[:, row_of_z] = z[:, row_of_z] + step
        z_tilde = z_copy.clone()

        rec = decoder(z_tilde).view(bs, 3, imsize, imsize)
        out_imgs = rec.mul_(127.5).add_(127.5).clamp(0.0, 255.0)
        print(out_imgs.shape)
        out_imgs = out_imgs.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

        for img_idx, img in enumerate(out_imgs):
            imsave('images/img'+str(img_idx)+str(iter_idx)+str(step)+'.png', img)

