import os
import sys
from math import ceil, floor

import torchvision.transforms as transforms
import imageio
from PIL import Image
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
from utils.datasets import get_background
from utils.viz_helpers import get_samples
from utils.viz_helpers import (read_loss_from_file, add_labels, make_grid_img,
                               sort_list_by_other, FPS_GIF, concatenate_pad)
import random
from tqdm import tqdm
import copy
from imageio import imsave

TRAIN_FILE = "train_losses.log"
DECIMAL_POINTS = 3
GIF_FILE = "training.gif"
PLOT_NAMES = dict(generate_samples="samples.png",
                  data_samples="data_samples.png",
                  reconstruct="reconstruct.png",
                  traversals="traversals.png",
                  reconstruct_traverse="reconstruct_traverse.png",
                  gif_traversals="posterior_traversals.gif",)

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

class Visualizer():
    def __init__(self, model, dataset, model_dir,
                 save_images=True,
                 loss_of_interest=None,
                 display_loss_per_dim=False,
                 max_traversal=0.475,  # corresponds to ~2 for standard normal
                 upsample_factor=1):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : disvae.vae.VAE

        dataset : str
            Name of the dataset.

        model_dir : str
            The directory that the model is saved to and where the images will
            be stored.

        save_images : bool, optional
            Whether to save images or return a tensor.

        loss_of_interest : str, optional
            The loss type (as saved in the log file) to order the latent dimensions by and display.

        display_loss_per_dim : bool, optional
            if the loss should be included as text next to the corresponding latent dimension images.

        max_traversal: float, optional
            The maximum displacement induced by a latent traversal. Symmetrical
            traversals are assumed. If `m>=0.5` then uses absolute value traversal,
            if `m<0.5` uses a percentage of the distribution (quantile).
            E.g. for the prior the distribution is a standard normal so `m=0.45` c
            orresponds to an absolute value of `1.645` because `2m=90%%` of a
            standard normal is between `-1.645` and `1.645`. Note in the case
            of the posterior, the distribution is not standard normal anymore.

        upsample_factor : floar, optional
            Scale factor to upsample the size of the tensor
        """
        self.model = model
        self.latent_dim = self.model.z_dim if dataset=='mnist' else 128
        self.max_traversal = max_traversal
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset
        self.upsample_factor = upsample_factor

        self.potentialSet = [1, 2, 3, 4, 5, 9, 12, 13, 16, 17]
        self.potentialSet_short = [9, 13, 16, 17] if self.dataset == "mnist" else [1,2,3,4]
        self.mnist_normalrange = [[-3.39912, 2.83359], [-2.23347, 1.76427], [-3.33213952, 2.4100915200000004],
                   [-3.1712631040000003, 2.2483367040000006], [-3.38752704, 3.9973510400000007],
                   [-4.353424, 3.9273840000000004], [-3.5737352000000002, 3.6870752],
                   [-3.9186784000000006, 4.770078400000001], [-4.384398400000001, 4.134798400000001],
                   [-3.1040488, 3.0058888]]

        self.total_dim = len(self.potentialSet_short)

        #if loss_of_interest is not None:
        #    self.losses = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE),
        #                                      loss_of_interest)

    def delta_fn(self,sample_original,classifier,output_classes):

        torch.set_grad_enabled(True)
        #classifier.train()
        self.model.VAE.decode.train()
        
        for name, param in classifier.named_parameters():           #because classifier.train() does not work..
            param.requires_grad_(True)
        
        delta = torch.zeros(sample_original.shape[0]).to(self.device)

        total_lk = 0

        for i in range(output_classes):

            Lk = 0

            sample = sample_original.clone().detach().requires_grad_(True).to(self.device)
        
            new_dim_sample = sample.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            xxx = self.model.VAE.decode(new_dim_sample)
            #xxx.retain_grad()

            reconstructed = torch.sigmoid(xxx)
            #print(reconstructed.shape)
            
            output = classifier((reconstructed).to(self.device))
            #print(output)

            loss = output[0][i]
            loss.backward(retain_graph = True,create_graph = True)      # to retain graph for second partial derivative
            
            #make_dot(loss).render("loss_fn",format = "png")

            for name, param in classifier.named_parameters():

                y = (param.grad).requires_grad_(True)
                #print(y)

                Lk +=  torch.norm(y,p = 2)*torch.norm(y, p=2)
                #print(Lk)
                param.grad = torch.zeros_like(param)

            
            for name,param in self.model.VAE.decode.named_parameters():
                x = (param.grad).requires_grad_(True)
                #print(x)
                
                Lk +=  torch.norm(x,p = 2)*torch.norm(x,p=2)
                #print(Lk)
                param.grad = torch.zeros_like(param)

            total_lk = total_lk + Lk
            #print(sample.grad)
            sample.grad = torch.zeros_like(sample)
            #print(sample.grad)
            
            Lk.backward()
            #print(sample.grad)
            #make_dot(Lk).render("Lk",format = "png")


            #torch.norm(sample.grad,p=2)            #maybe taking only the unit vector...
            delta = delta + sample.grad#/(torch.norm(sample.grad,p=2))

            # preparing for the next iteration.
            for name,param in self.model.VAE.decode.named_parameters():
                param.grad = torch.zeros_like(param)

            for name,param in classifier.named_parameters():
                param.grad = torch.zeros_like(param)

            sample.grad = torch.zeros_like(sample)
            loss = 0

        return delta,(total_lk.cpu().detach().numpy()+0)

    def update_rate(self,rate,iterations):

        if(iterations%5==0):
            return rate*(0.95)
        return rate


    def sensitive_encystSamples(self,classifier,samples_per_dim = 10,rate = 0.00005,max_iterations=1000,show_plots = False,sample_label = None,output_classes = 10):
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        for name,param in classifier.named_parameters():
            param = param.to(self.device)

        #self.model = self.model.to(self.device)

        print('\nGenerating sensitive samples\n')
        initial_rate = rate

        
        classifier.eval()
        seed = random.randint(1,1000)
        

        inner_boundary = {}
        outer_boundary = {}
        inner_sens = {}
        outer_sens = {}
        

        data,_ = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)
        img_size = data[0].shape
        
        inner_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)
        outer_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)


        dim_count = 0
        
        for dim in self.potentialSet_short:       

            print('The dimension number is:'+str(dim))
            random.seed(seed)
            seed = int(seed*4/3)

            data,_ = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)


            img_size = data[0].shape
            inner_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)
            outer_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)  
            inner_img_sens = torch.zeros(samples_per_dim).to(self.device)
            outer_img_sens = torch.zeros(samples_per_dim).to(self.device)


            data = data.to(self.device)
            
            for sample_num in range(samples_per_dim):

                #print(data.shape)
                sample0 = self.model.encode(data[sample_num])
                
                sample = sample0[0, :, 0, 0].to(self.device)
                #print(sample.shape)
                #sys.exit()

                rate = initial_rate
                first_lk = None
                
                iterations = 0

                Lk_list = []

                print('Increasing the sensitivity first for '+str(max_iterations/2)+' iterations')

                pbar = tqdm(total = max_iterations/2)

                while(iterations<max_iterations/2):

                    delta,total_lk = self.delta_fn(sample,classifier,output_classes)
                    delta = delta.to(self.device)

                    #print(delta)
                    sample = sample + rate*(delta)

                    xxx = self.model.decode(sample.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), toArray=False)
                    img  = torch.sigmoid(xxx).data

                    classifier.eval()
                    _,pred = torch.max(classifier((img).to(self.device)), 1)
                    #print(pred)
                    prev_pred = pred

                    iterations = iterations + 1

                    #print(total_lk)
                    Lk_list.append(total_lk)

                    pbar.update(1)

                    if first_lk is None:
                        first_lk = total_lk
                    
                    rate = self.update_rate(rate,iterations)
                
                pbar.close()

                init_Lk = total_lk

                if show_plots:
                    plt.title("Sensitivity vs. Iterations")
                    plt.xlabel("Iterations")
                    plt.ylabel("Sensitivity")
                    plt.plot(Lk_list,label = "Sensitivity")
                    plt.legend()
                    plt.show()

                print('Now will converge as soon as change of label takes place...')
                pbar = tqdm(total = max_iterations/2)

                rate = rate*(1.25)      #since the rate must have fallen a lot by now..
                while(torch.equal(pred,prev_pred) and (iterations)<max_iterations):

                    prev_img = img
                    init_Lk = total_lk

                    delta,total_lk = self.delta_fn(sample,classifier,output_classes)
                    delta = delta.to(self.device)

                    sample = sample + rate* (delta)

                    xxx = self.model.decode(sample.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), toArray=False)
                    img  = torch.sigmoid(xxx).data

                    _,pred = torch.max(classifier((img).to(self.device)), 1)

                    iterations = iterations + 1

                    rate = self.update_rate(rate,iterations)

                    pbar.update(1)
                
                pbar.close()
                print('For the sample = '+str(sample_num))
                #print(torch.sum(torch.square(img - prev_img))/torch.sum(torch.square(prev_img)))
                
                if(torch.equal(pred,prev_pred)):
                    print('No image found after changing this particular feature')
                    print('The Lk at iteration=0 was : '+str(first_lk))
                    print('Second last sensitivity : '+str(init_Lk))
                    print('Last sensitivity : '+str(total_lk))
                    #print('They have equal prediction')
                    
                    prev_img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    print('\n')
                else:
                    print('The previous prediction : '+str(prev_pred[0]))
                    print('The new prediction : '+str(pred[0]))
                    print('Iterations needed : '+str(iterations))
                    print('The Lk at iteration=0 was : '+str(first_lk))
                    print('Inner sensitivity : '+str(init_Lk))
                    print('Outer sensitivity : '+str(total_lk))
                    print('\n')
                    
                inner_grid[sample_num + dim_count*(samples_per_dim)] = prev_img[0]
                inner_img[sample_num] = prev_img
                inner_img_sens[sample_num] = init_Lk
                outer_grid[sample_num + dim_count*(samples_per_dim)] = img[0]
                outer_img[sample_num] = img
                outer_img_sens[sample_num] = total_lk
            
            inner_boundary[dim_count] = inner_img 
            inner_sens[dim_count] = inner_img_sens
            outer_boundary[dim_count] = outer_img
            outer_sens[dim_count] = outer_img_sens
            dim_count = dim_count + 1
        
        grid_size = (self.total_dim,samples_per_dim)
        trash = self._save_or_return(inner_grid,grid_size,"Inner_Boundary_Sensitive.png")
        trash = self._save_or_return(outer_grid,grid_size,"Outer_Boundary_Sensitive.png")
        return inner_boundary,inner_sens,outer_boundary,outer_sens

    def gray_encystSamples(self,classifier,attacked_clf,attacked_clf2,samples_per_dim=10,rate=0.05,max_iterations=5000,mutiple=False,gaussian_noise = False,sample_label = None):

        torch.set_grad_enabled(False)

        self.device = torch.device('cpu')

        for name,param in classifier.named_parameters():
            param = param.to(self.device)

        #self.model = self.model.to(self.device)

        print('\nGenerating gray noise encyst samples\n')
        if not mutiple:
            print("Adding noise to a single dim at a time\n")
        else:
            print('Adding noise to complete vector at a time\n')

        #classifier = classifier.to(self.device)
        classifier.eval()
        attacked_clf.eval()

        if attacked_clf2 is not None:
            attacked_clf2.eval()

        seed = random.randint(1,1000)
        
        inner_boundary = {}
        outer_boundary = {}
        inner_pred = {}
        outer_pred = {}
        
        data,_ = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)
        img_size = data[0].shape
        
        inner_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)
        outer_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)
        
        dim_count = 0
        
        for dim in self.potentialSet_short:
            
            print('The dimension number is:'+str(dim))
            random.seed(seed)
            seed = int(seed*4/3)

            data,label_list = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)
            
            img_size = data[0].shape
            inner_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)
            outer_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)  
            inner_img_pred = torch.zeros(samples_per_dim).to(self.device)
            outer_img_pred = torch.zeros(samples_per_dim).to(self.device)

            data = data.to(self.device)
            print(data.shape)

            if self.dataset == 'cifar':

                bs, imsize = data.shape[0], data.shape[2]

                mu_logvar = self.model.encoder(data).view(bs, -1)
                mu = mu_logvar[:, 0:self.latent_dim]
                
                logvar = mu_logvar[:, self.latent_dim:]
                sample_pool = reparameterize(mu, logvar)

                rec = self.model.decoder(sample_pool).view(bs, 3, imsize, imsize)
                img_pool = rec.mul_(127.5).add_(127.5).clamp(0.0, 255.0)          

            for sample_num in range(samples_per_dim):

                factor = random.randint(1,2)            #for ensuring a feature is both decreased and increased max_iter times..
                if factor==2:
                    factor = -1

                if self.dataset == "mnist":
                    sample0 = self.model.encode(data[sample_num])
                    
                    sample = sample0[:, :, 0, 0].to(self.device)
                                         
                    xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                    img = torch.sigmoid(xxx).data
                
                elif self.dataset=="cifar":
                    sample = sample_pool[sample_num].clone().unsqueeze_(0)
                    img = img_pool[sample_num].clone().unsqueeze_(0)                    
                                
                _,pred = torch.max(classifier((img).to(self.device)), 1)
                _,dirty_pred = torch.max(attacked_clf((img).to(self.device)),1)

                if attacked_clf2 is not None:
                    _,dirty_pred2 = torch.max(attacked_clf2((img).to(self.device)),1)
                else:
                    dirty_pred2 = dirty_pred
                
                initial_sample = sample
                initial_img = img
                iterations = 0
                
                while((torch.equal(pred,dirty_pred) and torch.equal(pred,dirty_pred2) and iterations<max_iterations) or iterations==0):
                    
                    prev_img = img

                    if not mutiple:
                        if not gaussian_noise:
                            noise = 1
                        else:
                            noise = abs(np.random.normal(loc = 0,scale = 1))
                        sample[:,dim] = sample[:,dim] + factor*rate*noise
                    else:
                        if not gaussian_noise:
                            noise = torch.ones_like(sample).to(self.device)
                        else:
                            noise = torch.randn_like(sample).to(self.device)
                        sample = sample + factor*rate*noise

                    if self.dataset == "mnist":
                        xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                        img = torch.sigmoid(xxx).data
                        
                    elif self.dataset == "cifar":
                        rec1 = self.model.decoder(sample).view(1, 3, 32,32)
                        img = rec1.mul_(127.5).add_(127.5).clamp(0.0, 255.0)  
                    
                    _,pred = torch.max(classifier((img).to(self.device)), 1)
                    _,dirty_pred = torch.max(attacked_clf((img).to(self.device)),1)

                    if attacked_clf2 is not None:
                        _,dirty_pred2 = torch.max(attacked_clf2((img).to(self.device)),1)
                    else:
                        dirty_pred2 = dirty_pred

                    iterations = iterations + 1
                
                if(torch.equal(pred,dirty_pred) and torch.equal(pred,dirty_pred2)):
                    
                    iterations = 0
                    factor = factor*(-1)
                    sample = initial_sample
                    img = initial_img
                    
                    while(torch.equal(pred,dirty_pred) and torch.equal(pred,dirty_pred2) and iterations<max_iterations):
                        prev_img = img

                        if not mutiple:
                            if not gaussian_noise:
                                noise = 1
                            else:
                                noise = abs(np.random.normal(loc = 0,scale = 1))
                            sample[:,dim] = sample[:,dim] + factor*rate*noise
                        else:
                            if not gaussian_noise:
                                noise = torch.ones_like(sample).to(self.device)
                            else:
                                noise = torch.randn_like(sample).to(self.device)
                            sample = sample + factor*rate*noise

                        if self.dataset == "mnist":
                            xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                            img = torch.sigmoid(xxx).data
                            
                        elif self.dataset == "cifar":
                            rec1 = self.model.decoder(sample).view(1, 3, 32,32)
                            img = rec1.mul_(127.5).add_(127.5).clamp(0.0, 255.0)

                        _,pred = torch.max(classifier((img).to(self.device)), 1)
                        _,dirty_pred = torch.max(attacked_clf((img).to(self.device)),1)

                        if attacked_clf2 is not None:
                            _,dirty_pred2 = torch.max(attacked_clf2((img).to(self.device)),1)
                        else:
                            dirty_pred2 = dirty_pred

                        iterations = iterations + 1
                
                print('For the sample = '+str(sample_num))
                #print(torch.sum(torch.square(img - prev_img))/torch.sum(torch.square(prev_img)))
                
                if(torch.equal(pred,dirty_pred) and torch.equal(pred,dirty_pred2)):
                    print('No image found after changing this particular feature')
                    #print('They have equal prediction')
                    
                    prev_img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    print('\n')
                else:
                    #print('The previous prediction : '+str(prev_pred[0]))
                    #print('The new prediction : '+str(pred[0]))
                    print('\n')
                    
                inner_grid[sample_num + dim_count*(samples_per_dim)] = prev_img[0]
                inner_img[sample_num] = prev_img
                inner_img_pred[sample_num] = label_list[sample_num]
                outer_grid[sample_num + dim_count*(samples_per_dim)] = img[0]
                outer_img[sample_num] = img
                outer_img_pred[sample_num] = label_list[sample_num]
            
            inner_boundary[dim_count] = inner_img 
            inner_pred[dim_count] = inner_img_pred
            outer_boundary[dim_count] = outer_img
            outer_pred[dim_count] = outer_img_pred
            dim_count = dim_count + 1
        
        grid_size = (self.total_dim,samples_per_dim)
        trash = self._save_or_return(inner_grid,grid_size,"Inner_Boundary_gray.png")
        trash = self._save_or_return(outer_grid,grid_size,"Outer_Boundary_gray.png")
        return inner_boundary,inner_pred,outer_boundary,outer_pred
        


    def encystSamples(self,classifier,samples_per_dim=10,rate=0.05,max_iterations=5000,mutiple=False,gaussian_noise = False,sample_label = None):

        torch.set_grad_enabled(False)
        self.device = torch.device('cpu')

        for name,param in classifier.named_parameters():
            param = param.to(self.device)

        print('\nGenerating random noise encyst samples\n')
        if not mutiple:
            print("Adding noise to a single dim at a time\n")
        else:
            print('Adding noise to complete vector at a time\n')

        #classifier = classifier.to(self.device)
        classifier.eval()
        seed = random.randint(1,1000)
        
        inner_boundary = {}
        outer_boundary = {}
        inner_pred = {}
        outer_pred = {}
        
        data,_ = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)
        img_size = data[0].shape
        
        inner_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)
        outer_grid = torch.zeros(self.total_dim*samples_per_dim,img_size[0],img_size[1],img_size[2]).to(self.device)
        
        dim_count = 0
        
        for dim in self.potentialSet_short:
            
            print('The dimension number is:'+str(dim))
            random.seed(seed)
            seed = int(seed*4/3)

            data,label_list = get_samples(self.dataset, samples_per_dim,LABELS = sample_label)
            

            img_size = data[0].shape
            inner_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)
            outer_img = torch.zeros(samples_per_dim,1,img_size[0],img_size[1],img_size[2]).to(self.device)  
            inner_img_pred = torch.zeros(samples_per_dim).to(self.device)
            outer_img_pred = torch.zeros(samples_per_dim).to(self.device)

            data = data.to(self.device)

            if self.dataset == 'cifar':

                bs, imsize = data.shape[0], data.shape[2]

                mu_logvar = self.model.encoder(data).view(bs, -1)
                mu = mu_logvar[:, 0:self.latent_dim]
                
                logvar = mu_logvar[:, self.latent_dim:]
                sample_pool = reparameterize(mu, logvar)

                rec = self.model.decoder(sample_pool).view(bs, 3, imsize, imsize)
                img_pool = rec.mul_(127.5).add_(127.5).clamp(0.0, 255.0)

                #print(sample_pool.shape)
                
            for sample_num in range(samples_per_dim):

                factor = random.randint(1,2)            #for ensuring a feature is both decreased and increased max_iter times..
                if factor==2:
                    factor = -1

                #print(data.shape)
                if self.dataset == "mnist":
                    sample0 = self.model.encode(data[sample_num])
                    
                    sample = sample0[:, :, 0, 0].to(self.device)
                    
                    xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                    img = torch.sigmoid(xxx).data
                elif self.dataset == "cifar":
                    sample = sample_pool[sample_num].unsqueeze_(0)
                    img = img_pool[sample_num].unsqueeze_(0)
                
                #print('\n\n')
                #print(img.get_device())

                _,pred = torch.max(classifier((img).to(self.device)), 1)
                prev_pred = pred
                
                initial_sample = sample
                initial_img = img
                iterations = 0
                
                while(torch.equal(pred,prev_pred) and iterations<max_iterations):
                    prev_img = img

                    if not mutiple:
                        if not gaussian_noise:
                            noise = 1
                        else:
                            noise = abs(np.random.normal(loc = 0,scale = 1))
                        sample[:,dim] = sample[:,dim] + factor*rate*noise
                    else:
                        if not gaussian_noise:
                            noise = torch.ones_like(sample).to(self.device)
                        else:
                            noise = torch.randn_like(sample).to(self.device)
                        sample = sample + factor*rate*noise

                    if self.dataset == "mnist":
                        xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                        img = torch.sigmoid(xxx).data
                        
                    elif self.dataset == "cifar":
                        rec1 = self.model.decoder(sample).view(1, 3, 32,32)
                        img = rec1.mul_(127.5).add_(127.5).clamp(0.0, 255.0)

                    _,pred = torch.max(classifier((img).to(self.device)), 1)
                    iterations = iterations + 1
                
                if(torch.equal(pred,prev_pred)):
                    
                    iterations = 0
                    factor = factor*(-1)
                    sample = initial_sample
                    img = initial_img
                    
                    while(torch.equal(pred,prev_pred) and iterations<max_iterations):
                        prev_img = img

                        if not mutiple:
                            if not gaussian_noise:
                                noise = 1
                            else:
                                noise = abs(np.random.normal(loc = 0,scale = 1))
                            sample[:,dim] = sample[:,dim] + factor*rate*noise
                        else:
                            if not gaussian_noise:
                                noise = torch.ones_like(sample).to(self.device)
                            else:
                                noise = torch.randn_like(sample).to(self.device)
                            sample = sample + factor*rate*noise

                        if self.dataset == "mnist":
                            xxx = self.model.decode(sample.unsqueeze(-1).unsqueeze(-1), toArray=False)
                            img = torch.sigmoid(xxx).data
                            
                        elif self.dataset == "cifar":
                            rec1 = self.model.decoder(sample).view(1, 3, 32, 32)
                            img = rec1.mul_(127.5).add_(127.5).clamp(0.0, 255.0)

                        _,pred = torch.max(classifier((img).to(self.device)), 1)
                        iterations = iterations + 1
                
                print('For the sample = '+str(sample_num))
                #print(torch.sum(torch.square(img - prev_img))/torch.sum(torch.square(prev_img)))
                
                if(torch.equal(pred,prev_pred)):
                    print('No image found after changing this particular feature')
                    #print('They have equal prediction')
                    
                    prev_img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    img = torch.zeros(1,img_size[0],img_size[1],img_size[2]).to(self.device)
                    print('\n')
                else:
                    out_imgs = img.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    imsave('cifar_images/cifar_live'+str(dim)+str(sample_num)+'.png', out_imgs[0])
                    #print('The previous prediction : '+str(prev_pred[0]))
                    #print('The new prediction : '+str(pred[0]))
                    print('\n')
                    
                inner_grid[sample_num + dim_count*(samples_per_dim)] = prev_img[0]
                inner_img[sample_num] = prev_img
                inner_img_pred[sample_num] = label_list[sample_num]
                outer_grid[sample_num + dim_count*(samples_per_dim)] = img[0]
                outer_img[sample_num] = img
                outer_img_pred[sample_num] = label_list[sample_num]
            
            inner_boundary[dim_count] = inner_img 
            inner_pred[dim_count] = inner_img_pred
            outer_boundary[dim_count] = outer_img
            outer_pred[dim_count] = outer_img_pred
            dim_count = dim_count + 1
        
        grid_size = (self.total_dim,samples_per_dim)
        trash = self._save_or_return(inner_grid,grid_size,"Inner_Boundary.png")
        trash = self._save_or_return(outer_grid,grid_size,"Outer_Boundary.png")
        return inner_boundary,inner_pred,outer_boundary,outer_pred


    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # from 0.05 to -1.645

        # symmetrical traversals
        return (-1 * max_traversal, max_traversal)

    def _traverse_line(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.

        n_samples : int
            Number of samples to generate.

        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
        if data is None:
            # mean of prior for other dimensions
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)

        else:
            if data.size(0) > 1:
                raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                   std=post_std_idx),
                                        steps=n_samples)

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        return samples

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        if self.save_images and not is_force_return:
            filename = os.path.join(self.model_dir, filename)
            save_image(to_plot, filename, **kwargs)
        else:
            make_grid_img(to_plot, **kwargs)
        return -1

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        generated = self._decode_latents(prior_samples)
        return self._save_or_return(generated.data, size, PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(8, 8)):
        """Plot samples from the dataset

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[:size[0] * size[1], ...]
        return self._save_or_return(data, size, PLOT_NAMES["data_samples"])

    def reconstruct(self, data, size=(8, 8), is_original=True, is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains

        is_original : bool, optional
            Whether to exclude the original plots.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        if is_original:
            if size[0] % 2 != 0:
                raise ValueError("Should be even number of rows when showing originals not {}".format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self._save_or_return(to_plot, size, PLOT_NAMES["reconstruct"],
                                    is_force_return=is_force_return)

    def traversals(self,
                   data=None,
                   is_reorder_latents=False,
                   n_per_latent=8,
                   n_latents=None,
                   is_force_return=False):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_reorder_latents : bool, optional
            If the latent dimensions should be reordered or not

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data)
                          for dim in range(self.latent_dim)]
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))

        if is_reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape)
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data, size, filename,
                                    is_force_return=is_force_return)

    def reconstruct_traverse(self, data,
                             is_posterior=True,
                             n_per_latent=8,
                             n_latents=None,
                             is_show_text=False):
        """
        Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_posterior : bool, optional
            Whether to sample from the posterior.

        is_show_text : bool, optional
            Whether the KL values next to the traversal rows.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim

        reconstructions = self.reconstruct(data[:2 * n_per_latent, ...],
                                           size=(2, n_per_latent),
                                           is_force_return=True)
        traversals = self.traversals(data=data[0:1, ...] if is_posterior else None,
                                     is_reorder_latents=True,
                                     n_per_latent=n_per_latent,
                                     n_latents=n_latents,
                                     is_force_return=True)

        concatenated = np.concatenate((reconstructions, traversals), axis=0)
        concatenated = Image.fromarray(concatenated)

        if is_show_text:
            losses = sorted(self.losses, reverse=True)[:n_latents]
            labels = ['orig', 'recon'] + ["KL={:.4f}".format(l) for l in losses]
            concatenated = add_labels(concatenated, labels)

        filename = os.path.join(self.model_dir, PLOT_NAMES["reconstruct_traverse"])
        concatenated.save(filename)

    def gif_traversals(self, data, n_latents=None, n_per_gif=15):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        data : bool
            Data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        """
        n_images, _, _, width_col = data.shape
        width_col = int(width_col * self.upsample_factor)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(data=data[i:i + 1, ...], is_reorder_latents=True,
                                   n_per_latent=n_per_gif, n_latents=n_latents,
                                   is_force_return=True)

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)

            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(grid[:, [(j + 1) * padding_width + j * width_col + i
                                            for i in range(width_col)], :])

        pad_values = (1 - get_background(self.dataset)) * 255
        all_cols = [concatenate_pad(cols, pad_size=2, pad_values=pad_values, axis=1)
                    for cols in all_cols]

        filename = os.path.join(self.model_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)


class GifTraversalsTraining:
    """Creates a Gif of traversals by generating an image at every training epoch.

    Parameters
    ----------
    model : disvae.vae.VAE

    dataset : str
        Name of the dataset.

    model_dir : str
        The directory that the model is saved to and where the images will
        be stored.

    is_reorder_latents : bool, optional
        If the latent dimensions should be reordered or not

    n_per_latent : int, optional
        The number of points to include in the traversal of a latent dimension.
        I.e. number of columns.

    n_latents : int, optional
        The number of latent dimensions to display. I.e. number of rows. If `None`
        uses all latents.

    kwargs:
        Additional arguments to `Visualizer`
    """

    def __init__(self, model, dataset, model_dir,
                 is_reorder_latents=False,
                 n_per_latent=10,
                 n_latents=None,
                 **kwargs):
        self.save_filename = os.path.join(model_dir, GIF_FILE)
        self.visualizer = Visualizer(model, dataset, model_dir,
                                     save_images=False, **kwargs)

        self.images = []
        self.is_reorder_latents = is_reorder_latents
        self.n_per_latent = n_per_latent
        self.n_latents = n_latents if n_latents is not None else model.latent_dim

    def __call__(self):
        """Generate the next gif image. Should be called after each epoch."""
        cached_training = self.visualizer.model.training
        self.visualizer.model.eval()
        img_grid = self.visualizer.traversals(data=None,  # GIF from prior
                                              is_reorder_latents=self.is_reorder_latents,
                                              n_per_latent=self.n_per_latent,
                                              n_latents=self.n_latents)
        self.images.append(img_grid)
        if cached_training:
            self.visualizer.model.train()

    def save_reset(self):
        """Saves the GIF and resets the list of images. Call at the end of training."""
        imageio.mimsave(self.save_filename, self.images, fps=FPS_GIF)
        self.images = []