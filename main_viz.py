import numpy
import argparse
import os
import sys
import torch
import torch.nn as nn

from disvae import init_specific_model
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
import not2imp
from not2imp.main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata
import torchvision.models as models
from scripts.test import FactorVAE
from net.models import LeNet_5
from models_32 import *
from cifar_misc import *
from resnet import ResNet18
import json

import argparse
from vqvae import Solver
from torch.autograd import Variable
import torch
from vqvae2 import VQVAE

from finetune import ModifiedVGG16Model

from utils import util_funcs
from models.model_utils import get_model, get_dataset
from utils.util_funcs import *

cifar_dim = 128
NUM_WORKERS = min(n_cpu, max(8, 12 * n_gpu))

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]

class CIFAR_VAE:
    def __init__(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder

class WrappedModel(nn.Module):
	def __init__(self,model):
		super(WrappedModel, self).__init__()
		self.module = model 
	def forward(self, x):
		return self.module(x)

def parse_arguments(args_to_parse):
    """Parse the command line arguments.
    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    description = "CLI for plotting using pretrained models of `disvae`"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,default='new_vae',
                        help="Name of the model for storing and loading purposes.")
    #parser.add_argument("plots", type=str, nargs='+', choices=PLOT_TYPES,
    #                    help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-r', '--n-rows', type=int, default=6,
                        help='The number of rows to visualize (if applicable).')
    parser.add_argument('-c', '--n-cols', type=int, default=7,
                        help='The number of columns to visualize (if applicable).')
    parser.add_argument('-t', '--max-traversal', default=2,
                        type=lambda v: check_bounds(v, lb=0, is_inclusive=False,
                                                    type=float, name="max-traversal"),
                        help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed. If `m>=0.5` then uses absolute value traversal, if `m<0.5` uses a percentage of the distribution (quantile). E.g. for the prior the distribution is a standard normal so `m=0.45` corresponds to an absolute value of `1.645` because `2m=90%%` of a standard normal is between `-1.645` and `1.645`. Note in the case of the posterior, the distribution is not standard normal anymore.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    parser.add_argument('-u', '--upsample-factor', default=1,
                        type=lambda v: check_bounds(v, lb=1, is_inclusive=True,
                                                    type=int, name="upsample-factor"),
                        help='The scale factor with which to upsample the image (if applicable).')
    parser.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    parser.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    
    # All the features here are added by me for novel watermark..
    parser.add_argument('--model', default='net', help='the model name')
    parser.add_argument('--dataset',type = str,default = 'mnist',help = 'The dataset you want to test on')
    parser.add_argument('--labels',type = str,default = None, help = 'If you want to generate watermarks only for a set of labels')
    parser.add_argument('--sensitive',action = 'store_true',help = 'generate sensitive watermark')
    parser.add_argument('--gray_box',action = 'store_true',help = 'generate gray_box watermark')
    parser.add_argument('--am_path',default = 'classifers/square_white_tar0_alpha0.00_mark(3,3).pth',help = 'for gray box model')
    parser.add_argument('--encyst',action = 'store_true',help = 'If you want to save a tensor containing samples of inner and outer boundary')
    parser.add_argument('--samples',type = int,default = 5,help = 'Number of boundary samples to be generated per latent dim')
    parser.add_argument('--gaussian',action = 'store_true',help = 'If examples are generated from gaussian noise in gray and black box')
    parser.add_argument('--rate',type = float,default = 0.1,help = 'the change in value of feature')
    parser.add_argument('--iter',type = int,default = 100,help = 'the change in value of feature')
    parser.add_argument('--arch_path', default='classifers/net_architecture.pth', help='the model architecture path')
    parser.add_argument('--model_path', default='classifers/net.pth', help='the classifier path')
    parser.add_argument('--multiple',action = 'store_true',help = 'If in case the random noise is added to all the dimensions, will always set to be tru in cifar10 and face')
    parser.add_argument('--show_plots',action = 'store_true',help = 'Show plots of sensitivity in sensitive samples')
    parser.add_argument('--am_path2',default = None,help = 'for gray box model the second attack model')
    parser.add_argument('--compress',action = 'store_true',help = 'If you want to test compression use diff. model')
    parser.add_argument('--scratch',action = 'store_true',help = 'If you want to take clean model as from scratch and attacked model as fine tuned one..')
    parser.add_argument('--mnist_gen',default = '200000',help = 'To choose among different generative models for mnist..')
    args = parser.parse_args()

    return args

def create_runCIFAR100(architecture, dataset, num_embeddings, num_workers, selection_fn, neighborhood, device, ckpt_epoch, embed_dim, data_path, size, download=False, **kwargs):
    #global args2, scheduler2
    vae_batch=128

    #print('creating data loaders')
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood,
                                                        selection_fn, size, **kwargs)
    checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)
    checkpoint_path = f'checkpoints/{checkpoint_name}'
    
    print('Loading model')
    model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=False, **kwargs)
    model.load_state_dict(torch.load(os.path.join('.', checkpoint_path),map_location = device), strict=False)

    return model

def cifar100_Generator(this_device):
    util_funcs.seed_generators(7)

    return create_runCIFAR100(alpha=0.3, architecture='vqvae', backward_dict=1, checkpoint_freq=5, checkpoint_path='./checkpoints', ckpt_epoch=195, data_path='./data', dataset='cifar100', device=this_device, dictionary_loss_weight=0.0, download=False, embed_dim=64, eval_iter=1, experiment_name='experiment_vq', is_enforce_sparsity=False, is_quantize_coefs=False, lr=0.0003, neighborhood=1, normalize_dict=True, normalize_x=True, normalize_z=False, num_embeddings=512, num_epochs=200, num_nonzero=2, num_strides=2, num_workers=8, parallelize=True, sample_gradients=1000, sample_size=25, sampling_iter=25, sched=None, seed=7, selection_fn='vanilla', size=64, summary_path='./summary', use_backwards_simd=True, vae_batch=128)


def main(args):
    
    set_seed(args.seed)
    experiment_name = args.name             #--------------------------------------------new_vae
    model_dir = os.path.join(RES_DIR, experiment_name)
    
    dataset = args.dataset
    
    if torch.cuda.is_available() and args.dataset!='mnist':
        device = 'cuda'
    else:
        device = 'cpu'


    if dataset != "mnist" and dataset != "cifar" and dataset != "face" and dataset!='cifar100':
        print('The dataset is not supported')
        sys.exit()

    

    if dataset=="mnist":
        if(args.mnist_gen=="200000"):
            model = FactorVAE(dataset,args.sensitive,args.mnist_gen)
        else:
            directory="results/"+args.mnist_gen
            MODEL_FILENAME = "model.pt"
            META_FILENAME = "specs.json"
            path_to_metadata = os.path.join(directory, META_FILENAME)
            with open(path_to_metadata) as metadata_file:
                metadata = json.load(metadata_file)
            img_size = metadata["img_size"]
            latent_dim = metadata["latent_dim"]
            model_type = metadata["model_type"]

            path_to_model = os.path.join(directory, MODEL_FILENAME)
            model = init_specific_model(model_type, img_size, latent_dim).to(device)
            model.load_state_dict(torch.load(path_to_model), strict=False)
            model.to(device)
            model.eval()
            print("Loading the model: "+args.mnist_gen)
    elif dataset=="cifar":

        args.multiple = True

        model = Solver(device)
        model.set_mode('eval')

        if device=='cuda':
            model.model = model.model.cuda()

        """
        encoder0 = WrappedModel(Encoder(cifar_dim))
        encoder0.load_state_dict(torch.load("cifar_vae/580_encoder.sd",map_location = device))
        encoder0.eval()
        decoder0 = WrappedModel(Decoder(cifar_dim))
        decoder0.load_state_dict(torch.load("cifar_vae/580_decoder.sd",map_location = device))
        decoder0.eval()

        model = CIFAR_VAE(encoder0,decoder0)
        """
    elif dataset=="cifar100":
        args.multiple = True
        model = cifar100_Generator(device)
        model = model.to(device)
        model.eval()
    elif dataset == "face":
        
        args.multiple = True
        model = VQVAE()
        model.load_state_dict(torch.load('distributed/vqvae_560.pt',map_location = device))
        model = model.to(device)
        model.eval()

    

    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     dataset=dataset,
                     max_traversal=args.max_traversal,
                     loss_of_interest='kl_loss_',
                     upsample_factor=args.upsample_factor,
                     mnist_gen = (args.mnist_gen!="200000"))
                     
    print('\nThe dataset used is : '+dataset)

    


    if args.dataset == 'cifar':
        if args.arch_path == 'classifers/net_architecture.pth':
            args.arch_path = 'classifers/resnet_architecture.pth'

        if args.model_path == 'classifers/net.pth':
            args.model_path = 'classifers/resnet18_comp.pth'
    
    elif args.dataset == 'cifar100':
        if args.arch_path == 'classifers/net_architecture.pth':
            args.arch_path = 'classifers/resnet_architecture100.pth'

        if args.model_path == 'classifers/net.pth':
            args.model_path = 'classifers/resnet18_comp100.pth'


    elif args.dataset == 'face':
        if args.arch_path == 'classifers/net_architecture.pth':
            args.arch_path = None

        if args.model_path == 'classifers/net.pth':
            if args.scratch:
                args.model_path = 'classifers/clean_face_model_scratch.pth'
            else:
                args.model_path = 'classifers/clean_face_model.pth'




    if args.encyst or True:

        PATH = args.arch_path

        if torch.cuda.is_available() and args.sensitive:        #only support with mnist
            this_device = torch.device("cuda")
            classifier = torch.load(PATH,map_location="cuda:0")
            PATH = args.model_path
            classifier.load_state_dict(torch.load(PATH,map_location="cuda:0"))
            classifier.cuda()

        elif not args.compress and not args.scratch:                 #only for cifar10 and mnist and cifar100..

            print("\nloading the model : "+args.model_path+"\n")

            classifier0 = torch.load(PATH,map_location=device)
            PATH = args.model_path
            classifier0.load_state_dict(torch.load(PATH,map_location=device))
            classifier = (classifier0)

            if args.dataset != 'mnist' and torch.cuda.is_available():
                classifier = classifier.cuda()
            # print(help(classifier))

        elif not args.scratch: 
            if args.dataset == "mnist":                  
                print('Loading net2.pth')
                classifier = LeNet_5(mask=True)
                classifier.load_state_dict(torch.load('compression_clfs/net2.pth',map_location='cpu'))
            elif args.dataset == "cifar":
                print('Loading the ResNet18')
                classifier = WrappedModel(ResNet18())
                classifier.load_state_dict(torch.load('compression_clfs/resnet2.pth',map_location = device)["net"])        
            elif args.dataset == 'face':
                print('Lading the vgg16 model')
                classifier = torch.load(args.model_path,map_location = device)
        else:
            print('Loading the vgg16 model trained from scratch')
            classifier = torch.load(args.model_path,map_location = device)

        if args.sensitive:
            inner_boundary,inner_sens,outer_boundary,outer_sens = viz.sensitive_encystSamples(classifier,args.samples,args.rate,args.iter,args.show_plots,sample_label = args.labels)

            dictionary = {}
            dictionary["inner_img"] = inner_boundary
            dictionary["inner_sens"] = inner_sens
            dictionary["outer_img"] = outer_boundary
            dictionary["outer_sens"] = outer_sens
            torch.save(dictionary,model_dir+f"/watermark_sens.pth")
         
        elif args.gray_box:

            print("\nloading the attacked model\n")
            attack_model = torch.load(args.arch_path,map_location=device)
            attack_model.load_state_dict(torch.load(args.am_path,map_location=device))

            if args.am_path2 is not None:
                attack_model2 = torch.load(args.arch_path,map_location=device)
                attack_model2.load_state_dict(torch.load(args.am_path2,map_location=device))
            else:
                attack_model2 = None

            inner_boundary,inner_pred,outer_boundary,outer_pred = viz.gray_encystSamples(classifier,attack_model,attack_model2,args.samples,args.rate,args.iter,args.multiple,args.gaussian,sample_label = args.labels)

            dictionary = {}
            dictionary["inner_img"] = inner_boundary
            dictionary["inner_pred"] = inner_pred
            dictionary["outer_img"] = outer_boundary
            dictionary["outer_pred"] = outer_pred
            torch.save(dictionary,model_dir+f"/watermark_gray.pth")

        else:
            inner_boundary,inner_pred,outer_boundary,outer_pred = viz.encystSamples(classifier,args.samples,args.rate,args.iter,args.multiple,args.gaussian,sample_label = args.labels)

            dictionary = {}
            dictionary["inner_img"] = inner_boundary
            dictionary["inner_pred"] = inner_pred
            dictionary["outer_img"] = outer_boundary
            dictionary["outer_pred"] = outer_pred
            torch.save(dictionary,model_dir+f"/watermark.pth")
    
    print("Program Terminated")

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)