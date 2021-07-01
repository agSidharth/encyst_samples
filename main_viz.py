import argparse
import os
import sys
import torch
import torch.nn as nn

from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
import not2imp
from not2imp.main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata
import torchvision.models as models


PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]


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

    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("plots", type=str, nargs='+', choices=PLOT_TYPES,
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
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
    parser.add_argument('--model', default='resnet', help='the model name')

    # All the features here are added by me for novel watermark..
    parser.add_argument('--sensitive',action = 'store_true',help = 'generate sensitive watermark')
    parser.add_argument('--encyst',action = 'store_true',help = 'If you want to save a tensor containing samples of inner and outer boundary')
    parser.add_argument('--samples',type = int,default = 5,help = 'Number of boundary samples to be generated per latent dim')
    parser.add_argument('--natural',action = 'store_true',help = 'If examples are generated from natural samples')
    parser.add_argument('--rate',type = float,default = 0.1,help = 'the change in value of feature')
    parser.add_argument('--iter',type = int,default = 100,help = 'the change in value of feature')
    parser.add_argument('--arch_path', default='classifers/net_architecture.pth', help='the model architecture path')
    parser.add_argument('--model_path', default='classifers/net.pth', help='the classifier path')
    args = parser.parse_args()

    return args


def main(args):
    """Main function for plotting fro pretrained models.
    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    set_seed(args.seed)
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    model.eval()  # don't sample from latent: use mean
    dataset = meta_data['dataset']
    
    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     dataset=dataset,
                     max_traversal=args.max_traversal,
                     loss_of_interest='kl_loss_',
                     upsample_factor=args.upsample_factor)
    size = (args.n_rows, args.n_cols)
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = args.n_cols * args.n_rows
    samples = get_samples(dataset, num_samples, idcs=args.idcs)

    print('\nThe dataset used is : '+ str(dataset))
    if args.encyst:

        if (dataset=='mnist'):
            #Load the classifer to tell if boundary has been crossed or not.
            if(args.model=='resnet'):

                print("loading the resnet model")
                classifier = models.resnet18(pretrained=False)
                classifier.fc = nn.Linear(512,10)
                classifier.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

                PATH = args.model_path
                classifier.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

            elif (args.model=='net'):

                print("\nloading the Net model\n")
                PATH = args.arch_path
                classifier = torch.load(PATH,map_location=torch.device('cpu'))

                PATH = args.model_path
                classifier.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

        if args.sensitive:
            inner_boundary,inner_pred,outer_boundary,outer_pred = viz.sensitive_encystSamples(classifier,args.samples,args.natural,args.rate,args.iter)
        else:
            inner_boundary,inner_pred,outer_boundary,outer_pred = viz.encystSamples(classifier,args.samples,args.natural,args.rate,args.iter)

        dictionary = {}
        dictionary["inner_img"] = inner_boundary
        dictionary["inner_pred"] = inner_pred
        dictionary["outer_img"] = outer_boundary
        dictionary["outer_pred"] = outer_pred
        torch.save(dictionary,model_dir+f"/watermark.pth")

    else:
        if "all" in args.plots:
            args.plots = [p for p in PLOT_TYPES if p != "all"]
    
        for plot_type in args.plots:
            if plot_type == 'generate-samples':
                viz.generate_samples(size=size)
            elif plot_type == 'data-samples':
                viz.data_samples(samples, size=size)
            elif plot_type == "reconstruct":
                viz.reconstruct(samples, size=size)
            elif plot_type == 'traversals':
                viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                               n_per_latent=args.n_cols,
                               n_latents=args.n_rows,
                               is_reorder_latents=True)
            elif plot_type == "reconstruct-traverse":
                viz.reconstruct_traverse(samples,
                                         is_posterior=args.is_posterior,
                                         n_latents=args.n_rows,
                                         n_per_latent=args.n_cols,
                                         is_show_text=args.is_show_loss)
            elif plot_type == "gif-traversals":
                viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
            else:
                raise ValueError("Unkown plot_type={}".format(plot_type))
        
    print("Program Terminated")

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)