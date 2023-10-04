#!/usr/bin/env python
"""client.py: argument parser for the ancestry_vae package"""

import argparse 

def argument_parser(argv_list):
    """Takes a list of arguments and outputs parameters or settings for the VAE
    Parameters
    ----------
    argv_list   : list of command line arguments
    Returns
    -------
    args         : list of arguments from the user
    """    
    parser=argparse.ArgumentParser()

    parser.add_argument("--name",default=None,
        help="name of dataset or analysis")

    parser.add_argument("--infile",
                        help="path to input H5Dpy file")
    
    parser.add_argument("--pwd",
                        help="path to position weight matrices directory")

    parser.add_argument("--out",default="results",
                        help="Path for the output directory")

    parser.add_argument("--max_epochs",default=500,type=int,
                        help="max training epochs, default=500")
    
    parser.add_argument("--model",default="full",type=str,
                        help="Model to use CNN+RNN or only CNN")

    parser.add_argument("--batch_size",default=20,type=int,
                        help="Batch size, default=20")

    parser.add_argument("--drop_out",default=0.1,type=float,
                        help="Drop-out, default=0.1")

    parser.add_argument("--save_model",default=False,action="store_true",
                        help="Save model as model.pt")

    parser.add_argument("--gpu",default=False,action="store_true",
                        help="Use GPU for computation")

    parser.add_argument("--seed",default=1,type=int,help="random seed, default: 1")

    parser.add_argument("--grid_search",default=False,action="store_true",
                        help='run grid search over network sizes and use the network with \
                              minimum test loss. default: False. ')

    parser.add_argument("--depth",default=3,type=int,
                        help='number of hidden layers, default=3')

    parser.add_argument("--width",default=1000,type=int,
                        help='nodes per hidden layer. default=1000')
    
    parser.add_argument("--tensorboard",default=False,action="store_true",
                        help='create log file to visualize in tensorboard')
        
    parser.add_argument("--save",default=False, action="store_true",
                        help="save model")

    return parser.parse_args(argv_list)
