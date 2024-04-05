import yaml
import urllib
import pandas as pd
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
import cv2
import warnings
import tqdm
import sys
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from parameter_and_input_saliency import save_gradients, compute_input_space_saliency, sort_filters_layer_wise
from datasets import load_dataset
from pacs import pacs
from pcsa import pcsa
from pasc import pasc
from acsp import acsp
from mnist_svhn import mnist_svhn
from imagenet_c import imagenet_c
from imagenet import imagenet


parser = argparse.ArgumentParser(description='Input Space Saliency')
parser.add_argument('--model', default='resnet50', type=str, help='name of architecture')
parser.add_argument('--data_to_use', default='ImageNet', type=str, help='which dataset to use (ImageNet or ImageNet_A)')

# Logging
# parser.add_argument('--project_name', default='input_space_saliency', type=str, help='project name for Comet ML')
parser.add_argument('--figure_folder_name', default='input_space_saliency', type=str, help='directory to save figures')

# Modes for the signed saliency model: by default, regular loss on the given example is used.
#All final experiments were done with the following options off
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true', help='Use logits to compute parameter saliency')
parser.add_argument('--logit_difference', action='store_true', help='Use logit difference as parameter saliency loss')


#Boosting for input-space saliency
parser.add_argument('--boost_factor', default=100.0, type=float, help='boost factor for salient filters')
parser.add_argument('--k_salient', default=10, type=int, help='num filters to boost')

parser.add_argument('--compare_random', action='store_true',
                    help='whether to boost k random filters for comparison')
# parser.add_argument('--least_salient', action='store_true',
#                     help='whether to boost k least salient filters for comparison to frying most salient')

#Smoothing input space saliency (SmoothGrad-like, should be set to default, off at all times)
parser.add_argument('--noise_iters', default=1, type=int, help='number of noises to average across')
parser.add_argument('--noise_percent', default=0, type=float, help='std of the noises')

#Pick reference image
#Either using an image from raw_images/ folder
parser.add_argument('--image_path', default='raw_images/great_white_shark_mispred_as_killer_whale.jpeg', type=str, help='image id from valset to use')
parser.add_argument('--image_target_label', default=None, type=int, help='image label (number from 0 to 999 according to ImageNet labels)')
#Or using the i-th image from ImageNet validation set, for this ImageNet validation set path must be specified
parser.add_argument('--reference_id', default=None, type=int, help='image id from valset to use') #107 for great white shark

#PATHS
parser.add_argument('--imagenet_val_path', default='path_to_imagenet', type=str, help='ImageNet validation set path')#/home/rilevin/data/ImageNet/val
# parser.add_argument('--testset_stats_path', default='', type=str, help='filter saliency over the testset (where to save)')
# parser.add_argument('--inference_file_path', default='', type=str, help='where to save network inference results')


if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)
    print('In main')
    # print('pytorch version: ', torch.__version__)
    ###########################################################
    ####Define net, testset, precompute testset avg saliency
    ###########################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    args.model = 'resnet50'
    experiment = None #Used to be a comet_ml experiment for logging

    print('device :', device)
    print('args.model=', args.model)
    print('args.data_to_use=', args.data_to_use)
    print('args.figure_folder_name=', args.figure_folder_name)
    print('args.signed=', args.signed)
    print('args.logit=', args.logit)
    print('args.logit_difference=', args.logit_difference)
    print('args.imagenet_val_path=', args.imagenet_val_path)

    # path for saving stats
    model_helpers_root_path = os.path.join('helper_objects', args.model)

    if not os.path.exists(model_helpers_root_path):
        print('No helper objects directory exists for this model, creating one\n')
        os.makedirs(model_helpers_root_path)

    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])

    inv_transform_test = transforms.Compose([
        transforms.Normalize(mean=(0., 0., 0.), std=(1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1., 1., 1.)),
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ImageNet validation set
    if args.data_to_use == 'ImageNet':
        dataset_path = args.imagenet_val_path
    else:
        raise NotImplementedError

    
    # Model
    print('==> Building model..', flush=True)
    print(args.model, flush=True)

    if args.model == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
    elif args.model == 'vgg19':
        net = torchvision.models.vgg19(pretrained=True)
    elif args.model == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
    elif args.model == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
    elif args.model == 'vit':
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    else:
        #Other torchvision models should be inserted here
        raise NotImplementedError

    net = net.to(device)
    net.eval()

    layer_to_filter_id = {}
    filter_id_to_layer = {} # tuple of (layer name, index)
    conv_filter_id_to_layer = {} # tuple of (layer name, index)
    conv_idx = []
    ind = 0
    conv_filter_num = 0

    if args.model != 'vit':
        for layer_num, (name, param) in enumerate(net.named_parameters()):
            # print(name, param.shape)
            if len(param.size()) == 4:
                if 'conv' not in name:
                    print('Not a conv layer {}: {}'.format(name, layer_num))

                for j in range(param.size()[0]):
                    if 'conv' in name:
                        conv_idx.append(ind + j)
                        conv_filter_id_to_layer[conv_filter_num] = (name, j)
                        conv_filter_num += 1

                    if name not in layer_to_filter_id:
                        layer_to_filter_id[name] = [ind + j]
                    else:
                        layer_to_filter_id[name].append(ind + j)

                    filter_id_to_layer[ind + j] = (name, j)

                ind += param.size()[0]
    else:
        num_heads = 12
        print('vit heads num:', num_heads)
        for layer_num, (name, param) in enumerate(net.named_parameters()):
            if len(param.size()) == 2 and (param.size()[0] == param.size()[1] * 3):
                for j in range(3 * num_heads):
                    conv_idx.append(ind + j)
                    filter_id_to_layer[ind + j] = (name, j)

                    if name not in layer_to_filter_id:
                        layer_to_filter_id[name] = [ind + j]
                    else:
                        layer_to_filter_id[name].append(ind + j)

                ind += 3 * num_heads

    imagenet(args, net, dataset_path, model_helpers_root_path, algo='POT', do_stats=True, do_finetuning=True, zero_clear=False)

    dataset_path = 'path_to_pac'
    model_helpers_root_path = os.path.join(model_helpers_root_path, 'pacs')
    if not os.path.exists(model_helpers_root_path):
        print('No helper objects directory exists for this model, creating one\n')
        os.makedirs(model_helpers_root_path)
    pacs(args, dataset_path, model_helpers_root_path, do_train=False, do_stats=False, do_finetuning=True)

    


