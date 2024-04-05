import yaml
import urllib
import pandas as pd
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
import cv2
import warnings
import tqdm
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from parameter_and_input_saliency import save_gradients, compute_input_space_saliency, sort_filters_layer_wise
from extremevalue import percentage, initialize

def stats(args, net, dataset_path, transform, aggregation, save_path, level=0.90, save=True, verbose=True):
    print('In stats', flush=True)
    print(save_path, dataset_path, args.imagenet_val_path)
    torch.manual_seed(40)
    np.random.seed(40)

    if dataset_path == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomRotation(20),
            # transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.7, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.MNIST(root='path_to_mnist', train=True, download=True, transform=transform)
        valid_dataset = datasets.MNIST(root='path_to_mnist', train=False, download=True, transform=transform)

        # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    elif dataset_path != args.imagenet_val_path:
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)

        train_size = int(0.8 * len(dataset))
        print('train_size:', train_size)
        print('valid_size:', len(dataset) - train_size)

        train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    else:
        valid_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)
    net.eval()

    filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=device, mode='naive',
                                          aggregation=aggregation, signed=args.signed, logit=args.logit,
                                          logit_difference=args.logit_difference)

    saliency = []
    sample_num = len(valid_loader)
    wrong = 0
    for i, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        wrong += (preds != labels).sum().item()

        filter_saliency = filter_saliency_model(
            inputs, labels,
            testset_mean_abs_grad=None,
            testset_std_abs_grad=None)
        # print(filter_saliency.shape)

        saliency.append(filter_saliency.detach())

        if i % 500 == 0:
            print('i = {}'.format(i), flush=True)

    saliency = torch.stack(saliency, dim=0).cpu().numpy()

    print('wrong:', wrong, flush=True)
    print('num:', sample_num, flush=True)

    gammas, sigmas, lls, peaks, Nt, init_threshold = initialize(saliency, level = level)

    print(save, save_path)
    if save:
        np.save(os.path.join(save_path, 'gammas'), gammas)
        np.save(os.path.join(save_path, 'sigmas'), sigmas)
        np.save(os.path.join(save_path, 'lls'), lls)
        np.save(os.path.join(save_path, 'Nt'), Nt)
        np.save(os.path.join(save_path, 'init_threshold'), init_threshold)
    if verbose:
        print('gammas:\n', gammas[:5])
        print('sigmas:\n', sigmas[:5])
        print('Nt:\n', Nt[:5])
        print('init_threshold:\n', init_threshold[:5])


    filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(net, valid_dataset, aggregation, args) 
    torch.save({'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad}, os.path.join(save_path, 'mean_std'))
