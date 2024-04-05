import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
import os
from train import train
from stats import stats
from finetuning import finetuning

def imagenet(args, model, dataset_path, model_helpers_root_path, algo='baseline', do_stats = False, do_finetuning=True, zero_clear=False):
    torch.manual_seed(40)
    np.random.seed(40)

    print(algo)
    print('zero_clear', zero_clear)

    lr = 0.001

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model != 'vit':
        aggregation = 'filter_wise'
    else:
        aggregation = 'vit'

    filter_id_to_layer = {}
    ind = 0
    for layer_num, (name, param) in enumerate(model.named_parameters()):
        # print(name, param.shape)
        if len(param.size()) == 4:
            if 'conv' not in name:
                print('Not a conv layer {}: {}'.format(name, layer_num))

            for j in range(param.size()[0]):
                filter_id_to_layer[ind + j] = (name, j)

            ind += param.size()[0]

    criterion = nn.CrossEntropyLoss()

    print('model_helpers_root_path', model_helpers_root_path)
    valid_len = 50000

    levels = [0.85, 0.95]
    for level in levels:
        if do_stats:
            save_path = os.path.join(model_helpers_root_path, 'stats', str(level))
            print(save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                stats(args, model, dataset_path, transform, aggregation, save_path, level=level, save=True, verbose=True)
            else:
                print(save_path, 'exists.')

        if do_finetuning:
            stat_path = os.path.join(model_helpers_root_path, 'stats', str(level))
            save_path = 'Figs/' + args.model + '/Imagenet/' + str(level)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            finetuning(args, model, filter_id_to_layer, aggregation, dataset_path, valid_len, transform, stat_path, save_path, algo=algo, zero_clear=zero_clear)

        print(level, 'done')
    