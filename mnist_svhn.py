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

def mnist_svhn(args, dataset_path, model_helpers_root_path, do_train=False, do_stats=False, do_finetuning=False):
    torch.manual_seed(40)
    np.random.seed(40)

    lr = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(
           in_features=model.fc.in_features,
           out_features=10)
    model.to(device)

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
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print('model_helpers_root_path', model_helpers_root_path)

    if do_train:
        dataset_path = 'MNIST'
        save_path = os.path.join(model_helpers_root_path, 'params')
        train(model, dataset_path, criterion, optimizer, save_path, epochs=10, best_acc=False)

    if do_stats:
        dataset_path = 'MNIST'
        save_path = os.path.join(model_helpers_root_path, 'stats')
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.load_state_dict(torch.load(os.path.join(model_helpers_root_path, 'params')))
        aggregation = 'filter_wise'
        stats(args, model, dataset_path, transform, aggregation, save_path, level=0.90, save=True, verbose=True)

    if do_finetuning:
        dataset_path = 'MNIST'
        valid_len = 10000
        model.load_state_dict(torch.load(os.path.join(model_helpers_root_path, 'params')))
        aggregation = 'filter_wise'
        stat_path = os.path.join(model_helpers_root_path, 'stats')
        save_path = 'Figs/resnet18/mnist_svhn'
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        finetuning(args, model, filter_id_to_layer, aggregation, dataset_path, valid_len, transform, stat_path, save_path, algo='baseline')
        finetuning(args, model, filter_id_to_layer, aggregation, dataset_path, valid_len, transform, stat_path, save_path, algo='POT')
        finetuning(args, model, filter_id_to_layer, aggregation, dataset_path, valid_len, transform, stat_path, save_path, algo='conv5')
    