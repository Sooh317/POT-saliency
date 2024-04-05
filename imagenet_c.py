from corrupt import corrupt
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
from finetuning import finetuning

def imagenet_c(args, dataset_path, model_helpers_root_path, corruption_name, method, severity=5):
    torch.manual_seed(40)
    np.random.seed(40)

    lr = 0.001

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])

    transform_noise = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: corrupt(x, severity=severity, corruption_name=corruption_name, corruption_number=-1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torchvision.models.resnet50(pretrained=True)
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

    valid_dataset_path = 'path_to_imagenet'
    original_dataset = torchvision.datasets.ImageFolder(valid_dataset_path, transform=transform)
    valid_len = len(original_dataset)
    aggregation = 'filter_wise'
    model_helpers_root_path = 'helper_objects/resnet50'
    stat_path = os.path.join(model_helpers_root_path, 'stats')
    save_path = f'Figs/resnet50/imagenet_c-{corruption_name}'
    print(stat_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    finetuning(args, model, filter_id_to_layer, aggregation, dataset_path, valid_len, transform_noise, stat_path, save_path, algo=method)