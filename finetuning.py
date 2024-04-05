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
from torch.optim import lr_scheduler
from torchvision import datasets
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
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from parameter_and_input_saliency import save_gradients, compute_input_space_saliency, sort_filters_layer_wise
from torch.utils.data import DataLoader, random_split
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
from parameter_and_input_saliency import save_gradients, compute_input_space_saliency, sort_filters_layer_wise
import copy
from extremevalue import percentage, initialize

def finetuning(args, net, filter_id_to_layer, aggregation, dataset_path, valid_len, transform, stat_path, save_path, algo='POT', zero_clear=False):
    # testset is the one in the file "parameter_and_input_saliency.py"
    print("In finetuning", flush=True)
    print('algo:', algo)

    torch.manual_seed(40)
    np.random.seed(40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.eval()

    filter_stats = torch.load(os.path.join(stat_path, 'mean_std'))
    testset_mean_stat = filter_stats['mean']
    testset_std_stat = filter_stats['std']
    print(testset_mean_stat.shape)
    print('filter_testset_mean_abs_grad:\n', testset_mean_stat[:10])

    if dataset_path == 'MNIST':
        # SVHNデータセットのダウンロード
        train_dataset = datasets.SVHN(root='path_to_svhn', split='train', transform=transform, download=True)
        test_dataset = datasets.SVHN(root='path_to_svhn', split='test', transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
        loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    else:
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    gammas = np.load(os.path.join(stat_path, 'gammas.npy'))
    sigmas = np.load(os.path.join(stat_path, 'sigmas.npy'))
    Nt = np.load(os.path.join(stat_path, 'Nt.npy'))
    init_threshold = np.load(os.path.join(stat_path, 'init_threshold.npy'))

    if algo == 'POT' or algo == 'conv5':
        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=device, mode='naive',
                                              aggregation=aggregation, signed=args.signed, logit=args.logit,
                                              logit_difference=args.logit_difference)

    else:
        filter_saliency_std_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=device, mode='std',
                                              aggregation=aggregation, signed=args.signed, logit=args.logit,
                                              logit_difference=args.logit_difference)

    lr = 0.001
    num_of_filter = 50 if algo == 'random' else 25
    incorrect, correct, flipped = np.zeros(num_of_filter), np.zeros(num_of_filter), np.zeros(num_of_filter)

    original_state_dict = copy.deepcopy(net.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    wrong_num = np.zeros(num_of_filter)
    print('len(loader):', len(loader))

    df_i = pd.DataFrame(np.nan, index=range(len(loader)), columns=range(num_of_filter))
    df_c = pd.DataFrame(np.nan, index=range(len(loader)), columns=range(num_of_filter))
    df_f = pd.DataFrame(np.nan, index=range(len(loader)), columns=range(num_of_filter))
    res = []
    grads = []

    for i, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        if preds == labels.data:
            continue

        # calculating the ranking
        old_confidence = outputs.softmax(dim=1).squeeze()

        if algo == 'POT':
            filter_saliency = filter_saliency_model(
                inputs, labels,
                testset_mean_abs_grad=None,
                testset_std_abs_grad=None).detach().cpu().numpy()
            index = np.squeeze(np.argwhere(filter_saliency > init_threshold), axis=1)
            probabilities = percentage(init_threshold[index], filter_saliency[index], gammas[index], sigmas[index], valid_len, Nt[index])
            index_order = (probabilities).argsort()
            d_order = index[index_order]
        elif algo == 'baseline':
            filter_saliency_std = filter_saliency_std_model(
                inputs, labels,
                testset_mean_abs_grad=testset_mean_stat,
                testset_std_abs_grad=testset_std_stat).detach().cpu().numpy()
            # By negating the whole elements, we get the correct order.
            d_order = np.argsort(-filter_saliency_std)
        elif algo == 'conv5':
            filter_saliency = filter_saliency_model(
                inputs, labels,
                testset_mean_abs_grad=None,
                testset_std_abs_grad=None).detach().cpu().numpy()
            d_order = np.argsort(-filter_saliency)
            d_order = d_order[d_order >= 2240] # in conv5_x of ResNet50
        elif algo == 'random':
            d_order = np.random.choice(26560, size=num_of_filter, replace=False)   


        res.append(d_order[:num_of_filter])

        # print(d_order[:num_of_filter])

        # calculate and save gradient
        loss = criterion(outputs, labels)
        loss.backward()
        grad_dict = {k:v.grad for k,v in net.named_parameters()}

        counter = 0
        count = 0
        modified_dict = copy.deepcopy(net.state_dict())

        while counter < num_of_filter and count < d_order.shape[0]:
            (name, idx) = filter_id_to_layer[d_order[count]]
            count += 1
            # print((name, idx))
            # if 'conv' not in name:
                # continue
            # print(grad_dict[name][idx])
            wrong_num[counter] += 1
            counter += 1
            if zero_clear:
                modified_dict[name][idx] = torch.zeros_like(modified_dict[name][idx])
            else:
                modified_dict[name][idx] -= lr * grad_dict[name][idx]

            net.load_state_dict(modified_dict)
            with torch.no_grad():
                outputs = net(inputs)
                _, pred = torch.max(outputs, 1)
                new_confidence = outputs.softmax(dim=1).squeeze()
            
            # print(new_confidence)
            df_i.iat[i, counter - 1] = (new_confidence[preds] - old_confidence[preds]).item()
            df_c.iat[i, counter - 1] = (new_confidence[labels.data] - old_confidence[labels.data]).item()
            df_f.iat[i, counter - 1] = 1 if pred == labels.data else 0
            # incorrect[counter - 1] += new_confidence[preds] - old_confidence[preds]
            # correct[counter - 1] += new_confidence[labels.data] - old_confidence[labels.data]
            # flipped[counter - 1] += 1 if pred == labels.data else 0
        net.load_state_dict(original_state_dict)
        if i % 100 == 0:
            print(i + 1, 'done', flush=True)
            # break
    
    print('finished calculation')
    save_path = os.path.join(save_path, algo)
    if zero_clear:
        save_path = os.path.join(save_path, 'zero_clear')

    print(save_path)
    if not os.path.exists(save_path):
        print('No helper objects directory exists for this model, creating one\n')
        os.makedirs(save_path)

    # # print(df_i)
    df_i = df_i.dropna(how='all')
    df_c = df_c.dropna(how='all')
    df_f = df_f.dropna(how='all')
    # print(df_i)

    df_i.to_csv(os.path.join(save_path, 'df_i'), index=False)  # CSV形式で保存（index列は保存しない）
    df_c.to_csv(os.path.join(save_path, 'df_c'), index=False)  # CSV形式で保存（index列は保存しない）
    df_f.to_csv(os.path.join(save_path, 'df_f'), index=False)  # CSV形式で保存（index列は保存しない）


    # incorrect /= wrong_num / 100.0
    # correct /= wrong_num / 100.0
    # flipped /= wrong_num / 100.0

    # x = np.linspace(1, 26, num_of_filter)
    # fig = plt.figure(figsize=(15, 4), dpi=360)

    # ax2 = fig.add_subplot(131)
    # ax2.plot(x, incorrect)
    # # ax2.plot(x, incorrect_random, label='random')
    # # ax2.plot(x, incorrect_least, label='least salient')
    # ax2.set_ylabel('incorrect class confidence')
    # ax2.set_xlabel('#finetuned filters')
    # ax2.legend()

    # ax3 = fig.add_subplot(132)
    # ax3.plot(x, correct)
    # # ax3.plot(x, correct_random, label='random')
    # # ax3.plot(x, correct_least, label='least salient')
    # ax3.set_ylabel('correct class confidence')
    # ax3.set_xlabel('#finetuned filters')
    # ax3.legend()

    # ax1 = fig.add_subplot(133)
    # ax1.plot(x, flipped)
    # # ax1.plot(x, flipped_random, label='random')
    # # ax1.plot(x, flipped_least, label='least salient')
    # ax1.set_ylabel('percentage of corrected samples')
    # ax1.set_xlabel('#finetuned filters')
    # ax1.legend()

    # np.save(os.path.join(save_path, f'res25_{algo}'), np.array(res))
    # np.save(os.path.join(save_path, f'incorrect_{algo}'), incorrect)
    # np.save(os.path.join(save_path, f'correct_{algo}'), correct)
    # np.save(os.path.join(save_path, f'flipped_{algo}'), flipped)
    # fig.savefig(os.path.join(save_path, f'result_{algo}.png'))