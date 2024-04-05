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
from torch.utils.data import DataLoader, random_split
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
from extremevalue import percentage, initialize

def train(model, dataset_path, criterion, optimizer, save_path, epochs=5, best_acc=False):
    # batch_size = 64
    batch_size = 256
    
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
        test_dataset = datasets.MNIST(root='path_to_mnist', train=False, download=True, transform=transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        train_size = int(0.8 * len(dataset))
        print('train_size:', train_size)
        print('valid_size:', len(dataset) - train_size)
        train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print('train_loader len:', len(train_loader))
    print('val_loader len:', len(val_loader))

    best_val_loss = float('inf') 
    best_val_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}")

        model.eval()
        val_loss = 0.0
        running_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_correct += (preds == labels).sum().item()

        average_val_loss = val_loss / len(val_loader)
        epoch_acc = 100.0 * (running_correct / len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {average_val_loss}, Validation Acc.: {running_correct}", flush=True)

        if not best_acc:
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss   
                torch.save(model.state_dict(), save_path)
        else:
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
