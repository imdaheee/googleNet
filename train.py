
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.googlenet import GoogLeNet


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

   
    DATASET_ROOT = "POC_Dataset"

    train_dir = os.path.join(DATASET_ROOT, "Training")
    test_dir = os.path.join(DATASET_ROOT, "Testing")

    
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform_train,
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform_test,
    )
