import torch
import os

def mnist():
    trainset = torch.load("data/processed/train.pt")
    testset = torch.load("data/processed/test.pt")

    return trainset, testset