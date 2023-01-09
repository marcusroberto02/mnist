from tests import _PATH_DATA
from src.data.mnist import mnist
import os
import torch


def test_data_exists():
    assert os.path.isfile(_PATH_DATA + "/processed/train.pt")
    assert os.path.isfile(_PATH_DATA + "/processed/test.pt")


def test_data():
    trainset, testset = mnist()
    N_train, N_test = 25000, 5000
    assert len(trainset) == N_train
    assert len(testset) == N_test

    # make sure images have correct shape
    assert trainset[:][0].shape == torch.Size([N_train, 1, 28, 28])
    assert testset[:][0].shape == torch.Size([N_test, 1, 28, 28])

    # make sure all labels are represented
    assert len(torch.unique(trainset[:][1])) == 10
    assert len(torch.unique(testset[:][1])) == 10
