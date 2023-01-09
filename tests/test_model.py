import pytest
from src.models.model import MyAwesomeModel
import torch

def test_model():
    model = MyAwesomeModel()
    img = torch.randn(torch.Size([1,28,28]))
    assert model(img).shape == torch.Size([1,10])

        