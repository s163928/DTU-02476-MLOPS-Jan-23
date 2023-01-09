import numpy as np
import pytest
import torch
from src.models.model import MyAwesomeModel
def test_model():
    model = MyAwesomeModel()
    tester = torch.randn(64, 784)
    print(tester.shape)
    model(tester)

        