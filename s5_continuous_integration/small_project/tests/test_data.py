import os
import pytest
import torch
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/processed_data.pth"), reason="Data files not found")
def test_check_train_data():
    data = torch.load(_PATH_DATA + "/processed/processed_data.pth")
    train_set = data["train"]
    assert train_set != None

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/processed_data.pth"), reason="Data files not found")
def test_check_test_data():
    data = torch.load(_PATH_DATA +"/processed/processed_data.pth")
    test_set = data["test"]
    assert test_set != None