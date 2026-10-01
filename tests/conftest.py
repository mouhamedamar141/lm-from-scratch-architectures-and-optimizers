import torch
import pytest


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
