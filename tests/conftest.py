import pytest
import torch

from scout.model import Scout


@pytest.fixture(scope="session")
def full_model():
    torch.manual_seed(0)
    return Scout()
