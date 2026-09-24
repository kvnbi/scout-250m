import pytest
import torch

from scout.model import Scout


def unchanged(model):
    before = {key: value.clone() for key, value in model.state_dict().items()}
    yield model
    after = model.state_dict()
    assert all(torch.equal(after[key], value) for key, value in before.items()), "a test changed a shared model"


@pytest.fixture(scope="session")
def full_model():
    torch.manual_seed(0)
    yield from unchanged(Scout())
