import pytest

from scout.config import ModelConfig


def test_parameter_count():
    config = ModelConfig()
    assert config.parameter_count == 247_088_768
    assert config.non_embedding_parameters == 217_728_640
    assert config.embedding_parameters == 29_360_128


def test_kv_cache_bytes_per_token():
    assert ModelConfig().kv_cache_bytes_per_token() == 13_312


def test_reserved_token_ids():
    ids = ModelConfig().reserved_token_ids
    assert len(ids) == 32
    assert ids[-1] == 32767


def test_rejects_uneven_grouping():
    with pytest.raises(ValueError):
        ModelConfig(n_kv_heads=3)


def test_rejects_mismatched_heads():
    with pytest.raises(ValueError):
        ModelConfig(head_dim=96)
