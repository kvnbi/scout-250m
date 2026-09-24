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


def test_norm_and_rope_defaults():
    config = ModelConfig()
    assert config.norm_eps == 1e-6
    assert config.rope_theta == 10000.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_dim": 63, "n_query_heads": 1, "n_kv_heads": 1, "d_model": 63},
        {"norm_eps": 0.0},
        {"rope_theta": 1.0},
        {"n_layers": 0},
        {"ffn_hidden": 0},
        {"vocab_size": 0, "reserved_slots": 0},
        {"reserved_slots": -1},
        {"reserved_slots": 32768},
    ],
)
def test_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        ModelConfig(**overrides)


def test_qwen3_config_fields():
    fields = ModelConfig().qwen3_config()
    assert fields == {
        "vocab_size": 32768,
        "hidden_size": 896,
        "intermediate_size": 2432,
        "num_hidden_layers": 26,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
    }


def test_qwen3_config_requires_qk_norm():
    with pytest.raises(ValueError):
        ModelConfig(qk_norm=False).qwen3_config()


def test_allows_zero_reserved_slots():
    assert len(ModelConfig(reserved_slots=0).reserved_token_ids) == 0
