import json

import pytest
import torch
from bert_files import TINY, huggingface_state, random_model, safetensors_bytes, write_config, write_model_directory

from scout.retrieval.bert import Bert, BertConfig, convert_key, load_bert


def test_the_default_configuration_is_bert_base_without_a_pooler():
    config = BertConfig()
    assert (config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.intermediate_size) == (768, 12, 12, 3072)
    with torch.device("meta"):
        model = Bert(config)
    assert sum(p.numel() for p in model.parameters()) == 108_891_648


def test_a_configuration_file_is_read_and_unknown_keys_are_ignored(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 32, "num_attention_heads": 4, "num_hidden_layers": 2, "intermediate_size": 64, "model_type": "bert", "pad_token_id": 0, "layer_norm_eps": 1e-5})
    )
    config = BertConfig.from_file(tmp_path / "config.json")
    assert (config.hidden_size, config.num_hidden_layers, config.layer_norm_eps, config.vocab_size) == (32, 2, 1e-5, 30522)


@pytest.mark.parametrize(
    "raw",
    [{"hidden_act": "relu"}, {"hidden_act": "gelu_new"}, {"position_embedding_type": "relative_key"}],
)
def test_unsupported_architectures_are_refused(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported"):
        BertConfig.from_file(tmp_path / "config.json")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hidden_size=0),
        dict(num_hidden_layers=0),
        dict(vocab_size=-1),
        dict(intermediate_size=0),
        dict(max_position_embeddings=0),
        dict(type_vocab_size=0),
        dict(num_attention_heads=0),
        dict(hidden_size=30, num_attention_heads=4),
        dict(layer_norm_eps=0.0),
    ],
)
def test_invalid_configurations_are_refused(kwargs):
    with pytest.raises(ValueError):
        BertConfig(**kwargs)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("embeddings.word_embeddings.weight", "word_embeddings.weight"),
        ("embeddings.position_embeddings.weight", "position_embeddings.weight"),
        ("embeddings.token_type_embeddings.weight", "token_type_embeddings.weight"),
        ("embeddings.LayerNorm.weight", "LayerNorm.weight"),
        ("embeddings.LayerNorm.gamma", "LayerNorm.weight"),
        ("embeddings.LayerNorm.beta", "LayerNorm.bias"),
        ("encoder.layer.3.attention.self.query.weight", "layers.3.query.weight"),
        ("encoder.layer.11.attention.self.key.bias", "layers.11.key.bias"),
        ("encoder.layer.0.attention.self.value.weight", "layers.0.value.weight"),
        ("encoder.layer.2.attention.output.dense.bias", "layers.2.attention_output.bias"),
        ("encoder.layer.2.attention.output.LayerNorm.weight", "layers.2.attention_norm.weight"),
        ("encoder.layer.2.attention.output.LayerNorm.gamma", "layers.2.attention_norm.weight"),
        ("encoder.layer.5.intermediate.dense.weight", "layers.5.intermediate.weight"),
        ("encoder.layer.5.output.dense.weight", "layers.5.output.weight"),
        ("encoder.layer.5.output.LayerNorm.beta", "layers.5.output_norm.bias"),
        ("bert.encoder.layer.1.output.LayerNorm.bias", "layers.1.output_norm.bias"),
        ("bert.embeddings.word_embeddings.weight", "word_embeddings.weight"),
        ("pooler.dense.weight", None),
        ("bert.pooler.dense.bias", None),
        ("embeddings.position_ids", None),
        ("cls.predictions.bias", None),
        ("cls.seq_relationship.weight", None),
    ],
)
def test_checkpoint_names_map_onto_ours(key, expected):
    assert convert_key(key) == expected


def test_our_parameter_names_are_exactly_the_converted_checkpoint_names():
    model = random_model()
    converted = {convert_key(key) for key in huggingface_state(model)} - {None}
    assert converted == set(model.state_dict())


def to_dtype(model, dtype):
    return model.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_outputs_equal_transformers_bit_for_bit_on_a_padded_batch(tmp_path, dtype):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(1)
    config = transformers.BertConfig(
        vocab_size=TINY.vocab_size, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, intermediate_size=64, max_position_embeddings=16
    )
    config._attn_implementation = "sdpa"
    theirs = transformers.BertModel(config).eval()
    with torch.no_grad():
        for name, parameter in theirs.named_parameters():
            if "LayerNorm" in name:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter) if name.endswith("weight") else 0.1 * torch.randn_like(parameter))
    theirs.save_pretrained(tmp_path)
    ours = load_bert(tmp_path, dtype=dtype)
    theirs = theirs.to(dtype)
    ids = torch.randint(0, TINY.vocab_size, (3, 12))
    mask = torch.ones(3, 12, dtype=torch.long)
    mask[0, 7:] = 0
    mask[2, 3:] = 0
    types = torch.randint(0, 2, (3, 12))
    with torch.no_grad():
        for kwargs in (dict(), dict(token_type_ids=types)):
            a = ours(ids, mask, **kwargs)
            b = theirs(input_ids=ids, attention_mask=mask, **kwargs).last_hidden_state
            assert a.dtype == dtype and torch.equal(a, b)
        assert torch.equal(ours(ids), theirs(input_ids=ids).last_hidden_state)


def test_older_checkpoints_with_prefixes_gamma_names_and_heads_load(tmp_path):
    model = write_model_directory(tmp_path / "plain")
    old = huggingface_state(model, prefix="bert.", norm_names=("gamma", "beta"), extras=False)
    old["bert.pooler.dense.weight"] = torch.zeros(32, 32)
    old["cls.predictions.bias"] = torch.zeros(TINY.vocab_size)
    old["cls.seq_relationship.weight"] = torch.zeros(2, 32)
    directory = tmp_path / "old"
    directory.mkdir()
    write_config(directory)
    (directory / "model.safetensors").write_bytes(safetensors_bytes(old))
    ids = torch.randint(0, TINY.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.equal(load_bert(directory)(ids), model(ids))


def test_missing_and_unexpected_weights_are_errors(tmp_path):
    model = write_model_directory(tmp_path / "ok")
    state = huggingface_state(model)
    missing = dict(state)
    del missing["encoder.layer.1.output.dense.weight"]
    (tmp_path / "missing").mkdir()
    write_config(tmp_path / "missing")
    (tmp_path / "missing" / "model.safetensors").write_bytes(safetensors_bytes(missing))
    with pytest.raises(RuntimeError, match="Missing"):
        load_bert(tmp_path / "missing")
    extra = dict(state)
    extra["encoder.layer.7.output.dense.weight"] = torch.zeros(1)
    (tmp_path / "extra").mkdir()
    write_config(tmp_path / "extra")
    (tmp_path / "extra" / "model.safetensors").write_bytes(safetensors_bytes(extra))
    with pytest.raises(RuntimeError, match="Unexpected"):
        load_bert(tmp_path / "extra")


def test_a_wrongly_shaped_weight_is_an_error(tmp_path):
    model = write_model_directory(tmp_path / "ok")
    state = huggingface_state(model)
    state["encoder.layer.0.intermediate.dense.weight"] = torch.zeros(5, 5)
    (tmp_path / "bad").mkdir()
    write_config(tmp_path / "bad")
    (tmp_path / "bad" / "model.safetensors").write_bytes(safetensors_bytes(state))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_bert(tmp_path / "bad")


def test_loading_gives_an_inference_only_model_in_the_requested_dtype(tmp_path):
    write_model_directory(tmp_path)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        model = load_bert(tmp_path, dtype=dtype)
        assert not model.training
        assert all(p.dtype == dtype and not p.requires_grad and p.device.type == "cpu" for p in model.parameters())


def test_padding_never_changes_the_real_tokens(tmp_path):
    model = write_model_directory(tmp_path)
    loaded = load_bert(tmp_path)
    torch.manual_seed(2)
    short = torch.randint(0, TINY.vocab_size, (1, 5))
    padded = torch.cat([short, torch.randint(0, TINY.vocab_size, (1, 6))], dim=1)
    mask = torch.tensor([[1] * 5 + [0] * 6])
    with torch.no_grad():
        alone = loaded(short)
        together = loaded(padded, mask)[:, :5]
        garbage = loaded(torch.cat([short, torch.zeros(1, 6, dtype=torch.long)], dim=1), mask)[:, :5]
    assert torch.allclose(alone, together, atol=1e-6) and torch.equal(together, garbage)
    assert torch.equal(model(short), alone)


def test_token_types_change_the_output_and_zero_is_the_default(tmp_path):
    write_model_directory(tmp_path)
    model = load_bert(tmp_path)
    ids = torch.randint(0, TINY.vocab_size, (2, 6))
    with torch.no_grad():
        default = model(ids)
        assert torch.equal(default, model(ids, token_type_ids=torch.zeros_like(ids)))
        assert not torch.equal(default, model(ids, token_type_ids=torch.ones_like(ids)))


def test_position_matters_and_too_long_inputs_are_refused(tmp_path):
    write_model_directory(tmp_path)
    model = load_bert(tmp_path)
    ids = torch.randint(0, TINY.vocab_size, (1, 6))
    with torch.no_grad():
        assert not torch.equal(model(ids)[0, 0], model(ids.flip(1))[0, 0])
        assert model(torch.zeros(1, 16, dtype=torch.long)).shape == (1, 16, 32)
        with pytest.raises(ValueError, match="positions"):
            model(torch.zeros(1, 17, dtype=torch.long))


def test_only_attended_tokens_influence_the_attended_ones(tmp_path):
    write_model_directory(tmp_path)
    model = load_bert(tmp_path)
    ids = torch.tensor([[2, 4, 7, 9, 3, 5]])
    changed = ids.clone()
    changed[0, 4:] = torch.tensor([8, 6])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        assert torch.equal(model(ids, mask)[:, :4], model(changed, mask)[:, :4])
        assert not torch.equal(model(ids)[:, :4], model(changed)[:, :4])
