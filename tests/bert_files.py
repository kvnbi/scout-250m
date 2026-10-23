import json
import re
import struct

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from scout.retrieval.bert import Bert, BertConfig
from scout.retrieval.weights import DTYPES

NAMES = {dtype: name for name, dtype in DTYPES.items()}
WORDS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "query", "passage", ":", "the", "cat", "sat", "on", "mat", "dog", "ran",
    "far", "away", "hello", "world", "##s", "##ing", "cafe", ".",
]
TINY = BertConfig(
    vocab_size=len(WORDS),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=16,
)
REVERSE = (
    (r"^layers\.(\d+)\.(query|key|value)\.", r"encoder.layer.\1.attention.self.\2."),
    (r"^layers\.(\d+)\.attention_output\.", r"encoder.layer.\1.attention.output.dense."),
    (r"^layers\.(\d+)\.attention_norm\.", r"encoder.layer.\1.attention.output.LayerNorm."),
    (r"^layers\.(\d+)\.intermediate\.", r"encoder.layer.\1.intermediate.dense."),
    (r"^layers\.(\d+)\.output\.", r"encoder.layer.\1.output.dense."),
    (r"^layers\.(\d+)\.output_norm\.", r"encoder.layer.\1.output.LayerNorm."),
    (r"^(word_embeddings|position_embeddings|token_type_embeddings|LayerNorm)\.", r"embeddings.\1."),
)


def safetensors_bytes(tensors, metadata=None, order=None):
    header, chunks, position = {}, [], 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name in order or list(tensors):
        tensor = tensors[name].contiguous().reshape(-1)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": NAMES[tensor.dtype],
            "shape": list(tensors[name].shape),
            "data_offsets": [position, position + len(raw)],
        }
        chunks.append(raw)
        position += len(raw)
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + b"".join(chunks)


def huggingface_state(model, prefix="", norm_names=("weight", "bias"), extras=True):
    state = {}
    for key, value in model.state_dict().items():
        for pattern, replacement in REVERSE:
            key = re.sub(pattern, replacement, key)
        if "LayerNorm" in key:
            key = key.replace(".weight", f".{norm_names[0]}").replace(".bias", f".{norm_names[1]}")
        state[prefix + key] = value.clone()
    if extras:
        hidden = model.config.hidden_size
        state[prefix + "pooler.dense.weight"] = torch.zeros(hidden, hidden)
        state[prefix + "pooler.dense.bias"] = torch.zeros(hidden)
        state[prefix + "embeddings.position_ids"] = torch.arange(model.config.max_position_embeddings).unsqueeze(0)
    return state


def write_tokenizer(directory):
    tokenizer = Tokenizer(models.WordPiece({word: number for number, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(directory / "tokenizer.json"))


def write_config(directory, config=TINY):
    raw = {
        "architectures": ["BertModel"],
        "hidden_act": "gelu",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "layer_norm_eps": config.layer_norm_eps,
        "max_position_embeddings": config.max_position_embeddings,
        "model_type": "bert",
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "position_embedding_type": "absolute",
        "type_vocab_size": config.type_vocab_size,
        "vocab_size": config.vocab_size,
    }
    (directory / "config.json").write_text(json.dumps(raw))


def random_model(seed=0, config=TINY):
    torch.manual_seed(seed)
    model = Bert(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "LayerNorm" in name or "norm" in name:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter) if name.endswith("weight") else 0.1 * torch.randn_like(parameter))
    return model.eval()


def write_model_directory(directory, seed=0, config=TINY):
    directory.mkdir(parents=True, exist_ok=True)
    model = random_model(seed, config)
    write_config(directory, config)
    write_tokenizer(directory)
    (directory / "model.safetensors").write_bytes(safetensors_bytes(huggingface_state(model), metadata={"format": "pt"}))
    return model
