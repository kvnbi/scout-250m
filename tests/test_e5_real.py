import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scout.retrieval.bert import load_bert
from scout.retrieval.dense import DenseIndex, build_dense_index
from scout.retrieval.embedder import Embedder
from scout.retrieval.model_fetch import E5_BASE_V2
from scout.retrieval.passages import Passage

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "models" / "e5-base-v2"
DIRECTORY = os.environ.get("SCOUT_E5_DIR") or (str(DEFAULT) if (DEFAULT / "model.safetensors").exists() else None)
pytestmark = pytest.mark.skipif(not DIRECTORY, reason="download e5-base-v2 to data/models or set SCOUT_E5_DIR")

QUESTION = "who wrote Animal Farm"
ANSWER = "Animal Farm is an allegorical novella by George Orwell, first published in England on 17 August 1945."
OTHER = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
HARD = [
    "who wrote Animal Farm",
    "caf\u00e9 na\u00efve R\u00c9SUM\u00c9 \u00fcber",
    "\u4e2d\u6587\u6587\u672c and \u65e5\u672c\u8a9e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0639\u0631\u0628\u064a",
    "emoji \U0001f600 and ligature \ufb01 and zero\u200bwidth",
    "   leading   and\ttabs\nand newlines   ",
    "snake_case kebab-case don't 3.14 1,000,000 U.S.A. e-mail@example.com https://example.org/a?b=c#d",
    "",
    "word " * 700,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
]


@pytest.fixture(scope="module")
def embedder():
    return Embedder(Path(DIRECTORY))


@pytest.fixture(scope="module")
def transformers_pair():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(DIRECTORY)
    model = transformers.AutoModel.from_pretrained(DIRECTORY, attn_implementation="sdpa").eval()
    return tokenizer, model


def reference(pair, texts, prefix, max_length=512):
    tokenizer, model = pair
    batch = tokenizer([prefix + t for t in texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    weights = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * weights).sum(1) / weights.sum(1), dim=1).numpy()


def test_every_file_matches_its_pinned_checksum():
    for name, digest in E5_BASE_V2.files.items():
        assert hashlib.sha256((Path(DIRECTORY) / name).read_bytes()).hexdigest() == digest, name


def test_hidden_states_equal_transformers_bit_for_bit(transformers_pair):
    tokenizer, model = transformers_pair
    ours = load_bert(Path(DIRECTORY))
    batch = tokenizer(["query: " + t for t in HARD], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        assert torch.equal(ours(batch["input_ids"], batch["attention_mask"]), model(**batch).last_hidden_state)


@pytest.mark.parametrize("kind, prefix", [("query", "query: "), ("passage", "passage: ")])
def test_embeddings_equal_the_reference_pipeline(embedder, transformers_pair, kind, prefix):
    texts = [t for t in HARD if t.strip()]
    ours = embedder.encode(texts, kind, batch_size=4)
    np.testing.assert_allclose(ours, reference(transformers_pair, texts, prefix), atol=2e-6)
    assert np.all(np.sum(ours * reference(transformers_pair, texts, prefix), axis=1) > 0.999999)


def test_tokenisation_matches_transformers_with_truncation(embedder, transformers_pair):
    tokenizer, _ = transformers_pair
    for limit in (512, 64, 8):
        embedder.tokenizer.enable_truncation(max_length=limit)
        for text in HARD:
            expected = tokenizer("query: " + text, truncation=True, max_length=limit)["input_ids"]
            assert embedder.tokenizer.encode("query: " + text).ids == expected, (limit, text[:30])
    embedder.tokenizer.enable_truncation(max_length=512)


def test_known_values_and_the_relevant_passage_ranks_higher(embedder):
    query = embedder.encode([QUESTION], "query")[0]
    answer = embedder.encode([ANSWER], "passage")[0]
    other = embedder.encode([OTHER], "passage")[0]
    np.testing.assert_allclose(query[:6], [0.0121, -0.0131, -0.0733, -0.0458, 0.025, -0.0129], atol=2e-4)
    np.testing.assert_allclose(answer[:6], [-0.0015, -0.0366, -0.0765, -0.0436, 0.0445, -0.018], atol=2e-4)
    assert float(query @ answer) == pytest.approx(0.8952, abs=2e-4)
    assert float(query @ other) == pytest.approx(0.6841, abs=2e-4)
    assert query @ answer > query @ other


def test_batching_and_padding_do_not_change_real_embeddings(embedder):
    texts = [ANSWER, "short", OTHER * 3, QUESTION]
    together = embedder.encode(texts, "passage", batch_size=4)
    for row, text in zip(together, texts):
        np.testing.assert_allclose(row, embedder.encode([text], "passage")[0], atol=2e-6)


CORPUS = [
    ("Animal Farm", "Animal Farm is an allegorical novella by George Orwell, first published in England on 17 August 1945."),
    ("Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, completed in 1889."),
    ("Photosynthesis", "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to produce sugar and oxygen."),
    ("Mount Everest", "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas on the border of Nepal and Tibet."),
    ("Albert Einstein", "Albert Einstein was born in Ulm in the Kingdom of Wuerttemberg on 14 March 1879 and developed the theory of relativity."),
    ("Great Barrier Reef", "The Great Barrier Reef is the world's largest coral reef system, off the coast of Queensland, Australia."),
    ("Printing press", "Johannes Gutenberg introduced movable type printing to Europe around 1440, starting the Printing Revolution."),
    ("Penicillin", "Penicillin was discovered by Alexander Fleming in 1928 and became the first widely used antibiotic."),
    ("Pacific Ocean", "The Pacific Ocean is the largest and deepest of the oceans, covering about a third of the surface of the Earth."),
    ("Marie Curie", "Marie Curie was a physicist and chemist who researched radioactivity and won Nobel Prizes in both physics and chemistry."),
]
QUESTIONS = [
    ("which author wrote the novella about farm animals who take over the farm", "Animal Farm"),
    ("how tall is the iron tower in the French capital", "Eiffel Tower"),
    ("how do plants turn light into food", "Photosynthesis"),
    ("what is the tallest peak on the planet", "Mount Everest"),
    ("where was the physicist who developed relativity born", "Albert Einstein"),
    ("which is the biggest coral system in the world", "Great Barrier Reef"),
    ("who brought movable type to Europe", "Printing press"),
    ("which mould led to the first antibiotic drug", "Penicillin"),
    ("what is the biggest ocean", "Pacific Ocean"),
    ("which scientist won Nobel Prizes for studying radioactivity", "Marie Curie"),
]


@pytest.mark.parametrize("dtype, device", [("float16", "cpu"), ("float32", "cpu")])
def test_a_dense_index_over_real_embeddings_answers_paraphrased_questions(tmp_path, embedder, dtype, device):
    passages = [Passage(f"{i}:0", title, "", text, "https://en.wikipedia.org/wiki/X", "2018-12-20") for i, (title, text) in enumerate(CORPUS)]
    build_dense_index(passages, tmp_path / "idx", embedder, dtype=dtype, model="intfloat/e5-base-v2", chunk_passages=4)
    index = DenseIndex(tmp_path / "idx", embedder, name="e5", device=device)
    for question, title in QUESTIONS:
        hits = index.search(question, 3)
        assert hits[0].passage.title == title, (question, [h.passage.title for h in hits])
        assert hits[0].score > hits[-1].score
