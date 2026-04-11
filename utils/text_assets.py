import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from utils.text_processing import tokenize_sentence


GLOVE_6B_300D_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
SPECIAL_TOKENS = ("<pad>", "<go>", "<eos>", "<unk>")
TEXT_SPECS = {
    "Gref": {
        "vocab_file": "vocabulary_Gref.txt",
        "embedding_file": "Gref_emb.npy",
        "text_len": 20,
    },
    "referit": {
        "vocab_file": "vocabulary_referit.txt",
        "embedding_file": "referit_emb.npy",
        "text_len": 20,
    },
    "plantseg": {
        "vocab_file": "vocabulary_plantseg.txt",
        "embedding_file": "plantseg_emb.npy",
        "text_len": 64,
    },
}
DATASET_ALIASES = {
    "Gref": "Gref",
    "unc": "Gref",
    "unc+": "Gref",
    "referit": "referit",
    "plantseg": "plantseg",
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_plantseg_root():
    return repo_root().parent / "plantseg"


def canonical_dataset_name(dataset_name):
    if dataset_name not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return DATASET_ALIASES[dataset_name]


def get_dataset_text_spec(dataset_name, data_root="data"):
    canonical_name = canonical_dataset_name(dataset_name)
    spec = TEXT_SPECS[canonical_name].copy()
    root = Path(data_root)
    spec["canonical_name"] = canonical_name
    spec["vocab_path"] = root / spec["vocab_file"]
    spec["embedding_path"] = root / spec["embedding_file"]
    return spec


def ensure_plantseg_text_assets(data_root="data", plantseg_root=None, cache_dir=None):
    spec = get_dataset_text_spec("plantseg", data_root=data_root)
    if spec["vocab_path"].exists() and spec["embedding_path"].exists():
        return spec

    plantseg_root = Path(plantseg_root) if plantseg_root else default_plantseg_root()
    annotations_path = plantseg_root / "main.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"PlantSeg annotation file not found: {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    vocabulary = build_plantseg_vocabulary(annotations)
    spec["vocab_path"].parent.mkdir(parents=True, exist_ok=True)
    with open(spec["vocab_path"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(vocabulary))
        handle.write("\n")

    glove_txt = ensure_glove_6b_300d(cache_dir or (Path(data_root) / ".cache"))
    embeddings = build_embedding_matrix(vocabulary, glove_txt)
    np.save(spec["embedding_path"], embeddings.astype(np.float32))
    return spec


def build_plantseg_vocabulary(annotations):
    tokens = set()
    for sample in annotations:
        captions = sample.get("caption", [])
        if len(captions) <= 3:
            raise ValueError(f"caption[3] is missing for sample {sample.get('id', '<unknown>')}")
        tokens.update(tokenize_sentence(captions[3]))

    ordered_tokens = sorted(token for token in tokens if token not in SPECIAL_TOKENS)
    return list(SPECIAL_TOKENS) + ordered_tokens


def ensure_glove_6b_300d(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "glove.6B.zip"
    txt_path = cache_dir / "glove.6B.300d.txt"

    if not txt_path.exists():
        if not zip_path.exists():
            urllib.request.urlretrieve(GLOVE_6B_300D_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract("glove.6B.300d.txt", path=cache_dir)

    return txt_path


def build_embedding_matrix(vocabulary, glove_txt_path):
    needed_tokens = {token for token in vocabulary if token not in SPECIAL_TOKENS}
    found_vectors = {}

    with open(glove_txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            token, vector = parse_glove_line(line)
            if token in needed_tokens:
                found_vectors[token] = vector
                if len(found_vectors) == len(needed_tokens):
                    break

    if not found_vectors:
        raise RuntimeError(f"No GloVe vectors were found in {glove_txt_path}")

    mean_vector = np.mean(np.stack(list(found_vectors.values()), axis=0), axis=0, dtype=np.float32)
    zero_vector = np.zeros_like(mean_vector)
    embedding_rows = []

    for token in vocabulary:
        if token == "<pad>":
            embedding_rows.append(zero_vector)
        else:
            embedding_rows.append(found_vectors.get(token, mean_vector))

    return np.stack(embedding_rows, axis=0)


def parse_glove_line(line):
    parts = line.rstrip().split(" ")
    token = parts[0]
    vector = np.asarray(parts[1:], dtype=np.float32)
    return token, vector
