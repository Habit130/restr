import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import skimage

from utils import im_processing, text_processing


SPECIAL_TOKENS = ["<pad>", "<go>", "<eos>", "<unk>"]
MAX_TOKENS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare plant disease train/test batches for ReSTR."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../dataset"),
        help="Sibling dataset root that contains train.json/test.json and train|test folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data"),
        help="Output root for generated vocabulary, embeddings, and batch npz files.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="plantseg",
        help="Dataset prefix used for generated assets.",
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        required=True,
        help="Path to a 300d GloVe text file.",
    )
    parser.add_argument(
        "--caption-index",
        type=int,
        default=2,
        help="Zero-based caption index to use from each JSON record.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=480,
        help="Square image size used for train batches.",
    )
    return parser.parse_args()


def sentence_to_tokens(sentence):
    words = text_processing.SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [word.lower() for word in words if len(word.strip()) > 0]
    if words and words[-1] == ".":
        words = words[:-1]
    return words


def load_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(records, caption_index):
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)

    for record in records:
        caption = record["caption"][caption_index]
        for token in sentence_to_tokens(caption):
            if token not in seen:
                seen.add(token)
                vocab.append(token)

    return vocab


def write_vocab(vocab, vocab_path):
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def build_embedding_matrix(vocab, glove_path, embedding_path):
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    embedding_matrix = np.zeros((len(vocab), 300), dtype=np.float32)
    found_tokens = 0

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            pieces = line.rstrip().split()
            if len(pieces) != 301:
                continue
            token = pieces[0]
            if token not in token_to_index:
                continue
            embedding_matrix[token_to_index[token]] = np.asarray(pieces[1:], dtype=np.float32)
            found_tokens += 1

    known_rows = embedding_matrix[len(SPECIAL_TOKENS):]
    non_zero_rows = known_rows[np.any(known_rows != 0, axis=1)]
    unk_vector = non_zero_rows.mean(axis=0) if len(non_zero_rows) > 0 else np.zeros(300, dtype=np.float32)
    embedding_matrix[token_to_index["<unk>"]] = unk_vector

    for token, idx in token_to_index.items():
        if token in SPECIAL_TOKENS:
            continue
        if not np.any(embedding_matrix[idx]):
            embedding_matrix[idx] = unk_vector

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embedding_matrix)
    return found_tokens


def relpath_to_path(root, relpath):
    return root.joinpath(*PurePosixPath(relpath).parts)


def resolve_mask_path(dataset_root, mask_relpath):
    original_mask = relpath_to_path(dataset_root, mask_relpath)
    parts = list(PurePosixPath(mask_relpath).parts)
    binary_parts = ["lbl_binary_255" if part == "lbl" else part for part in parts]
    binary_mask = dataset_root.joinpath(*binary_parts)
    return binary_mask if binary_mask.exists() else original_mask


def ensure_three_channels(image):
    if image.ndim == 2:
        return np.tile(image[:, :, np.newaxis], (1, 1, 3))
    if image.shape[2] > 3:
        return image[:, :, :3]
    return image


def clear_existing_batches(split_dir):
    if not split_dir.exists():
        return
    for npz_path in split_dir.glob("*.npz"):
        npz_path.unlink()


def save_split(records, split, args, vocab_dict, output_dir):
    split_dir = output_dir / (split + "_batch")
    split_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_batches(split_dir)

    for idx, record in enumerate(records):
        image_path = relpath_to_path(args.dataset_root, record["image"])
        mask_path = resolve_mask_path(args.dataset_root, record["mask"])
        sentence = record["caption"][args.caption_index]

        image = skimage.io.imread(str(image_path))
        image = ensure_three_channels(image)
        mask = skimage.io.imread(str(mask_path))
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)

        if split == "train":
            image = skimage.img_as_ubyte(
                im_processing.resize_and_pad(image, args.img_size, args.img_size)
            )
            mask = im_processing.resize_and_pad(mask, args.img_size, args.img_size)

        text = text_processing.preprocess_sentence(sentence, vocab_dict, MAX_TOKENS)
        sample_path = split_dir / f"{args.dataset_name}_{split}_{idx}.npz"
        np.savez(
            file=sample_path,
            text_batch=np.asarray(text, dtype=np.int64),
            im_batch=image,
            mask_batch=(mask > 0),
            sent_batch=[sentence],
            im_name_batch=Path(record["image"]).name,
        )


def main():
    args = parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not args.glove_path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {args.glove_path}")

    train_records = load_records(args.dataset_root / "train.json")
    test_records = load_records(args.dataset_root / "test.json")

    for split_name, records in {"train": train_records, "test": test_records}.items():
        for record in records:
            if len(record["caption"]) <= args.caption_index:
                raise IndexError(
                    f"{split_name} sample '{record['id']}' does not have caption index {args.caption_index}"
                )
            image_path = relpath_to_path(args.dataset_root, record["image"])
            mask_path = resolve_mask_path(args.dataset_root, record["mask"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    vocab = build_vocab(train_records + test_records, args.caption_index)
    vocab_path = args.output_root / f"vocabulary_{args.dataset_name}.txt"
    embedding_path = args.output_root / f"{args.dataset_name}_emb.npy"
    batch_root = args.output_root / f"{args.dataset_name}_{args.img_size}_batch"

    write_vocab(vocab, vocab_path)
    found_tokens = build_embedding_matrix(vocab, args.glove_path, embedding_path)
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_path)

    save_split(train_records, "train", args, vocab_dict, batch_root)
    save_split(test_records, "test", args, vocab_dict, batch_root)

    print(
        "Prepared dataset '{}' with {} train samples and {} test samples.".format(
            args.dataset_name, len(train_records), len(test_records)
        )
    )
    print("Vocabulary size: {} | GloVe hits: {}".format(len(vocab), found_tokens))
    print("Vocabulary file: {}".format(vocab_path))
    print("Embedding file: {}".format(embedding_path))
    print("Batch root: {}".format(batch_root))


if __name__ == "__main__":
    main()
