import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import skimage

from utils import im_processing, text_processing
from utils.io import load_referit_gt_mask as load_gt_mask
from utils.text_assets import (
    default_plantseg_root,
    ensure_plantseg_text_assets,
    get_dataset_text_spec,
)


def save_batch(data_folder, data_prefix, n_batch, image, mask, text, sent, im_name):
    np.savez(
        file=os.path.join(data_folder, f"{data_prefix}_{n_batch}.npz"),
        text_batch=text,
        im_batch=image,
        mask_batch=(mask > 0),
        sent_batch=[sent],
        im_name_batch=im_name,
    )


def maybe_resize_for_training(setname, image, mask, input_h, input_w):
    if "train" not in setname:
        return image, mask

    image = skimage.img_as_ubyte(im_processing.resize_and_pad(image, input_h, input_w))
    mask = im_processing.resize_and_pad(mask, input_h, input_w)
    return image, mask


def ensure_rgb_image(image):
    if image.ndim == 2:
        return np.tile(image[:, :, np.newaxis], (1, 1, 3))
    return image


def load_coco_dependencies():
    coco_api_path = "./external/coco/PythonAPI"
    refer_path = "./external/refer"
    if coco_api_path not in sys.path:
        sys.path.append(coco_api_path)
    if refer_path not in sys.path:
        sys.path.append(refer_path)

    from refer import REFER
    from pycocotools import mask as cocomask

    return REFER, cocomask


def build_referit_batches(output_dir, setname, text_len, input_h, input_w):
    im_dir = "./data/referit/images/"
    mask_dir = "./data/referit/mask/"
    query_file = "./data/referit/referit_query_" + setname + ".json"
    spec = get_dataset_text_spec("referit", data_root=output_dir)

    data_folder = os.path.join(output_dir, "referit", setname + "_batch")
    data_prefix = "referit_" + setname
    os.makedirs(data_folder, exist_ok=True)

    with open(query_file, "r", encoding="utf-8") as handle:
        query_dict = json.load(handle)
    vocab_dict = text_processing.load_vocab_dict_from_file(spec["vocab_path"])

    samples = []
    for name in query_dict.keys():
        im_name = name.split("_", 1)[0] + ".jpg"
        mask_name = name + ".mat"
        for sent in query_dict[name]:
            samples.append((im_name, mask_name, sent))

    for n_batch, (im_name, mask_name, sent) in enumerate(samples):
        print("saving batch %d / %d" % (n_batch + 1, len(samples)))
        im = skimage.io.imread(im_dir + im_name)
        mask = load_gt_mask(mask_dir + mask_name).astype(np.float32)
        im, mask = maybe_resize_for_training(setname, im, mask, input_h, input_w)
        im = ensure_rgb_image(im)
        text = text_processing.preprocess_sentence(sent, vocab_dict, text_len)
        save_batch(data_folder, data_prefix, n_batch, im, mask, text, sent, im_name)


def build_coco_batches(output_dir, dataset, setname, text_len, input_h, input_w):
    refer_cls, cocomask = load_coco_dependencies()
    spec = get_dataset_text_spec(dataset, data_root=output_dir)
    im_dir = "./data/coco/images"
    im_type = "train2014"

    data_folder = os.path.join(output_dir, dataset, setname + "_batch")
    data_prefix = dataset + "_" + setname
    os.makedirs(data_folder, exist_ok=True)

    if dataset == "Gref":
        refer = refer_cls("./external/refer/data", dataset="refcocog", splitBy="google")
    elif dataset == "unc":
        refer = refer_cls("./external/refer/data", dataset="refcoco", splitBy="unc")
    elif dataset == "unc+":
        refer = refer_cls("./external/refer/data", dataset="refcoco+", splitBy="unc")
    else:
        raise ValueError("Unknown dataset %s" % dataset)
    refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]["split"] == setname]
    vocab_dict = text_processing.load_vocab_dict_from_file(spec["vocab_path"])

    n_batch = 0
    for ref in refs:
        im_name = "COCO_" + im_type + "_" + str(ref["image_id"]).zfill(12)
        im = skimage.io.imread("%s/%s/%s.jpg" % (im_dir, im_type, im_name))
        seg = refer.Anns[ref["ann_id"]]["segmentation"]
        rle = cocomask.frPyObjects(seg, im.shape[0], im.shape[1])
        mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)

        im, mask = maybe_resize_for_training(setname, im, mask, input_h, input_w)
        im = ensure_rgb_image(im)

        for sentence in ref["sentences"]:
            print("saving batch %d" % (n_batch + 1))
            sent = sentence["sent"]
            text = text_processing.preprocess_sentence(sent, vocab_dict, text_len)
            save_batch(data_folder, data_prefix, n_batch, im, mask, text, sent, im_name)
            n_batch += 1


def build_plantseg_batches(output_dir, setname, text_len, input_h, input_w, plantseg_root):
    plantseg_root = Path(plantseg_root)
    spec = ensure_plantseg_text_assets(data_root=output_dir, plantseg_root=plantseg_root)
    with open(plantseg_root / "main.json", "r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    samples = [sample for sample in annotations if sample["split"] == setname]
    data_root = Path(output_dir) / "plantseg" / f"plantseg_{input_h}_batch"
    data_folder = data_root / f"{setname}_batch"
    data_folder.mkdir(parents=True, exist_ok=True)
    data_prefix = f"plantseg_{setname}"
    vocab_dict = text_processing.load_vocab_dict_from_file(spec["vocab_path"])

    for n_batch, sample in enumerate(samples):
        print("saving batch %d / %d" % (n_batch + 1, len(samples)))
        image = skimage.io.imread(plantseg_root / sample["image"])
        mask = skimage.io.imread(plantseg_root / sample["mask"]).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 127).astype(np.float32)
        image, mask = maybe_resize_for_training(setname, image, mask, input_h, input_w)
        image = ensure_rgb_image(image)

        sent = sample["caption"][3]
        text = text_processing.preprocess_sentence(sent, vocab_dict, text_len)
        save_batch(str(data_folder), data_prefix, n_batch, image, mask, text, sent, Path(sample["image"]).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="referit")
    parser.add_argument("-t", type=str, default="trainval")
    parser.add_argument("--img-size", type=int, default=480)
    parser.add_argument("--text-len", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--plantseg-root", type=str, default=str(default_plantseg_root()))

    args = parser.parse_args()
    spec = get_dataset_text_spec(args.d, data_root=args.output_dir)
    text_len = spec["text_len"] if args.text_len is None else args.text_len
    input_h = args.img_size
    input_w = args.img_size
    if args.d == "referit":
        build_referit_batches(args.output_dir, setname=args.t, text_len=text_len, input_h=input_h, input_w=input_w)
    elif args.d == "plantseg":
        build_plantseg_batches(
            args.output_dir,
            setname=args.t,
            text_len=text_len,
            input_h=input_h,
            input_w=input_w,
            plantseg_root=args.plantseg_root,
        )
    else:
        build_coco_batches(args.output_dir, dataset=args.d, setname=args.t, text_len=text_len, input_h=input_h, input_w=input_w)
