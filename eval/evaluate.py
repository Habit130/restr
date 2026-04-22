import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.visualization import save_img_gt, save_mask_soft_torch, save_mask_softpatch_torch, save_mask_torch
import os
import os.path as osp
import argparse
import json
from torch.utils import data
from dataset.referit_dataset_vit import ReferDataSet_vit

import numpy as np
import torch
from torch.autograd import Variable
import os
from PIL import Image
from tqdm import tqdm
from utils.torchutils import decode_image_vit, resize_and_crop_nearest, resize_and_pad, resize_and_crop
import torch.backends.cudnn as cudnn
import config
from model.factory import create_restr
from utils.text_assets import get_dataset_text_spec


STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "oldbgr": {"mean": (0.485, 0.456, 0.406), "std": (1, 1, 1)}
}
DATA_DIRECTORY = '../data/referit/referit_480_batch'
SET = 'test'
INPUT_SIZE = '480,480'
BATCH_SIZE = 8

DROPOUT = 0
DROP_PATH = 0.1
IGNORE_LABEL = 255
THRESHOLD = 0.5


def infer_dataset_name(data_dir):
    return Path(data_dir).name.split("_")[0]


def infer_data_root(data_dir):
    return str(Path(data_dir).resolve().parent.parent)


def resolve_text_len(dataset_name, override=None):
    if override is not None:
        return override
    return get_dataset_text_spec(dataset_name)["text_len"]


def ensure_legacy_eval_dirs(output_eval_dir):
    os.makedirs(output_eval_dir, exist_ok=True)
    for subdir in ("img", "gt", "pred_sigm", "pred_h", "pred_cmap"):
        os.makedirs(os.path.join(output_eval_dir, subdir), exist_ok=True)


def resolve_mask_relative_path(dataset_name, sample_name):
    stem = Path(str(sample_name)).stem
    if dataset_name == "plantseg":
        return Path("ann") / f"{stem}.png"
    return Path(f"{stem}.png")


def save_binary_mask_file(mask_tensor, output_path):
    mask_array = mask_tensor.detach().cpu().numpy()
    out_image = Image.fromarray(np.uint8(mask_array * 255), "L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(output_path)


def update_confusion_counts(pred_mask, target_mask):
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    tp = torch.logical_and(pred_mask, target_mask).sum().item()
    fp = torch.logical_and(pred_mask, torch.logical_not(target_mask)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred_mask), target_mask).sum().item()
    tn = torch.logical_and(torch.logical_not(pred_mask), torch.logical_not(target_mask)).sum().item()
    return tp, fp, fn, tn


def safe_ratio(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def summarize_binary_metrics(tp, fp, fn, tn):
    fg_iou = safe_ratio(tp, tp + fp + fn)
    bg_iou = safe_ratio(tn, tn + fp + fn)
    fg_acc = safe_ratio(tp, tp + fn)
    bg_acc = safe_ratio(tn, tn + fp)
    metrics = {
        "IoU": 100.0 * fg_iou,
        "Dice": 100.0 * safe_ratio(2 * tp, 2 * tp + fp + fn),
        "Recall": 100.0 * fg_acc,
        "mIoU": 100.0 * ((fg_iou + bg_iou) / 2.0),
        "mACC": 100.0 * ((fg_acc + bg_acc) / 2.0),
    }
    return metrics


def evaluate(output_eval_dir, i_iter, model
            , H = 320, W =320, valloader=None, is_vis = False, save_im_sent = False
            , threshold=THRESHOLD, is_prec=False, dataname = None, mask_output_dir=None,
            save_masks_only=False):
    os.makedirs(output_eval_dir, exist_ok=True)
    if is_vis or save_im_sent:
        ensure_legacy_eval_dirs(output_eval_dir)

    mask_output_root = None
    if mask_output_dir is not None:
        mask_output_root = Path(mask_output_dir)
        mask_output_root.mkdir(parents=True, exist_ok=True)


    model.eval()
    tp_sum, fp_sum, fn_sum, tn_sum = 0, 0, 0, 0
    if is_prec is True:
        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
        seg_total = 0.
    with torch.no_grad():
        t= tqdm(valloader, desc='Evaluating', leave=True)
        for index, batch in enumerate(t):
            images, labels, size, texts, sents, name = batch
            orig_images = images
            B, _, orig_H, orig_W = images.size()
            images = resize_and_pad(images, H, W)
            
            
            images = Variable(images).cuda()
            labels = Variable(labels.float()).cuda()
            texts =Variable(texts).cuda()
        
            # 1. Extract visual feature
            pred = model(images, texts)
            is_local_up = False
            if type(pred) is tuple:
                pred, _, l_pred = pred
                is_local_up = True if l_pred is not None else False
            # 2. Soft map for pred and local pred
            sigm_pred = torch.sigmoid(pred)
            if is_local_up is True:
                sigm_l_pred = torch.sigmoid(l_pred)
                sigm_l_pred = resize_and_crop_nearest(sigm_l_pred, orig_H, orig_W)
            
            pred = resize_and_crop(pred, orig_H, orig_W)
            sigm_pred = resize_and_crop(sigm_pred, orig_H, orig_W)

            orig_images = decode_image_vit(orig_images[0], normalization=STATS['vit'])

            hard_pred = (sigm_pred >= threshold) if 0.0 <= threshold <= 1.0 else (pred >= threshold)
            if not save_masks_only:
                batch_tp, batch_fp, batch_fn, batch_tn = update_confusion_counts(hard_pred, labels)
                tp_sum += batch_tp
                fp_sum += batch_fp
                fn_sum += batch_fn
                tn_sum += batch_tn
                batch_iou = safe_ratio(batch_tp, batch_tp + batch_fp + batch_fn)
                if is_prec is True:
                    for n_eval_iou in range(len(eval_seg_iou_list)):
                        eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                        seg_correct[n_eval_iou] += (batch_iou >= eval_seg_iou)
                    seg_total += 1

            if mask_output_root is not None:
                for i in range(B):
                    hard_output = hard_pred[i].permute(1,2,0).squeeze()
                    relative_path = resolve_mask_relative_path(dataname, name[i])
                    save_binary_mask_file(hard_output, mask_output_root / relative_path)

            if is_vis and index < 400:
                for i in range(B):
                    sigm_output = sigm_pred[i].permute(1,2,0).squeeze()         # C H W -> H W C
                    save_mask_soft_torch(sigm_output, name[i], output_eval_dir)

                    if is_local_up is True:
                        sigm_l_output = sigm_l_pred[i].permute(1,2,0).squeeze()
                        save_mask_softpatch_torch(sigm_l_output, name[i], output_eval_dir)

                    hard_output = hard_pred[i].permute(1,2,0).squeeze()
                    save_mask_torch(hard_output, Path(str(name[i])).stem, output_eval_dir)

                    if save_im_sent:
                        sent = "_".join(sents[i].split(" ")).replace("/", "_")
                        save_img_gt(orig_images, labels[i].cpu().data.numpy().squeeze(), Path(str(name[i])).stem, sent, output_eval_dir)

            if not save_masks_only:
                running_metrics = summarize_binary_metrics(tp_sum, fp_sum, fn_sum, tn_sum)
                t.set_postfix({"IoU ": " %.3f%% " % running_metrics["IoU"]})  
            
            if i_iter == 1:
                t.close()
                break

        if save_masks_only:
            return None

        if is_prec is True:
            result_str = ''
            for n_eval_iou in range(len(eval_seg_iou_list)):
                result_str += 'Prec@%s = %f | \t' % \
                            (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
            print(result_str)
        metrics = summarize_binary_metrics(tp_sum, fp_sum, fn_sum, tn_sum)
        print(
            "IoU={IoU:.3f} Dice={Dice:.3f} Recall={Recall:.3f} mIoU={mIoU:.3f} mACC={mACC:.3f}".format(
                **metrics
            )
        )
        with open(os.path.join(output_eval_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics


def get_arguments():
    """
    Returns: A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--set", type=str, default=SET)


    parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--restore_refseg", type=str, default=None)
    parser.add_argument('--iters', type=int, nargs='*', default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--text_len", type=int, default=None)
    parser.add_argument("--mask_output_dir", type=str, default=None)
    parser.add_argument("--save_masks_only", action="store_true")

    parser.add_argument("--v_backbone", type=str, default="vit_base_patch16_384")
    parser.add_argument("--l_backbone", type=str, default="transformer_glove")
    parser.add_argument("--mm_fusion", type=str, default="decoder_transformer")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--adamW", action="store_true")

    parser.add_argument("--is_shared", action="store_true")
    parser.add_argument("--no_decoder", action="store_true")

    parser.add_argument("--is_vis", action="store_true")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()

def make_model_cfg(args, cfg, input_size, dataset_name):

    model_cfg = {}
    v_backbone = args.v_backbone
    v_model_cfg = cfg["v_backbone"][v_backbone].copy()
    v_model_cfg["image_size"] = input_size
    v_model_cfg["backbone"] = v_backbone
    model_cfg["v_backbone"] = v_model_cfg
    
    l_backbone = args.l_backbone
    l_model_cfg = cfg["l_backbone"][l_backbone].copy()
    l_model_cfg["n_heads"] = v_model_cfg["n_heads"]
    l_model_cfg["emb_name"] = dataset_name
    l_model_cfg["d_model"] = v_model_cfg["d_model"]
    l_model_cfg["data_root"] = infer_data_root(args.data_dir)
    l_model_cfg["text_len"] = resolve_text_len(dataset_name, args.text_len)
    model_cfg["l_backbone"] = l_model_cfg

    fusion_module = args.mm_fusion
    mm_fusion_cfg = cfg["fusion_module"].copy()
    mm_fusion_cfg["name"] = fusion_module
    mm_fusion_cfg["is_shared"] =args.is_shared
    mm_fusion_cfg["is_decoder"] = not args.no_decoder
    mm_fusion_cfg["n_heads"] = v_model_cfg["n_heads"]
    model_cfg["mm_fusion"] = mm_fusion_cfg
    
    return model_cfg

def main():
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    if args.checkpoint is None and args.restore_refseg is None:
        raise ValueError("Either --checkpoint or --restore_refseg must be provided.")
    if args.checkpoint is None and not args.iters:
        raise ValueError("--iters is required when --restore_refseg is used.")
    if args.checkpoint is None and len(args.iters) == 1 and args.iters[0] == 0:
        for i in range(50):
            args.iters.append((i + 1) * 5000)
        del args.iters[0]

    dataset_name = infer_dataset_name(args.data_dir)
    print("Training dataset: {}".format(dataset_name))
    output_dir = osp.basename(args.restore_refseg) if args.output_dir is None and args.restore_refseg else dataset_name
    if args.output_dir is not None:
        output_dir = args.output_dir

    # Create network.
    cfg = config.load_config()
    model_cfg = make_model_cfg(args, cfg, input_size, dataset_name)

    model = create_restr(model_cfg)
    checkpoint_entries = []
    if args.checkpoint is not None:
        checkpoint_entries.append(("best", args.checkpoint))
    else:
        print(str(args.iters) + ' will be evaluated!')
        for i_iter in args.iters:
            weight_path_vis = args.restore_refseg+"/"+osp.basename(args.restore_refseg)+"_"+str(i_iter)+".pth"
            checkpoint_entries.append((str(i_iter), weight_path_vis))

    output_root = Path("../eval_dir") / output_dir / args.set
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for i_iter, checkpoint_path in checkpoint_entries:
        saved_state_dict_vis = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(saved_state_dict_vis)
        model.eval()
        model.cuda()
        num_model = sum(p.numel() for p in model.parameters())
        print("# of parameters: ", num_model)

        valloader = data.DataLoader(ReferDataSet_vit(args.data_dir, args.set), 
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        output_eval_dir = str(output_root / str(i_iter))
        metrics = evaluate(output_eval_dir, i_iter, model, valloader=valloader, H=input_size[0], W=input_size[1]
            , is_vis=args.is_vis, save_im_sent = True, threshold=args.threshold, is_prec=True, dataname = dataset_name,
            mask_output_dir=args.mask_output_dir, save_masks_only=args.save_masks_only)
        if metrics is not None:
            summary[str(i_iter)] = {"checkpoint": checkpoint_path, "metrics": metrics}

    if summary:
        with open(output_root / "summary_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
