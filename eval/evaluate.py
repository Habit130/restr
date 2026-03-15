import sys
sys.path.append('../')
from eval.visualization import save_img_gt, save_mask_soft_torch, save_mask_softpatch_torch, save_mask_torch
import json
import os
import os.path as osp
import argparse
from torch.utils import data
from dataset.referit_dataset_vit import ReferDataSet_vit

import numpy as np
import torch
from torch.autograd import Variable
import os
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils.torchutils import compute_mask_IU, compute_mask_IU_torch, decode_image_vit, resize_and_crop_nearest, resize_and_pad, resize_and_crop, vis_torch_mask
from utils.imutils import crf_inference_label
import torch.nn as nn
import torch.backends.cudnn as cudnn
import config
from model.factory import create_restr
from utils.binary_metrics import BinarySegMeter


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
THRESHOLD = 1e-9


def infer_dataset_name(data_dir):
    return osp.basename(osp.normpath(data_dir)).split("_")[0]


def infer_data_root(data_dir):
    return osp.dirname(osp.normpath(data_dir))


def resolve_checkpoint_path(restore_refseg, dataset_name, i_iter, checkpoint_prefix=None):
    prefixes = []
    if checkpoint_prefix:
        prefixes.append(checkpoint_prefix)
    if dataset_name not in prefixes:
        prefixes.append(dataset_name)
    restore_name = osp.basename(osp.normpath(restore_refseg))
    if restore_name not in prefixes:
        prefixes.append(restore_name)

    for prefix in prefixes:
        candidate = osp.join(restore_refseg, prefix + "_" + str(i_iter) + ".pth")
        if osp.exists(candidate):
            return candidate

    return osp.join(restore_refseg, prefixes[0] + "_" + str(i_iter) + ".pth")


def evaluate(output_eval_dir, i_iter, model
            , H = 320, W =320, valloader=None, is_vis = False, save_im_sent = False
            , threshold=1e-8, is_prec=False, dataname = None, binary_metrics=False
            , metrics_output_path=None):
    os.makedirs(output_eval_dir+'/img', exist_ok=True)
    os.makedirs(output_eval_dir+'/gt', exist_ok=True)
    os.makedirs(output_eval_dir+'/pred_sigm', exist_ok=True)
    os.makedirs(output_eval_dir+'/pred_h', exist_ok=True)
    os.makedirs(output_eval_dir+'/pred_cmap', exist_ok=True)


    model.eval()
    cum_I, cum_U = 0., 0.
    cum_I_sigm, cum_U_sigm = 0., 0.
    cum_I_crf, cum_U_crf = 0., 0.
    if is_prec is True:
        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
        seg_total = 0.
    binary_meter = BinarySegMeter() if binary_metrics else None
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
            
            labels_np = labels.cpu().data.numpy()

            orig_images = decode_image_vit(orig_images[0], normalization=STATS['vit'])

            hard_pred = (pred >= threshold)
            try:
                I, U = compute_mask_IU_torch(hard_pred, labels)
                I = int(I.item())
                U = int(U.item())
            except Exception as exc:
                pred_np = hard_pred.detach().cpu().numpy().astype(bool)
                label_np = labels.detach().cpu().numpy().astype(bool)
                if pred_np.shape != label_np.shape:
                    raise RuntimeError(
                        "Failed to compute IoU for sample {} with pred shape {} and label shape {}".format(
                            name, pred_np.shape, label_np.shape
                        )
                    ) from exc
                print("Falling back to numpy IoU for sample {}".format(name))
                I = int(np.logical_and(pred_np, label_np).sum())
                U = int(np.logical_or(pred_np, label_np).sum())
            cum_I += I
            cum_U += U
            if binary_meter is not None:
                binary_meter.update(hard_pred, labels)
            sample_iou = (I / U) if U > 0 else 1.0
            if is_prec is True:
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (sample_iou >= eval_seg_iou)
                seg_total += 1

            if is_vis and index < 400:
                for i in range(B):
                    sigm_output = sigm_pred[i].permute(1,2,0).squeeze()         # C H W -> H W C
                    save_mask_soft_torch(sigm_output, name[i], output_eval_dir)

                    if is_local_up is True:
                        sigm_l_output = sigm_l_pred[i].permute(1,2,0).squeeze()
                        save_mask_softpatch_torch(sigm_l_output, name[i], output_eval_dir)

                    hard_output = hard_pred[i].permute(1,2,0).squeeze()
                    save_mask_torch(hard_output, name[i], output_eval_dir)

                    if save_im_sent:
                        sent = "_".join(sents[i].split(" ")).replace("/", "_")
                        save_img_gt(orig_images, labels[i].cpu().data.numpy().squeeze(), name[i], sent, output_eval_dir)

            if cum_U > 0:
                t.set_postfix({"mIoU ":" %.3f%% " % (100*(cum_I/cum_U))})  
            
            if i_iter == 1 and dataname is None:
                t.close()
                break

        metrics = {}
        if is_prec is True:
            result_str = ''
            for n_eval_iou in range(len(eval_seg_iou_list)):
                result_str += 'Prec@%s = %f | \t' % \
                            (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
            print(result_str)
            print("Cumulated_IoU = %.3f" %(100*(cum_I/cum_U)))
            metrics["precision_at"] = {
                str(eval_seg_iou_list[n_eval_iou]): float(seg_correct[n_eval_iou] / seg_total)
                for n_eval_iou in range(len(eval_seg_iou_list))
            }
        cum_IoU = 100*(cum_I/cum_U) if cum_U > 0 else 0.0
        metrics["cum_IoU"] = float(cum_IoU)

        if binary_meter is not None:
            binary_summary = binary_meter.compute()
            metrics.update(binary_summary)
            print(
                "IoU = {iou:.4f} | Dice = {dice:.4f} | Recall = {recall:.4f} | "
                "mIoU = {miou:.4f} | mAcc = {macc:.4f}".format(
                    iou=binary_summary["IoU"],
                    dice=binary_summary["Dice"],
                    recall=binary_summary["Recall"],
                    miou=binary_summary["mIoU"],
                    macc=binary_summary["mAcc"],
                )
            )

        if metrics_output_path is not None:
            with open(metrics_output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)

        if binary_metrics:
            return metrics
        return cum_IoU


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
    parser.add_argument("--restore_refseg", type=str, required=True)
    parser.add_argument('--iters', type=int, nargs='*', required=True)
    parser.add_argument("--checkpoint_prefix", type=str, default=None)
    parser.add_argument("--binary_metrics", action="store_true")

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
    v_model_cfg = cfg["v_backbone"][v_backbone]
    v_model_cfg["image_size"] = input_size
    v_model_cfg["backbone"] = v_backbone
    model_cfg["v_backbone"] = v_model_cfg
    
    l_backbone = args.l_backbone
    l_model_cfg = cfg["l_backbone"][l_backbone]
    # l_model_cfg["backbone"] = l_backbone
    l_model_cfg["n_heads"] = v_model_cfg["n_heads"]
    l_model_cfg["emb_name"] = dataset_name
    l_model_cfg["d_model"] = v_model_cfg["d_model"]
    l_model_cfg["data_root"] = infer_data_root(args.data_dir)
    model_cfg["l_backbone"] = l_model_cfg

    fusion_module = args.mm_fusion
    mm_fusion_cfg = cfg["fusion_module"]
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

    if len(args.iters) == 1 and args.iters[0] == 0:
        for i in range(50):
            (args.iters).append((i+1)*5000)
        del (args.iters)[0]

    print(str(args.iters) + ' will be evaluated!')
    output_dir = osp.basename(osp.normpath(args.restore_refseg)) if args.output_dir is None else args.output_dir
    


    dataset_name = infer_dataset_name(args.data_dir)
    print("Training dataset: {}".format(dataset_name))
    print(resolve_checkpoint_path(args.restore_refseg, dataset_name, args.iters[0], args.checkpoint_prefix))

    # Create network.
    cfg = config.load_config()
    model_cfg = make_model_cfg(args, cfg, input_size, dataset_name)

    model = create_restr(model_cfg)
    output_root = os.path.join("../eval_dir/", output_dir, args.set)
        
    for i_iter in args.iters:
        weight_path_vis = resolve_checkpoint_path(args.restore_refseg, dataset_name, i_iter, args.checkpoint_prefix)
        saved_state_dict_vis = torch.load(weight_path_vis)


        model.load_state_dict(saved_state_dict_vis)
        model.eval()
        model.cuda()
        num_model = sum(p.numel() for p in model.parameters())
        print("# of parameters: ", num_model)

        valloader = data.DataLoader(ReferDataSet_vit(args.data_dir, args.set), 
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        output_eval_dir = os.path.join(output_root, str(i_iter))
        evaluate(output_eval_dir, i_iter, model, valloader=valloader, H=input_size[0], W=input_size[1]
            , is_vis=args.is_vis, save_im_sent = True, threshold=args.threshold, is_prec=not args.binary_metrics
            , dataname = dataset_name, binary_metrics=args.binary_metrics
            , metrics_output_path=os.path.join(output_eval_dir, "metrics.json"))


if __name__ == '__main__':
    main()
