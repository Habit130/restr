# PlantSeg Linux Server Workflow

本仓库已经适配到单卡 RTX 4090 / CUDA 11.8 / Python 3.10 的 Linux 服务器路径，目标数据集固定为与仓库同级的 `plantseg` 目录。

## 目录约定

- 当前仓库根目录：`restr`
- 数据集根目录：`../plantseg`
- 标注真源：`../plantseg/main.json`
- 文本输入：每条样本的 `caption[3]`

## 环境

使用 [environment.linux.4090.yml](/E:/Projects/DisSeg/restr/environment.linux.4090.yml) 创建 conda 环境。该环境面向：

- Python 3.10
- PyTorch 2.1
- CUDA 11.8
- timm 0.6.11
- transformers 4.22.2

## 自动下载与自动生成

首次构建 `plantseg` batch 时会自动完成：

- 下载官方 `glove.6B.zip`
- 生成 `data/vocabulary_plantseg.txt`
- 生成 `data/plantseg_emb.npy`

首次训练时会由 `timm` 自动下载 `vit_base_patch16_384` 的官方预训练权重。

## 数据准备

训练、验证、测试都使用 `plantseg/main.json` 自带的 `split` 字段，不重切分。

构建 batch：

```bash
python build_batches.py -d plantseg -t train --img-size 480
python build_batches.py -d plantseg -t val --img-size 480
python build_batches.py -d plantseg -t test --img-size 480
```

构建结果位于：

- `data/plantseg/plantseg_480_batch/train_batch`
- `data/plantseg/plantseg_480_batch/val_batch`
- `data/plantseg/plantseg_480_batch/test_batch`

## 训练

默认关闭 W&B；需要时显式加 `--enable_wandb`。

```bash
python train_restr.py \
  --data_dir ./data/plantseg/plantseg_480_batch \
  --set train \
  --valset val \
  --text_len 64 \
  --adamW \
  --exp_name restr_plantseg
```

训练会在 `weights/restr_plantseg/` 下产出：

- `best_iou.pth`
- `best_iou_metrics.json`
- 兼容旧流程的迭代 checkpoint

## 正式评估

正式评估固定使用 `val` 上前景 IoU 最优的 `best_iou.pth`，在 `test` 上汇报：

- IoU
- Dice
- Recall
- mIoU
- mACC

```bash
cd eval
python evaluate.py \
  --data_dir ../data/plantseg/plantseg_480_batch \
  --set test \
  --text_len 64 \
  --checkpoint ../weights/restr_plantseg/best_iou.pth
```

评估结果位于：

- `eval_dir/plantseg/test/best/metrics.json`
- `eval_dir/plantseg/test/summary_metrics.json`

## 指标口径

- `IoU`：前景 IoU
- `Dice`：前景 Dice
- `Recall`：前景 Recall
- `mIoU`：前景/背景 IoU 平均
- `mACC`：前景/背景 Accuracy 平均

所有指标都按全数据集累计像素级 `TP/FP/FN/TN` 后统一计算，不做逐图平均。
