# Server Usage

This repository is prepared for Linux server execution with a single RTX 4090.

## Manual prerequisites

- Clone this repository to the server.
- Place the custom dataset as a sibling directory: `../dataset`.
- Provide a 300d GloVe text file on the server.
- Allow `timm` to access or reuse pretrained ViT weights.

## Environment

Create the server environment from `environment.server.yml`.

If you already created the environment before this pin was added, downgrade NumPy once:

```bash
conda activate restr-server
conda install -y -c conda-forge "numpy<2" "mkl<2024.1" "intel-openmp<2024.1"
```

## Dataset preparation

Generate custom ReSTR batches, vocabulary, and embeddings from the sibling dataset:

```bash
python prepare_plantseg_batches.py \
  --dataset-root ../dataset \
  --output-root ./data \
  --dataset-name plantseg \
  --caption-index 2 \
  --img-size 480 \
  --glove-path /path/to/glove.6B.300d.txt
```

Generated assets:

- `data/vocabulary_plantseg.txt`
- `data/plantseg_emb.npy`
- `data/plantseg_480_batch/train_batch/*.npz`
- `data/plantseg_480_batch/test_batch/*.npz`

Mask handling is binary by contract: background is `0`, and any non-zero source pixel becomes foreground.

## Training

Train without validation and save checkpoints under `weights/plantseg/`:

```bash
python train_restr.py \
  --data_dir ./data/plantseg_480_batch \
  --set train \
  --no_val \
  --adamW \
  --exp_name plantseg
```

Notes:

- `--no_val` disables validation loaders and validation-time checkpoint selection.
- The final training step always writes a checkpoint for later testing.
- If you want offline tracking, set `WANDB_MODE=offline` before launching training.

## Post-training evaluation

Run formal testing on the last checkpoint and write `metrics.json` into the evaluation output directory:

```bash
cd eval
python evaluate.py \
  --data_dir ../data/plantseg_480_batch \
  --set test \
  --restore_refseg ../weights/plantseg \
  --checkpoint_prefix plantseg \
  --iters <final_step> \
  --binary_metrics
```

Reported metrics:

- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mAcc`

## Output contract

- Training checkpoints: `weights/plantseg/plantseg_<step>.pth`
- Evaluation directory: `eval_dir/plantseg/test/<step>/`
- Metric summary: `eval_dir/plantseg/test/<step>/metrics.json`
