# BrainIAC on the local CN/MCI/AD slices

This repository implements the 4-task evaluation investigating whether pretraining with contrastive learning on structural brain MRI provides transferable representations for 3-class CN/MCI/AD classification, and whether further SimCLR adaptation on the dataset improves performance.

## Checkpoints and Dataset Downloads

Because model checkpoints and raw array datasets exceed GitHub's 100 MB file limit, download them from Dropbox:

🔗 **Dropbox Download Link:** [BrainIAC Checkpoints & Data](https://www.dropbox.com/scl/fo/0ajukuwdtvej404iorjgf/ACmRsYirh_PXUq975fnoznw?rlkey=pgdo6lryibb5fefegoz9nj6nv&st=n3cm0zry&dl=0)

Place the downloaded files into their respective directories:
- **Checkpoints** -> `src/checkpoints/`:
  - `BrainIAC.ckpt` (pretrained foundation model)
  - `BrainIAC_simclr_finetuned.ckpt` (SimCLR-adapted model)
- **Data** -> `src/data/`:
  - `brain_train_image_final.npy`
  - `brain_train_label.npy`
  - `brain_test_image_final.npy`
  - `brain_test_label.npy`

---

## Setup

```bash
pip install -r requirements.txt
cd src
```

*Note: Add `--device cpu` to extraction and adaptation commands if running on a machine without a GPU.*

---

## The 4 Tasks

### Task 1: Pretrained Feature Extraction
Load the pretrained BrainIAC encoder (`checkpoints/BrainIAC.ckpt`), freeze its weights, and extract 768-d latent feature representations for the training, validation, and test splits:
```bash
python extract_features.py --checkpoint checkpoints/BrainIAC.ckpt --output features/pretrained.npz --batch-size 4 --val-fraction 0.2 --seed 0
```

### Task 2: Linear Probing on Pretrained Features
Freeze the pretrained encoder representations, train a linear classification layer on the extracted training features to predict the three classes (CN, MCI, AD), and evaluate performance on the validation and test sets:
```bash
python linear_probe.py --features features/pretrained.npz --output-dir results/pretrained --seed 0
```
Outputs accuracy, precision, recall, F1, AUROC, and confusion matrix saved to `results/pretrained/metrics.txt`.

### Task 3: Fine-Tuning (SimCLR Contrastive Adaptation)
Continue contrastive learning (SimCLR-style NT-Xent loss) using only the brain MRI training data to adapt the pretrained encoder to the target dataset. The test set is completely held out:
```bash
python finetune_simclr.py --epochs 50 --batch-size 8 --lr 0.0001 --weight-decay 0.00003 --temperature 0.1 --warmup-epochs 5 --val-fraction 0.2 --seed 0
```
The best adapted encoder checkpoint is saved to `checkpoints/BrainIAC_simclr_finetuned.ckpt`.

### Task 4: Post-Fine-Tuning Evaluation
Freeze the adapted encoder from Task 3, extract its latent feature representations:
```bash
python extract_features.py --checkpoint checkpoints/BrainIAC_simclr_finetuned.ckpt --output features/adapted.npz --batch-size 4 --val-fraction 0.2 --seed 0
```
Then train and evaluate the linear classification layer on the adapted features:
```bash
python linear_probe.py --features features/adapted.npz --output-dir results/adapted --seed 0
```
Outputs accuracy, precision, recall, F1, AUROC, and confusion matrix saved to `results/adapted/metrics.txt`.

---

## Separate Testing Script (Parts 2 & 4 Evaluation)

To run the full evaluation across both the pretrained and adapted encoders in a single command, use `test_linear_probe.py`. It uses `extract_features` from parts 1 and 3, runs the linear probes for parts 2 and 4, and pretty-prints the metrics separately in the terminal:

```bash
python test_linear_probe.py --batch-size 4 --val-fraction 0.2 --seed 0
```

*(Add `--skip-extract` if `features/pretrained.npz` and `features/adapted.npz` have already been generated).*
