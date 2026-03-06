# AT-BPTT++: Improving Inner-loop Dataset Distillation with Automatic Reinforcement

#### Project Page | Paper (coming soon)

This repository contains a clean and runnable implementation of **AT-BPTT++** based on the RaT-BPTT code structure from `Simple_Dataset_Distillation`.

## Abstract

Dataset distillation is effective but expensive because outer updates require differentiating through long inner optimization trajectories.  
AT-BPTT++ improves this inner-loop process with two components:

1. **WTPO selector** (Within-Trajectory Policy Optimization):  
   It adaptively selects truncation position and window size using standardized trajectory-wise gradient signals and a reference-policy proximal update.
2. **CR-LRHA** (Cache-and-Refresh Low-Rank Hessian Approximation):  
   It amortizes second-order computation by caching low-rank factors and refreshing only when drift is significant.

Together, they improve robustness and efficiency for truncated meta-gradient optimization.

## Introduction

Key points of this implementation:

- Implements a practical **AT-BPTT++ training path** under `--ddtype at_bptt_pp`.
- Keeps **legacy compatibility** with `standard` and `curriculum` modes.
- Provides configurable WTPO and CR-LRHA hyperparameters directly from CLI.
- Preserves original dataset/model framework for CIFAR/Tiny-ImageNet/ImageNet style experiments.

## Performance

This repository focuses on reproducible code and training workflow.  
For full quantitative comparisons and ablations, please refer to your manuscript tables and figures.

## Getting Started

### Code Structure

```text
.
+-- framework
|   +-- base.py              # Training worker, test, logging
|   +-- config.py            # Datasets and model factory
|   +-- distill_higher.py    # Distill model + WTPO + CR-LRHA
|   +-- convnet.py
|   +-- model.py
|   +-- vgg.py
|   +-- util.py
|   +-- metrics.py
+-- environment.yml
+-- main.py                  # Main entrypoint
+-- README.md
```

### Environment Preparation

```bash
conda env create -f environment.yml
conda activate ffcv
```

### Data Preparation

Datasets are handled via `torchvision` in `framework/config.py`.  
If needed, adjust `--root` to your local dataset directory.

### Example Usage

#### CIFAR-10 (AT-BPTT++)

```bash
python main.py \
  --dataset cifar10 \
  --arch convnet \
  --num_per_class 10 \
  --batch_per_class 10 \
  --task_sampler_nc 10 \
  --ddtype at_bptt_pp \
  --window 60 \
  --totwindow 200 \
  --window_radius 20 \
  --selector_kl 1.0 \
  --selector_ref_ema 0.05 \
  --selector_tau_w 1.0 \
  --cr_rank 32 \
  --cr_delta 0.05 \
  --cr_period 20 \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --batch_size 5000 \
  --epochs 60000 \
  --test_freq 25 \
  --print_freq 10 \
  --zca \
  --syn_strategy flip_rotate \
  --real_strategy flip_rotate \
  --fname cifar10_at_bptt_pp
```

#### Legacy RaT-BPTT-compatible mode

```bash
python main.py \
  --dataset cifar10 \
  --ddtype curriculum \
  --cctype 2 \
  --window 60 \
  --totwindow 200
```

### Important Arguments

- `--ddtype {at_bptt_pp, standard, curriculum}`: selects distillation mode.
- `--window`, `--totwindow`, `--window_radius`: truncation horizon controls.
- `--selector_kl`, `--selector_ref_ema`, `--selector_tau_w`: WTPO policy controls.
- `--cr_rank`, `--cr_delta`, `--cr_period`, `--cr_blend`: CR-LRHA controls.
- `--disable_cr_lrha`: disables cache-and-refresh low-rank module.

## Output

Checkpoints and distilled data are saved to:

```text
./save/<dataset>/
```

## Citation

If you use this code, please cite your AT-BPTT++ paper and the RaT-BPTT baseline paper.

```bibtex
@inproceedings{feng2024embarrassingly,
  title={Embarrassingly Simple Dataset Distillation},
  author={Feng, Yunzhen and Vedantam, Ramakrishna and Kempe, Julia},
  booktitle={ICLR},
  year={2024}
}
```

## Acknowledgement

- RaT-BPTT baseline code: `fengyzpku/Simple_Dataset_Distillation`
