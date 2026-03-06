import argparse

import torch
import torch.multiprocessing as mp

from framework.base import main_worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AT-BPTT++ dataset distillation")
    parser.add_argument("--seed", default=0, type=int)

    # Distributed training
    parser.add_argument("--mp_distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="Number of processes")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str, help="Distributed URL")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="Distributed backend")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id for single-GPU training")

    # Core optimization and data settings
    parser.add_argument("--root", default="./dataset", type=str, help="Dataset root directory")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset name")
    parser.add_argument("--arch", default="convnet", type=str, help="Backbone architecture")
    parser.add_argument("--lr", default=1e-3, type=float, help="Outer learning rate for distilled data")
    parser.add_argument("--inner_optim", default="Adam", type=str, help="Inner optimizer type")
    parser.add_argument("--outer_optim", default="Adam", type=str, help="Outer optimizer type")
    parser.add_argument("--inner_lr", default=1e-3, type=float, help="Inner learning rate")
    parser.add_argument("--label_lr_scale", default=1.0, type=float, help="Label learning rate divisor")
    parser.add_argument("--num_per_class", default=10, type=int, help="Images per class (IPC)")
    parser.add_argument("--batch_per_class", default=10, type=int, help="Samples per class in each synthetic batch")
    parser.add_argument("--task_sampler_nc", default=10, type=int, help="Classes sampled per synthetic batch")
    parser.add_argument("--window", default=60, type=int, help="Base truncation window")
    parser.add_argument("--minwindow", default=0, type=int, help="Legacy argument for curriculum mode")
    parser.add_argument("--totwindow", default=200, type=int, help="Total trajectory length")
    parser.add_argument("--num_train_eval", default=8, type=int, help="Model retrains for evaluation")
    parser.add_argument("--train_y", action="store_true", help="Enable soft-label optimization")
    parser.add_argument("--batch_size", default=5000, type=int, help="Mini-batch size for real data")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--test_freq", default=25, type=int, help="Evaluation frequency (epochs)")
    parser.add_argument("--print_freq", default=10, type=int, help="Logging frequency (steps)")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=60000, type=int)
    parser.add_argument("--epoch", dest="epochs", type=int, help="Alias for --epochs")

    # Distillation mode
    parser.add_argument(
        "--ddtype",
        default="at_bptt_pp",
        type=str,
        choices=["standard", "curriculum", "at_bptt_pp"],
        help="Distillation strategy",
    )
    parser.add_argument("--cctype", default=0, type=int, help="Legacy curriculum type for RaT-BPTT compatibility")
    parser.add_argument("--window_radius", default=20, type=int, help="Radius for adaptive window around base window")

    # WTPO selector hyperparameters
    parser.add_argument("--selector_kl", default=1.0, type=float, help="KL coefficient in proximal policy update")
    parser.add_argument("--selector_ref_ema", default=0.05, type=float, help="EMA rate for reference policy")
    parser.add_argument("--selector_tau_w", default=1.0, type=float, help="Temperature for window scaling")

    # CR-LRHA hyperparameters
    parser.add_argument("--disable_cr_lrha", action="store_true", help="Disable cache-and-refresh low-rank module")
    parser.add_argument("--cr_rank", default=32, type=int, help="Low-rank dimension for CR-LRHA")
    parser.add_argument("--cr_delta", default=0.05, type=float, help="Refresh threshold for relative drift")
    parser.add_argument("--cr_period", default=20, type=int, help="Periodic refresh interval")
    parser.add_argument("--cr_blend", default=1.0, type=float, help="Blend weight for low-rank projected gradient")

    # Augmentation and logging
    parser.add_argument("--zca", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--clip_coef", default=0.9, type=float, help="EMA coefficient for gradient clipping")
    parser.add_argument("--fname", default="at_bptt_pp", type=str, help="Run name suffix for checkpoint files")
    parser.add_argument("--name", default="at_bptt_pp", type=str, help="wandb experiment name")
    parser.add_argument("--comp_aug", action="store_true", help="Compose synthetic augmentations")
    parser.add_argument("--comp_aug_real", action="store_true", help="Compose real-data augmentations")
    parser.add_argument("--syn_strategy", default="flip_rotate", type=str, help="Synthetic data augmentation strategy")
    parser.add_argument("--real_strategy", default="flip_rotate", type=str, help="Real data augmentation strategy")
    parser.add_argument("--ckptname", default="none", type=str, help="Checkpoint for distilled data initialization")
    parser.add_argument("--limit_train", action="store_true", help="Use a reduced training split")
    parser.add_argument("--load_ckpt", action="store_true")
    parser.add_argument("--complete_random", action="store_true", help="Legacy random curriculum behavior")

    args = parser.parse_args()
    args.use_cr_lrha = not args.disable_cr_lrha

    args.distributed = args.world_size > 1 or args.mp_distributed
    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.num_train_eval = int(args.num_train_eval / ngpus_per_node)

    if args.mp_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        for _ in range(5):
            torch.cuda.empty_cache()
    else:
        main_worker(args.gpu, ngpus_per_node, args)
