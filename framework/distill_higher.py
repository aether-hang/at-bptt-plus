import random
from typing import Dict, Optional, Tuple

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from framework.config import get_arch


def _global_grad_norm(parameters) -> torch.Tensor:
    grad_sq_sum = None
    device = None
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        device = grad.device
        norm_term = grad.pow(2).sum()
        grad_sq_sum = norm_term if grad_sq_sum is None else grad_sq_sum + norm_term
    if grad_sq_sum is None:
        return torch.tensor(0.0, device=device if device is not None else "cpu")
    return torch.sqrt(grad_sq_sum + 1e-12)


class WTPOSelector:
    """Within-trajectory policy optimization selector for truncation and window size."""

    def __init__(
        self,
        trajectory_steps: int,
        base_window: int,
        window_radius: int,
        kl_lambda: float,
        ref_ema: float,
        tau_w: float,
        eps: float = 1e-8,
    ):
        self.trajectory_steps = max(1, int(trajectory_steps))
        self.base_window = max(1, int(base_window))
        self.window_radius = max(0, int(window_radius))
        self.kl_lambda = max(float(kl_lambda), eps)
        self.ref_ema = float(ref_ema)
        self.tau_w = max(float(tau_w), eps)
        self.eps = eps

        init_policy = torch.full((self.trajectory_steps,), 1.0 / self.trajectory_steps)
        self.policy = init_policy.clone()
        self.ref_policy = init_policy.clone()

        self.running_g_min: Optional[float] = None
        self.running_g_max: Optional[float] = None
        self.prev_grad_norms: Optional[torch.Tensor] = None

        self.last_beta = 1.0
        self.last_advantage_w: Optional[torch.Tensor] = None

    def _ensure_device(self, device: torch.device) -> None:
        if self.policy.device != device:
            self.policy = self.policy.to(device)
            self.ref_policy = self.ref_policy.to(device)
            if self.prev_grad_norms is not None:
                self.prev_grad_norms = self.prev_grad_norms.to(device)

    def _standardize(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean()
        std = values.std(unbiased=False)
        return (values - mean) / (std + self.eps)

    def _compute_beta(self, mean_grad: float) -> float:
        if self.running_g_min is None or self.running_g_max is None:
            self.running_g_min = mean_grad
            self.running_g_max = mean_grad
            return 1.0
        self.running_g_min = min(self.running_g_min, mean_grad)
        self.running_g_max = max(self.running_g_max, mean_grad)
        span = self.running_g_max - self.running_g_min
        if span < self.eps:
            progress = 1.0
        else:
            progress = (mean_grad - self.running_g_min) / (span + self.eps)
        return float(2.0 * progress - 1.0)

    def update_and_sample(self, grad_norms: torch.Tensor, variation: torch.Tensor) -> Dict[str, float]:
        device = grad_norms.device
        self._ensure_device(device)

        adv_n = self._standardize(grad_norms)
        adv_w = self._standardize(variation)

        beta = self._compute_beta(float(grad_norms.mean().item()))
        effective_adv = beta * adv_n

        logits = torch.log(self.ref_policy.clamp_min(self.eps)) + effective_adv / self.kl_lambda
        self.policy = torch.softmax(logits, dim=0).detach()
        self.ref_policy = ((1.0 - self.ref_ema) * self.ref_policy + self.ref_ema * self.policy).detach()

        truncation_index = int(torch.multinomial(self.policy, 1).item())

        eta_n = torch.sigmoid(adv_w[truncation_index] / self.tau_w)
        low = max(1, self.base_window - self.window_radius)
        high = min(self.trajectory_steps, self.base_window + self.window_radius)
        window_size = int(round(self.base_window - self.window_radius + 2 * self.window_radius * float(eta_n.item())))
        window_size = max(low, min(high, window_size))
        window_size = max(1, min(window_size, truncation_index + 1))

        if self.prev_grad_norms is None:
            rel_change = 1.0
        else:
            prev_g = self.prev_grad_norms[truncation_index].abs() + self.eps
            rel_change = float(variation[truncation_index] / prev_g)
        self.prev_grad_norms = grad_norms.detach().clone()

        self.last_beta = beta
        self.last_advantage_w = adv_w.detach().clone()

        return {
            "truncation_index": truncation_index,
            "window_size": window_size,
            "rel_change": rel_change,
            "beta": beta,
        }


class CacheRefreshLRHA:
    """
    Cache-and-refresh low-rank surrogate for gradient preconditioning.
    This acts as a lightweight approximation layer in place of repeated full updates.
    """

    def __init__(
        self,
        rank: int = 32,
        refresh_delta: float = 0.05,
        refresh_interval: int = 20,
        blend: float = 1.0,
        eps: float = 1e-8,
    ):
        self.rank = max(1, int(rank))
        self.refresh_delta = float(refresh_delta)
        self.refresh_interval = max(1, int(refresh_interval))
        self.blend = float(max(0.0, min(1.0, blend)))
        self.eps = eps

        self.u: Optional[torch.Tensor] = None
        self.s: Optional[torch.Tensor] = None
        self.vh: Optional[torch.Tensor] = None
        self.refresh_count = 0

    def _fit(self, flat_grad: torch.Tensor) -> None:
        max_rank = min(flat_grad.shape[0], flat_grad.shape[1])
        rank = min(self.rank, max_rank)
        if rank <= 0:
            self.u = None
            self.s = None
            self.vh = None
            return
        u, s, vh = torch.linalg.svd(flat_grad, full_matrices=False)
        self.u = u[:, :rank].detach()
        self.s = s[:rank].detach()
        self.vh = vh[:rank, :].detach()
        self.refresh_count += 1

    def _project(self, flat_grad: torch.Tensor) -> torch.Tensor:
        if self.u is None or self.s is None or self.vh is None:
            return flat_grad
        approx = (self.u * self.s.unsqueeze(0)) @ self.vh
        return self.blend * approx + (1.0 - self.blend) * flat_grad

    def apply(self, grad: torch.Tensor, step: int, rel_change: float) -> Tuple[torch.Tensor, bool]:
        original_shape = grad.shape
        if grad.ndim == 1:
            flat_grad = grad.view(1, -1)
        else:
            flat_grad = grad.view(grad.shape[0], -1)

        needs_refresh = (
            self.u is None
            or rel_change >= self.refresh_delta
            or step % self.refresh_interval == 0
        )
        if needs_refresh:
            self._fit(flat_grad.detach())
        projected = self._project(flat_grad)
        return projected.view(original_shape), needs_refresh


class Distill(nn.Module):
    def __init__(
        self,
        x_init,
        y_init,
        arch,
        window,
        lr,
        num_train_eval,
        img_pc,
        batch_pc,
        num_classes=2,
        task_sampler_nc=2,
        train_y=False,
        channel=3,
        im_size=(32, 32),
        inner_optim="SGD",
        syn_intervention=None,
        real_intervention=None,
        cctype=0,
        total_unroll_steps=20,
        window_radius=20,
        selector_kl=1.0,
        selector_ref_ema=0.05,
        selector_tau_w=1.0,
        selector_eps=1e-8,
        use_cr_lrha=True,
        cr_rank=32,
        cr_delta=0.05,
        cr_period=20,
        cr_blend=1.0,
    ):
        super().__init__()
        self.data = nn.Embedding(img_pc * num_classes, int(channel * np.prod(im_size)))
        self.data.weight.data = x_init.float().clone()

        self.train_y = train_y
        if train_y:
            self.label = nn.Embedding(img_pc * num_classes, num_classes)
            self.label.weight.data = y_init.float().clone()
        else:
            self.label = y_init

        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)
        self.img_pc = img_pc
        self.batch_pc = batch_pc
        self.arch = arch
        self.lr = lr
        self.window = max(1, int(window))
        self.total_unroll_steps = max(self.window, int(total_unroll_steps))
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.num_train_eval = num_train_eval
        self.curriculum = self.total_unroll_steps - self.window
        self.inner_optim = inner_optim
        self.batch_id = 0
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention
        self.task_sampler_nc = task_sampler_nc
        self.cctype = cctype

        self.selector = WTPOSelector(
            trajectory_steps=self.total_unroll_steps,
            base_window=self.window,
            window_radius=window_radius,
            kl_lambda=selector_kl,
            ref_ema=selector_ref_ema,
            tau_w=selector_tau_w,
            eps=selector_eps,
        )
        self.use_cr_lrha = bool(use_cr_lrha)
        self.cr_lrha = (
            CacheRefreshLRHA(
                rank=cr_rank,
                refresh_delta=cr_delta,
                refresh_interval=cr_period,
                blend=cr_blend,
                eps=selector_eps,
            )
            if self.use_cr_lrha
            else None
        )
        self.last_rel_change = 1.0
        self.last_truncation_index = self.total_unroll_steps - 1
        self.last_window_size = self.window
        self.last_beta = 1.0
        self.last_cr_refreshed = False
        self.dd_type = "at_bptt_pp"

    def _model_device(self) -> torch.device:
        return self.data.weight.device

    def _build_inner_optimizer(self, network: nn.Module):
        if self.inner_optim == "SGD":
            return optim.SGD(network.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if self.inner_optim == "Adam":
            return optim.Adam(network.parameters(), lr=self.lr)
        raise NotImplementedError(f"Unsupported inner optimizer: {self.inner_optim}")

    def _apply_intervention(self, batch: torch.Tensor, dtype: str) -> torch.Tensor:
        intervention = self.syn_intervention if dtype == "syn" else self.real_intervention
        if intervention is None:
            return batch
        return intervention(batch, dtype=dtype, seed=random.randint(0, 10000))

    def get_task_indices(self):
        task_indices = list(range(self.num_classes))
        if self.task_sampler_nc < self.num_classes:
            random.shuffle(task_indices)
            task_indices = task_indices[: self.task_sampler_nc]
            task_indices.sort()
        return task_indices

    def subsample(self):
        indices = []
        if self.task_sampler_nc == self.num_classes:
            task_indices = list(range(self.num_classes))
        else:
            task_indices = self.get_task_indices()

        for class_idx in task_indices:
            class_indices = torch.randperm(self.img_pc, device=self._model_device())[: self.batch_pc].sort()[0]
            indices.append(class_indices + self.img_pc * class_idx)

        indices = torch.cat(indices)
        imgs = self.data(indices)
        imgs = imgs.view(indices.numel(), self.channel, self.im_size[0], self.im_size[1]).contiguous()

        if self.train_y:
            labels = self.label(indices)
            labels = labels.view(indices.numel(), self.num_classes).contiguous()
        else:
            labels = self.label[indices]
        return imgs, labels

    def _collect_trajectory_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        probe_net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).to(self._model_device())
        probe_net.train()
        probe_optim = self._build_inner_optimizer(probe_net)
        grad_norms = []

        for _ in range(self.total_unroll_steps):
            probe_optim.zero_grad(set_to_none=True)
            imgs, labels = self.subsample()
            imgs = imgs.detach()
            imgs = self._apply_intervention(imgs, dtype="syn")
            out, _ = probe_net(imgs)
            loss = self.criterion(out, labels)
            loss.backward()
            grad_norms.append(_global_grad_norm(probe_net.parameters()))
            probe_optim.step()

        grad_norms = torch.stack(grad_norms).detach()
        variation = torch.zeros_like(grad_norms)
        variation[1:] = (grad_norms[1:] - grad_norms[:-1]).abs()
        return grad_norms, variation

    def _plan_unroll(self) -> Tuple[int, int]:
        if self.dd_type == "at_bptt_pp":
            grad_norms, variation = self._collect_trajectory_stats()
            selector_state = self.selector.update_and_sample(grad_norms, variation)
            self.last_truncation_index = int(selector_state["truncation_index"])
            self.last_window_size = int(selector_state["window_size"])
            self.last_rel_change = float(selector_state["rel_change"])
            self.last_beta = float(selector_state["beta"])
            prefix_steps = max(0, self.last_truncation_index + 1 - self.last_window_size)
            self.curriculum = prefix_steps
            return prefix_steps, self.last_window_size

        if self.dd_type == "curriculum":
            return max(0, int(self.curriculum)), self.window

        if self.dd_type == "standard":
            return 0, self.window

        raise NotImplementedError(f"Unsupported dd_type: {self.dd_type}")

    def _inner_train_step(self, network: nn.Module, optimizer, use_higher: bool = False, diffopt=None) -> None:
        imgs, labels = self.subsample()
        if not use_higher:
            imgs = imgs.detach()
        imgs = self._apply_intervention(imgs, dtype="syn")
        out, _ = network(imgs)
        loss = self.criterion(out, labels)
        if use_higher:
            diffopt.step(loss)
            return
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def forward(self, x):
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).to(self._model_device())
        self.net.train()
        self.optimizer = self._build_inner_optimizer(self.net)

        prefix_steps, diff_steps = self._plan_unroll()
        for _ in range(prefix_steps):
            self._inner_train_step(self.net, self.optimizer)

        with higher.innerloop_ctx(self.net, self.optimizer, copy_initial_weights=True) as (fnet, diffopt):
            for _ in range(diff_steps):
                self._inner_train_step(fnet, self.optimizer, use_higher=True, diffopt=diffopt)
            x = self._apply_intervention(x, dtype="real")
            return fnet(x)

    def init_train(self, epoch, init=False, lim=True):
        del lim
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).to(self._model_device())
            self.optimizer = self._build_inner_optimizer(self.net)
        for _ in range(epoch):
            self._inner_train_step(self.net, self.optimizer)

    def apply_cr_lrha(self, step: int) -> bool:
        if not self.use_cr_lrha or self.cr_lrha is None:
            self.last_cr_refreshed = False
            return False
        if self.data.weight.grad is None:
            self.last_cr_refreshed = False
            return False

        with torch.no_grad():
            projected_grad, refreshed = self.cr_lrha.apply(
                self.data.weight.grad.detach(),
                step=step,
                rel_change=self.last_rel_change,
            )
            self.data.weight.grad.copy_(projected_grad)
            self.last_cr_refreshed = bool(refreshed)
        return self.last_cr_refreshed

    def ema_init(self, ema_coef):
        self.shadow = -1e5
        self.ema_coef = ema_coef

    def ema_update(self, grad_norm):
        if self.shadow == -1e5:
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow

    def test(self, x):
        with torch.no_grad():
            out = self.net(x)
        return out


def random_indices(y, nclass=10, intraclass=False, device="cuda"):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index
