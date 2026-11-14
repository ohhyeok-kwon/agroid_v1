"""DAgger student distillation on rsl_rl 5.4.2 (Distillation / DistillationRunner).

The student acts (beta = 0, fixed std for state diversity) and the frozen teacher's mean action on
the same observation is the label. Changes against the native algorithm:
- The student's observation normalizer is updated once per update with the moments of the whole
  rollout on every rank (all_reduce), not per step and per rank, so multi-GPU checksums agree.
- Each update first records the acting student's imitation error on its own rollout by joint group,
  command kind and command-change window (from the "diag" observation group).
- R1DistillationRunner strips None model options for student/teacher (mjlab only does it for
  actor/critic) and loads the teacher strictly without restoring the teacher run's step counter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from mjlab.rl.config import RslRlBaseRunnerCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.algorithms import Distillation
from rsl_rl.runners import DistillationRunner

from r1lab import contract as C

COMMAND_KINDS = ("stand", "forward", "forward_turn", "heading", "backward")
WALK_KINDS = ("forward", "forward_turn", "backward")


def joint_groups():
    """POLICY-order joint indices: legs (hips, knees, ankles), waist, arms."""
    groups = {"legs": [], "waist": [], "arms": []}
    for index, name in enumerate(C.POLICY_JOINTS):
        key = ("waist" if name.startswith("waist") else
               "arms" if any(part in name for part in ("shoulder", "elbow", "wrist")) else "legs")
        groups[key].append(index)
    return groups


# ----------------------------------------------------------------------------- configs
@dataclass
class R1DistillModelCfg:
    class_name: str = "MLPModel"
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict | None = field(default_factory=lambda: {
        "class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar", "learn_std": False})


def teacher_model_cfg():
    """Matching teacher actor layout: mlp.{0,2,4,6} + distribution.std_param, no normalizer."""
    return R1DistillModelCfg(hidden_dims=(512, 256, 128), obs_normalization=False,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                               "std_type": "scalar"})


@dataclass
class R1DistillationAlgorithmCfg:
    class_name: str = "r1lab.distillation:R1Distillation"
    num_learning_epochs: int = 3
    gradient_length: int = 4
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"
    command_kinds: tuple[str, ...] = COMMAND_KINDS
    transition_window_s: float = 1.0
    # "constant" (D1) or "cosine": learning_rate -> lr_final_ratio * learning_rate over lr_total_updates
    # (D1 plateaued at walk MSE ~.195 from update ~80 with a fixed 1e-3).
    lr_schedule: str = "constant"
    lr_final_ratio: float = 0.1
    lr_total_updates: int = 600


@dataclass
class R1DistillationRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "r1lab.distillation:R1DistillationRunner"
    student: R1DistillModelCfg = field(default_factory=R1DistillModelCfg)
    teacher: R1DistillModelCfg = field(default_factory=teacher_model_cfg)
    algorithm: R1DistillationAlgorithmCfg = field(default_factory=R1DistillationAlgorithmCfg)
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"student": ("student",), "teacher": ("actor",)})
    num_steps_per_env: int = 24
    max_iterations: int = 600
    save_interval: int = 50
    experiment_name: str = "r1_careful_walk"
    logger: str = "tensorboard"
    upload_model: bool = False


def strip_none_options(train_cfg):
    """Drop None-valued model options of student/teacher (MLPModel rejects them)."""
    for key in ("student", "teacher"):
        if isinstance(train_cfg.get(key), dict):
            for option in [k for k, v in train_cfg[key].items() if v is None]:
                train_cfg[key].pop(option)
    return train_cfg


# ----------------------------------------------------------------------------- normalizer
def moment_sums(x):
    """[count, sum, sum of squares] of x [m, d] in float64 (summable across ranks)."""
    x = x.detach().double()
    return torch.cat((x.new_tensor([x.shape[0]]), x.sum(0), x.square().sum(0)))


def moments_from_sums(sums):
    dimension = (sums.numel() - 1) // 2
    count = sums[0]
    mean = sums[1:1 + dimension] / count
    var = (sums[1 + dimension:] / count - mean.square()).clamp_min(0.0)
    return count, mean, var


def global_moments(x, distributed=False):
    """Population count/mean/variance of x [m, d] over every sample on every rank."""
    sums = moment_sums(x)
    if distributed:
        torch.distributed.all_reduce(sums)
    return moments_from_sums(sums)


def merge_moments(normalizer, count, mean, var):
    """EmpiricalNormalization.update with given batch moments (same running-moment formula)."""
    if normalizer.until is not None and normalizer.count >= normalizer.until:
        return False
    total = normalizer.count.double() + count
    rate = count / total
    old_mean = normalizer._mean.double()[0]
    old_var = normalizer._var.double()[0]
    delta = mean - old_mean
    new_mean = old_mean + rate * delta
    new_var = old_var + rate * (var - old_var + delta * (mean - new_mean))
    normalizer._mean.copy_(new_mean[None].to(normalizer._mean.dtype))
    normalizer._var.copy_(new_var[None].to(normalizer._var.dtype))
    normalizer._std.copy_(new_var.sqrt()[None].to(normalizer._std.dtype))
    normalizer.count.fill_(int(round(float(total))))
    return True


# ----------------------------------------------------------------------------- algorithm
class R1Distillation(Distillation):
    def __init__(self, student, teacher, storage, *, command_kinds=COMMAND_KINDS, transition_window_s=1.0,
                 lr_schedule="constant", lr_final_ratio=0.1, lr_total_updates=600, **kwargs):
        super().__init__(student, teacher, storage, **kwargs)
        if lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if not 0.0 < lr_final_ratio <= 1.0 or lr_total_updates < 1:
            raise ValueError("lr_final_ratio must be in (0, 1] and lr_total_updates >= 1")
        self.lr_schedule, self.lr_final_ratio, self.lr_total_updates = lr_schedule, float(lr_final_ratio), int(lr_total_updates)
        self.base_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.updates_done = 0
        self.command_kinds = tuple(command_kinds)
        self.transition_window_s = float(transition_window_s)
        self.joint_group_index = {name: torch.tensor(index, device=self.device)
                                  for name, index in joint_groups().items()}
        self.last_metrics = {}

    def process_env_step(self, obs, rewards, dones, extras):
        """Native bookkeeping without the per-step, per-rank normalizer update."""
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def scheduled_learning_rate(self):
        """Deterministic per update (identical on every rank)."""
        if self.lr_schedule == "constant":
            return self.base_learning_rate
        t = min(self.updates_done / max(self.lr_total_updates - 1, 1), 1.0)
        return self.base_learning_rate * (self.lr_final_ratio + (1.0 - self.lr_final_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))

    def update(self):
        rate = self.scheduled_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        metrics = self.imitation_metrics()
        metrics["imitation/learning_rate"] = rate
        self.update_normalizer()
        losses = super().update()
        self.updates_done += 1
        self.last_metrics = {**{key: float(value) for key, value in losses.items()}, **metrics}
        return {**losses, **metrics}

    def update_normalizer(self):
        student = self._raw_student
        if not student.obs_normalization:
            return None
        rollout = self.storage.observations[: self.storage.step]
        x = torch.cat([rollout[group] for group in student.obs_groups], dim=-1).flatten(0, 1)
        count, mean, var = global_moments(x, distributed=self.is_multi_gpu)
        merge_moments(student.obs_normalizer, count, mean, var)
        return count

    @torch.no_grad()
    def imitation_metrics(self):
        """Deterministic acting student vs teacher labels on this rollout (before the update)."""
        steps = self.storage.step
        obs = self.storage.observations[:steps].flatten(0, 1)
        labels = self.storage.privileged_actions[:steps].flatten(0, 1)
        error = (self._raw_student(obs) - labels).square()
        everything = torch.ones(error.shape[0], dtype=torch.bool, device=error.device)
        windows = {"all": everything}
        if "diag" in obs.keys():
            kind = obs["diag"][:, 0].round().long()
            switch = obs["diag"][:, 1] < self.transition_window_s
            steady = ~switch
            names = self.command_kinds
            index = {name: i for i, name in enumerate(names)}
            walk = torch.zeros_like(everything)
            for name in WALK_KINDS:
                if name in index:
                    walk |= kind == index[name]
            windows.update(switch=switch, steady=steady, walk=walk & steady)
            if "stand" in index:
                windows["stand"] = (kind == index["stand"]) & steady
            if "heading" in index:
                windows["turn"] = (kind == index["heading"]) & steady
            for name, i in index.items():
                windows["kind_" + name] = kind == i
        groups = {"all": torch.arange(error.shape[1], device=error.device), **self.joint_group_index}
        sums, keys = [], []
        for window, mask in windows.items():
            sums.append(mask.sum().to(error.dtype))
            keys.append(("count", window, None))
            for group, columns in groups.items():
                sums.append(error[mask][:, columns].sum())
                keys.append(("sum", window, group))
        totals = torch.stack(sums).double()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(totals)
        counts = {window: float(totals[i]) for i, (kind_, window, _) in enumerate(keys) if kind_ == "count"}
        metrics = {}
        for i, (kind_, window, group) in enumerate(keys):
            if kind_ != "sum" or counts[window] <= 0:
                continue
            mse = float(totals[i]) / (counts[window] * len(groups[group]))
            metrics[f"imitation/mse_{group}" if window == "all" else f"imitation/mse_{group}_{window}"] = mse
            metrics[f"imitation/rmse_{group}" if window == "all" else f"imitation/rmse_{group}_{window}"] = mse ** 0.5
        total = counts["all"]
        for window, count in counts.items():
            if window != "all":
                metrics[f"imitation/share_{window}"] = count / max(total, 1.0)
        if "imitation/rmse_all_switch" in metrics and metrics.get("imitation/rmse_all_steady", 0.0) > 0:
            metrics["imitation/switch_steady_ratio"] = metrics["imitation/rmse_all_switch"] / metrics[
                "imitation/rmse_all_steady"]
        return metrics


# ----------------------------------------------------------------------------- runner
class R1DistillationRunner(DistillationRunner, MjlabOnPolicyRunner):
    """DistillationRunner with mjlab checkpoints (env step counter, upload gate)."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, strip_none_options(train_cfg), log_dir, device)

    def load_teacher(self, path, map_location=None):
        """Strict teacher-only load from an RL checkpoint's actor_state_dict.

        runner.load would also restore the teacher run's common_step_counter; this does not.
        """
        loaded = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if "actor_state_dict" not in loaded:
            raise ValueError("Teacher checkpoint has no actor_state_dict")
        self.alg.load(loaded, {"teacher": True, "iteration": False}, strict=True)
        if not self.alg.teacher_loaded:
            raise RuntimeError("Teacher parameters were not loaded")
        return {"iteration": int(loaded.get("iter", -1))}
