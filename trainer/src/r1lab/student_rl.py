"""D6: PPO fine-tune of a distilled 810-input student, anchored to its frozen teacher.

Single concept against the DAgger student (imitation only): the student's imitation error is not priced by
physics. D1 slipped 1.3-1.6x the manufacturer while its teacher slipped .54-.81x, and its upper body turned
1.44-1.84x the manufacturer's rate against the teacher's 1.06-1.41x (same physics, matched speed). PPO on the
teacher task's own rewards (stance slip, landing speed, sway excess, anchor height, arm/body tracking, falls)
lets the student trade imitation error for outcome, while bc_coef * MSE(student mean, teacher mean) on the
student's own rollout states (the DAgger loss) keeps it on the teacher's motion.

- actor: the distilled student (810 -> 24, normalizer frozen as distilled, fixed std = its DAgger acting std)
- critic: the teacher run's critic weights on the teacher scene's privileged group; for actor_warmup_updates
  updates only the value loss is optimized (no actor gradient, so Adam has no stale actor moments)
- teacher: the frozen teacher actor (233 inputs); never in the optimizer, the gradient all-reduce or the
  PPO parameter broadcast (every rank loads it strictly from the same file)
- fixed learning rates per parameter group: rsl_rl's adaptive schedule multiplies the rate by 1.5 per
  minibatch while the KL is below desired_kl / 2, which a near-zero-KL fine-tune hits every minibatch
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from mjlab.rl.config import RslRlBaseRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from r1lab.distillation import COMMAND_KINDS, WALK_KINDS, R1DistillModelCfg, joint_groups, teacher_model_cfg

STUDENT_ACTING_STD = 0.1  # the DAgger acting std of the distilled students (R1DistillModelCfg)


def student_actor_cfg():
    """The distilled student's exact layout; std fixed at its DAgger acting std."""
    return R1DistillModelCfg(distribution_cfg={"class_name": "GaussianDistribution", "init_std": STUDENT_ACTING_STD,
                                               "std_type": "scalar", "learn_std": False})


def teacher_critic_cfg():
    """The matching teacher's critic layout (motion_teacher_rl_cfg: 512-256-128, no normalizer)."""
    return R1DistillModelCfg(hidden_dims=(512, 256, 128), obs_normalization=False, distribution_cfg=None)


def terrain_critic_cfg():
    """A fresh critic for the careful_walk terrain task (its own privileged group). No normalizer: per-rank running
    statistics would make the replicated critics differ (distributed checksum audit), as the teachers' critics avoid."""
    return R1DistillModelCfg(hidden_dims=(512, 256, 128), obs_normalization=False, distribution_cfg=None)


@dataclass
class StudentPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "r1lab.student_rl:StudentPPO"
    schedule: str = "fixed"
    learning_rate: float = 2e-5          # actor, after the critic warm-up
    critic_learning_rate: float = 3e-4
    actor_warmup_updates: int = 40
    bc_coef: float = 1.0
    clip_param: float = 0.1
    entropy_coef: float = 0.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    command_kinds: tuple[str, ...] = COMMAND_KINDS
    transition_window_s: float = 1.0
    acting_std: float = STUDENT_ACTING_STD  # fixed exploration std of the fine-tuned actor


@dataclass
class StudentRLRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    actor: R1DistillModelCfg = field(default_factory=student_actor_cfg)
    critic: R1DistillModelCfg = field(default_factory=teacher_critic_cfg)
    teacher: R1DistillModelCfg = field(default_factory=teacher_model_cfg)
    algorithm: StudentPPOAlgorithmCfg = field(default_factory=StudentPPOAlgorithmCfg)
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("student",), "critic": ("critic",), "teacher": ("actor",)})
    num_steps_per_env: int = 24
    max_iterations: int = 300
    save_interval: int = 25
    experiment_name: str = "r1_careful_walk"
    logger: str = "tensorboard"
    upload_model: bool = False


def model_options(cfg):
    """MLPModel keyword arguments (None-valued options removed)."""
    return {key: value for key, value in cfg.items() if value is not None}


def imitation_report(error, obs, command_kinds, transition_window_s, joint_group_index, distributed):
    """Per-window / per-joint-group MSE of error [m, 24] (same keys as R1Distillation.imitation_metrics)."""
    everything = torch.ones(error.shape[0], dtype=torch.bool, device=error.device)
    windows = {"all": everything}
    if "diag" in obs.keys():
        kind = obs["diag"][:, 0].round().long()
        switch = obs["diag"][:, 1] < transition_window_s
        steady = ~switch
        index = {name: i for i, name in enumerate(command_kinds)}
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
    groups = {"all": torch.arange(error.shape[1], device=error.device), **joint_group_index}
    sums, keys = [], []
    for window, mask in windows.items():
        sums.append(mask.sum().to(error.dtype))
        keys.append(("count", window, None))
        for group, columns in groups.items():
            sums.append(error[mask][:, columns].sum())
            keys.append(("sum", window, group))
    totals = torch.stack(sums).double()
    if distributed:
        torch.distributed.all_reduce(totals)
    counts = {window: float(totals[i]) for i, (kind_, window, _) in enumerate(keys) if kind_ == "count"}
    metrics = {}
    for i, (kind_, window, group) in enumerate(keys):
        if kind_ != "sum" or counts[window] <= 0:
            continue
        mse = float(totals[i]) / (counts[window] * len(groups[group]))
        suffix = group if window == "all" else f"{group}_{window}"
        metrics[f"imitation/mse_{suffix}"] = mse
        metrics[f"imitation/rmse_{suffix}"] = mse ** 0.5
    return metrics


class StudentPPO(PPO):
    teacher: nn.Module

    def __init__(self, actor, critic, storage, *, teacher, bc_coef=1.0, critic_learning_rate=3e-4,
                 actor_warmup_updates=40, command_kinds=COMMAND_KINDS, transition_window_s=1.0,
                 acting_std=STUDENT_ACTING_STD, **kwargs):
        if kwargs.get("schedule") != "fixed":
            raise ValueError("StudentPPO uses fixed per-group learning rates (schedule='fixed')")
        if kwargs.get("symmetry_cfg") or kwargs.get("rnd_cfg"):
            raise ValueError("StudentPPO supports neither symmetry augmentation nor RND")
        if bc_coef < 0 or actor_warmup_updates < 0:
            raise ValueError("bc_coef and actor_warmup_updates must be non-negative")
        super().__init__(actor, critic, storage, **kwargs)
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("StudentPPO expects feed-forward actor and critic")
        self.teacher = teacher.to(self.device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self._raw_teacher = self.teacher
        self.teacher_loaded = False
        self.bc_coef = float(bc_coef)
        if not 0.0 < acting_std <= 1.0:
            raise ValueError("acting_std must be in (0, 1]")
        self.acting_std = float(acting_std)
        self.actor_learning_rate = float(self.learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.actor_warmup_updates = int(actor_warmup_updates)
        self.command_kinds = tuple(command_kinds)
        self.transition_window_s = float(transition_window_s)
        self.joint_group_index = {name: torch.tensor(index, device=self.device)
                                  for name, index in joint_groups().items()}
        self.updates_done = 0
        self.last_metrics = {}
        # Two groups: the actor (fine-tune rate, 0 while warming up) and the critic.
        self.optimizer = resolve_optimizer(kwargs.get("optimizer", "adam"))([
            {"params": list(self.actor.parameters()), "lr": self.actor_learning_rate},
            {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate}])

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        models = {}
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic", "teacher"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        cfg["algorithm"].pop("share_cnn_encoders", None)
        for name, outputs in (("actor", env.num_actions), ("critic", 1), ("teacher", env.num_actions)):
            options = model_options(dict(cfg[name]))
            model_class = resolve_callable(options.pop("class_name"))
            if options.get("distribution_cfg") is not None:
                options["distribution_cfg"] = dict(options["distribution_cfg"])
            models[name] = model_class(obs, cfg["obs_groups"], name, outputs, **options).to(device)
            print(f"{name.capitalize()} Model: {models[name]}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(models["actor"], models["critic"], storage, teacher=models["teacher"], device=device,
                         **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

    # ------------------------------------------------------------------ loading
    def load_initialization(self, student_checkpoint, teacher_checkpoint, map_location=None):
        """actor <- distilled student, critic and teacher <- the teacher run's checkpoint (all strict)."""
        student = torch.load(student_checkpoint, map_location=map_location or self.device, weights_only=False)
        teacher = torch.load(teacher_checkpoint, map_location=map_location or self.device, weights_only=False)
        if "student_state_dict" not in student:
            raise ValueError("The initial student checkpoint has no student_state_dict (not a distillation run)")
        if "actor_state_dict" not in teacher or "critic_state_dict" not in teacher:
            raise ValueError("The teacher checkpoint has no actor/critic state")
        self._raw_actor.load_state_dict(student["student_state_dict"], strict=True)
        # The distilled students act with their fixed DAgger std; keep the PPO std identical (acting_std).
        self._raw_actor.distribution.std_param.data.fill_(self.acting_std)
        self._raw_critic.load_state_dict(teacher["critic_state_dict"], strict=True)
        self._raw_teacher.load_state_dict(teacher["actor_state_dict"], strict=True)
        self.teacher_loaded = True
        return {"student_iteration": int(student.get("iter", -1)), "teacher_iteration": int(teacher.get("iter", -1))}

    def load_self_anchor(self, student_checkpoint, map_location=None):
        """actor and frozen anchor <- the same distilled student (strict); the critic stays freshly initialized."""
        student = torch.load(student_checkpoint, map_location=map_location or self.device, weights_only=False)
        if "student_state_dict" not in student:
            raise ValueError("The initial student checkpoint has no student_state_dict (not a distillation run)")
        self._raw_actor.load_state_dict(student["student_state_dict"], strict=True)
        self._raw_actor.distribution.std_param.data.fill_(self.acting_std)
        self._raw_teacher.load_state_dict(student["student_state_dict"], strict=True)
        self.teacher_loaded = True
        return {"student_iteration": int(student.get("iter", -1))}

    def save(self):
        saved = super().save()
        saved["teacher_state_dict"] = self._raw_teacher.state_dict()
        saved["student_rl_state"] = {"updates_done": self.updates_done}
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        resumed = super().load(loaded_dict, load_cfg, strict)
        if "teacher_state_dict" in loaded_dict:
            self._raw_teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=True)
            self.teacher_loaded = True
        self.updates_done = int(loaded_dict.get("student_rl_state", {}).get("updates_done", self.updates_done))
        return resumed

    def train_mode(self):
        super().train_mode()
        self.teacher.eval()

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def imitation_metrics(self):
        """Deterministic acting student vs teacher on this rollout (before the update)."""
        obs = self.storage.observations.flatten(0, 1)
        error = (self._raw_actor(obs) - self._raw_teacher(obs)).square()
        return imitation_report(error, obs, self.command_kinds, self.transition_window_s,
                                self.joint_group_index, self.is_multi_gpu)

    def update(self):
        if not self.teacher_loaded:
            raise RuntimeError("StudentPPO requires the strictly loaded teacher")
        warm = self.updates_done < self.actor_warmup_updates
        self.optimizer.param_groups[0]["lr"] = 0.0 if warm else self.actor_learning_rate
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate
        metrics = self.imitation_metrics()
        totals = {"value": 0.0, "surrogate": 0.0, "bc": 0.0, "clip_fraction": 0.0, "kl": 0.0}
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                       stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            mean = self.actor.output_mean
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            with torch.no_grad():
                label = self.teacher(batch.observations)
                kl = self.actor.get_kl_divergence(batch.old_distribution_params,
                                                  self.actor.output_distribution_params).mean()
            ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
            advantages = torch.squeeze(batch.advantages)
            surrogate = torch.max(-advantages * ratio,
                                  -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            bc_loss = (mean - label).square().mean()
            loss = self.value_loss_coef * value_loss
            if not warm:
                loss = loss + surrogate + self.bc_coef * bc_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate.item()
            totals["bc"] += bc_loss.item()
            totals["kl"] += kl.item()
            totals["clip_fraction"] += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
        count = self.num_learning_epochs * self.num_mini_batches
        losses = {key: value / count for key, value in totals.items()}
        losses["actor_learning_rate"] = self.optimizer.param_groups[0]["lr"]
        losses["actor_warmup"] = float(warm)
        self.storage.clear()
        self.updates_done += 1
        self.last_metrics = {**losses, **metrics}
        return {**losses, **metrics}

    def reduce_parameters(self):
        """Average actor and critic gradients over ranks (the frozen teacher has none)."""
        params = [p for p in list(self.actor.parameters()) + list(self.critic.parameters()) if p.grad is not None]
        if not params:
            return
        flat = torch.cat([p.grad.view(-1) for p in params])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in params:
            n = p.numel()
            p.grad.data.copy_(flat[offset:offset + n].view_as(p.grad.data))
            offset += n
