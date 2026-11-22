"""S1 learner for S0b (R12-G; positive 22; coordinator 2026-11-20 04:1x-05:2x; negative 18 addenda 31-34).

S1PPO = rsl_rl 5.4.2 PPO + records + in-run gates; the optimisation is S0's (numbers.LR_FLOOR = rsl_rl's own 1e-5;
the 1e-7 floor of 04:4x was withdrawn at 05:2x). update() below is rsl_rl 5.4.2 PPO.update verbatim except the lines
marked '# S1' (tools/s1ppo_diff.txt is the diff against the installed library).

  records  per update -> loss_dict, written by the rsl_rl logger as tfevents Loss/<key> (learning rate: the existing
           Loss/learning_rate):
             kl_mean, kl_max, kl_mean_after_first, kl_over_share   the 20 minibatch KLs rsl_rl's rule reads
                                                                   (minibatch 0 = before any weight change)
             kl_dim_max_share, kl_dim_argmax                      the per-action-dimension KL summed over the 20
                                                                   minibatches: share and index of the largest
             sigma_00 .. sigma_23, sigma_min, sigma_pair_ratio_max the policy std per joint after the update (order
                                                                   r1lab.contract.ENTITY_JOINTS) and the largest
                                                                   left/right mirror-pair ratio
             sigma_collapse_ratio, sigma_collapse                 min / median std over the 24 joints and the flag
                                                                   ratio < SIGMA_COLLAPSE (negative 18 addendum 37);
                                                                   -1 = not applicable (fixed std, S0b-fixed)
             sigma_max_abs_change                                 largest |change| of the 24 std since the last update
             kl_dim_share_over_half                               1 if one dimension carries > 0.5 of the KL
                                                                   (a '[S1PPO] READOUT NOTE' line at readouts)
             lr_at_floor                                           1 if the rate ended the update at the floor
           stdout (= train.log): one '[S1PPO]' line and one '[S1PPO-sigma]' line (24 std values) per update.
  gates    (in-run, negative 18 addenda 33-34 and 37-38; coordinator 05:2x / 05:4x; the KL streak is recorded only;
           the S0b-fixed arm (FixedGaussian) follows addenda 43-45 instead: S1 and S2 stop at once (M1), the
           50-update mean of the largest one-dimension KL share > 0.5 stops (M2), a failed fixed-std check stops (M0))
           S1: learning rate at the floor for GATE_STREAK consecutive updates (checked again every further
           GATE_STREAK). If the last saved checkpoint is a sigma collapse, STOP at once; otherwise print
           '[S1PPO] GATE S1 READOUT ...' (the readout of addendum 34 decides) and go on. S2: at a readout update
           (1,000, 2,000, then every 2,000) the floor share of the last GATE_WINDOW updates >= GATE_SHARE -> STOP.
           A sigma collapse alone is recorded, not a stop. A STOP prints '[S1PPO] GATE STOP ...'; S1GateStop is raised
           at the start of the NEXT update, so the firing update is logged and (on a save boundary) saved first.
           The update index is Adam's step count / 20.
"""
from __future__ import annotations

import collections

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from r1s1 import numbers as N
from r1s1.distribution import FixedGaussian, sigma_of
from r1wb.symmetry import JOINT_PERM


class S1GateStop(RuntimeError):
    """An in-run gate fired at the previous update (see the module docstring)."""


def is_readout(it: int) -> bool:
    return it in N.READOUTS_FIRST or (it > max(N.READOUTS_FIRST) and it % N.READOUT_EVERY == 0)


class S1PPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._s1 = {"floor_streak": 0, "kl_streak": 0, "window": collections.deque(maxlen=N.GATE_WINDOW),
                    "stop": None, "collapse_at_save": False, "prev_sigma": None}

    def _s1_iteration(self) -> int:
        steps = [int(state["step"]) for state in self.optimizer.state.values() if "step" in state]
        return (max(steps) if steps else 0) // (self.num_learning_epochs * self.num_mini_batches)

    def _s1_record(self, it: int, lr_start: float, kls: list[float], kl_dims: torch.Tensor | None = None
                   ) -> dict[str, float]:
        s = self._s1
        lr_end = float(self.learning_rate)
        at_floor = lr_end <= N.LR_FLOOR * (1.0 + 1e-6)
        over = 2.0 * float(self.desired_kl)
        kl_mean = sum(kls) / len(kls)
        after_first = kls[1:] or kls
        s["floor_streak"] = s["floor_streak"] + 1 if at_floor else 0
        s["kl_streak"] = s["kl_streak"] + 1 if kl_mean > over else 0
        s["window"].append(at_floor)
        sigma = sigma_of(self._raw_actor.distribution).float().cpu()   # any std parametrisation (r1s1.distribution)
        collapse_ratio = float(sigma.min() / sigma.quantile(0.5))   # median with the two middle values averaged
        fixed = isinstance(self._raw_actor.distribution, FixedGaussian)
        fixed_check = None
        if fixed and (s["prev_sigma"] is None or it % N.SAVE_INTERVAL == 0):
          dist = self._raw_actor.distribution
          dev = float((dist.std_param.detach().float() - dist.std_target.float()).abs().max())
          in_opt = any(p is dist.std_param for g in self.optimizer.param_groups for p in g["params"])
          fixed_check = dev <= 1e-7 and not in_opt
          print(f"[S1PPO] FIXED STD CHECK it={it}: max|std - fixed| = {dev:.3g}, std in optimizer groups: {in_opt} -> "
                f"{'ok' if fixed_check else 'FAILED'}", flush=True)
        collapse = (collapse_ratio < N.SIGMA_COLLAPSE) and not fixed    # S0b-fixed: not applicable (coordinator 07:5x)
        change = 0.0 if s["prev_sigma"] is None else float((sigma - s["prev_sigma"]).abs().max())
        s["prev_sigma"] = sigma.clone()
        if it % N.SAVE_INTERVAL == 0:            # the runner saves model_<it> after this update
            s["collapse_at_save"] = collapse
        fired, readout = [], None
        if s["floor_streak"] >= N.GATE_STREAK and s["floor_streak"] % N.GATE_STREAK == 0:
            text = f"S1 learning rate at the floor {N.LR_FLOOR:g} for {s['floor_streak']} consecutive updates"
            if s["collapse_at_save"]:
                fired.append(text + " with a sigma collapse at the last saved checkpoint")
            else:
                readout = text + f"; no sigma collapse at the last saved checkpoint (model_{it - it % N.SAVE_INTERVAL}.pt)"
        if is_readout(it):
            share = sum(s["window"]) / len(s["window"])
            if share >= N.GATE_SHARE:
                fired.append(f"S2 floor share {share:.3f} over the last {len(s['window'])} updates at readout {it}")
        # the KL streak is recorded only (coordinator 05:2x: S1 and S2 are the stop gates)
        record = {"kl_mean": kl_mean, "kl_max": max(kls), "kl_mean_after_first": sum(after_first) / len(after_first),
                  "kl_over_share": sum(k > over for k in kls) / len(kls), "lr_at_floor": float(at_floor)}
        dim_text = ""
        if kl_dims is not None:
            total = float(kl_dims.sum())
            top = int(torch.argmax(kl_dims))
            record["kl_dim_max_share"] = float(kl_dims[top]) / total if total > 0 else 0.0
            record["kl_dim_argmax"] = float(top)
            dim_text = f"kl_dim_max_share={record['kl_dim_max_share']:.3f}@{top} "
        pair = torch.maximum(sigma / sigma[list(JOINT_PERM)], sigma[list(JOINT_PERM)] / sigma)
        record.update({f"sigma_{j:02d}": float(v) for j, v in enumerate(sigma)})
        record.update({"sigma_min": float(sigma.min()), "sigma_pair_ratio_max": float(pair.max()),
                       "sigma_collapse_ratio": collapse_ratio, "sigma_collapse": -1.0 if fixed else float(collapse),
                       "gate_s1_readout": float(readout is not None), "sigma_max_abs_change": change})
        if "kl_dim_max_share" in record:
          record["kl_dim_share_over_half"] = float(record["kl_dim_max_share"] > 0.5)
        if fixed_check is not None:
          record["sigma_fixed_check"] = float(fixed_check)
        print(f"[S1PPO] it={it} lr_start={lr_start:.4e} lr_end={lr_end:.4e} kl_mb0={kls[0]:.3e} "
              f"kl_mean={kl_mean:.3e} kl_max={record['kl_max']:.3e} kl_after_first={record['kl_mean_after_first']:.3e} "
              f"kl_over_share={record['kl_over_share']:.2f} {dim_text}sigma_min={float(sigma.min()):.4g}"
              f"@{int(sigma.argmin())} sigma_pair_ratio_max={float(pair.max()):.3g}@{int(pair.argmax())} "
              f"sigma_collapse_ratio={collapse_ratio:.3g} sigma_collapse={'n/a' if fixed else int(collapse)} "
              f"sigma_max_abs_change={change:.3g} "
              f"floor_streak={s['floor_streak']} kl_streak={s['kl_streak']}", flush=True)
        print(f"[S1PPO-sigma] it={it} " + " ".join(f"{float(v):.4g}" for v in sigma), flush=True)
        if fixed:   # S0b-fixed stop gates (negative 18 addenda 43-45, freezes 78-81): M0, M1 (S1 / S2 stop), M2
            win = s.setdefault("kl_win", collections.deque(maxlen=50))
            win.append(record.get("kl_dim_max_share", 0.0))
            ma = sum(win) / len(win)
            record["kl_dim_max_share_ma50"] = ma
            if len(win) >= 50 and ma > 0.5:
                fired.append(f"M2 one action dimension carries {ma:.3f} of the KL (50-update mean) > 0.5")
            if fixed_check is False:
                fired.append("M0 the std is not fixed (|std - fixed| > 1e-7 or std in the optimizer)")
            if readout is not None:                                  # M1: S1 stops this arm (no progress condition)
                fired.append("M1 " + readout.split(";")[0])
                readout = None
                record["gate_s1_readout"] = 0.0
        if readout is not None:
            print(f"[S1PPO] GATE S1 READOUT it={it}: {readout} - the run goes on", flush=True)
        if (is_readout(it) or it == 500) and record.get("kl_dim_max_share", 0.0) > 0.5:
            print(f"[S1PPO] READOUT NOTE it={it}: one action dimension carries {record['kl_dim_max_share']:.3f} of the "
                  f"update's KL (dimension {int(record['kl_dim_argmax'])})", flush=True)
        if fired:
            s["stop"] = (f"it={it}: " + "; ".join(fired)
                         + f" (last checkpoint model_{it - it % N.SAVE_INTERVAL}.pt or later)")
            print(f"[S1PPO] GATE STOP {s['stop']}", flush=True)
        return record

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        if self._s1["stop"] is not None:  # S1
            raise S1GateStop(self._s1["stop"])  # S1
        s1_it, s1_lr_start, s1_kls, s1_kl_dims = self._s1_iteration(), float(self.learning_rate), [], None  # S1
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    s1_kls.append(float(kl_mean))  # S1
                    s1_om, s1_os = batch.old_distribution_params[0], batch.old_distribution_params[1]  # S1
                    s1_nm, s1_ns = distribution_params[0], distribution_params[1]  # S1
                    s1_d = (torch.log(s1_ns / s1_os) + (s1_os**2 + (s1_om - s1_nm) ** 2) / (2 * s1_ns**2) - 0.5).mean(0)  # S1
                    s1_kl_dims = s1_d if s1_kl_dims is None else s1_kl_dims + s1_d  # S1

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(N.LR_FLOOR, self.learning_rate / 1.5)  # S1: N.LR_FLOOR = 1e-5
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if s1_kls:  # S1
            loss_dict.update(self._s1_record(s1_it, s1_lr_start, s1_kls, s1_kl_dims))  # S1

        # Clear the storage
        self.storage.clear()

        return loss_dict
