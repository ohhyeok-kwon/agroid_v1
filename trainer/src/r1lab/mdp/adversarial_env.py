"""Physical environment-policy bridge; actor observation contract is unchanged.

Protocol 1 costs per transition: [termination indicator, seconds not recovered
following a real disturbance, stable commanded stall time (s) after two seconds without net progress]. Teacher
scores use four matched-program outcomes; legacy mode retains its old score. Scores are heuristics, not a proof of solvability or PAIRED regret.
Only update() trains; reset()/step merely sample, apply physics and collect.
Physical survival is necessary but not sufficient: acceptance also requires
purposeful careful motion, an appropriate emergency departure, and its return.
Raw command error is telemetry, never this module's locomotion objective.
"""
from __future__ import annotations

import math
import torch
from r1lab.mdp.adversary import AdversaryController, challenge_score, decode_actions
from r1lab.mdp.projectiles import ProjectileBurst, ballistic_velocity
from r1lab.mdp.recovery import mark_disturbance, reset_recovery
from r1lab.mdp.progress import commanded_motion, command_changed, STALL_PROTOCOL_VERSION


class LegacyAdversarialScenario:
  _park = ProjectileBurst._park
  _ids = ProjectileBurst._ids

  def __init__(self, cfg, env):
    self.env = env
    self.count = cfg.params.get("count", 8)
    self.clean_fraction = cfg.params.get("clean_fraction", .25)
    if self.count < 3 or not 0 <= self.clean_fraction <= 1:
      raise ValueError("At least three projectile slots and a valid clean fraction required")
    self.controller = AdversaryController(context_dim=8, device=env.device)
    self.robot = env.scene["robot"]
    self.balls = [env.scene[f"ball{i}"] for i in range(self.count)]
    self.body = self.robot.find_bodies("torso_link")[0][0]
    self.force_bodies = []
    for pattern in ("torso_link", "pelvis", ".*wrist_roll_link"):
      self.force_bodies.extend(self.robot.find_bodies(pattern)[0])
    self.force_bodies = list(dict.fromkeys(self.force_bodies)) or [self.body]
    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)
    self.active = torch.zeros(n, self.count, dtype=torch.bool, device=dev)
    self.hit = torch.zeros_like(self.active)
    self.age = torch.zeros(n, self.count, device=dev)
    self.elapsed, self.next_shot, self.force_remaining = z(), z(), z()
    self.launched, self.actual_hits = z(), z()
    self.started = torch.zeros(n, dtype=torch.bool, device=dev)
    self.recovering = torch.zeros_like(self.started)
    self.delivered = torch.zeros_like(self.started)
    self.finished = torch.zeros_like(self.started)
    self.reserved = torch.zeros_like(self.started)
    self.pending_mark = torch.zeros_like(self.started)
    self.stable, self.recovery_time, self.stall_cost, self.return_sum = z(), z(), z(), z()
    self.body_slot = torch.zeros(n, dtype=torch.long, device=dev)
    self.force_vector = torch.zeros(n, 3, device=dev)
    self.force_offset = torch.zeros_like(self.force_vector)
    self.previous_command = torch.zeros(n, 3, device=dev)
    self.command_age = z()
    self.stall_age = z()
    self.tracking_error_integral = z()
    self.progress_xy = torch.zeros(n, 2, device=dev)
    self.progress_yaw = z()
    self.progress_translation, self.progress_rotation = z(), z()
    self.progress_translation_mark, self.progress_rotation_mark = z(), z()
    self.translation_stall_age, self.rotation_stall_age = z(), z()
    self.sample = None
    self.decoded = None
    self.cap = 0
    self.success_ema = torch.full((4, 5), .75, device=dev)
    self.trials = torch.zeros(4, 5, device=dev)
    self.stage_trials = torch.zeros(4, 5, device=dev)
    self.stage_success = torch.zeros(4, 5, device=dev)
    self.last_step = -1
    self.last_observed = -1
    env._r1_adversary = self
    env._r1_projectile_event = self  # existing same-tick reward contact hook
    # Managers and reset poses are not all ready during event construction.

  def _command(self):
    manager = getattr(self.env, "command_manager", None)
    if manager is None:
      return self.previous_command.clone()
    return manager.get_command("twist")[:, :3]

  def _terrain(self, ids, levels):
    terrain = getattr(self.env.scene, "terrain", None)
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
      return
    old = self.env.scene.env_origins[ids].clone()
    target = levels.clamp_max(terrain.terrain_origins.shape[0]-1)
    terrain.terrain_levels[ids] = target
    terrain.update_env_origins(ids, torch.zeros_like(target, dtype=torch.bool),
                                torch.zeros_like(target, dtype=torch.bool))
    # This reset hook follows reset_base. Move the already reset pose with origin.
    delta = self.env.scene.env_origins[ids] - old
    pose = torch.cat((self.robot.data.root_link_pos_w[ids]+delta,
                      self.robot.data.root_link_quat_w[ids]), -1)
    self.robot.write_root_link_pose_to_sim(pose, env_ids=ids)

  def reset(self, env_ids=None):
    ids = self._ids(env_ids)
    if len(ids) == 0:
      return
    command = self._command()
    context = torch.cat((command[ids], self.success_ema.mean(-1).expand(len(ids), -1),
                         torch.full((len(ids), 1), self.cap/4, device=self.env.device)), -1)
    self.reserved[ids] = torch.rand(len(ids), device=self.env.device) < self.clean_fraction
    sampled = self.controller.sample(context, clean_mask=self.reserved[ids],
                                      max_difficulty=self.cap, max_terrain_level=self.cap)
    if self.sample is None:
      self.sample = {k: torch.zeros((self.env.num_envs, *v.shape[1:]), dtype=v.dtype,
                                    device=v.device) for k, v in sampled.items()}
    for key, value in sampled.items():
      self.sample[key][ids] = value
    self.decoded = decode_actions(self.sample["action"])
    self._terrain(ids, self.decoded["terrain_level"][ids])
    for value in (self.active, self.hit, self.started, self.recovering, self.delivered, self.finished, self.pending_mark):
      value[ids] = False
    for value in (self.age, self.elapsed, self.force_remaining, self.launched, self.actual_hits,
                  self.stable, self.recovery_time, self.stall_cost, self.return_sum, self.command_age, self.stall_age, self.tracking_error_integral):
      value[ids] = 0
    reset_recovery(self.env, ids)
    self.next_shot[ids] = self.decoded["delay_s"][ids]
    self.previous_command[ids] = command[ids]
    self._reset_progress(ids, command)
    self.body_slot[ids] = torch.randint(len(self.force_bodies), (len(ids),), device=self.env.device)
    self.force_vector[ids] = 0
    self.force_offset[ids] = 0
    zeros = torch.zeros(len(ids), len(self.force_bodies), 3, device=self.env.device)
    self.robot.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self.force_bodies)
    for i in range(self.count):
      self._park(i, ids)

  def _reset_progress(self, ids, command):
    self.progress_xy[ids] = self.robot.data.root_link_pos_w[ids, :2]
    yaw = self._yaw()[ids]
    self.progress_yaw[ids] = yaw
    for value in (self.progress_translation, self.progress_rotation,
                  self.progress_translation_mark, self.progress_rotation_mark,
                  self.translation_stall_age, self.rotation_stall_age):
      value[ids] = 0

  def _stall_seconds(self, command, changed, stable, recovering, fall):
    """Credit new directed progress, never repeated sway; recovery remains exempt.

    Translation follows the current body-frame command along a curved path.
    Credited cumulative high-water marks survive backtracking. Mixed commands
    retain OR semantics; separate ages expose rotation without translation.
    """
    translation, rotation = commanded_motion(command)
    xy, yaw = self.robot.data.root_link_pos_w[:, :2], self._yaw()
    delta_yaw = yaw-self.progress_yaw
    delta_yaw = torch.atan2(delta_yaw.sin(), delta_yaw.cos())
    middle_yaw = self.progress_yaw+.5*delta_yaw
    direction = command[:, :2]/command[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine, sine = middle_yaw.cos(), middle_yaw.sin()
    direction_w = torch.stack((cosine*direction[:, 0]-sine*direction[:, 1],
                               sine*direction[:, 0]+cosine*direction[:, 1]), -1)
    self.progress_translation += ((xy-self.progress_xy)*direction_w).sum(-1)
    self.progress_rotation += delta_yaw*command[:, 2].sign()
    translated = translation & (self.progress_translation-self.progress_translation_mark >= .02)
    rotated = rotation & (self.progress_rotation-self.progress_rotation_mark >= .06)
    self.progress_translation_mark = torch.where(translated, self.progress_translation,
                                                self.progress_translation_mark)
    self.progress_rotation_mark = torch.where(rotated, self.progress_rotation,
                                             self.progress_rotation_mark)
    self.progress_xy.copy_(xy); self.progress_yaw.copy_(yaw)
    progressed = translated | rotated
    eligible = (translation | rotation) & stable & ~recovering & ~fall
    reset = changed | ~eligible
    self.stall_age = torch.where(reset | progressed, 0., self.stall_age+self.env.step_dt)
    self.translation_stall_age = torch.where(reset | ~translation | translated, 0.,
                                             self.translation_stall_age+self.env.step_dt)
    self.rotation_stall_age = torch.where(reset | ~rotation | rotated, 0.,
                                          self.rotation_stall_age+self.env.step_dt)
    self._reset_progress(reset.nonzero(as_tuple=False).flatten(), command)
    return ((self.stall_age >= 2.-1e-6) & eligible).float()*self.env.step_dt

  def _progress_checkpoint(self):
    return {name:getattr(self, name).clone() for name in
            ('stall_age','stall_cost','tracking_error_integral','progress_xy',
             'progress_yaw','progress_translation','progress_rotation',
             'progress_translation_mark','progress_rotation_mark',
             'translation_stall_age','rotation_stall_age',
             'previous_command','command_age')} | ({'normal_return_time':self.normal_return_time.clone(),'purpose_return_time':self.purpose_return_time.clone()} if hasattr(self,'normal_return_time') else {})

  def _restore_progress(self, state=None):
    if state is not None and 'progress_translation_mark' not in state:
      # Old physical-episode progress has incompatible accounting semantics.
      state = None
    if state is not None:
      for name, value in state.items():
        target = getattr(self, name)
        if target.shape != value.shape: raise ValueError('Progress checkpoint environment count mismatch')
        target.copy_(value)
    else:
      self.stall_age.zero_(); self.stall_cost.zero_(); self.tracking_error_integral.zero_()
      if hasattr(self,'normal_return_time'):
        self.normal_return_time.zero_(); self.purpose_return_time.zero_()
      self._reset_progress(self._ids(), self._command())

  def _mark(self, ids):
    if len(ids):
      self.started[ids] = True
      self.recovering[ids] = True
      self.stable[ids] = 0
      mark_disturbance(self.env, ids)

  def observe_contacts(self):
    if self.last_observed == self.env.common_step_counter:
      return
    self.last_observed = self.env.common_step_counter
    touched = self.pending_mark.clone()
    self.pending_mark[:] = False
    for i in range(self.count):
      force = self.env.scene[f"projectile_hit_{i}"].data.force_history
      contact = (force.norm(dim=-1).amax(dim=(1, 2)) > 1e-3) & self.active[:, i]
      self.actual_hits += (contact & ~self.hit[:, i]).float()
      self.hit[:, i] |= contact
      touched |= contact
    self._mark(touched.nonzero(as_tuple=False).flatten())

  def _yaw(self):
    q = self.robot.data.root_link_quat_w
    return torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                       1-2*(q[:, 2].square()+q[:, 3].square()))

  def _apply(self, terminal):
    env, dt, d = self.env, self.env.step_dt, self.decoded
    kind = d["kind"]
    self.age += self.active*dt
    self.active &= self.age < 5.
    due = (self.elapsed >= self.next_shot-1e-6) & ~terminal
    angle = self._yaw()+d["angle"]
    direction = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), -1)
    # Force duration uses a scaled final tick, preserving the prescribed impulse.
    self.force_remaining = (self.force_remaining-dt).clamp_min(0)
    start_force = due & (kind == 2) & ~self.delivered
    ids = start_force.nonzero(as_tuple=False).flatten()
    if len(ids):
      self.force_remaining[ids] = d["force_duration_s"][ids]
      self.force_vector[ids] = direction[ids]*d["force_n"][ids, None]
      self.force_offset[ids] = torch.stack((-.04*angle[ids].sin(), .04*angle[ids].cos(),
                                           torch.full_like(angle[ids], .05)), -1)
      self.delivered[ids] = True
    force_active = (self.force_remaining > 0) & ~terminal
    forces = torch.zeros(env.num_envs, len(self.force_bodies), 3, device=env.device)
    batch = torch.arange(env.num_envs, device=env.device)
    f = self.force_vector*(self.force_remaining/dt).clamp(0, 1)[:, None]*force_active[:, None]
    forces[batch, self.body_slot] = f
    torques = torch.zeros_like(forces)
    torques[batch, self.body_slot] = torch.cross(self.force_offset, f, dim=-1)
    self.robot.write_external_wrench_to_sim(forces, torques, body_ids=self.force_bodies)
    self.pending_mark |= force_active
    ids = (due & (kind == 3) & ~self.delivered).nonzero(as_tuple=False).flatten()
    if len(ids):
      v = torch.cat((self.robot.data.root_link_lin_vel_w[ids],
                     self.robot.data.root_link_ang_vel_w[ids]), -1).clone()
      v[:, :3] += direction[ids]*d["push_delta_v_mps"][ids, None]
      self.robot.write_root_link_velocity_to_sim(v, env_ids=ids)
      self.delivered[ids] = True
      self._mark(ids)
    ready = due & (kind == 1) & (self.launched < d["ball_shots"]) & (~self.active).any(-1)
    ids = ready.nonzero(as_tuple=False).flatten()
    if len(ids):
      slots = (~self.active[ids]).long().argmax(-1)
      target = self.robot.data.body_com_pos_w[ids, self.body].clone()
      origin = target+2*direction[ids]+target.new_tensor([0, 0, .1])
      velocity = ballistic_velocity(origin, target, d["ball_speed_mps"][ids])
      for i in range(self.count):
        mask = slots == i
        selected = ids[mask]
        if len(selected):
          state = self.balls[i].data.default_root_state[selected].clone()
          state[:, :3] = origin[mask]
          state[:, 7:] = 0
          state[:, 7:10] = velocity[mask]
          self.balls[i].write_root_state_to_sim(state, env_ids=selected)
          self.active[selected, i] = True
          self.hit[selected, i] = False
          self.age[selected, i] = 0
      self.launched[ids] += 1
      self.next_shot[ids] = self.elapsed[ids]+d["ball_interval_s"][ids]
    for i in range(self.count):
      self._park(i, (~self.active[:, i]).nonzero(as_tuple=False).flatten())

  def __call__(self, env, env_ids=None, **params):
    if self.last_step == env.common_step_counter:
      return
    self.last_step = env.common_step_counter
    if self.sample is None:
      self.reset()
    self.observe_contacts()
    dt = env.step_dt
    self.elapsed += dt
    command = self._command()
    changed = command_changed(command, self.previous_command)
    self.command_age = torch.where(changed, 0., self.command_age+dt)
    self.previous_command = command.clone()
    data = self.robot.data
    speed_error = (data.root_link_lin_vel_b[:, :2]-command[:, :2]).norm(dim=-1)
    tilt = data.projected_gravity_b[:, :2].norm(dim=-1)
    good = (tilt < math.sin(math.radians(12))) & (data.root_link_ang_vel_b[:, :2].norm(dim=-1) < 1.)
    self.stable = torch.where(good, self.stable+dt, 0.)
    self.recovering &= self.stable < .5
    unresolved = self.recovering
    fall = env.termination_manager.terminated.bool()
    done = fall | env.termination_manager.time_outs.bool()
    stall_seconds = self._stall_seconds(command, changed, good, unresolved, fall)
    costs = torch.stack(((fall & ~self.finished).float(), unresolved.float()*dt, stall_seconds), -1)
    self.tracking_error_integral += speed_error*dt
    log = env.extras.setdefault("log", {})
    log["Metrics/stable_stall_fraction"] = (stall_seconds > 0).float().mean()
    log["Metrics/translation_stall_fraction"] = (self.translation_stall_age >= 2.-1e-6).float().mean()
    log["Metrics/rotation_stall_fraction"] = (self.rotation_stall_age >= 2.-1e-6).float().mean()
    log["Metrics/raw_tracking_error_mps"] = speed_error.mean()
    self.recovery_time += costs[:, 1]
    self.stall_cost += costs[:, 2]
    self.return_sum += getattr(env, "reward_buf", torch.zeros_like(self.elapsed))
    kind = self.decoded["kind"]
    valid = self._valid_attack(fall)
    success = self._episode_success(fall, kind)
    difficulty = self.sample["action"][:, 1]
    p = self.success_ema[kind, difficulty]
    burden = .5*fall.float()+.25*(self.recovery_time/2).clamp_max(1)+.25*(self.stall_cost/self.elapsed.clamp_min(dt)).clamp_max(1)
    score = self._teacher_score(p, difficulty, burden)
    score = torch.where(valid, score, -torch.ones_like(score))
    completed = done & ~self.finished
    ids = completed.nonzero(as_tuple=False).flatten()
    if len(ids):
      batch = {k:v[ids] for k,v in self.sample.items()}
      # Invalid/missed attacks receive a negative score, not an easy success.
      # Randomized initial episode lengths may time out before any attack was
      # possible. Do not teach the adversary from those censored schedules.
      exposed = ((self.elapsed >= self.decoded["delay_s"]+.5) | self.started | (kind == 0))
      self._observe_completed(ids,batch,score,valid,success,exposed,fall)
      eligible = self._curriculum_eligible(ids,valid)
      flat = kind[eligible]*5+difficulty[eligible]
      self.stage_trials.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
      self.stage_success.view(-1).scatter_add_(0, flat, success[eligible].float())
      self.finished[ids] = True
    env.extras["robust"] = dict(protocol_version=1, costs=costs.detach().clone(), groups=kind.clone(),
        completed=completed.clone(), valid=valid.clone(), success=success.clone(),
        score=score.detach().clone(), episode_return=self.return_sum.clone(),
        recovery_seconds=self.recovery_time.clone(), difficulty=difficulty.clone(),
        stable_stall_seconds=self.stall_cost.clone(), stall_no_progress_seconds=self.stall_age.clone(),
        tracking_error_mps=speed_error.clone(), tracking_error_integral_m=self.tracking_error_integral.clone(),
        cost_protocol_version=3, stall_protocol_version=STALL_PROTOCOL_VERSION,
        reserved_clean=self.reserved.clone(), launched=self.launched.clone(), hits=self.actual_hits.clone())
    self._apply(done)

  def _curriculum_eligible(self,ids,valid):
    return ids[valid[ids] & ~self.reserved[ids]]

  def _observe_completed(self,ids,batch,score,valid,success,exposed,fall):
    self.controller.observe(batch,score[ids],valid=exposed[ids],success=success[ids].float())

  def _teacher_score(self, probability, difficulty, burden):
    return challenge_score(probability, difficulty)+probability*burden

  def _episode_success(self, fall, kind):
    return ~fall & ((kind == 0) | (self.started & ~self.recovering))

  def _valid_attack(self, fall):
    kind = self.decoded["kind"]
    return torch.where(kind == 1, ((self.actual_hits > 0) & fall) |
        ((self.launched >= self.decoded["ball_shots"]) & (self.actual_hits >= self.decoded["ball_shots"])),
        torch.where(kind == 0, torch.ones_like(fall), self.delivered))

  def update(self, reduce_sum=None):
    # Fixed collective order across all ranks, even with no completed episodes.
    with torch.inference_mode(False), torch.enable_grad():
      counts = self.controller._sum(self.stage_trials.clone(), reduce_sum)
      successes = self.controller._sum(self.stage_success.clone(), reduce_sum)
      seen = counts > 0
      rate = successes/counts.clamp_min(1)
      self.success_ema[seen] = .9*self.success_ema[seen]+.1*rate[seen]
      self.trials += counts
      # Require evidence for each physical attack kind before raising the ceiling.
      supported = self.trials[1:, self.cap] >= 8
      if supported.all() and (self.success_ema[1:, self.cap] > .8).all() and self.cap < 4:
        self.cap += 1
      elif supported.all() and (self.success_ema[1:, self.cap] < .2).any() and self.cap > 0:
        self.cap -= 1
      self.stage_trials.zero_()
      self.stage_success.zero_()
      self.controller.pending = [{k:v.clone() for k,v in b.items()} for b in self.controller.pending]
      result = self.controller.update(reduce_sum)
      result.update(difficulty_cap=self.cap, success_ema=self.success_ema.detach().cpu().tolist())
      return result

  def checkpoint(self):
    return dict(protocol_version=1, controller=self.controller.checkpoint(), cap=self.cap,
                success_ema=self.success_ema.clone(), trials=self.trials.clone(), progress=self._progress_checkpoint())

  def restore(self, state, restore_episode_state=False):
    if state["protocol_version"] != 1:
      raise ValueError("Unsupported scenario score protocol")
    self.controller.restore(state["controller"])
    self.cap = int(state["cap"])
    self.success_ema.copy_(state["success_ema"])
    self.trials.copy_(state["trials"])
    self.stage_trials.zero_()
    self.stage_success.zero_()
    self._restore_progress(state.get("progress") if restore_episode_state else None)


from mjlab.managers.event_manager import requires_model_fields
from r1lab.mdp.scene_program import (GenerativeSceneController, decode_program,
                                    MODEL_FIELDS, OBJECTS, EVENTS, program_json, validate_program)


@requires_model_fields(*MODEL_FIELDS)
class AdversarialScenario(LegacyAdversarialScenario):
  """Continuous object/event-program adversary; the four-mode class is legacy.

  Requires scene_entities()['adversary_scene']. Programs can combine multiple
  obstacles, balls, finite forces, velocity jumps and timed material changes.
  All mutable geometry bounds are updated together; no topology changes occur.
  """
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    if self.count != EVENTS: raise ValueError('Generative programs require exactly eight projectile slots')
    self.controller=GenerativeSceneController(context_dim=8,device=env.device)
    self.success_ema=torch.full((5,5),.75,device=env.device)
    self.trials=torch.zeros_like(self.success_ema)
    self.stage_trials=torch.zeros_like(self.success_ema)
    self.stage_success=torch.zeros_like(self.success_ema)
    asset=env.scene['adversary_scene']
    ids=asset.find_geoms('object_.*')[0]
    self.object_ids=asset.indexing.geom_ids[ids].long()
    if len(self.object_ids)!=OBJECTS: raise ValueError('Expected twelve procedural object geoms')
    model=env.sim.mj_model
    floors=[]
    for i in range(model.ngeom):
      body=model.body(int(model.geom_bodyid[i])).name
      if body == 'terrain' or body.startswith('terrain/') or (body == 'world' and int(model.geom_type[i]) == 0):
        floors.append(i)
    if not floors: raise ValueError('Material events require a terrain/floor geom')
    self.floor_ids=torch.tensor(floors,dtype=torch.long,device=env.device)
    self.floor_friction=torch.as_tensor(model.geom_friction[floors].copy(),dtype=torch.float32,device=env.device)
    robot_geoms=[i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')
                 and model.geom(i).name.endswith('_collision')]
    self.material_ids=torch.tensor(floors+robot_geoms,dtype=torch.long,device=env.device)
    baseline=torch.as_tensor(model.geom_friction[floors+robot_geoms].copy(),dtype=torch.float32,device=env.device)
    self.episode_friction=baseline[None,:,:].expand(env.num_envs,-1,-1).clone()
    self.material_dirty=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    # Revalidated by the runner after prior warmup, never inferred from survival alone.
    self.quality_gate=False
    self.normal_return_time=torch.zeros(env.num_envs,device=env.device)
    self.purpose_return_time=torch.zeros_like(self.normal_return_time)
    self.program=None
    self.event_done=torch.zeros(env.num_envs,EVENTS,dtype=torch.bool,device=env.device)
    self.event_applied=torch.zeros_like(self.event_done)
    self.force_end=torch.zeros(env.num_envs,EVENTS,device=env.device)
    self.event_force=torch.zeros(env.num_envs,EVENTS,3,device=env.device)
    self.event_torque=torch.zeros_like(self.event_force)
    self.event_body=torch.zeros(env.num_envs,EVENTS,dtype=torch.long,device=env.device)
    self.ball_event=torch.full((env.num_envs,self.count),-1,dtype=torch.long,device=env.device)
    self.raw_programs=[]
    from r1lab.mdp.scenario_sampling import ReplicaQueue,anchor_bank
    self.replicas=ReplicaQueue(4)
    self.anchor_bank=anchor_bank(env.device)
    self.anchor_fraction=cfg.params.get('anchor_fraction',.35)
    if self.clean_fraction+self.anchor_fraction>1:raise ValueError('Invalid fixed-slice fractions')
    order=torch.arange(env.num_envs,device=env.device)
    # Spread fixed groups across small and large smoke batches alike.
    nclean=round(env.num_envs*self.clean_fraction)
    nanchor=round(env.num_envs*self.anchor_fraction)
    self.slice_id=torch.full_like(order,2)
    self.slice_id[:nclean]=0;self.slice_id[nclean:nclean+nanchor]=1
    self.anchor_group=torch.zeros_like(order)
    self.anchor_group[nclean:nclean+nanchor]=torch.arange(nanchor,device=env.device)%4+1
    self.anchor_counter=order.clone()
    self.replica_id=torch.full_like(order,-1)
    self.commanded_seconds=torch.zeros(env.num_envs,device=env.device)
    self.geometry_opportunity=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    self.geometry_contact=torch.zeros_like(self.geometry_opportunity)
    self.program_valid=torch.ones_like(self.geometry_opportunity)
    self.episode_censored=torch.zeros_like(self.geometry_opportunity)
    self.teacher_censored=torch.zeros_like(self.geometry_opportunity)
    self.replica_start_step=torch.zeros(env.num_envs,device=env.device)
    self.replica_scores=torch.full((env.num_envs,),float('nan'),device=env.device)
    self._configure_rsi(cfg.params)

  def _configure_rsi(self,params):
    """Reserve only existing clean rows; fixed anchors/generator stay unchanged."""
    self.rsi_data_file=params.get('rsi_data_file',params.get('reference_file'))
    self.rsi_clean_fraction=float(params.get('rsi_clean_fraction',params.get('rsi_fraction',.4))) if self.rsi_data_file else 0.
    if not 0 <= self.rsi_clean_fraction <= 1:
      raise ValueError('rsi_fraction must be within [0,1]')
    self.rsi_seed=int(params.get('rsi_seed',0))
    # Environments precede the native runner's process-group initialization.
    from r1lab.distributed import distributed_context
    self.rsi_rank=distributed_context().rank
    self.rsi_episode=torch.zeros_like(self.slice_id,dtype=torch.bool)
    clean=(self.slice_id==0).nonzero(as_tuple=False).flatten()
    self.rsi_episode[clean[:round(len(clean)*self.rsi_clean_fraction)]]=True
    self.rsi_source_frame=torch.full_like(self.slice_id,-1)
    self.rsi_initial_root_state=torch.zeros(self.env.num_envs,13,device=self.env.device)
    self.rsi_initial_command=torch.zeros(self.env.num_envs,3,device=self.env.device)
    self.rsi_initial_velocity_b=torch.zeros_like(self.rsi_initial_command)
    self.rsi_initializer=None
    self._rsi_pending_state=None
    self._rsi_expected_hash=None

  def _apply_rsi(self,ids):
    """EventManager.reset follows observation/action/reward/command resets.

    The caller performs the normal final forward/sense before observations.
    No trajectory is played back and no reference frame enters the actor.
    """
    ids=ids[self.rsi_episode[ids]]
    if not len(ids):return
    if (self.slice_id[ids]!=0).any():raise ValueError('RSI must remain in the clean slice')
    if self.rsi_initializer is None:
      from r1lab.mdp.reference_initialization import ReferenceStateInitializer
      self.rsi_initializer=ReferenceStateInitializer(self.env,self.rsi_data_file,
                                                    seed=self.rsi_seed+100003*self.rsi_rank)
      if (self._rsi_expected_hash is not None and
          self.rsi_initializer.provenance['artifact_sha256']!=self._rsi_expected_hash):
        raise ValueError('RSI checkpoint reference artifact mismatch')
      if self._rsi_pending_state is not None:
        self.rsi_initializer.load_state_dict(self._rsi_pending_state)
        self._rsi_pending_state=None
    self.rsi_initializer.begin_reset(ids)
    result=self.rsi_initializer.apply(ids,at_reset=True)
    self.rsi_source_frame[ids]=result['source_frame']
    self.rsi_initial_root_state[ids]=result['root_state_world']
    self.rsi_initial_command[ids]=self._command()[ids]
    root=result['root_state_world'];qvec=-root[:,4:7];qw=root[:,3:4]
    def inverse_rotate(velocity):
      twice_cross=2*torch.cross(qvec,velocity,dim=-1)
      return velocity+qw*twice_cross+torch.cross(qvec,twice_cross,dim=-1)
    linear=inverse_rotate(root[:,7:10]);angular=inverse_rotate(root[:,10:13])
    self.rsi_initial_velocity_b[ids]=torch.cat((linear[:,:2],angular[:,2:3]),-1)
    # Discard displacement from the ordinary reset pose to the reference seed.
    self._reset_progress(ids,self._command())
    # Derived body transforms may still contain the terminal pose until the
    # caller's final forward(). The initializer returns the actual written seed.
    self.progress_xy[ids]=root[:,:2]
    q=root[:,3:7]
    self.progress_yaw[ids]=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),
                                     1-2*(q[:,2].square()+q[:,3].square()))

  def restore_material(self,env_ids=None):
    """Run BEFORE reset DR; restoring afterwards would overwrite newly sampled DR."""
    ids=self._ids(env_ids)
    ids=ids[self.material_dirty[ids]]
    if len(ids):
      self.env.sim.model.geom_friction[ids[:,None],self.material_ids[None,:]]=self.episode_friction[ids]
      self.material_dirty[ids]=False

  def _write_objects(self,ids,friction_override=None):
    p=self.program;model=self.env.sim.model
    row=ids[:,None];col=self.object_ids[None,:]
    size=p['object_size'][ids]
    pos=p['object_pos'][ids]+self.env.scene.env_origins[ids,None,:]+p['anchor_offset'][ids,None,:]
    pos=pos.clone();pos[~p['object_mask'][ids],2]=-5.
    model.geom_pos[row,col]=pos
    model.geom_quat[row,col]=p['object_quat'][ids]
    model.geom_size[row,col]=size
    friction=torch.zeros_like(size);friction[...,0]=(p['object_friction'][ids] if friction_override is None else friction_override[:,None])
    friction[...,1]=.02;friction[...,2]=.01
    model.geom_friction[row,col]=friction
    radius=size.norm(dim=-1)
    radius[:,8:]=(size[:,8:,0].square()+size[:,8:,1].square()).sqrt()
    model.geom_rbound[row,col]=radius
    half=size.clone();half[:,8:,1]=size[:,8:,0];half[:,8:,2]=size[:,8:,1]
    model.geom_aabb[row,col,0]=0
    model.geom_aabb[row,col,1]=half

  def reset(self,env_ids=None):
    ids=self._ids(env_ids)
    if len(ids)==0:return
    # Manual resets cancel unfinished replicas explicitly; do not leak cohorts.
    abandoned=ids[(self.replica_id[ids]>=0)&~self.finished[ids]]
    for key in self.replica_id[abandoned].tolist():
      self.replicas.complete(key,False,False,True,0.,self.controller.updates)
    self.restore_material(ids)
    self.episode_friction[ids]=self.env.sim.model.geom_friction[ids[:,None],self.material_ids[None,:]].clone()
    command=self._command()
    context=torch.cat((command[ids],self.success_ema.mean(-1).expand(len(ids),-1)),-1)
    self.reserved[ids]=self.slice_id[ids]==0
    sample=self.controller.sample(context,clean_mask=self.reserved[ids],max_difficulty=self.cap)
    anchor_rows=(self.slice_id[ids]==1).nonzero(as_tuple=False).flatten()
    anchor_ids=ids[anchor_rows]
    bank_ids=((self.anchor_counter[anchor_ids]//4)%16)*4+self.anchor_group[anchor_ids]-1
    for key in ('raw','object_mask','event_types'):sample[key][anchor_rows]=self.anchor_bank[key][bank_ids]
    sample['trainable'][anchor_rows]=False
    sample['action'][anchor_rows,0]=self.anchor_group[anchor_ids]
    sample['action'][anchor_rows,1]=4
    self.anchor_counter[anchor_ids]+=4
    shared_rows=[];replica_keys=[]
    rows_meta=zip(ids.tolist(),self.slice_id[ids].tolist())
    for row,(env_id,slice_id) in enumerate(rows_meta):
      chosen={k:v[row:row+1] for k,v in sample.items()}
      key=-1
      if slice_id==2:
        key,chosen=self.replicas.assign(chosen,version=self.controller.updates)
      shared_rows.append(chosen);replica_keys.append(key)
    self.replica_id[ids]=torch.tensor(replica_keys,device=self.env.device)
    sample={k:torch.cat([row[k] for row in shared_rows],0) for k in sample}
    if self.sample is None:
      self.sample={k:torch.zeros((self.env.num_envs,*v.shape[1:]),dtype=v.dtype,device=v.device) for k,v in sample.items()}
    for k,v in sample.items():self.sample[k][ids]=v
    # Freeze each replica's contextual command for its complete episode.
    self._hold_replica_command(ids)
    command=self._command()
    new=decode_program(sample['raw'],sample['object_mask'],sample['event_types'],ceiling=1.)
    for cap in range(5):
      rows=(sample['action'][:,1]==cap)&(self.slice_id[ids]==2)
      if rows.any():
        part=decode_program(sample['raw'][rows],sample['object_mask'][rows],sample['event_types'][rows],ceiling=.15+.85*cap/4)
        for k,v in part.items():
          if isinstance(v,torch.Tensor):new[k][rows]=v
    # All geometry starts in the same robot-yaw frame across replicas.
    q=self.robot.data.root_link_quat_w[ids]
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    c,si=yaw.cos()[:,None],yaw.sin()[:,None]
    x,y=new['object_pos'][...,0].clone(),new['object_pos'][...,1].clone()
    new['object_pos'][...,0]=c*x-si*y;new['object_pos'][...,1]=si*x+c*y
    qw,qx,qy,qz=new['object_quat'].unbind(-1);cy,sy=(yaw/2).cos()[:,None],(yaw/2).sin()[:,None]
    new['object_quat']=torch.stack((cy*qw-sy*qz,cy*qx-sy*qy,cy*qy+sy*qx,cy*qz+sy*qw),-1)
    anchor=self.robot.data.root_link_pos_w[ids]-self.env.scene.env_origins[ids]
    anchor=anchor.clone();anchor[:,2]=0
    new['anchor_offset']=anchor
    if self.program is None:
      self.program={k:(torch.zeros((self.env.num_envs,*v.shape[1:]),dtype=v.dtype,device=v.device) if isinstance(v,torch.Tensor) else v) for k,v in new.items()}
    for k,v in new.items():
      if isinstance(v,torch.Tensor):self.program[k][ids]=v
      else:self.program[k]=v
    self.program_valid[ids]=validate_program(new)
    invalid=ids[~self.program_valid[ids]]
    # Invalid programs become clean physical scenes, not rank-local exceptions.
    self.program['object_mask'][invalid]=False;self.program['event_type'][invalid]=0
    self.sample['trainable'][invalid]=False
    first=torch.where(self.program['event_type']>0,self.program['event_time'],float('inf')).amin(-1)
    first=torch.where(torch.isfinite(first),first,torch.zeros_like(first))
    self.decoded=dict(kind=self.sample['action'][:,0],delay_s=first,
                      ball_shots=(self.program['event_type']==1).sum(-1))
    for value in (self.active,self.hit,self.started,self.recovering,self.delivered,self.finished,self.pending_mark,self.event_done,self.event_applied):value[ids]=False
    for value in (self.age,self.elapsed,self.force_remaining,self.launched,self.actual_hits,self.stable,self.recovery_time,self.stall_cost,self.return_sum,self.command_age,self.stall_age,self.tracking_error_integral,self.force_end,self.event_force,self.event_torque):value[ids]=0
    self.ball_event[ids]=-1
    self.commanded_seconds[ids]=0
    self.geometry_opportunity[ids]=False;self.geometry_contact[ids]=False
    self.episode_censored[ids]=False;self.teacher_censored[ids]=False
    self.replica_scores[ids]=float('nan')
    self.replica_start_step[ids]=self.env.common_step_counter
    self.normal_return_time[ids]=0
    self.purpose_return_time[ids]=0
    self.previous_command[ids]=command[ids]
    self._reset_progress(ids,command)
    reset_recovery(self.env,ids)
    zeros=torch.zeros(len(ids),len(self.force_bodies),3,device=self.env.device)
    self.robot.write_external_wrench_to_sim(zeros,zeros,env_ids=ids,body_ids=self.force_bodies)
    self._write_objects(ids)
    for i in range(self.count):self._park(i,ids)
    self._apply_rsi(ids)

  def _effective_difficulty(self):
    p=self.program;kind=p['event_type']
    strength=torch.where(kind==1,(p['ball_speed']-6)/12,
               torch.where(kind==2,(p['force_n']-20)/40,
               torch.where(kind==3,(p['push_dv']-.1)/.5,torch.zeros_like(p['ball_speed']))))
    attack=(strength+.1)*(kind>0)
    objects=p['object_mask'].float()*self.geometry_contact[:,None]
    delivered=self.event_applied & ((kind!=1)|(self.actual_hits[:,None]>0))
    attack=attack*delivered
    # Credit realized active geometry/events, never the ceiling label alone.
    return 4*((attack.sum(-1)/3 + objects.sum(-1)/12)/2).clamp(0,1)

  def _teacher_score(self,probability,difficulty,burden):
    # Score exists only when all K matched-program replicas have finished.
    return torch.full_like(probability,float('nan'))

  def _hold_replica_command(self,ids):
    ids=ids[self.slice_id[ids]==2]
    manager=self.env.command_manager
    if len(ids)==0 or not hasattr(manager,'get_term'):return
    term=manager.get_term('twist')
    term.vel_command_b[ids]=self.sample['context'][ids,:3]
    term.time_left[ids]=1e9
    for name in ('is_heading_env','is_world_env','is_standing_env','is_forward_env'):
      value=getattr(term,name,None)
      if value is not None:value[ids]=False

  def _curriculum_eligible(self,ids,valid):
    return ids[valid[ids] & (self.slice_id[ids]==2) & ~self.rsi_episode[ids] & ~self.episode_censored[ids] & ~self.teacher_censored[ids]]

  def _observe_completed(self,ids,batch,score,valid,success,exposed,fall):
    # A time limit that truncates the scheduled interaction+return phase is
    # censored. An early fall on fixed anchors still counts as a safety failure.
    planned=self.program['event_type']>0
    last=torch.where(planned,self.program['event_time']+self.program['duration'],0.).amax(-1)
    horizon=float(getattr(self.env,'max_episode_length_s',20.))
    required=torch.maximum(last+3.,torch.full_like(last,horizon-2*self.env.step_dt))
    self.episode_censored[ids]=(~fall[ids])&(self.elapsed[ids]<required[ids])
    self.teacher_censored[ids]=self.episode_censored[ids] | (fall[ids]&~valid[ids])
    severity=(self._effective_difficulty()/4).clamp(0,1)
    generated=ids[(self.slice_id[ids]==2)&~self.rsi_episode[ids]]
    records=zip(generated.tolist(),self.replica_id[generated].tolist(),success[generated].tolist(),
                valid[generated].tolist(),self.teacher_censored[generated].tolist(),severity[generated].tolist())
    for env_id,key,passed,is_valid,censored,burden in records:
      result=self.replicas.complete(key,passed,is_valid,censored,burden,self.controller.updates)
      if result is not None:
        sample,value,success_rate=result
        self.replica_scores[env_id]=value
        self.controller.observe(sample,torch.tensor([value],device=self.env.device),
                                success=torch.tensor([success_rate],device=self.env.device))

  def _episode_success(self,fall,kind):
    # End the disturbance sequence with a sustained normal interval. Report
    # valid moving-only careful-quality evidence separately; missing evidence
    # must neither certify purpose nor erase physical teacher data in warmup.
    complete=(self.event_done | (self.program['event_type']==0)).all(-1)
    complete &= (self.force_end <= self.elapsed[:,None]).all(-1)
    data=self.robot.data
    balanced=(data.projected_gravity_b[:,:2].norm(dim=-1)<math.sin(math.radians(12)))
    balanced &= data.root_link_ang_vel_b[:,:2].norm(dim=-1)<1.
    normal=complete & balanced & ~self.recovering & ~fall
    self.normal_return_time=torch.where(normal,self.normal_return_time+self.env.step_dt,0.)
    quality=getattr(self.env,'_r1_careful_quality',None)
    valid=getattr(self.env,'_r1_careful_quality_valid',None)
    self.purpose_observed=torch.zeros_like(normal)
    purposeful=torch.zeros_like(normal)
    if quality is not None and valid is not None:
      if quality.shape != normal.shape or valid.shape != normal.shape:
        raise ValueError('Careful quality and validity must have one boolean per environment')
      moving=self._command().norm(dim=-1)>.05
      self.purpose_observed=valid.bool() & moving
      purposeful=normal & self.purpose_observed & quality.bool()
    self.purpose_return_time=torch.where(purposeful,self.purpose_return_time+self.env.step_dt,0.)
    # Keep physical teacher data during prior warmup. Purpose validation is
    # separately reported, never fabricated from unavailable/standing evidence.
    return ~fall & ~self.recovering & (self.normal_return_time>=2.-1e-5)

  def _valid_attack(self,fall):
    planned=self.program['event_type']>0
    all_done=(self.event_applied|~planned).all(-1)
    ball_ok=self.actual_hits >= (self.program['event_type']==1).sum(-1)
    applied_nonball=(self.event_applied & (self.program['event_type']!=1)).any(-1)
    exposed=self.started|self.geometry_contact|applied_nonball|(self.actual_hits>0)
    geometry_ok=~self.program['object_mask'].any(-1)|self.geometry_opportunity
    pure_clean=~planned.any(-1)&~self.program['object_mask'].any(-1)
    return self.program_valid & (pure_clean | (all_done&ball_ok&geometry_ok) | (fall&exposed))

  def _apply(self,terminal):
    p=self.program;env=self.env;dt=env.step_dt
    self.age+=self.active*dt;self.active&=self.age<5.
    due=(self.elapsed[:,None]>=p['event_time']-1e-6)&~self.event_done&~terminal[:,None]
    yaw=self._yaw()
    for slot in range(EVENTS):
      kind=p['event_type'][:,slot]
      angle=yaw+p['event_angle'][:,slot]
      direction=torch.stack((angle.cos(),angle.sin(),torch.zeros_like(angle)),-1)
      ids=(due[:,slot]&(kind==1)).nonzero(as_tuple=False).flatten()
      if len(ids):
        # Eight program events and eight ball slots prevent active-slot theft.
        target=self.robot.data.body_com_pos_w[ids,self.body].clone()
        target[:,2]+=p['height_offset'][ids,slot]
        origin=target+2*direction[ids]+target.new_tensor([0.,0.,.1])
        velocity=ballistic_velocity(origin,target,p['ball_speed'][ids,slot])
        state=self.balls[slot].data.default_root_state[ids].clone()
        state[:,:3]=origin;state[:,7:]=0;state[:,7:10]=velocity
        self.balls[slot].write_root_state_to_sim(state,env_ids=ids)
        self.active[ids,slot]=True;self.hit[ids,slot]=False;self.age[ids,slot]=0
        self.launched[ids]+=1;self.ball_event[ids,slot]=slot;self.event_applied[ids,slot]=True
      ids=(due[:,slot]&(kind==2)).nonzero(as_tuple=False).flatten()
      if len(ids):
        self.force_end[ids,slot]=self.elapsed[ids]+p['duration'][ids,slot]
        self.event_force[ids,slot]=direction[ids]*p['force_n'][ids,slot,None]
        offset=torch.stack((-direction[ids,1]*p['point_offset'][ids,slot],direction[ids,0]*p['point_offset'][ids,slot],p['height_offset'][ids,slot]),-1)
        self.event_torque[ids,slot]=torch.cross(offset,self.event_force[ids,slot],dim=-1)
        self.event_body[ids,slot]=p['body_choice'][ids,slot]%len(self.force_bodies)
        self.event_applied[ids,slot]=True;self.delivered[ids]=True
      ids=(due[:,slot]&(kind==3)).nonzero(as_tuple=False).flatten()
      if len(ids):
        v=torch.cat((self.robot.data.root_link_lin_vel_w[ids],self.robot.data.root_link_ang_vel_w[ids]),-1).clone()
        v[:,:3]+=direction[ids]*p['push_dv'][ids,slot,None]
        self.robot.write_root_link_velocity_to_sim(v,env_ids=ids)
        self.event_applied[ids,slot]=True;self.delivered[ids]=True;self._mark(ids)
      ids=(due[:,slot]&(kind==4)).nonzero(as_tuple=False).flatten()
      if len(ids):
        # Surface material changes only, never geometry teleported into the robot.
        self._write_objects(ids,p['material_friction'][ids,slot])
        self.env.sim.model.geom_friction[ids[:,None],self.material_ids[None,:],0]=p['material_friction'][ids,slot,None]
        self.material_dirty[ids]=True
        self.event_applied[ids,slot]=True
      self.event_done[:,slot]|=due[:,slot]
    forces=torch.zeros(env.num_envs,len(self.force_bodies),3,device=env.device);torques=torch.zeros_like(forces)
    batch=torch.arange(env.num_envs,device=env.device)
    for slot in range(EVENTS):
      fraction=((self.force_end[:,slot]-self.elapsed)/dt).clamp(0,1)*~terminal
      forces[batch,self.event_body[:,slot]]+=self.event_force[:,slot]*fraction[:,None]
      torques[batch,self.event_body[:,slot]]+=self.event_torque[:,slot]*fraction[:,None]
      self.pending_mark|=fraction>0
    self.robot.write_external_wrench_to_sim(forces,torques,body_ids=self.force_bodies)
    for i in range(self.count):self._park(i,(~self.active[:,i]).nonzero(as_tuple=False).flatten())

  def __call__(self,env,env_ids=None,**params):
    if self.last_step==env.common_step_counter:return
    if self.sample is None:self.reset()
    self._hold_replica_command(self._ids(None))
    translating, rotating = commanded_motion(self._command())
    self.commanded_seconds+=(translating | rotating)*env.step_dt
    positions=env.sim.model.geom_pos[:,self.object_ids,:2]
    distance=(positions-self.robot.data.root_link_pos_w[:,None,:2]).norm(dim=-1)
    self.geometry_opportunity|=((distance<.75)&self.program['object_mask']).any(-1)
    try:
      contact=env.scene['adversary_object_contact'].data.found
      self.geometry_contact|=contact.reshape(env.num_envs,-1).gt(0).any(-1)
    except KeyError:pass
    super().__call__(env,env_ids,**params)
    robust=env.extras['robust'];ids=robust['completed'].nonzero(as_tuple=False).flatten()
    data=getattr(env.sim,'data',None)
    robust['physics_overflow']=getattr(data,'overflow',torch.zeros_like(self.slice_id)).clone()
    robust['constraint_rows']=getattr(data,'nefc',torch.zeros_like(self.slice_id)).clone()
    robust['calibration_mask']=((self.slice_id!=2)&~self.rsi_episode).clone()
    robust['rsi_episode']=self.rsi_episode.clone()
    robust['rsi_source_frame']=self.rsi_source_frame.clone()
    robust['rsi_initial_root_state']=self.rsi_initial_root_state.clone()
    robust['rsi_initial_command']=self.rsi_initial_command.clone()
    robust['rsi_initial_velocity_b']=self.rsi_initial_velocity_b.clone()
    robust['rsi_initial_linear_mismatch_mps']=(self.rsi_initial_velocity_b[:,:2]-self.rsi_initial_command[:,:2]).norm(dim=-1)
    robust['rsi_initial_yaw_mismatch_rad_s']=(self.rsi_initial_velocity_b[:,2]-self.rsi_initial_command[:,2]).abs()
    robust['rsi_protocol_version']=1
    robust['rsi_sampler_seed']=self.rsi_seed+100003*self.rsi_rank
    robust['rsi_sampler_rank']=self.rsi_rank
    if self.rsi_data_file:
      mask=self.rsi_episode;count=mask.sum().clamp_min(1)
      env.extras['log']['RSI/fraction']=mask.float().mean()
      env.extras['log']['RSI/completed']=(mask&robust['completed']).sum()
      env.extras['log']['RSI/falls']=(mask&robust['completed']&env.termination_manager.terminated.bool()).sum()
      env.extras['log']['RSI/initial_linear_mismatch_mps']=(robust['rsi_initial_linear_mismatch_mps']*mask).sum()/count
      env.extras['log']['RSI/initial_yaw_mismatch_rad_s']=(robust['rsi_initial_yaw_mismatch_rad_s']*mask).sum()/count
    robust['slice_id']=self.slice_id.clone()
    robust['anchor_group']=self.anchor_group.clone()
    robust['episode_censored']=self.episode_censored.clone()
    robust['teacher_censored']=self.teacher_censored.clone()
    robust['episode_fall']=env.termination_manager.terminated.bool().clone()
    robust['episode_recovery_seconds']=self.recovery_time.clone()
    robust['episode_stall_seconds']=self.stall_cost.clone()
    robust['episode_commanded_seconds']=self.commanded_seconds.clone()
    robust['planned_exposure']=(self.program['event_type']>0).sum(-1).clamp_min(1)
    robust['geometry_opportunity']=self.geometry_opportunity.clone()
    robust['geometry_contact']=self.geometry_contact.clone()
    robust['replica_score']=self.replica_scores.clone()
    robust['teacher_version_age']=self.controller.updates-self.sample['version']
    robust['replica_step_span']=env.common_step_counter-self.replica_start_step
    robust['frontier_estimator_assumption']='IID only under fixed actor; evolving actor, independent DR/noise'
    robust['score']=self.replica_scores.clone()
    robust['program_env_ids']=ids.clone()
    robust['program']={k:(v[ids].clone() if isinstance(v,torch.Tensor) else v) for k,v in self.program.items()}
    robust['protocol_version']=2
    robust['cost_names']=('fall_event','unrecovered_seconds','stable_stall_seconds')
    robust['failure']=env.termination_manager.terminated.bool().clone()
    robust['normal_return_seconds']=self.normal_return_time.clone()
    robust['physical_success']=robust['success'].clone()
    robust['purpose_observed']=self.purpose_observed.clone()
    robust['purpose_return_seconds']=self.purpose_return_time.clone()
    robust['purposeful_success']=robust['success'] & (self.purpose_return_time>=2.-1e-5)
    robust['purpose_evidence']='learned proxy, not final perceptual validation'
    completed_program=(self.event_done | (self.program['event_type']==0)).all(-1)
    robust['phase']=torch.where(completed_program & ~self.recovering,2,
                       torch.where(self.event_done.any(-1) | self.recovering,1,0))
    self.restore_material(robust['completed'].nonzero(as_tuple=False).flatten())
    robust['event_applied']=self.event_applied.clone()
    robust['effective_difficulty']=self._effective_difficulty().clone()
    robust['pure_clean']=(~self.program['object_mask'].any(-1)) & (~(self.program['event_type']>0).any(-1))

  def update(self,reduce_sum=None):
    with torch.inference_mode(False),torch.enable_grad():
      counts=self.controller._sum(self.stage_trials.clone(),reduce_sum)
      successes=self.controller._sum(self.stage_success.clone(),reduce_sum)
      seen=counts>0;rate=successes/counts.clamp_min(1)
      self.success_ema[seen]=.9*self.success_ema[seen]+.1*rate[seen]
      self.trials+=counts
      supported=self.trials[:,self.cap].sum()>=32
      weights=counts[:,self.cap];total=weights.sum()
      p=(rate[:,self.cap]*weights).sum()/total.clamp_min(1)
      gate=self.controller._sum(torch.tensor([float(bool(self.quality_gate)),1.],device=self.env.device),reduce_sum)
      approved=bool(gate[0] == gate[1])
      if approved and supported and total>=8 and p>.8 and self.cap<4:self.cap+=1
      elif supported and total>=8 and p<.2 and self.cap>0:self.cap-=1
      self.stage_trials.zero_();self.stage_success.zero_()
      self.controller.pending=[{k:v.clone() for k,v in b.items()} for b in self.controller.pending]
      result=self.controller.update(reduce_sum)
      result.update(difficulty_cap=self.cap,quality_gate=approved,success_ema=self.success_ema.cpu().tolist())
      return result

  def replay_program(self,env_id,record):
    """Install validated decoded JSON; caller restores the recorded robot seed.

    Replay contains exact geometry and events, but reproducing randomized robot
    dynamics/initial pose also requires the caller's episode seed/state record.
    """
    candidate={}
    for key,value in self.program.items():
      if isinstance(value,torch.Tensor):
        restored=torch.as_tensor(record[key],dtype=value.dtype,device=value.device)
        if restored.shape != value[env_id].shape or not torch.isfinite(restored).all():
          raise ValueError('Malformed replay field: '+key)
        candidate[key]=restored.unsqueeze(0)
      else:candidate[key]=record[key]
    if not validate_program(candidate).all():raise ValueError('Replay scene failed physical envelope validation')
    ids=torch.tensor([env_id],device=self.env.device,dtype=torch.long)
    self.reset(ids)
    for key,value in candidate.items():
      if isinstance(value,torch.Tensor):self.program[key][env_id]=value[0]
    self.sample['trainable'][env_id]=False
    types=self.program['event_type'][env_id]
    present=torch.stack([(types==i).any() for i in range(1,5)])
    group=4 if present.sum()>1 else int(present.long().argmax())+1 if present.any() else 0
    if self.program['object_mask'][env_id].any():group=4
    self.sample['action'][env_id,0]=group
    self.decoded['ball_shots'][env_id]=(types==1).sum()
    times=self.program['event_time'][env_id][types>0]
    self.decoded['delay_s'][env_id]=times.min() if len(times) else 0
    self._write_objects(ids)

  def checkpoint(self):
    return dict(protocol_version=2,controller=self.controller.checkpoint(),cap=self.cap,quality_gate=bool(self.quality_gate),
                success_ema=self.success_ema.clone(),trials=self.trials.clone(),progress=self._progress_checkpoint(),
                anchor_counter=self.anchor_counter.clone(),sampler_protocol=1,
                restore_metadata=getattr(self,'restore_metadata',None),
                rsi_config={'data_file':self.rsi_data_file,'clean_fraction':self.rsi_clean_fraction,
                            'seed':self.rsi_seed,'rank':self.rsi_rank,
                            'effective_seed':self.rsi_seed+100003*self.rsi_rank},
                rsi_initializer_rank=self.rsi_rank,
                rsi_initializer=self.rsi_initializer.state_dict() if self.rsi_initializer is not None else self._rsi_pending_state)

  def restore(self,state,restore_episode_state=False):
    if state['protocol_version']!=2 or not 0<=int(state['cap'])<=4:
      raise ValueError('Unsupported generative scenario checkpoint')
    self.controller.restore(state['controller']);self.cap=int(state['cap'])
    self.quality_gate=False  # fixed-anchor quality must be revalidated after a load
    from r1lab.mdp.scenario_sampling import ReplicaQueue
    self.replicas=ReplicaQueue(4)  # no physical world restoration: discard incomplete cohorts
    saved_counter=state.get('anchor_counter')
    self.restore_metadata={'anchor_counter':'initialized_no_saved_counter'}
    if saved_counter is not None and saved_counter.shape==self.anchor_counter.shape:
      self.anchor_counter.copy_(saved_counter)
      self.restore_metadata={'anchor_counter':'restored_same_env_count'}
    elif saved_counter is not None:
      self.anchor_counter.copy_(torch.arange(self.env.num_envs,device=self.env.device))
      self.restore_metadata={'anchor_counter':'reset_env_count_changed',
          'saved_env_count':int(saved_counter.numel()),'current_env_count':self.env.num_envs}
    self.success_ema.copy_(state['success_ema']);self.trials.copy_(state['trials'])
    self.stage_trials.zero_();self.stage_success.zero_()
    self._restore_progress(state.get("progress") if restore_episode_state else None)
    saved_rsi=state.get('rsi_initializer')
    self._rsi_expected_hash=saved_rsi.get('artifact_sha256') if saved_rsi is not None else None
    same_rsi_rank=int(state.get('rsi_initializer_rank',0))==self.rsi_rank
    if saved_rsi is not None and self.rsi_data_file and same_rsi_rank:
      if self.rsi_initializer is None:self._rsi_pending_state=saved_rsi
      else:self.rsi_initializer.load_state_dict(saved_rsi)
      self.restore_metadata['rsi']='restored_same_rank_sampler'
    elif saved_rsi is not None and self.rsi_data_file:
      self._rsi_pending_state=None
      self.rsi_initializer=None
      self.restore_metadata['rsi']='reset_sampler_for_different_rank'
    else:
      self.restore_metadata['rsi']='initialized_new_sampler' if self.rsi_data_file else 'disabled'
    # Physical episodes are discarded on resume, just like replica cohorts.
    self.rsi_source_frame.fill_(-1)
    self.rsi_initial_root_state.zero_()
    self.rsi_initial_command.zero_();self.rsi_initial_velocity_b.zero_()


def restore_adversary_material(env, env_ids=None):
  """First reset event, before foot-friction DR; safe before scenario creation."""
  scenario=getattr(env,'_r1_adversary',None)
  if scenario is not None and hasattr(scenario,'restore_material'):
    scenario.restore_material(env_ids)
