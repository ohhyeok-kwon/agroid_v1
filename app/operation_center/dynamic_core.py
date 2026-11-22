"""DynamicMode 의 핵심: 평가에서 재는 것과 같은 전신 스택을 mjlab 장면 하나에서 돌린다(로봇 수는 인자).

  지시(목표 5값: 전후·좌우 속도, 회전 속도, 골반 높이, 높이 사용 여부)
    -> 조향 고리(r1wb.s2.steer, 학습된 느린 고리) -> 운동 토큰 머리(r1wb.s2.small) -> 토큰기 복원
    -> 참조 프레임(r1wb.s2.runtime.S2View) -> 추종 정책 System 1(관측은 r1wb.tracking.env_cfg.teacher_observation)

걸음마다의 순서는 평가 도구(r1wb.eval.eval_stream --policy s2)와 같다: view.update -> 관측 -> 정책 -> 물리 한 걸음 ->
고리의 진행 기록. 달라지는 것은 장면뿐이다: 넘어져도 자동으로 다시 세우지 않고(넘어짐을 표시만 한다), 앱의 지형 물체와
공(1 kg, 반지름 7 cm; 날아가는 동안 무게 없음 = 평가에서 잰 던지기와 같은 물체)을 둔다. 계획의 높이는 선택 파일의
ground_reference 대로 발밑 지면 기준으로 읽힌다(r1wb.world.ground_reference).
Simulation 작업 프로세스(dynamic_worker, 로봇 1대)와 비교 시험(.agent/tests/operation_center, 로봇 여러 대)이 같이 쓴다.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
    wp.config.ptx_target_arch = 90
    wp.config.cuda_output = 'ptx'

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

BALL_LIFETIME = 5.0            # s, 앱의 공 수명(simulation.PROJECTILE_SETTINGS)
BALLS = 8
FLOOR = 60.0                   # m, 지형 물체가 놓이는 바닥 한 변(출발점이 한가운데)
STAND = (0.0, 0.0, 0.0, 0.731, 0.0)


@dataclass(kw_only=True)
class AppTerrainCfg(SubTerrainCfg):
    """앱의 지형 물체(simulation_worker.terrain: 큰 돌 = 타원체, 계단 = 상자)를 평평한 바닥 위에 같은 자리·크기로 둔다."""
    kind: str = 'rocks'

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        from simulation_worker import terrain
        body = spec.body('terrain')
        centre = (self.size[0] / 2, self.size[1] / 2)
        geometries = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=(0.5, 0.5, 0.5, 1.0))]
        for item in terrain(self.kind):
            x, y, z = item['position']
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX if item['shape'] == 'box' else mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=item['size'], pos=[centre[0] + x, centre[1] + y, z],
                quat=[math.cos(item['yaw'] / 2), 0.0, 0.0, math.sin(item['yaw'] / 2)])
            geometries.append(TerrainGeometry(geom=geom, color=(0.44, 0.51, 0.6, 1.0) if item['shape'] == 'box' else (0.45, 0.48, 0.52, 1.0)))
        return TerrainOutput(origin=np.array([centre[0], centre[1], 0.0]), geometries=geometries)


class DynamicCore:
    def __init__(self, selection: dict, environment: str = 'flat', envs: int = 1, device: str = 'cuda:0',
                 camera: bool = False, system1: str | None = None):
        """selection: dynamic_mode.load_manifest 의 결과. system1: 체크포인트 경로를 바꿔 끼울 때(비교 시험)."""
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        from r1wb import world
        from r1wb.s2.runtime import S2View, goal_brain
        from r1wb.s2.small import SmallS2
        from r1wb.s2.steer import Steer, SteerPolicy, steered_goal
        from r1wb.s2.tokenizer import MotionTokenizer
        from r1wb.student.observations import load_teacher
        from r1wb.student.stream import stream_env_cfg

        self.selection, self.environment, self.count, self.device = selection, environment, envs, device
        checkpoint = str(Path(system1 or selection['system1']['path']).resolve())
        cfg = stream_env_cfg(play=True, clean=True, matched=True, teacher=checkpoint)
        motion = cfg.commands['motion']
        # as eval_stream --policy s2 --match-pose reference --commit --library <library>
        motion.match_pose, motion.match_commit, motion.match_height, motion.match_stop = 'reference', True, 'stream', 'room'
        motion.motion_file = selection['library']['path']
        stand = motion.clip_list
        motion.clip_list = tuple(stand[i % len(stand)] for i in range(envs))
        cfg.scene.num_envs = envs
        cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ('root_ground', 'feet_ground_height')) + world.ray_sensors()
        if camera:
            from r1wb.s2.collect import head_camera
            cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(224),)
        cfg.terminations = {'time_out': cfg.terminations['time_out']}           # a fall is shown, not reset
        if environment != 'flat':
            terrain = cfg.scene.terrain
            terrain.terrain_type = 'generator'
            terrain.max_init_terrain_level = None
            terrain.terrain_generator = TerrainGeneratorCfg(
                seed=17, size=(FLOOR, FLOOR), border_width=1.0, num_rows=1, num_cols=1, color_scheme='none',
                sub_terrains={'app': AppTerrainCfg(kind=environment, size=(FLOOR, FLOOR))})
        self.ball_names = world.add_throwables(cfg, 'ball', BALLS)
        cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 400)
        cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.wrapped = RslRlVecEnvWrapper(self.env)
        env = self.env
        self.term = env.command_manager.get_term('motion')
        table = torch.zeros(envs, 1, 8)
        table[:, 0] = torch.tensor([1.0e6, 0.0, 0.0, 0.0, STAND[3], 0.0, 0.0, 0.0])
        self.term.set_script(table)
        env.reset()
        self.wrapped.get_observations()
        self.robot = env.scene['robot']
        self.dt = float(env.step_dt)

        self.goal = torch.tensor(STAND, device=device).repeat(envs, 1)           # what the robots are told now
        tokenizer = MotionTokenizer.load(Path(selection['tokens']['path']), device)
        head = SmallS2.load(Path(selection['head']['path']), device)
        steer_policy = SteerPolicy.load(Path(selection['steer']['path']), device)
        self.steerer = Steer(env, self.term, steer_policy.see, generator=False, leak=steer_policy.leak,
                             remember=steer_policy.remember, turned=steer_policy.turned)
        self.view = S2View(env, self.term, goal_brain(head, steered_goal(lambda decision: self.goal.clone(), self.steerer, steer_policy)), tokenizer)
        self.tracker = load_teacher(checkpoint, device)
        self.joint_action = env.action_manager.get_term('joint_pos')
        if selection.get('ground_reference'):
            world.ground_reference(env, self.term, selection['ground_reference'])
            world.ground_reference(env, self.view, selection['ground_reference'])
        self._world = world
        self.fallen = torch.zeros(envs, dtype=torch.bool, device=device)
        self.steps = 0
        self.balls = [env.scene[name] for name in self.ball_names]
        self.ball_used = torch.zeros(envs, dtype=torch.long, device=device)
        self.ball_age = torch.full((BALLS, envs), 1.0e3, device=device)       # s since thrown (1e3 = parked)
        self.ball_flight = torch.zeros(BALLS, envs, 3, device=device)
        self.ball_measure = torch.zeros(BALLS, envs, device=device)           # s after the throw at which the hit is read
        self.ball_given = torch.full((BALLS, envs), float('nan'), device=device)   # N s the ball lost along its flight
        self.ball_mass = float(world.BALL['mass'])

    # ------------------------------------------------------------------ control
    def tell(self, goal) -> None:
        """goal: (5,) for every robot or (envs, 5): vx, vy, yaw rate, height, whether the height counts."""
        goal = torch.as_tensor(goal, dtype=torch.float, device=self.device)
        self.goal = goal.repeat(self.count, 1) if goal.dim() == 1 else goal.clone()

    def step(self) -> None:
        """One control step (0.02 s), in the evaluation's order."""
        from tensordict import TensorDict

        from r1wb.tracking.env_cfg import teacher_observation
        with torch.no_grad():
            self.view.update()
            seen = teacher_observation(self.view.env, 'motion', self.joint_action.plain_action, local=True,
                                       width=self.tracker.obs_dim)
            action = self.joint_action.from_plain_action(self.tracker(TensorDict({'actor': seen}, batch_size=[self.count])))
            self.wrapped.step(action)
            self.steerer.progress.after_step()
            self._balls_after_step()
            self.fallen |= self._world.fallen(self.env)
        self.steps += 1

    def reset(self) -> None:
        everyone = torch.arange(self.count, device=self.device)
        self.env.reset()
        self.wrapped.get_observations()
        self.steerer.restart(everyone)
        self.goal = torch.tensor(STAND, device=self.device).repeat(self.count, 1)
        self.fallen[:] = False
        for index in range(BALLS):
            self._park(index, everyone)

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------ balls
    def throw(self, envs: torch.Tensor, origin: torch.Tensor, target: torch.Tensor, speed: torch.Tensor) -> None:
        """A ball for each of `envs` from `origin` (n, 3) straight at `target` (n, 3) at `speed` (n,) m/s (world)."""
        along = target - origin
        distance = along.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
        along = along / distance
        pose = torch.cat([origin, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(envs), 4)], dim=1)
        velocity = torch.cat([along * speed.unsqueeze(1), torch.zeros(len(envs), 3, device=self.device)], dim=1)
        which = self.ball_used[envs] % BALLS
        for index, body in enumerate(self.balls):
            chosen = which == index
            if chosen.any():
                body.write_root_link_pose_to_sim(pose[chosen], envs[chosen])
                body.write_root_link_velocity_to_sim(velocity[chosen], envs[chosen])
                self.ball_age[index, envs[chosen]] = 0.0
                self.ball_flight[index, envs[chosen]] = velocity[chosen, :3]
                self.ball_measure[index, envs[chosen]] = distance[chosen, 0] / speed[chosen].clamp_min(0.1) + 0.15
                self.ball_given[index, envs[chosen]] = float('nan')
        self.ball_used[envs] += 1

    def _park(self, index: int, envs: torch.Tensor) -> None:
        pose = torch.zeros(len(envs), 7, device=self.device)
        pose[:, 2], pose[:, 3] = self._world.PARK + index, 1.0
        self.balls[index].write_root_link_pose_to_sim(pose, envs)
        self.balls[index].write_root_link_velocity_to_sim(torch.zeros(len(envs), 6, device=self.device), envs)
        self.ball_age[index, envs] = 1.0e3

    def _balls_after_step(self) -> None:
        self.ball_age += self.dt
        for index, body in enumerate(self.balls):
            flying = self.ball_age[index] < 1.0e3
            due = flying & (self.ball_age[index] >= self.ball_measure[index]) & torch.isnan(self.ball_given[index])
            if due.any():
                speed = torch.norm(self.ball_flight[index], dim=1).clamp_min(1.0e-6)
                kept = (body.data.root_link_lin_vel_w * self.ball_flight[index]).sum(1) / speed
                self.ball_given[index] = torch.where(due, self.ball_mass * (speed - kept), self.ball_given[index])
            old = (flying & (self.ball_age[index] > BALL_LIFETIME)).nonzero().flatten()
            if old.numel():
                self._park(index, old)

    def ball_positions(self, env: int = 0) -> list:
        """[(slot, [x, y, z])] of the balls in flight in one world."""
        return [(index, body.data.root_link_pos_w[env].tolist()) for index, body in enumerate(self.balls)
                if float(self.ball_age[index, env]) < 1.0e3]

    # ------------------------------------------------------------------ what the caller reads
    def yaw(self) -> torch.Tensor:
        w, x, y, z = self.robot.data.root_link_quat_w.unbind(dim=1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def position(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w - self.env.scene.env_origins

    def plan_minus_robot_height(self) -> torch.Tensor:
        """Height of the plan System 1 reads minus the robot's (m): near zero when the ground reference is applied."""
        return self.view.anchor_pos_w[:, 2] - self.term.robot_anchor_pos_w[:, 2]

    def told_and_command(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(the goal told (envs, 5), the command the slow loop passes on (envs, 3))."""
        return self.goal, self.steerer.command


class SceneRenderer:
    """한 세계의 상태를 CPU MuJoCo 로 옮겨 앱 화면(960 x 600)을 그린다. 로봇을 따라가는 자유 시점."""

    def __init__(self, core: DynamicCore, width: int = 960, height: int = 600):
        self.core, self.width, self.height = core, width, height
        # Parked projectiles inflate the simulation bounds to ~67 m, centred 33 m
        # above the floor. OpenGL's resulting shadow map produces striped planes.
        # Keep display bounds local, on a copy: physics and the policy camera use
        # the original model, including its terrain, materials and sensor settings.
        self.model = copy.copy(core.env.sim.mj_model)
        self.model.stat.extent = 10.0
        self.model.vis.map.znear = 0.02 / self.model.stat.extent
        self.model.vis.map.zfar = 200.0 / self.model.stat.extent
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 0.0, -15.0, 2.8
        root = self.model.body('robot/pelvis')
        self.root_body = int(root.id)
        self.robot_bodies = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith('robot/')}

    def sync(self, env: int = 0) -> None:
        data = self.core.env.sim.data
        self.data.qpos[:] = data.qpos[env].cpu().numpy()
        self.data.qvel[:] = data.qvel[env].cpu().numpy()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def scene(self):
        self.camera.lookat[:] = self.data.xpos[self.root_body]
        self.model.stat.center[:] = self.camera.lookat
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.scene

    def render(self) -> np.ndarray:
        self.scene()
        return self.renderer.render()

    def pick(self, x: float, y: float):
        """화면 위치(0..1) -> (발사 원점, 조준점, 조준점이 로봇 위인가). 앱의 발사 계산과 같다(시점에서 0.25 m 앞에서 출발)."""
        scene = self.scene()
        camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        aspect = self.width / self.height
        up, forward = np.array(camera.up, dtype=float), np.array(camera.forward, dtype=float)
        half = (camera.frustum_top - camera.frustum_bottom) / 2
        direction = forward * camera.frustum_near + np.cross(forward, up) * (2 * x - 1) * half * aspect + up * (1 - 2 * y) * half
        direction /= np.linalg.norm(direction)
        origin = np.array(camera.pos, dtype=float) + direction * 0.25
        target = np.array(camera.pos, dtype=float) + direction * 12.0
        point = np.zeros(3)
        geom = np.zeros(1, dtype=np.int32)
        flex, skin = geom.copy(), geom.copy()
        selected = mujoco.mjv_select(self.model, self.data, mujoco.MjvOption(), aspect, x, 1 - y, scene, point, geom, flex, skin)
        on_robot = False
        if selected >= 0 and 0.4 < np.linalg.norm(point - camera.pos) < 12:
            target = point.copy()
            on_robot = int(geom[0]) >= 0 and int(self.model.geom_bodyid[int(geom[0])]) in self.robot_bodies
        return origin, target, on_robot

    def close(self) -> None:
        self.renderer.close()
