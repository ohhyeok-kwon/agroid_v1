"""Local MuJoCo simulation. JSON state and binary JPEG pipes; no robot transport."""
import json
import io
import math
from collections import deque
from pathlib import Path
import sys
import threading
import time

if __name__ == '__main__':
    import ctypes
    import os
    import signal
    owner = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != owner or owner == 1:
        raise SystemExit(0)

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image
from r1lab import contract as C
from r1lab.terrain_observation import actor_input_dimension, height_scan_numpy
from simulation import PROJECTILE_SETTINGS, validate_launch

PROJECT = Path(__file__).resolve().parents[2]
MOTION = PROJECT / 'src/motion_manager'
ACTIONS = ('walking', 'careful_walk', 'safe_crouch', 'locked_standing', 'damping', 'zero_moment')
EXPERIMENTS = ('reference_teacher', 'unified_policy')
ENVIRONMENTS = ('flat', 'rocks', 'stairs', 'mixed')


class TeacherLoader:
    """사람 모션 교사(실험)는 학습 모듈(torch·mjlab)이 무거워 처음 선택할 때 별도 스레드에서 불러온다."""

    def __init__(self, project=PROJECT):
        self.project, self.thread, self.selection, self.error = project, None, None, None

    def poll(self, wait=False):
        """준비되면 검증된 선택을, 불러오는 중이면 None을 돌려준다. 실패는 ValueError."""
        if self.thread is None:
            self.thread = threading.Thread(target=self.load, daemon=True)
            self.thread.start()
        if wait:
            self.thread.join()
        if self.thread.is_alive():
            return None
        if self.error:
            raise ValueError(self.error)
        return self.selection

    def load(self):
        try:
            import torch
            torch.set_num_threads(1)
            import reference_teacher
            self.selection = reference_teacher.load_selection(self.project)
        except Exception as error:
            self.error = str(error) if isinstance(error, ValueError) else f'사람 모션 교사를 불러올 수 없습니다: {error}'


def terrain(kind):
    objects = []
    rng = np.random.default_rng(17)
    if kind in ('rocks', 'mixed'):
        for _ in range(60 if kind == 'rocks' else 34):
            x, y = rng.uniform(-7, 9), rng.uniform(-6, 6)
            if abs(x) < 1.4 and abs(y) < 1.4:
                continue
            if kind == 'mixed' and 1.5 < x < 10 and abs(y) < 1.8:
                continue
            height = float(rng.uniform(.045, .16))
            objects.append({'shape': 'ellipsoid', 'size': [float(rng.uniform(.12,.3)), float(rng.uniform(.1,.26)), height/2],
                            'position': [float(x), float(y), height/2], 'yaw': float(rng.uniform(-math.pi,math.pi)), 'color': '#727b85'})
    if kind in ('stairs', 'mixed'):
        for i in range(7):
            h = (i + 1) * .06
            objects.append({'shape': 'box', 'size': [.2, 1.35, h/2], 'position': [1.8+i*.4,0,h/2], 'yaw': 0, 'color': '#70829a'})
            objects.append({'shape': 'box', 'size': [.2, 1.35, h/2], 'position': [8.2-i*.4,0,h/2], 'yaw': 0, 'color': '#70829a'})
        objects.append({'shape': 'box', 'size': [.8,1.35,.21], 'position': [5,0,.21], 'yaw': 0, 'color': '#8396ad'})
    return objects


class UnifiedLoader(TeacherLoader):
    """통합 정책 S1(실험)도 학습 모듈을 쓰므로 처음 선택할 때 별도 스레드에서 불러온다."""

    def load(self):
        try:
            import torch
            torch.set_num_threads(1)
            import unified_policy
            self.selection = unified_policy.load_selection(self.project)
        except Exception as error:
            self.error = str(error) if isinstance(error, ValueError) else f'통합 정책을 불러올 수 없습니다: {error}'


class Simulation:
    def __init__(self, environment='flat', policies=None, teacher_loader=None, unified_loader=None):
        self.sessions = {}
        self.policies = policies
        self.teacher_loader = teacher_loader or TeacherLoader()
        self.teacher = None
        self.unified_loader = unified_loader or UnifiedLoader()
        self.unified = None
        self.session_policy_info = {}
        self.session_input_dimensions = {}
        self.settings = {}
        self.stance = json.loads((MOTION/'locked_standing_action_manager/config/stance.json').read_text())
        self.environment = environment
        self.objects = terrain(environment)
        spec = mujoco.MjSpec.from_file(str(PROJECT/'trainer/src/r1lab/assets/r1/r1.xml'))
        spec.option.timestep = .001
        spec.add_texture(name='simulation_floor',type=mujoco.mjtTexture.mjTEXTURE_2D,builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         width=128,height=128,rgb1=[.15,.18,.23],rgb2=[.21,.25,.31])
        textures = ['']*int(mujoco.mjtTextureRole.mjNTEXROLE)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = 'simulation_floor'
        spec.add_material(name='simulation_floor',textures=textures,texuniform=True,texrepeat=[1,1],reflectance=.05)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30,30,.1], friction=[.8,.02,.01],material='simulation_floor')
        spec.worldbody.add_light(pos=[0,-2,4],dir=[0,0,-1],type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,diffuse=[.6,.6,.6])
        for item in self.objects:
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX if item['shape']=='box' else mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                    size=item['size'], pos=item['position'], euler=[0,0,item['yaw']], friction=[.8,.02,.01])
        for i in range(PROJECTILE_SETTINGS['capacity']):
            body = spec.worldbody.add_body(name=f'projectile_{i}',pos=[0,0,-20-i],gravcomp=1)
            body.add_freejoint(name=f'projectile_joint_{i}')
            body.add_geom(name=f'projectile_geom_{i}',type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[PROJECTILE_SETTINGS['radius'],0,0],mass=PROJECTILE_SETTINGS['mass'],
                          contype=1,conaffinity=1,friction=[.6,.02,.01],rgba=[1,.55,.16,1])
        self.model = spec.compile()
        self.model.stat.extent = 2
        self.model.vis.global_.offwidth = 960
        self.model.vis.global_.offheight = 600
        self.model.vis.headlight.ambient[:] = [.35,.35,.35]
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 0; self.camera.elevation = -15; self.camera.distance = 2.8
        self.data = mujoco.MjData(self.model)
        self.q_indices = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in C.ENTITY_JOINTS])
        self.v_indices = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in C.ENTITY_JOINTS])
        self.projectiles = []
        for i in range(PROJECTILE_SETTINGS['capacity']):
            joint = self.model.joint(f'projectile_joint_{i}').id
            self.projectiles.append({'body':self.model.body(f'projectile_{i}').id,'geom':self.model.geom(f'projectile_geom_{i}').id,
                                     'q':self.model.jnt_qposadr[joint],'v':self.model.jnt_dofadr[joint],'slot':i,'active':False})
            start = self.model.jnt_dofadr[joint]
            self.model.dof_damping[start:start+6] = 0
            self.model.dof_frictionloss[start:start+6] = 0
            self.model.dof_armature[start:start+6] = 0
        self.projectile_geoms = {item['geom']:item for item in self.projectiles}
        for name in C.ENTITY_JOINTS:
            j = self.model.joint(name)
            group = C.GROUPS[C._group_of(name)]
            self.model.dof_armature[j.dofadr] = group['armature']
            self.model.dof_damping[j.dofadr] = .05
            self.model.dof_frictionloss[j.dofadr] = .1
        self.effort = np.array([C.GROUPS[C._group_of(name)]['effort'] for name in C.ENTITY_JOINTS])
        self.reset()

    def configuration(self, action):
        if action not in self.settings:
            package = 'walking_action_manager' if action == 'walking' else f'{action}_action_manager'
            filename = 'amp_locomotion' if action == 'walking' else action
            self.settings[action] = json.loads((MOTION/package/'config'/f'{filename}.json').read_text())
        return self.settings[action]

    def session(self, action, variant='walk'):
        key = (action, variant)
        if key not in self.sessions:
            package = 'walking_action_manager' if action == 'walking' else f'{action}_action_manager'
            filename = variant if action == 'walking' else action
            options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
            selected = self.policies.open(action, options) if self.policies else None
            if selected is not None:
                self.sessions[key], info = selected
                self.session_policy_info[id(self.sessions[key])] = info
            else:
                self.sessions[key] = ort.InferenceSession(str(MOTION/package/'config'/f'{filename}.onnx'), options, providers=['CPUExecutionProvider'])
        return self.sessions[key]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0,0,C.STAND_HEIGHT+.01]
        self.data.qpos[3:7] = [1,0,0,0]
        self.data.qpos[self.q_indices] = [C.DEFAULT_POS_BY_NAME[name] for name in C.ENTITY_JOINTS]
        for item in self.projectiles:
            self.hide_projectile(item)
        mujoco.mj_forward(self.model,self.data)
        self.action = None; self.tick = 0; self.message = ''; self.velocity = np.zeros(3)
        self.physics_steps = 0; self.control_due = True; self.heading_target = None; self.allow_turn = True
        self.shots = 0; self.hits = 0; self.last_shot = None
        self.select('walking')

    def select(self, action):
        if action == self.action:
            return
        if action in ('walking','careful_walk','safe_crouch','reference_teacher','unified_policy'):
            if self.fallen():
                self.message = '넘어진 상태입니다. 초기화한 뒤 다시 시작하세요.'
                return
            if action == 'reference_teacher':
                if not self.prepare_teacher():
                    return
            elif action == 'unified_policy':
                if not self.prepare_unified():
                    return
            else:
                self.session(action)
        self.action = action; self.started = self.data.time
        self.heading_target = None; self.control_due = True
        if action == 'careful_walk' and self.policies and self.policies.heading_hold(action):
            # Contract v2 students hold the heading they start with until the user steers (as trained).
            self.heading_target = self.heading()
        if action == 'unified_policy':
            # 선택한 순간의 방향을 유지한다(우클릭으로 바꿀 때까지). 정책 출력은 건드리지 않고 회전 속도 명령만 만든다.
            self.heading_target = self.heading()
        self.start_q = self.data.qpos[self.q_indices].copy(); self.velocity = np.zeros(3)
        self.history = np.zeros((10,81),dtype=np.float32); self.previous = np.zeros(24,dtype=np.float32)
        self.fill = True; self.message = ''
        self.target = self.start_q.copy(); self.kp = np.zeros(24); self.kd = np.zeros(24)

    def prepare_teacher(self):
        """실험 교사를 현재 서 있는 자세에 맞춰 시작한다. 준비 중·실패면 기존 액션을 유지한다."""
        if self.environment != 'flat':
            self.message = '사람 모션 교사는 평지 자료로 학습해 평지에서만 시험합니다.'
            return False
        try:
            selection = self.teacher_loader.poll()
        except ValueError as error:
            self.message = str(error)
            return False
        if selection is None:
            self.message = '사람 모션 교사를 불러오는 중입니다.'
            return False
        if self.teacher is None:
            self.teacher = selection.controller(self.model)
        self.teacher.reset(self.model, self.data)
        return True

    def prepare_unified(self):
        """통합 정책(실험)을 현재 자세에서 시작한다. 준비 중·실패면 기존 액션을 유지한다."""
        if self.environment != 'flat':
            self.message = '통합 정책은 평지에서 학습해 평지에서만 시험합니다.'
            return False
        try:
            selection = self.unified_loader.poll()
        except ValueError as error:
            self.message = str(error)
            return False
        if selection is None:
            self.message = '통합 정책을 불러오는 중입니다.'
            return False
        if self.unified is None:
            self.unified = selection.controller(self.model)
        self.unified.reset(self.model, self.data)
        return True

    def fallen(self):
        _, x, y, _ = self.data.qpos[3:7]
        return bool(self.data.qpos[2] < .25 or 1-2*(x*x+y*y) < .5)

    def hide_projectile(self, item):
        item['active'] = False
        q, v = item['q'], item['v']
        self.data.qpos[q:q+7] = [0,0,-20-item['slot'],1,0,0,0]
        self.data.qvel[v:v+6] = 0
        self.model.geom_contype[item['geom']] = 0
        self.model.geom_conaffinity[item['geom']] = 0
        self.model.body_gravcomp[item['body']] = 1

    def scene(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=600,width=960)
        self.camera.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data,camera=self.camera)
        return self.renderer.scene

    def render(self):
        self.scene()
        output = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(output,format='JPEG',quality=85)
        return output.getvalue()

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer = None

    def launch(self, request):
        validate_launch(request)
        scene = self.scene()
        camera = mujoco.mjv_averageCamera(scene.camera[0],scene.camera[1])
        x,y = request['point']; aspect = 960/600
        up = np.array(camera.up,dtype=float); forward = np.array(camera.forward,dtype=float)
        half_height = (camera.frustum_top-camera.frustum_bottom)/2
        direction = forward*camera.frustum_near + np.cross(forward,up)*(2*x-1)*half_height*aspect + up*(1-2*y)*half_height
        direction /= np.linalg.norm(direction)
        origin = np.array(camera.pos,dtype=float)+direction*.25
        target = np.array(camera.pos,dtype=float)+direction*12
        point = np.zeros(3); geom = np.zeros(1,dtype=np.int32); flex=geom.copy(); skin=geom.copy()
        selected = mujoco.mjv_select(self.model,self.data,mujoco.MjvOption(),aspect,x,1-y,scene,point,geom,flex,skin)
        if selected>=0 and .4<np.linalg.norm(point-camera.pos)<12:
            target = point
        self.launch_at(origin,target,request['speed'])

    def launch_at(self, origin, target, speed):
        item = self.projectiles[self.shots % len(self.projectiles)]
        self.hide_projectile(item)
        origin = np.array(origin,dtype=float)
        delta = np.array(target,dtype=float)-origin
        gravity = self.model.opt.gravity
        # Aim along the lower ballistic arc when the selected speed can reach the target.
        b = speed*speed+float(delta @ gravity)
        discriminant = b*b-float(gravity @ gravity)*float(delta @ delta)
        flight = math.sqrt(2*float(delta @ delta)/(b+math.sqrt(discriminant))) if b>0 and discriminant>=0 else 0
        velocity = delta/flight-gravity*flight/2 if flight else delta/np.linalg.norm(delta)*speed
        self.data.qpos[item['q']:item['q']+3] = origin
        self.data.qvel[item['v']:item['v']+3] = velocity
        self.model.geom_contype[item['geom']] = 1
        self.model.geom_conaffinity[item['geom']] = 1
        self.model.body_gravcomp[item['body']] = 0
        self.shots += 1
        self.last_shot = {'id':self.shots,'speed':speed,'momentum':speed*PROJECTILE_SETTINGS['mass'],
                          'time':self.data.time,'hit':False}
        item.update(active=True,record=self.last_shot)
        mujoco.mj_forward(self.model,self.data)

    def update_projectiles(self):
        if not any(item['active'] for item in self.projectiles):
            return
        for contact in self.data.contact:
            for ball, other in ((contact.geom1,contact.geom2),(contact.geom2,contact.geom1)):
                item = self.projectile_geoms.get(ball)
                if item and item['active'] and not item['record']['hit'] and other not in self.projectile_geoms and self.model.geom_bodyid[other]!=0:
                    item['record']['hit'] = True
                    self.hits += 1
        for item in self.projectiles:
            if item['active'] and self.data.time-item['record']['time']>=PROJECTILE_SETTINGS['lifetime']:
                self.hide_projectile(item)

    def command(self, axes):
        speed = .35 if self.action == 'walking' else .3
        yaw_rate = 0.
        if self.heading_target is not None and self.allow_turn:
            error = math.remainder(self.heading_target-self.heading(),2*math.pi)
            yaw_rate = float(np.clip(error*3,-1.5 if self.action=='walking' else -1.,1.5 if self.action=='walking' else 1.))
            if abs(error)<.025:
                yaw_rate = 0.
        limits = self.policies.limits(self.action) if self.policies else {}
        longitudinal = limits.get('forward' if axes[0] >= 0 else 'reverse', speed)
        lateral = limits.get('lateral', speed)
        if 'yaw' in limits:
            yaw_rate = float(np.clip(yaw_rate, -limits['yaw'], limits['yaw']))
        desired = np.array([axes[0]*longitudinal,axes[1]*lateral,yaw_rate])
        self.velocity += np.clip(desired-self.velocity,[-.025,-.025,-.08],[.025,.025,.08])
        return self.velocity.copy()

    def heading(self):
        w,x,y,z = self.data.qpos[3:7]
        return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

    def set_heading(self, heading):
        if self.action not in ('walking','careful_walk','reference_teacher','unified_policy'):
            return
        self.heading_target = math.remainder(heading,2*math.pi)

    def policy(self, axes):
        action = self.action; elapsed = self.data.time-self.started
        if action == 'zero_moment':
            self.target = self.data.qpos[self.q_indices].copy(); self.kp[:] = 0; self.kd[:] = 0; return
        if action == 'damping':
            self.target = self.data.qpos[self.q_indices].copy(); self.kp[:] = 0; self.kd[:] = 4; return
        if action == 'locked_standing':
            final = np.array(self.stance['position_rad'][:24])
            duration = max(np.max(np.abs(final-self.start_q))/.1,.02)
            self.target = self.start_q+(final-self.start_q)*min(1,elapsed/duration)
            self.kp = np.array(self.stance['kp'][:24]); self.kd = np.array(self.stance['kd'][:24]); return
        if action == 'reference_teacher':
            # 학습처럼 명령을 계단식으로 주고(매처가 0.3초로 완화) 0.3초 목표각 혼합 없이 바로 적용한다.
            command = self.teacher.command_from_axes(axes)
            heading = self.heading_target if self.allow_turn else None
            self.target, self.kp, self.kd = self.teacher.step(self.model, self.data, command, heading); return
        if action == 'unified_policy':
            # 명령은 정책이 학습한 그대로(가속 한계는 제어기 안에서). 우클릭 방향은 회전 속도 명령으로 바꾼다:
            # 이 정책은 0.6 rad/s 아래의 제자리 회전 명령을 따르지 않으므로 그 아래로는 주지 않는다.
            command = self.unified.command_from_axes(axes)
            self.unified_lateral_ignored = bool(self.unified.lateral_ignored(axes))
            if self.heading_target is not None and self.allow_turn:
                error = math.remainder(self.heading_target-self.heading(),2*math.pi)
                if abs(command[0]) > .05 or abs(command[1]) > .05:
                    command[2] = float(np.clip(1.5*error,-.5,.5))        # 걷는 중: 오차에 비례한 완만한 회전
                elif abs(error) >= .1:
                    command[2] = math.copysign(min(.8,max(.6,abs(error)*3)),error)
            self.velocity = self.unified.twist.copy()
            self.target, self.kp, self.kd = self.unified.step(self.model, self.data, command); return
        p = self.configuration(action)
        command = self.command(axes)
        if action == 'safe_crouch':
            cmd = p['command']; recommended = cmd['recommended']; h0 = cmd['H0']
            height = max(recommended['h_target'],h0-recommended['rate']*max(0,elapsed-recommended['delay']))
            command = np.array([height-h0,recommended['h_target']-h0,recommended['rate']])
        permutation = np.argsort(p['action_idx'])
        w,x,y,z = self.data.qpos[3:7]
        gravity = [2*(w*y-x*z),-2*(w*x+y*z),1-2*(w*w+z*z)]
        frame = np.concatenate([self.data.qvel[3:6],gravity,command,self.data.qpos[self.q_indices][permutation]-p['default_joint_pos'],self.data.qvel[self.v_indices][permutation],self.previous]).astype(np.float32)
        if self.fill:
            self.history[:] = frame; self.fill = False
        else:
            self.history[:-1] = self.history[1:]; self.history[-1] = frame
        variant = 'turn' if action=='walking' and abs(command[2])>=1 and max(abs(command[0]),abs(command[1]))<=.3 else 'walk'
        session = self.session(action,variant)
        session_key = id(session)
        if session_key not in self.session_input_dimensions:
            self.session_input_dimensions[session_key] = actor_input_dimension(session)
        observation = self.history.reshape(1,810)
        if self.session_input_dimensions[session_key] == 927:
            observation = np.concatenate((observation, height_scan_numpy(self.model,self.data).reshape(1,117)),axis=1)
        raw = session.run(['actions'],{'obs':observation})[0][0]
        if not np.isfinite(raw).all():
            raise ValueError('정책 출력에 유효하지 않은 값이 있습니다.')
        idx = np.array(p['action_idx'])
        q = (np.array(p['default_joint_pos'])+raw*np.array(p['action_scale']))[idx]
        self.target = self.start_q*(1-min(1,elapsed/.3))+q*min(1,elapsed/.3)
        self.kp = np.array(p['stiffness'])[idx]; self.kd = np.array(p['damping'])[idx]
        self.previous = raw.astype(np.float32)

    def step(self, axes):
        self.advance(axes,20)

    def advance(self, axes, steps, allow_turn=True):
        self.allow_turn = allow_turn
        for _ in range(steps):
            if self.control_due or self.physics_steps%20==0:
                self.policy(axes);self.control_due = False
            torque = self.kp*(self.target-self.data.qpos[self.q_indices])-self.kd*self.data.qvel[self.v_indices]
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.v_indices] = np.clip(torque,-self.effort,self.effort)
            mujoco.mj_step(self.model,self.data)
            self.update_projectiles()
            self.physics_steps += 1
        self.tick = self.physics_steps//20

    def teacher_state(self):
        if self.action != 'reference_teacher' or self.teacher is None:
            return None
        info = self.teacher.info
        return {**self.teacher.diagnostics(), 'label':info['label'], 'run_id':info['run_id'],
                'iteration':info['iteration'], 'library':info['library'],
                'library_matches_training':info['library_matches_training']}

    def unified_state(self):
        if self.action != 'unified_policy' or self.unified is None:
            return None
        info = self.unified.info
        return {**self.unified.diagnostics(), 'label':info['label'], 'run_id':info['run_id'], 'iteration':info['iteration'],
                'lateral_ignored':getattr(self,'unified_lateral_ignored',False)}

    def snapshot(self, active=True):
        session = self.sessions.get((self.action, 'walk'))
        policy = self.session_policy_info.get(id(session)) if session is not None else None
        return {'state':'running' if active else 'paused','action':self.action,'environment':self.environment,
                'time':round(self.data.time,3),'tick':self.tick,'fallen':self.fallen(), 'message':self.message,
                'policy':policy,'teacher':self.teacher_state(),'unified':self.unified_state(),
                'velocity':self.data.qvel[:3].tolist(),'obstacles':self.objects,
                'launcher':{'shots':self.shots,'hits':self.hits,'last':self.last_shot},
                'render':{'engine':'MuJoCo','width':960,'height':600,'fps':60},
                'heading':self.heading(),'heading_target':self.heading_target,
                'camera':{key:float(getattr(self.camera,key)) for key in ('azimuth','elevation','distance')},
                'projectiles':[{'id':item['record']['id'],'position':self.data.qpos[item['q']:item['q']+3].tolist(),
                                'hit':item['record']['hit']} for item in self.projectiles if item['active']],
                'pose':{'tick':self.tick,'joints':dict(zip(C.ENTITY_JOINTS,map(float,self.data.qpos[self.q_indices]))),
                        'quaternion':self.data.qpos[3:7].tolist(),'position':self.data.qpos[:3].tolist()}}


def main():
    from simulation_policies import SimulationPolicies
    policies = SimulationPolicies(PROJECT)
    lock = threading.Lock()
    command = {'action':'walking','environment':'flat','axes':[0,0],'active':True,'reset':False,'received':time.monotonic(),
               'camera':{'azimuth':0,'elevation':-15,'distance':2.8}}
    pending_launches = deque(maxlen=32)
    pending_heading = None
    closed = threading.Event()
    def inputs():
        nonlocal pending_heading
        for line in sys.stdin:
            try:
                message = json.loads(line)
                with lock:
                    if 'launch' in message:
                        pending_launches.append(message.pop('launch'))
                    if 'heading' in message:
                        pending_heading = message.pop('heading')
                    command.update(message); command['received'] = time.monotonic()
            except (ValueError,TypeError):
                pass
        closed.set()
    threading.Thread(target=inputs,daemon=True).start()
    teacher_loader = TeacherLoader()
    unified_loader = UnifiedLoader()
    engine = Simulation(policies=policies, teacher_loader=teacher_loader, unified_loader=unified_loader)
    deadline = time.monotonic();fraction = 0.
    while not closed.is_set():
        started = time.monotonic()
        with lock:
            current = command.copy(); command['reset'] = False
            launches = list(pending_launches); pending_launches.clear()
            heading = pending_heading; pending_heading = None
        if current['environment'] != engine.environment:
            engine.close()
            engine = Simulation(current['environment'], policies=policies, teacher_loader=teacher_loader,
                                unified_loader=unified_loader)
        elif current['reset']:
            engine.reset()
        engine.select(current['action'])
        camera_changed = any(getattr(engine.camera,key)!=value for key,value in current['camera'].items())
        for key,value in current['camera'].items():
            setattr(engine.camera,key,value)
        active = current['active'] and started-current['received']<2
        if active:
            if started-current['received']<.4:
                if heading is not None:
                    engine.set_heading(heading)
                for launch in launches:
                    engine.launch(launch)
            fraction += 1000/60
            steps = int(fraction);fraction -= steps
            engine.advance(current['axes'] if started-current['received']<.4 else [0,0],steps,started-current['received']<.4)
        state = engine.snapshot(active)
        frame = engine.render() if active or camera_changed or engine.renderer is None else b''
        state['frame_bytes'] = len(frame)
        sys.stdout.buffer.write(json.dumps(state,allow_nan=False,separators=(',',':')).encode()+b'\n'+frame)
        sys.stdout.buffer.flush()
        deadline += 1/60
        remaining = deadline-time.monotonic()
        if remaining < -1/60:
            deadline = time.monotonic()
        time.sleep(max(0,remaining))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'state':'error','message':str(error)}),flush=True)
        raise
