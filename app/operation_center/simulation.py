"""Owns the local simulation subprocess; never sends requests to the physical robot."""
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

ACTIONS = [('walking','Walking'),('careful_walk','Careful Walk'),('safe_crouch','Safe Crouch'),
           ('locked_standing','Standing'),('damping','Damping'),('zero_moment','Zero Torque')]
EXPERIMENTS = [('dynamic_mode','DynamicMode'), ('reference_teacher','사람 모션 교사(실험)'), ('unified_policy','통합 정책 S1(실험)')]
# DynamicMode 는 평가와 같은 전신 스택을 별도 작업 프로세스(dynamic_worker.py, 트레이너의 평가 가상환경 · PC GPU)로 돌린다.
DYNAMIC = 'dynamic_mode'
DYNAMIC_SPEEDS = ('slow','normal','fast')
DYNAMIC_POSTURES = ('stand','rise','slight','deep')
DYNAMIC_KEYS = ('instruction','speed','posture')
ENVIRONMENTS = [('flat','평지'),('rocks','큰 돌 지형'),('stairs','계단 · 6cm'),('mixed','돌 + 계단')]
PROJECTILE_SETTINGS = {'min_speed':5, 'max_speed':25, 'default_speed':12, 'mass':1., 'radius':.07, 'lifetime':5., 'capacity':8}


def validate_launch(launch):
    if not isinstance(launch,dict) or set(launch) != {'point','speed'}:
        raise ValueError('발사에는 화면 위치·속도가 필요합니다.')
    point = launch['point']
    if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in point):
        raise ValueError('발사 위치는 화면 안이어야 합니다.')
    speed = launch['speed']
    if type(speed) not in (int,float) or not math.isfinite(speed) or not PROJECTILE_SETTINGS['min_speed'] <= speed <= PROJECTILE_SETTINGS['max_speed']:
        raise ValueError('발사 속도는 5~25 m/s 범위여야 합니다.')


class Simulation:
    def __init__(self, project, app):
        self.root, self.app = Path(project), Path(app)
        self.lock = threading.RLock(); self.write_lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.process = None; self.worker = None; self.log = None
        self.current = {'state':'stopped','message':'시뮬레이션을 시작합니다.'}
        self.latest_frame = None
        self.frame_id = 0
        self.closed = False
        self.mode = 'standard'; self.environment = 'flat'; self.camera = None; self.dynamic_fresh = False
        self.dynamic_process = None; self.dynamic_worker = None; self.dynamic_log = None; self.dynamic_environment = None
        self.dynamic_current = {'state':'stopped','action':DYNAMIC,'message':'DynamicMode 를 시작합니다.'}

    def start(self):
        with self.lock:
            if self.closed:
                return
            if self.process and self.process.poll() is None:
                return
            python = self.root / '.agent/trainer/.venv/bin/python'
            if not python.exists():
                self.current = {'state':'error','message':'트레이너 Python 환경이 없습니다.'}; return
            if self.log:
                self.log.close()
            directory = self.root/'.agent/operation_center'; directory.mkdir(parents=True,exist_ok=True)
            self.log = (directory/'simulation.log').open('a')
            self.current = {'state':'loading','message':'물리 모델과 Walking 정책을 불러오는 중입니다.'}
            self.latest_frame = None
            command = [str(python),'-u',str(self.app/'simulation_worker.py')]
            env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MUJOCO_GL':'egl'}
            # The worker's PR_SET_PDEATHSIG follows the *thread* that forks it. HTTP request threads end
            # with their connection, so fork from this reader thread, which lives until the worker exits.
            spawned = {}; ready = threading.Event()
            self.worker = threading.Thread(target=self._run,args=(command,env,self.log,spawned,ready),
                                           name='simulation-worker',daemon=True)
            self.worker.start(); ready.wait()
            if 'error' in spawned:
                self.current = {'state':'error','message':f'시뮬레이션을 시작하지 못했습니다: {spawned["error"]}'}; return
            self.process = spawned['process']

    def _run(self, command, env, log, spawned, ready):
        try:
            process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env)
        except OSError as error:
            spawned['error'] = error; ready.set(); return
        spawned['process'] = process; ready.set()
        for line in process.stdout:
            try:
                state = json.loads(line)
                if isinstance(state,dict) and 'state' in state:
                    size = state.pop('frame_bytes',0)
                    if not isinstance(size,int) or not 0<=size<=4*1024*1024:
                        raise ValueError('잘못된 프레임 크기입니다.')
                    frame = process.stdout.read(size) if size else None
                    if frame is not None and len(frame)!=size:
                        break
                    with self.lock:
                        if self.process is process:
                            self.current = state
                            if frame is not None and self.mode == 'standard':
                                self.latest_frame = frame
                                self.frame_id += 1
                                self.frame_condition.notify_all()
            except ValueError:
                pass
        # Stdout EOF can precede the exit status; wait so the worker is reaped and its death is shown.
        if process.wait() != 0:
            with self.lock:
                if self.process is process and not self.closed and self.current.get('state') != 'error':
                    self.current = {**self.current,'state':'error','message':'시뮬레이션이 종료되었습니다. 초기화를 눌러 다시 시작하세요.'}
        process.stdout.close()

    # ------------------------------------------------------------ DynamicMode: its own worker process
    def start_dynamic(self, environment):
        """Start the DynamicMode worker for `environment` (a worker for another terrain is replaced)."""
        with self.lock:
            if self.closed:
                return
            running = self.dynamic_process and self.dynamic_process.poll() is None
            if running and self.dynamic_environment == environment:
                return
        self.stop_dynamic()
        with self.lock:
            python = self.root / '.agent/trainer/rlinf/.venv/bin/python'
            if not python.exists():
                self.dynamic_current = {'state':'error','action':DYNAMIC,'message':'DynamicMode: 트레이너의 평가 Python 환경(.agent/trainer/rlinf/.venv)이 없습니다.'}; return
            directory = self.root/'.agent/operation_center'; directory.mkdir(parents=True,exist_ok=True)
            if self.dynamic_log:
                self.dynamic_log.close()
            self.dynamic_log = (directory/'dynamic_mode.log').open('a')
            self.dynamic_current = {'state':'loading','action':DYNAMIC,'environment':environment,
                                    'message':'DynamicMode 스택을 불러오는 중입니다(처음 10~20초).'}
            self.dynamic_environment = environment
            command = [str(python),'-u','-B',str(self.app/'dynamic_worker.py'),'--environment',environment]
            env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MUJOCO_GL':'egl',
                   'PYTHONPATH':str(self.root/'trainer/src')}
            # As for the standard worker: fork from the reader thread, which lives as long as the process.
            spawned = {}; ready = threading.Event()
            self.dynamic_worker = threading.Thread(target=self._run_dynamic,args=(command,env,self.dynamic_log,spawned,ready),
                                                   name='dynamic-worker',daemon=True)
            self.dynamic_worker.start(); ready.wait()
            if 'error' in spawned:
                self.dynamic_current = {'state':'error','action':DYNAMIC,'message':f'DynamicMode 를 시작하지 못했습니다: {spawned["error"]}'}; return
            self.dynamic_process = spawned['process']
            self.dynamic_fresh = True

    def _run_dynamic(self, command, env, log, spawned, ready):
        try:
            process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env)
        except OSError as error:
            spawned['error'] = error; ready.set(); return
        spawned['process'] = process; ready.set()
        for line in process.stdout:
            try:
                state = json.loads(line)
                if isinstance(state,dict) and 'state' in state:
                    size = state.pop('frame_bytes',0)
                    if not isinstance(size,int) or not 0<=size<=4*1024*1024:
                        raise ValueError('잘못된 프레임 크기입니다.')
                    frame = process.stdout.read(size) if size else None
                    if frame is not None and len(frame)!=size:
                        break
                    with self.lock:
                        if self.dynamic_process is process:
                            self.dynamic_current = state
                            if frame is not None and self.mode == 'dynamic':
                                self.latest_frame = frame
                                self.frame_id += 1
                                self.frame_condition.notify_all()
            except ValueError:
                pass
        code = process.wait()
        with self.lock:
            if self.dynamic_process is process and not self.closed:
                if code != 0 and self.dynamic_current.get('state') != 'error':
                    self.dynamic_current = {**self.dynamic_current,'state':'error','action':DYNAMIC,
                                            'message':'DynamicMode 가 종료되었습니다. DynamicMode 를 다시 누르세요(.agent/operation_center/dynamic_mode.log).'}
                elif code == 0:
                    self.dynamic_current = {**self.dynamic_current,'state':'stopped','action':DYNAMIC,'message':'DynamicMode 를 쉬는 중입니다.'}
        process.stdout.close()

    def stop_dynamic(self):
        """End the DynamicMode worker (it holds the PC's GPU while it runs)."""
        with self.lock:
            process, worker = self.dynamic_process, self.dynamic_worker
            self.dynamic_process = None; self.dynamic_worker = None
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if worker: worker.join(timeout=2)
        if process:
            for stream in (process.stdin,process.stdout):
                if stream:
                    try: stream.close()
                    except OSError: pass

    def _send(self, process, data):
        process.stdin.write((json.dumps(data,allow_nan=False)+'\n').encode()); process.stdin.flush()

    def snapshot(self):
        with self.lock:
            current = self.dynamic_current if self.mode == 'dynamic' else self.current
            return {**current,'actions':[{'id':key,'label':label} for key,label in ACTIONS],
                    'experiments':[{'id':key,'label':label} for key,label in EXPERIMENTS],
                    'environments':[{'id':key,'label':label} for key,label in ENVIRONMENTS],
                    'launcher':{**PROJECTILE_SETTINGS,**current.get('launcher',{})}}

    def frame(self):
        with self.lock:
            return self.latest_frame

    def wait_frame(self, after, timeout=1):
        with self.frame_condition:
            self.frame_condition.wait_for(lambda:self.frame_id>after or self.closed,timeout)
            if self.closed or self.frame_id<=after:
                return None
            return self.frame_id,self.latest_frame

    def command(self, data):
        if not isinstance(data,dict) or set(data)-{'action','environment','axes','active','reset','launch','camera','heading',*DYNAMIC_KEYS}:
            raise ValueError('지원하지 않는 시뮬레이션 명령입니다.')
        if 'instruction' in data and (not isinstance(data['instruction'],str) or not 0<len(data['instruction'].strip())<=200):
            raise ValueError('지시문은 1~200자의 문장이어야 합니다.')
        if 'speed' in data and data['speed'] not in DYNAMIC_SPEEDS:
            raise ValueError('알 수 없는 속도입니다.')
        if 'posture' in data and data['posture'] not in DYNAMIC_POSTURES:
            raise ValueError('알 수 없는 자세입니다.')
        if 'action' in data and data['action'] not in dict(ACTIONS) and data['action'] not in dict(EXPERIMENTS):
            raise ValueError('알 수 없는 액션입니다.')
        if 'environment' in data and data['environment'] not in dict(ENVIRONMENTS):
            raise ValueError('알 수 없는 환경입니다.')
        if 'axes' in data and (not isinstance(data['axes'],list) or len(data['axes'])!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1 for v in data['axes'])):
            raise ValueError('이동 입력은 -1~1 범위의 두 숫자여야 합니다.')
        if any(key in data and not isinstance(data[key],bool) for key in ('active','reset')):
            raise ValueError('상태 입력이 올바르지 않습니다.')
        if 'heading' in data and (type(data['heading']) not in (int,float) or not math.isfinite(data['heading']) or abs(data['heading'])>math.pi):
            raise ValueError('보행 방향은 -π~π 범위여야 합니다.')
        if 'launch' in data:
            validate_launch(data['launch'])
            if data.get('active') is False:
                raise ValueError('시뮬레이션 실행 중에만 발사할 수 있습니다.')
        if 'camera' in data:
            camera = data['camera']
            limits = {'azimuth':(-360,360),'elevation':(-85,5),'distance':(.75,8)}
            if not isinstance(camera,dict) or set(camera)!=set(limits) or any(type(camera[k]) not in (int,float) or not math.isfinite(camera[k]) or not lo<=camera[k]<=hi for k,(lo,hi) in limits.items()):
                raise ValueError('카메라 위치가 올바르지 않습니다.')
        if 'environment' in data:
            self.environment = data['environment']
        if 'camera' in data:
            self.camera = data['camera']          # (a DynamicMode worker started later begins with the view in use)
        if 'action' in data:
            entering = data['action'] == DYNAMIC and self.mode != 'dynamic'
            self.mode = 'dynamic' if data['action'] == DYNAMIC else 'standard'
        else:
            entering = False
        if self.mode == 'dynamic':
            return self.command_dynamic(data, entering)
        self.stop_dynamic()                       # another action: the DynamicMode worker gives the GPU back
        data = {key:value for key,value in data.items() if key not in DYNAMIC_KEYS}
        self.start()
        with self.write_lock:
            if not self.process or self.process.poll() is not None:
                raise ValueError(self.current.get('message','시뮬레이션을 시작하지 못했습니다.'))
            try:
                self._send(self.process, data)
            except (BrokenPipeError,OSError) as error:
                raise ValueError('시뮬레이션이 종료되었습니다. 초기화를 눌러 다시 시작하세요.') from error

    def command_dynamic(self, data, entering):
        """DynamicMode has its own worker; the standard worker never sees this action (it is paused meanwhile)."""
        with self.lock:
            process = self.dynamic_process
            running = process is not None and process.poll() is None
            failed = self.dynamic_current.get('state') == 'error'
            changed = self.dynamic_environment != self.environment
        if (not running or changed) and (entering or changed or data.get('reset') or 'action' in data or not failed):
            # (after an error it is started again only when the user asks: the action or the reset button)
            self.start_dynamic(self.environment)
        if entering:
            with self.write_lock:
                if self.process and self.process.poll() is None:
                    try: self._send(self.process, {'active':False,'axes':[0,0]})
                    except (BrokenPipeError,OSError): pass
        data = {key:value for key,value in data.items() if key not in ('action','environment')}
        if self.dynamic_fresh:
            self.dynamic_fresh = False
            if self.camera is not None and 'camera' not in data:
                data['camera'] = self.camera
        with self.write_lock:
            process = self.dynamic_process
            if not process or process.poll() is not None:
                if self.dynamic_current.get('state') == 'error':
                    raise ValueError(self.dynamic_current.get('message','DynamicMode 를 시작하지 못했습니다.'))
                return
            try:
                self._send(process, data)
            except (BrokenPipeError,OSError) as error:
                raise ValueError('DynamicMode 가 종료되었습니다. DynamicMode 를 다시 누르세요.') from error

    def close(self):
        with self.lock:
            self.closed = True
            self.frame_condition.notify_all()
        self.stop_dynamic()
        if self.dynamic_log: self.dynamic_log.close()
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if self.worker: self.worker.join(timeout=2)
        if process:
            for stream in (process.stdin,process.stdout):
                if stream: stream.close()
        if self.log: self.log.close()
        with self.lock:
            self.process = None
            self.current = {'state': 'stopped', 'message': '시뮬레이션이 종료되었습니다.'}
