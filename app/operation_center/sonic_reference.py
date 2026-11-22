"""DynamicMode 의 SONIC-R1 연결이 쓰는 참조 공급원(reference source). Simulation 전용이며 R1 배포와는 별개다.

SONIC 추종기는 매 틱(50 Hz) 지금 프레임부터 0.9 s 앞까지 0.1 s 간격 10장의 참조를 본다. 공급원은 그 프레임을 낸다.
프레임 = qpos (31,) = 뿌리 xyz, 쿼터니언 wxyz, 관절 24(R1 MJCF 순서 = clip_eval.R1_MJ). 작은 인터페이스(ReferenceSource)로
두어 나중에 배포용 계획기(RP1 ONNX)를 같은 자리에 끼운다.

  CalmCausalReference  calm v4.3s 인과 합성기(P38 calm43s_causal.Gen = P34 calm4_causal + '닿는 걸음 가운데 명령에 가장 가까운 것'
                       고르기(v4.1) + 마지막 걸음 제동(v4.2) + 출발 고침: 발자국 + LIP(DCM) + 다리 IK, 걷는 폭 = 서는 폭 0.20 m)를
                       실시간으로. 학습·평가용 참조이며 배포 계획기가 아니다. 생성기는 결정 하나(서기 한 프레임, 무게 옮김,
                       걸음 결정, 걸음·마지막 지지의 한 프레임)마다 그 프레임의 명령만 읽는다(발이 떠 있을 때 멈춤 명령이면 그
                       프레임에서 디딜 곳을 바꾼다). 걸음 결정은 한 번에 계산하므로(PC 최대 수십 ms, 50 Hz 한 틱보다 길다) 별도
                       프로세스가 명령을 그대로 둔다고 보고 MARGIN 만큼 미리 하고, 명령이 바뀌면 그 명령이 효력을 갖는 프레임 이후의
                       첫 결정부터 다시 한다(그 전 결정은 그 명령을 볼 수 없었으므로 그대로). 한 프로세스에서 calm2·calm3 과 섞지
                       않는다(v4 가 calm2_synth.kin() 의 서기 상수를 바꾼다): 합성기 프로세스는 GENERATOR 하나만 불러온다.
                       명령은 받은 뒤 DELAY_TICKS(5틱 = 0.1 s) 뒤에 효력을 가진다: 다시 계산하는 시간을 숨기는 고정 지연이다.
                       결과는 '같은 명령을 0.1 s 늦게 준 인과 합성기'와 프레임 단위로 같고(계산이 늦으면 틱이 기다릴 뿐 값은
                       바뀌지 않는다), 같은 명령 순서의 묶음 생성(calm2_causal.synth)과는 1e-9 안으로 같다.
  ClipReference        고정 클립(시험·양성 대조: 서버 판과 같은 npz 를 그대로).
꼴(CENTER)은 박자 3.0 걸음/s · 폭 0.20 m · 발 높이 5 cm · 보폭 상한 0.25 m(넘으면 박자가 오름)이고, 서기 자세는 v4 의 것(폭
0.20 m, 발바닥 평평, R1 기본과 다리 최대 2°)이라 로봇도 그 자세로 시작한다. 합성기 범위(LIMITS)는 v4.1 · v4.3s 로 사람 같은 무작위 조작을
돌려 확인한 값이다(.agent/operation_center/sonic/tests/stress_synth.py · stress_core.py). 넘는 명령은 방향을 유지해 줄여 쓰고 화면에
표시한다. 생성기가 닿지 않는 걸음을 스스로 줄여 고르므로(v4.1 부터, P38) 아래 SCALES 의 다시 풀기는 마지막 안전망으로만 남는다.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

LOOK = 46                    # 50 Hz frames past the current one the tracker reads (positions to +45; the velocity at +45 needs +46)
DELAY_TICKS = 5              # a command acts 0.1 s after it is told (fixed, so results do not depend on the compute time)
MARGIN = 80                  # 50 Hz frames the synthesizer process keeps decided ahead of the worker (>= LOOK + DELAY_TICKS)
GENERATOR = 'calm43s_causal' # P38's causal generator v4.3s (v4.1's reachable-step choice + v4.2's last-step braking + the start fix)
CENTER = dict(cadence=3.0, width=0.20, swing_h=0.05, ds=0.20, step_len_max=0.25)   # calm v4 shape (P34 USAGE v4)
LIMITS = dict(forward=0.75, backward=0.50, side=0.30, turn=1.0)                 # m/s, m/s, m/s, rad/s the synthesizer is asked for
# (v4.1: with these limits and the walk-turn share, human-like random driving through this wrapper with the retries off gave no
#  generator failure in 240 min (P38 README 3.2). v4 failed 1.2 / min at side 0.30 and was held at 0.20. Re-runs here, also with
#  the tracker and physics: .agent/operation_center/sonic/tests/results/stress41_* (v4.1) and stress43s_* (v4.3s)
#  (tests/stress_synth.py, tests/stress_core.py))
WAIT_S = 20.0                # longest wait for the synthesizer before the worker reports it stuck
SCALES = tuple(json.loads(os.environ.get('P36_SYNTH_SCALES', '[0.75, 0.5, 0.25, 0.0]')))   # a decision the generator cannot
# solve is solved again with the command scaled by these (the variable is a test hook: [] turns the retries off)


class ReferenceFailed(RuntimeError):
    """The reference could not go on (the synthesizer gave up or stopped); the core restarts it standing."""


class ReferenceSource:
    """참조 공급원. tell(틱, 명령) -> ready(틱) -> qpos[틱 - base .. 틱 - base + LOOK] 이 그 틱에 알 수 있는 최종 값이다."""
    name = ''
    note = ''
    length = None            # frames of a fixed clip (None: endless)
    base = 0                 # global 50 Hz frame of qpos[0]

    def reset(self) -> None:
        raise NotImplementedError

    def tell(self, tick: int, twist) -> tuple:
        """(vx, vy, wz) body frame from tick on; returns the twist the source will use (after its limits)."""
        raise NotImplementedError

    def ready(self, tick: int):
        """Make frames up to tick + LOOK final for what is known at tick. Returns the first local frame index whose value
        changed since the previous call (None: none)."""
        raise NotImplementedError

    def shift(self, frames: int) -> None:
        """Drop `frames` old frames from the front of qpos (base += frames)."""
        raise NotImplementedError

    def state(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def limit_twist(twist, limits=LIMITS):
    """The synthesizer's range: forward / backward / sideways speed scaled together (the direction is kept), turn rate
    clipped, and walking + turning at once sharing the range (|v| / v_lim(direction) + |w| / w_lim <= 1; the key
    vocabulary never asks both, a sentence or a later planner might). Returns (twist, note), note '' when unchanged."""
    asked = tuple(float(v) for v in twist)
    vx, vy, wz = asked
    scale = 1.0
    if vx > limits['forward']:
        scale = min(scale, limits['forward'] / vx)
    if vx < -limits['backward']:
        scale = min(scale, limits['backward'] / -vx)
    if abs(vy) > limits['side']:
        scale = min(scale, limits['side'] / abs(vy))
    vx, vy = vx * scale, vy * scale
    w = max(-limits['turn'], min(limits['turn'], wz))
    speed = math.hypot(vx, vy)
    if speed > 1e-9 and abs(w) > 1e-9:
        ux, uy = vx / speed, vy / speed
        reach = min(((limits['forward'] if ux > 0 else limits['backward']) / abs(ux)) if abs(ux) > 1e-9 else 9.0,
                    (limits['side'] / abs(uy)) if abs(uy) > 1e-9 else 9.0)
        share = speed / reach + abs(w) / limits['turn']
        if share > 1.0:
            vx, vy, w = vx / share, vy / share, w / share
    out = (round(vx, 4), round(vy, 4), round(w, 4))
    note = ''
    if any(abs(a - b) > 1e-4 for a, b in zip(out, asked)):
        note = (f"합성기 범위로 줄임: 앞 {limits['forward']:g} · 뒤 {limits['backward']:g} · 옆 {limits['side']:g} m/s · "
                f"회전 {limits['turn']:g} rad/s, 걸으며 돌 때는 둘이 범위를 나눔 "
                f"({asked[0]:g}, {asked[1]:g}, {asked[2]:g} → {out[0]:g}, {out[1]:g}, {out[2]:g})")
    return out, note


# ------------------------------------------------------------------ the synthesizer process
class _Online:
    """calm2_causal.Gen driven online (inside the synthesizer process). Global 100 Hz frame = base + local frame.
    A snapshot of the generator's decision state is kept before every tick() (one standing frame or one whole segment);
    change(k, twist) restores the first snapshot at or after k (decisions before k never read the command at k)."""
    SCALARS = ('k', 'state', 'closing', 'sw', 'k_on', 'psi_now', 'z_now', 'wtrans', 'close_ok',
               'fin', 'resume', 'n_since_init', 'stp', 'mid')       # (v4: stp / mid / fin are replaced, never changed in place)
    VECTORS = ('vs', 'xi', 'x', 'c_now', 'e_now', 'p_now')
    LISTS = ('steps', 'spans', 'feet', 'inits', 'fins')
    FRAMES = ('com', 'zmpt', 'dcm', 'e', 'c', 'psi', 'zp', 'Q', 'vsm', 'moving')

    def __init__(self, CC, P, seconds):
        self.CC, self.P, self.seconds = CC, P, seconds
        self.base = 0
        self.g = CC.Gen([(seconds, 0.0, 0.0, 0.0)], P).start()
        self.fresh = {name: getattr(self.g, name)[0].copy() for name in self.FRAMES}
        self.snaps, self.keys = [], []
        self.renewed = 0
        self.reductions = []                           # decisions solved with a scaled command: {k (local), told, scale, state}

    def _save(self):
        g = self.g
        s = {name: getattr(g, name) for name in self.SCALARS if hasattr(g, name)}
        s.update({name: np.array(getattr(g, name), float).copy() for name in self.VECTORS if hasattr(g, name)})
        s.update({name: list(getattr(g, name)) for name in self.LISTS if hasattr(g, name)})
        s.update(stand=dict(g.stand), Dprev=dict(g.Dprev), prof=dict(g.prof), stats=dict(g.stats), stand0=list(g.fs.stand0))
        if hasattr(g, 'settles'):                      # (v4: its rows are changed in place)
            s['settles'] = [list(r) for r in g.settles]
        return s

    def _restore(self, s):
        g = self.g
        for name in self.SCALARS + self.VECTORS + self.LISTS:
            if name in s:
                value = s[name]
                setattr(g, name, value.copy() if isinstance(value, (np.ndarray, list)) else value)
        g.stand, g.Dprev, g.prof, g.stats = dict(s['stand']), dict(s['Dprev']), dict(s['prof']), dict(s['stats'])
        g.fs.steps, g.fs.stand0 = g.steps, list(s['stand0'])
        if 'settles' in s:
            g.settles = [list(r) for r in s['settles']]
            g.fs.settles = g.settles
        for name in self.FRAMES:                       # the frames from here on as a fresh generator has them
            getattr(g, name)[g.k:] = self.fresh[name]

    def decided(self) -> int:
        """Global 100 Hz frames below this one are final for the commands told so far."""
        return self.base + self.g.k

    def tick(self) -> None:
        """One decision of the generator. If it cannot be solved (a landing out of the legs' reach for this command from
        this state), the same decision is solved again with the command at this frame scaled by SCALES (0 = stopping
        steps); the told command is kept for later decisions. The generator's own code is not changed, and the result is a
        function of the state and the command only (deterministic). If nothing solves, ReferenceFailed."""
        g = self.g
        if g.k >= g.N - 400:
            raise ReferenceFailed('참조 버퍼가 찼습니다(서지 않고 너무 오래 걸었습니다).')
        saved = self._save()
        self.keys.append(g.k)
        self.snaps.append(saved)
        try:
            g.tick()
            return
        except Exception as error:                     # noqa: BLE001
            failed = error
        k = saved['k']
        told = g.cmd[k].copy()
        for scale in SCALES:
            self._restore(saved)
            g.cmd[k] = told * scale
            try:
                g.tick()
                solved = True
            except Exception:                          # noqa: BLE001
                solved = False
            g.cmd[k] = told
            if solved:
                self.reductions.append(dict(k=int(k), told=[round(float(v), 3) for v in told], scale=scale, state=saved['state']))
                return
        self._restore(saved)
        raise ReferenceFailed(f'참조 합성기가 이 결정을 풀지 못했습니다({failed}).')

    def change(self, k_global: int, twist) -> None:
        k = k_global - self.base
        i = bisect.bisect_left(self.keys, k)
        if i < len(self.keys):
            self._restore(self.snaps[i])
            self.reductions = [r for r in self.reductions if r['k'] < self.keys[i]]   # (those of the discarded future)
            del self.snaps[i:], self.keys[i:]
        self.g.cmd[max(k, 0):] = np.asarray(twist, float)

    def frame(self, k_global: int) -> np.ndarray:
        return self.g.Q[k_global - self.base]

    def forget(self, k_global: int) -> list:
        """No command can act before global frame k any more: drop older snapshots and steps foot_state no longer needs,
        and renew the buffer at a standing moment once it is half used (P34 Stream._renew). Returns the reductions that
        are now in the past (final), with global frames."""
        g, k = self.g, k_global - self.base
        done = [dict(r, k=r['k'] + self.base) for r in self.reductions if r['k'] <= k]
        self.reductions = [r for r in self.reductions if r['k'] > k]
        i = bisect.bisect_left(self.keys, k)
        if i:
            del self.snaps[:i], self.keys[:i]
        t_old = k * self.CC.H - 1.0
        drop = 0
        for st in ([] if hasattr(g, 'settles') else g.steps[:-4]):   # (v4: foot_state's settles need the old landings)
            if st['t1'] >= t_old:
                break
            drop += 1
        if drop:                                       # foot_state starts from stand0: the trimmed steps' last footprints
            for st in g.steps[:drop]:
                g.fs.stand0[st['f']] = st['dst']
            del g.steps[:drop]
            g.fs.steps = g.steps
        if len(g.spans) > 4:
            del g.spans[:-4]
        ended = g.spans[-1][1] if g.spans else 0
        if (g.state == 'STAND' and k > g.N // 2 and ended < k - 200 and not self.reductions
                and all(s['state'] == 'STAND' for s in self.snaps)):    # standing since well before now: renew at k
            self._renew(k)
        return done

    def _renew(self, k):
        CC, old = self.CC, self.g
        g = CC.Gen([(self.seconds, 0.0, 0.0, 0.0)], self.P)
        g.stand = dict(B=old.stand['B'].copy(), psi=old.stand['psi'])
        g.fs.stand0 = [g._stand_fp(0), g._stand_fp(1)]
        g.feet = [g.fs.stand0[0], g.fs.stand0[1]]
        g.start()
        g.psi_now = g.stand['psi']
        g.cmd[:old.N - k] = old.cmd[k:]               # commands already told for later frames
        horizon = old.k - k
        self.g, self.base = g, self.base + k
        self.snaps, self.keys = [], []
        self.fresh = {name: getattr(g, name)[0].copy() for name in self.FRAMES}
        while g.k < horizon:
            self.tick()
        self.renewed += 1


def _synth_main(conn, harness, shape, seconds, margin):
    """The synthesizer process. Messages in: ('tick', epoch, n) / ('cmd', epoch, j_eff, twist) / ('reset', epoch, tick) / ('stop',).
    Messages out: ('ready',) / ('frames', epoch, version, j0, block (n, 31)) / ('info', epoch, what, data) /
    ('failed', epoch, text): the generator cannot go on in this epoch (it waits for a reset; the process stays)."""
    import ctypes
    import signal
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG: end with the DynamicMode worker
    if os.getppid() == 1:
        return
    os.environ.setdefault('R1_BODY', 'r1wb')
    if harness not in sys.path:
        sys.path.insert(0, harness)
    try:
        import importlib
        CC = importlib.import_module(GENERATOR)
        P = dict(CC.CDEF, **shape)
        online = _Online(CC, P, seconds)
        conn.send(('ready',))
    except Exception as error:                        # noqa: BLE001
        conn.send(('error', f'참조 합성기를 시작하지 못했습니다: {type(error).__name__}: {error}'))
        return
    epoch, version, now, sent, broken = 0, 0, 0, 0, False

    def send_until(j_max):
        nonlocal sent
        while online.decided() <= 2 * j_max:
            online.tick()
        if j_max >= sent:
            block = np.stack([online.frame(2 * j) for j in range(sent, j_max + 1)])
            conn.send(('frames', epoch, version, sent, block))
            sent = j_max + 1

    try:
        while True:
            target = now + margin
            if conn.poll(0 if (sent <= target and not broken) else None):
                message = conn.recv()
                kind = message[0]
                if kind == 'stop':
                    return
                if kind == 'reset':                   # a fresh generator standing at frame `tick` (worker numbering kept)
                    epoch, tick = message[1], int(message[2]) if len(message) > 2 else 0
                    online = _Online(CC, P, seconds)
                    online.base = 2 * tick
                    version, now, sent, broken = 0, tick, tick, False
                    continue
                if message[1] != epoch or broken:
                    continue
                try:
                    if kind == 'tick':
                        now = max(now, int(message[2]))
                        renewed = online.renewed
                        done = online.forget(2 * now)
                        if online.renewed != renewed:
                            conn.send(('info', epoch, 'renew', online.base))
                        if done:
                            conn.send(('info', epoch, 'reduced', done))
                    elif kind == 'cmd':
                        _, _, j_eff, twist = message
                        send_until(j_eff - 1 + LOOK)  # what the worker still reads of the regime in force until j_eff
                        version += 1
                        online.change(2 * j_eff, twist)
                        sent = min(sent, j_eff)
                except ReferenceFailed as error:
                    conn.send(('failed', epoch, str(error)))
                    broken = True
                continue
            want = min(sent + 24, target)             # the next block
            try:
                if online.decided() <= 2 * want:
                    online.tick()                     # one decision at a time: messages are read between decisions
                    continue
                send_until(want)
            except ReferenceFailed as error:
                conn.send(('failed', epoch, str(error)))
                broken = True
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return
    except Exception as error:                        # noqa: BLE001
        try:
            conn.send(('error', f'참조 합성기 오류: {type(error).__name__}: {error}'))
        except (BrokenPipeError, OSError):
            pass


# ------------------------------------------------------------------ the worker side
class CalmCausalReference(ReferenceSource):
    name = 'calm v4.3s 인과 합성기'
    note = '참조: calm v4.3s 인과 합성기 (학습·평가용, 배포 계획기 아님)'

    def __init__(self, harness, shape=None, first='left', delay_ticks=DELAY_TICKS, margin=MARGIN, seconds=600.0,
                 capacity=30000, limits=LIMITS):
        self.shape = dict(CENTER if shape is None else shape, first=first)
        self.delay, self.margin, self.capacity, self.limits = int(delay_ticks), int(margin), int(capacity), dict(limits)
        assert self.margin >= LOOK + self.delay
        self.harness, self.seconds = os.path.abspath(harness), float(seconds)
        self.qpos = np.zeros((self.capacity, 31))
        self.epoch = 0
        self._spawn()
        self._clear()

    def _spawn(self):
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_synth_main, args=(child, self.harness, self.shape, self.seconds, self.margin),
                                       daemon=True, name='calm4-synth')
        self.process.start()
        child.close()
        self._started = False

    def _clear(self):
        self.base, self.valid, self.version, self.told_version = 0, 0, 0, 0
        self.changes, self.pending, self.dirty = deque(), {}, None
        self.twist, self.asked, self.limit_note = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ''
        self.last_tick, self.waited_s, self.waits, self.renews = -1, 0.0, 0, 0
        self.reductions, self.last_reduction = 0, None

    def _receive(self, block):
        if not block and not self.conn.poll():
            return False
        try:
            if block and not self.conn.poll(WAIT_S):
                raise ReferenceFailed('참조 합성기가 응답하지 않습니다.')
            message = self.conn.recv()
        except (EOFError, OSError) as error:
            raise ReferenceFailed(f'참조 합성기 프로세스가 끝났습니다({type(error).__name__}).') from error
        if message[0] == 'error':
            raise ReferenceFailed(message[1])
        if message[0] == 'ready':
            self._started = True
            return True
        if message[0] == 'info':
            if message[1] == self.epoch and message[2] == 'renew':
                self.renews += 1
            if message[1] == self.epoch and message[2] == 'reduced':
                self.reductions += len(message[3])
                self.last_reduction = message[3][-1]
            return True
        if message[0] == 'failed':
            if message[1] == self.epoch:
                raise ReferenceFailed(message[2])
            return True
        _, epoch, version, j0, block_ = message
        if epoch != self.epoch:
            return True
        if version == self.version:
            self._write(j0, block_)
        elif version > self.version:
            start, parts = self.pending.setdefault(version, (j0, []))
            parts.append(block_)
        return True

    def _write(self, j0, block):
        a = j0 - self.base
        if a < 0:
            block, a = block[-a:], 0
        n = len(block)
        if a > self.valid or a + n > self.capacity:
            raise RuntimeError(f'참조 프레임 순서가 어긋났습니다({a} > {self.valid}).')
        if n:
            self.qpos[a:a + n] = block
            self.valid = max(self.valid, a + n)
            self.dirty = a if self.dirty is None else min(self.dirty, a)

    def _send(self, message):
        try:
            self.conn.send(message)
        except (BrokenPipeError, EOFError, OSError) as error:
            raise ReferenceFailed(f'참조 합성기 프로세스에 보낼 수 없습니다({type(error).__name__}).') from error

    def _pending_end(self, version):
        if version not in self.pending:
            return -1
        start, parts = self.pending[version]
        return start + sum(len(p) for p in parts)

    def reset(self):
        self.restart(0)

    def restart(self, tick):
        """A fresh generator standing still, its first frame at `tick` (the worker keeps counting ticks). A synthesizer
        process that has ended is started again."""
        if not self.process.is_alive():
            self._spawn()
            wait_started(self)
        self.epoch += 1
        self._send(('reset', self.epoch, int(tick)))
        self._clear()
        self.base = int(tick)

    def tell(self, tick, twist):
        self.asked = tuple(round(float(v), 4) for v in twist)
        twist, self.limit_note = limit_twist(twist, self.limits)
        if twist != self.twist:
            self.twist = twist
            self.told_version += 1
            j_eff = tick + self.delay
            self._send(('cmd', self.epoch, j_eff, twist))
            self.changes.append((self.told_version, j_eff))
        return twist

    def ready(self, tick):
        if tick != self.last_tick:
            self._send(('tick', self.epoch, tick))
            self.last_tick = tick
        while self._receive(False):
            pass
        began = time.perf_counter()
        while self.changes and self.changes[0][1] <= tick:
            version, j_eff = self.changes[0]
            while self._pending_end(version) <= tick + LOOK:
                self._receive(True)
            start, parts = self.pending.pop(version)
            self.version = version
            self.valid = start - self.base                # frames from j_eff on belong to the new regime
            self._write(start, np.concatenate(parts))
            self.changes.popleft()
            for v in [v for v in self.pending if v < version]:
                del self.pending[v]
        while self.valid <= tick - self.base + LOOK:
            self._receive(True)
        spent = time.perf_counter() - began
        if spent > 0.002:
            self.waited_s += spent
            self.waits += 1
        dirty, self.dirty = self.dirty, None
        return dirty

    def shift(self, frames):
        self.qpos[:-frames] = self.qpos[frames:]
        self.base += frames
        self.valid -= frames

    def state(self):
        return {'source': self.name, 'note': self.note, 'twist': list(self.twist), 'limit_note': self.limit_note,
                'shape': {k: self.shape[k] for k in ('cadence', 'width', 'swing_h')}, 'limits': self.limits,
                'delay_s': round(self.delay * 0.02, 2), 'waits': self.waits, 'waited_s': round(self.waited_s, 3),
                'renews': self.renews, 'reductions': self.reductions}

    def close(self):
        try:
            self.conn.send(('stop',))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)


class ClipReference(ReferenceSource):
    """A fixed clip (qpos_m1 at 50 Hz), as the server panels play it (tests and positive controls)."""
    name = 'clip'
    note = '참조: 고정 클립(시험용)'

    def __init__(self, qpos):
        self.qpos = np.asarray(qpos, float)
        self.length = len(self.qpos)
        self.twist = (0.0, 0.0, 0.0)
        self._first = True

    def reset(self):
        self._first = True

    def tell(self, tick, twist):
        return self.twist

    def ready(self, tick):
        first, self._first = self._first, False
        return 0 if first else None

    def shift(self, frames):
        raise RuntimeError('a fixed clip is not shifted')

    def state(self):
        return {'source': self.name, 'note': self.note, 'twist': list(self.twist), 'limit_note': '', 'delay_s': 0.0}


def wait_started(reference, timeout=60.0):
    """Block until the synthesizer process has imported its harness (the worker shows 'loading' meanwhile)."""
    if not isinstance(reference, CalmCausalReference):
        return
    began = time.monotonic()
    while not reference._started:
        if time.monotonic() - began > timeout:
            raise RuntimeError('참조 합성기가 시작되지 않습니다.')
        reference._receive(True)


def heading_of(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
