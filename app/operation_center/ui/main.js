import { RobotView } from './robot.js';

const $ = (id) => document.getElementById(id);
const labels = { disconnected: '연결 안 됨', connecting: '연결 중', connected: '연결됨', stale: '수신 중단', error: '연결 실패' };
const groups = { left_hip: '왼쪽 다리', right_hip: '오른쪽 다리', waist: '허리', left_shoulder: '왼쪽 팔', right_shoulder: '오른쪽 팔', head: '머리' };
const names = { hip_pitch: '고관절 Pitch', hip_roll: '고관절 Roll', hip_yaw: '고관절 Yaw', knee: '무릎', ankle_pitch: '발목 Pitch', ankle_roll: '발목 Roll', waist_roll: '허리 Roll', waist_yaw: '허리 Yaw', shoulder_pitch: '어깨 Pitch', shoulder_roll: '어깨 Roll', shoulder_yaw: '어깨 Yaw', elbow: '팔꿈치', wrist_roll: '손목 Roll', head_pitch: '머리 Pitch', head_yaw: '머리 Yaw' };
let view;
let lastState;
let pending = false;
let stopped = false;
let failed = false;
let seeded = false;
let actionError = '';

function degrees(radians) { return `${(radians * 180 / Math.PI).toFixed(1)}°`; }

function paint(state) {
  lastState = state;
  const { status, pose } = state;
  const live = status === 'connected';
  $('connection-label').textContent = labels[status] || '연결 확인';
  $('connection-badge').className = `badge ${live ? 'live' : status === 'connecting' || status === 'stale' ? 'waiting' : status === 'error' ? 'error' : ''}`;
  $('connection-badge').dataset.state = status;
  $('connection-badge').title = state.message;
  $('connection-state-mark').setAttribute('d', live ? 'M3.5 8 6.5 11 12.5 5' : status === 'error' ? 'M5 5l6 6M11 5l-6 6' : status === 'stale' ? 'M8 3.5v5M8 11.5v1' : 'M4 8h8');
  $('connection-message').textContent = actionError || state.message;
  $('rate').textContent = live ? `${state.hz.toFixed(1)} Hz` : '—';
  $('age').textContent = state.age_ms == null ? '—' : state.age_ms < 1000 ? `${state.age_ms} ms` : `${(state.age_ms / 1000).toFixed(1)} s`;
  $('telemetry-dot').classList.toggle('live', live);
  ['roll', 'pitch', 'yaw'].forEach((axis, index) => { $(axis).textContent = pose?.rpy ? degrees(pose.rpy[index]) : '—'; });
  const active = status === 'connecting' || status === 'connected' || status === 'stale';
  $('disconnect-button').hidden = !active;
  $('connect-button').hidden = active;
  $('connect-caption').textContent = status === 'error' ? '다시 연결' : '로봇 연결';
  $('robot-host').disabled = active || pending;
  $('pose-indicator').className = `pose-indicator ${live ? 'live' : pose ? 'stale' : ''}`;
  const missing = pose?.missing?.length ?? 0;
  $('pose-label').textContent = live ? (missing ? `LIVE · 관절 ${26 - missing}/26 · 골반 기준` : 'LIVE · 관절 자세 · 골반 기준') : pose ? '마지막 수신 자세 · 실시간 아님' : '기준 모델 · 실제 자세 미수신';
  $('footer-status').textContent = live ? (missing ? `관절 상태 수신 중 · 미수신 ${missing}개` : '실시간 관절 상태 수신 중') : pose ? '마지막 수신 자세 표시 중' : status === 'connecting' ? '로봇 상태 연결 중' : '실제 자세 수신 대기';
  $('footer-status-dot').classList.toggle('live', live);
  if (view?.loaded && (view.tick !== pose?.tick || failed)) view.setPose(pose);
  document.querySelectorAll('[data-joint]').forEach((output) => {
    const angle = pose?.joints[output.dataset.joint];
    output.textContent = angle === undefined ? '—' : degrees(angle);
  });
  failed = false;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? { cache: 'no-store' } : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `요청 실패 (${response.status})`);
  return data;
}

async function action(path, body) {
  if (pending) return;
  pending = true;
  actionError = '';
  $('connect-button').disabled = $('disconnect-button').disabled = true;
  try { paint(await request(path, body)); }
  catch (error) { actionError = error.message; $('connection-message').textContent = actionError; }
  finally {
    pending = false;
    $('connect-button').disabled = $('disconnect-button').disabled = false;
    if (lastState) $('robot-host').disabled = ['connecting', 'connected', 'stale'].includes(lastState.status);
  }
}

$('connect-form').addEventListener('submit', (event) => {
  event.preventDefault();
  action('/api/connect', { host: $('robot-host').value.trim() });
});
$('disconnect-button').addEventListener('click', () => action('/api/disconnect', {}));
$('robot-host').addEventListener('input', () => { actionError = ''; });

async function poll() {
  if (stopped) return;
  try {
    const state = await request('/api/status');
    if (!seeded) {
      if (!$('robot-host').value && state.host) $('robot-host').value = state.host;
      seeded = true;
    }
    if (!pending) paint(state);
  } catch (error) {
    failed = true;
    paint({ ...(lastState || {}), status: 'error', message: '앱의 상태 서버에 연결할 수 없습니다.', hz: 0, age_ms: lastState?.age_ms ?? null });
  }
  if (!stopped) setTimeout(poll, 100);
}

async function initializeView() {
  try {
    view = new RobotView($('robot-view'));
    const joints = await view.load();
    for (const name of joints) {
      for (const [prefix, title] of Object.entries(groups)) {
        if (name.startsWith(prefix) && (prefix === 'waist' ? name === 'waist_roll_joint' : name.endsWith('pitch_joint'))) {
          const header = document.createElement('div');
          header.className = 'joint-group'; header.textContent = title;
          $('joint-values').append(header); break;
        }
      }
      const row = document.createElement('div'); row.className = 'joint-row';
      const label = document.createElement('span');
      const shortName = name.replace(/^(left|right)_/, '').replace(/_joint$/, '');
      label.textContent = names[shortName] || shortName;
      const output = document.createElement('output'); output.dataset.joint = name; output.textContent = '—';
      row.append(label, output); $('joint-values').append(row);
    }
    $('model-loading').hidden = true;
    if (lastState) paint(lastState);
    for (const name of ['perspective', 'front', 'side']) {
      $(`view-${name}`).addEventListener('click', () => {
        view.preset(name);
        for (const item of ['perspective', 'front', 'side']) {
          $(`view-${item}`).classList.toggle('active', item === name);
          $(`view-${item}`).setAttribute('aria-pressed', String(item === name));
        }
      });
    }
    $('view-reset').addEventListener('click', () => $('view-perspective').click());
    $('show-grid').addEventListener('change', (event) => { view.grid.visible = event.target.checked; view.dirty = true; });
    $('show-joints').addEventListener('change', (event) => { view.setMarkersVisible(event.target.checked); });
  } catch (error) {
    $('model-loading').querySelector('.spinner').hidden = true;
    $('loading-message').textContent = `3D 화면을 열지 못했습니다: ${error.message}`;
  }
}
initializeView();
poll();
window.addEventListener('pagehide', () => { stopped = true; view?.destroy(); });
