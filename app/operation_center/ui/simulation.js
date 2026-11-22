import { MujocoView } from './mujoco.js';

// Both tabs own one backend. Serialize commands, including tab-exit pauses.
let commands = Promise.resolve();
function sendInput(payload) {
  const next = commands.catch(() => {}).then(async () => {
    const response = await fetch('/api/simulation/input', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),
    });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || '시뮬레이션 연결 실패');
    return state;
  });
  commands = next;
  return next;
}

const terrainNotes = { flat:'장애물 없는 평평한 바닥', rocks:'높이 4.5~16cm의 큰 돌과 불규칙한 발 디딤', stairs:'단높이 6cm · 올라가기와 내려가기', mixed:'큰 돌, 평지, 6cm 계단이 섞인 지형' };
const teacherNote = (teacher) => [
  `참조 ${teacher.clip} #${teacher.frame}${teacher.idle_frame ? ' (대기)' : ''}`,
  `전환 ${teacher.switches} · 재고정 ${teacher.reanchors} · 방향 오차 ${(teacher.heading_error_rad*180/Math.PI).toFixed(0)}° · 추종 오차 ${(teacher.tracking_error_m*100).toFixed(0)}cm`,
  teacher.lateral_ignored ? 'A/D 옆걸음은 사람 모션 자료가 없어 무시합니다.' : 'W/S 전진·후진과 우클릭 방향만 따릅니다. A/D 옆걸음 자료는 없습니다.',
  teacher.library_matches_training ? '' : '학습 때와 다른 매칭 라이브러리로 시험 중입니다.',
].filter(Boolean).join('\n');

const unifiedNote = (unified) => [
  `명령 ${unified.twist.map((value)=>value.toFixed(2)).join(' / ')} (전후 m/s · 좌우 m/s · 회전 rad/s)`,
  'W/S 전진·후진, W/S와 함께 A/D는 대각 이동, 우클릭 방향. 신경망 하나가 관측 이력과 목표만 보고 움직입니다(참조 동작·전환 규칙 없음).',
  unified.lateral_ignored ? 'A/D만 누른 옆걸음은 아직 학습되지 않아 무시합니다(모션 자료 공백).' : '',
].filter(Boolean).join('\n');

const dynamicNote = (dynamic) => [
  `지시 ${dynamic.goal} · ${dynamic.source==='text'?'문장':'키·버튼'} / 고리가 내보내는 명령 ${dynamic.command.map((value)=>value.toFixed(2)).join(' / ')} (전후 m/s · 좌우 m/s · 회전 rad/s)`,
  `구성 ${dynamic.parts} · System 1 입력 ${dynamic.inputs} · 실시간의 ${dynamic.realtime.toFixed(2)}배`,
  'W/A/S/D 8방향 · 우클릭 방향(먼저 그쪽으로 돌고 나서 걷습니다) · 자세 버튼 · 문장 지시. 마지막 입력이 우선이고, 지시는 0.3~0.6초 뒤 몸에 나타납니다(0.3초보다 짧은 키 입력은 놓칠 수 있음).',
  '평가와 같은 실행기를 PC GPU 로 돌립니다. Simulation 전용이며 R1 배포 정책과는 별개입니다.',
].join('\n');

// B0(S1 목적): 같은 입력 사상의 5값 명령을 S1 목표로 바꿔 정책 하나에 준다(app/operation_center/dynamic_b0.py).
const b0Note = (dynamic) => {
  const s1 = dynamic.s1;
  const goal = !s1 ? '' : s1.kind==='move'
    ? `이동 목표 · 앞 ${s1.ahead_m.toFixed(2)} m · 왼쪽 ${s1.left_m.toFixed(2)} m · 방향 오차 ${(s1.heading_error_rad*180/Math.PI).toFixed(0)}°`
    : `${s1.kind==='crouch'?'높이 목표':'버티기 목표'} · 골반 ${s1.height_target_m.toFixed(2)} m (지금 ${s1.height_m.toFixed(2)} m)`;
  return [
    `지시 ${dynamic.goal} · ${dynamic.source==='text'?'문장':'키·버튼'} → S1 ${goal}${s1?.deadline_s>0?` · 마감 ${s1.deadline_s.toFixed(1)}s`:''}`,
    `구성 ${dynamic.parts} · 입력 ${dynamic.inputs} · 실시간의 ${dynamic.realtime.toFixed(2)}배`,
    '이동은 명령 속도로 가는 점의 2초 앞(1초마다 갱신), 회전은 회전율×2초, 멈춤은 그 높이 버티기, 앉기·일어서기는 골반 높이 목표입니다. 평지 전용.',
    '알려진 결함(그대로 보여 줌): 버티기에서 제자리 디딤이 잦고 몸 방향이 돌며, 이동 중 두 발이 함께 뜨는 순간이 있습니다.',
  ].join('\n');
};

// SONIC-R1: SONIC 추종기 + calm 인과 합성기 참조(판은 sonic_reference.GENERATOR; app/operation_center/dynamic_sonic.py · sonic_reference.py).
const sonicNote = (dynamic) => {
  const s = dynamic.sonic || {};
  const twist = (s.twist || [0, 0, 0]).map((value)=>value.toFixed(2)).join(' / ');
  const heading = typeof s.heading_error_rad === 'number' ? ` · 참조와 몸 방향 차 ${(s.heading_error_rad*180/Math.PI).toFixed(0)}°` : '';
  return [
    `지시 ${dynamic.goal} · ${dynamic.source==='text'?'문장':'키·버튼'} → 참조 명령 ${twist} (전후 m/s · 좌우 m/s · 회전 rad/s)${heading}`,
    s.limit_note || '',
    `${s.reference_note || '참조: calm 인과 합성기 (학습·평가용, 배포 계획기 아님)'} — 명령은 ${(s.delay_s ?? 0.1).toFixed(1)}초 뒤 효력을 갖고, 걸을 때는 발을 드는 순간에만 읽습니다. 명령이 없으면 두 걸음으로 발을 모아 섭니다.`,
    `구성 ${dynamic.parts} · 입력 ${dynamic.inputs} · 실시간의 ${dynamic.realtime.toFixed(2)}배`,
    s.shape ? `참조 꼴: ${s.shape.cadence} 걸음/s · 폭 ${s.shape.width} m · 발 높이 ${Math.round(s.shape.swing_h*100)} cm` : '',
    s.limits ? `합성기 범위: 앞 ${s.limits.forward} · 뒤 ${s.limits.backward} · 옆 ${s.limits.side} m/s · 회전 ${s.limits.turn} rad/s(넘으면 방향을 유지해 줄임).` : '',
    '앉기·일어서기는 이 모델에 없어 서 있습니다. 평지 전용.',
  ].filter(Boolean).join('\n');
};

const instructionNote = (text) => {
  if (!text || text.state==='none') return '이 구성에는 목표 모델이 없어 문장 지시를 쓸 수 없습니다.';
  if (text.state==='loading') return `목표 모델을 불러오는 중입니다(수십 초). 그동안 키 조작은 됩니다.${text.words?`\n대기 중인 문장: “${text.words}”`:''}`;
  if (text.state==='error') return '목표 모델을 쓸 수 없습니다. 키·버튼 조작은 그대로 됩니다.';
  if (text.awaiting) return `“${text.words}” 읽는 중…`;
  if (text.read) return `“${text.words}” → ${text.read}${text.seconds?` (읽는 데 ${Math.round(text.seconds*1000)} ms)`:''}\n다음 키 입력·방향·자세 버튼이나 새 문장까지 유지됩니다.`;
  return '문장을 보내면 목표 모델이 머리 카메라와 함께 읽어 지시로 바꿉니다. 처음 한 번은 모델을 불러옵니다.';
};

export class SimulationPanel {
  constructor(openLibrary, { prefix = 'simulation', dynamic = false } = {}) {
    this.$ = (id) => document.getElementById(id.replace(/^simulation-/, `${prefix}-`));
    this.root = this.$('simulation-panel');
    this.dynamic = dynamic;
    this.defaultAction = dynamic ? 'dynamic_mode' : 'walking';
    this.paused = false;
    this.visible = false; this.keys = new Set(); this.stopped = false;
    this.launches = []; this.physicsState = 'stopped';
    this.pending = { action:this.defaultAction, environment:'flat' };
    this.$('simulation-library').addEventListener('click', openLibrary);
    // DynamicMode 에서는 초기화·지형 변경 뒤에도 그 모드에 머문다(다른 액션은 Walking 으로 돌아간다).
    const home = () => this.dynamic ? { action:'dynamic_mode', speed:this.$('simulation-dynamic-speed').value } : { action:'walking' };
    const reset = () => { this.keys.clear(); this.launches=[]; this.paintKeys(); this.pending = { ...home(), reset:true }; };
    this.$('simulation-reset').addEventListener('click', reset);
    this.$('simulation-reset-view').addEventListener('click', reset);
    this.$('simulation-environment').addEventListener('change', () => {
      this.keys.clear(); this.launches=[]; this.pending = { environment:this.$('simulation-environment').value, ...home(), reset:true };
      this.paintKeys();
      this.$('simulation-loading').hidden = false;
      this.$('simulation-message').textContent = '새 지형을 준비하고 있습니다.';
    });
    this.$('simulation-instruction-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const words=this.$('simulation-instruction').value.trim();
      if (event.isComposing || !words || this.action!=='dynamic_mode' || this.paused) return;
      this.pending={...this.pending,instruction:words.slice(0,200)};
      this.$('simulation-instruction-status').textContent='문장을 보내는 중…';
    });
    this.$('simulation-instruction').addEventListener('keydown', (event) => {
      if (event.key==='Enter' && !event.shiftKey && !event.isComposing && event.keyCode!==229) {
        event.preventDefault(); this.$('simulation-instruction-form').requestSubmit();
      }
    });
    for (const button of this.root.querySelectorAll('[data-instruction]')) {
      button.addEventListener('click', () => {
        this.$('simulation-instruction').value=button.dataset.instruction;
        this.$('simulation-instruction').focus();
      });
    }
    this.$('simulation-stop')?.addEventListener('click', () => this.stopMotion());
    this.$('simulation-pause')?.addEventListener('click', () => {
      this.paused=!this.paused;
      this.stopMotion();
      this.paintControls();
    });
    this.$('simulation-dynamic-speed').addEventListener('change', () => { this.pending={...this.pending,speed:this.$('simulation-dynamic-speed').value}; });
    this.$('simulation-model-select')?.addEventListener('change', () => {
      this.keys.clear(); this.launches=[]; this.paintKeys();
      this.pending={...this.pending,model:this.$('simulation-model-select').value};
      this.$('simulation-loading').hidden=false;
      this.$('simulation-message').textContent='연결 모델을 바꾸는 중입니다.';
    });
    for (const button of this.$('simulation-dynamic-posture').querySelectorAll('button')) {
      button.addEventListener('click', () => { this.keys.clear(); this.paintKeys(); this.pending={...this.pending,posture:button.dataset.posture}; });
    }
    window.addEventListener('keydown', (event) => {
      if (!this.visible || this.paused || event.ctrlKey || event.altKey || event.metaKey || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
      if (this.dynamic && event.code==='Space' && document.activeElement.tagName!=='BUTTON') { event.preventDefault(); this.stopMotion(); return; }
      if (['KeyW','KeyA','KeyS','KeyD'].includes(event.code)) { event.preventDefault(); this.keys.add(event.code); this.paintKeys(); }
    });
    window.addEventListener('keyup', (event) => { this.keys.delete(event.code); this.paintKeys(); });
    window.addEventListener('blur', () => { this.keys.clear(); this.paintKeys(); });
    this.root.addEventListener('focusin', (event) => {
      if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) { this.keys.clear(); this.paintKeys(); }
    });
    this.$('simulation-view').addEventListener('contextmenu', (event) => event.preventDefault());
    this.$('simulation-launch-speed').addEventListener('input',()=>this.paintLaunchSpeed());
    this.paintLaunchSpeed();
    this.$('simulation-view').addEventListener('pointerdown',(event)=>{
      this.pointerStart = event.button===0 ? {id:event.pointerId,x:event.clientX,y:event.clientY} : null;
    });
    this.$('simulation-view').addEventListener('pointercancel',()=>{this.pointerStart=null;});
    this.$('simulation-view').addEventListener('pointerup',(event)=>{
      const start=this.pointerStart; this.pointerStart=null;
      if (!this.visible || this.paused || this.physicsState!=='running' || !this.view?.loaded || event.button!==0 || event.buttons ||
          !start || start.id!==event.pointerId || Math.hypot(event.clientX-start.x,event.clientY-start.y)>6) return;
      const rect=this.$('simulation-view').getBoundingClientRect();
      if(event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) return;
      const point=this.view.point(event.clientX,event.clientY); if(!point) return;
      this.$('simulation-view').focus({preventScroll:true});
      this.launches.push({point,speed:Number(this.$('simulation-launch-speed').value)});
    });
  }

  paintKeys() { this.root.querySelectorAll('[data-key]').forEach((key)=>key.classList.toggle('pressed',this.keys.has(key.dataset.key))); }

  stopMotion() {
    this.keys.clear(); this.launches=[]; this.paintKeys();
    delete this.pending.instruction;
    delete this.pending.heading;
    this.pending.posture='stand';
  }

  paintControls() {
    if (!this.dynamic) return;
    const ready=this.physicsState==='running' && !this.paused;
    this.$('simulation-pause').textContent=this.paused ? '시뮬레이션 재개' : '일시정지';
    this.$('simulation-pause').setAttribute('aria-pressed', String(this.paused));
    for (const node of this.root.querySelectorAll('[data-motion-control]')) node.disabled=!ready;
    this.$('simulation-instruction').disabled=!ready || this.noLanguage;
    this.$('simulation-send').disabled=!ready || this.noLanguage;
    this.$('simulation-paused').hidden=!this.paused;
  }

  paintLaunchSpeed() {
    const speed=Number(this.$('simulation-launch-speed').value);
    this.$('simulation-launch-speed-value').textContent=`${speed} m/s`;
    this.$('simulation-launch-speed').setAttribute('aria-valuetext',`${speed} 미터 매 초`);
    this.$('simulation-launch-momentum').textContent=`${this.launcher?.mass||1} kg 공 · 운동량 ${(speed*(this.launcher?.mass||1)).toFixed(0)} kg·m/s`;
  }

  async initialize() {
    if (!this.view) {
      this.view=new MujocoView(this.$('simulation-view'),camera=>{this.pending.camera=camera;},heading=>{if(!this.paused)this.pending.heading=heading;});
    }
    this.view.setVisible(true);
  }

  setVisible(visible) {
    const wasVisible=this.visible;
    if (visible===wasVisible) return;
    this.visible=visible; this.keys.clear(); this.launches=[]; this.pointerStart=null; this.paintKeys(); clearTimeout(this.timer);
    if (visible) {
      this.pending={ action:this.defaultAction, environment:this.$('simulation-environment').value,
        ...(this.dynamic ? {speed:this.$('simulation-dynamic-speed').value,posture:'stand'} : {}) };
      this.initialize().then(()=>{ if (this.visible) this.poll(); }).catch((error)=>{ this.$('simulation-message').textContent=error.message; });
    } else if (wasVisible) {
      this.view?.setVisible(false);
      this.pending={};
      sendInput({active:false,axes:[0,0]}).catch(()=>{});
    }
  }

  async poll() {
    if (!this.visible || this.stopped || this.inflight) return;
    this.inflight=true;
    try {
      let forward=Number(this.keys.has('KeyW'))-Number(this.keys.has('KeyS'));
      let side=Number(this.keys.has('KeyA'))-Number(this.keys.has('KeyD'));
      const length=Math.hypot(forward,side); if(length>1){forward/=length;side/=length;}
      const payload={...this.pending,active:!this.paused,axes:this.paused?[0,0]:[forward,side]};
      if(this.launches.length && !this.paused) payload.launch=this.launches.shift();
      this.pending={};
      const state=await sendInput(payload);
      if(this.visible) this.paint(state);
    } catch(error) {
      this.$('simulation-alert').hidden=false; this.$('simulation-alert').textContent=error.message;
    } finally {
      this.inflight=false;
      if(this.visible&&!this.stopped) this.timer=setTimeout(()=>this.poll(),16);
    }
  }

  paint(state) {
    this.physicsState=state.state;
    if(state.launcher && !this.launcher) {
      this.launcher=state.launcher;
      this.$('simulation-launch-speed').min=state.launcher.min_speed;
      this.$('simulation-launch-speed').max=state.launcher.max_speed;
      this.paintLaunchSpeed();
    }
    const shots=state.launcher?.shots||0, hits=state.launcher?.hits||0;
    this.$('simulation-shot-status').textContent=`발사 ${shots} · 적중 ${hits}`;
    this.$('simulation-shot-status').classList.toggle('hit',!!hits);
    this.$('simulation-launch-status').textContent=shots ? `최근 발사 ${state.launcher.last.speed} m/s · ${state.launcher.last.hit?'로봇 적중':'발사 완료'}` : '로봇을 향해 왼쪽 클릭으로 발사하세요.';
    const loading=['loading','stopped'].includes(state.state) || (state.state==='running'&&!this.view?.loaded);
    this.$('simulation-loading').hidden=!loading;
    this.$('simulation-message').textContent=state.message||(this.dynamic?'DynamicMode 모델을 불러오는 중입니다.':'물리 모델과 Walking 정책을 불러오는 중입니다.');
    this.$('simulation-badge').className=`badge ${state.state==='running'?'live':''}`;
    this.$('simulation-badge').textContent={loading:'준비 중',stopped:'준비',running:'시뮬레이션 중',paused:'일시정지',error:'실행 오류'}[state.state]||state.state;
    this.$('simulation-alert').hidden=!state.fallen&&(!state.message||loading);
    this.$('simulation-alert').textContent=state.message||(state.fallen?'로봇이 넘어졌습니다. 초기화로 다시 시작하세요.':'');
    if(state.actions && !this.actionButtons && !this.dynamic) {
      this.actionButtons=new Map();
      for(const [container,actions] of [['simulation-actions',state.actions],['simulation-experiments',state.experiments||[]]]) {
        for(const action of actions) {
          if(action.id==='dynamic_mode') continue;
          const button=document.createElement('button'); button.type='button'; button.textContent=action.label; button.dataset.action=action.id;
          button.addEventListener('click',()=>{
            this.keys.clear();this.paintKeys();this.pending={...this.pending,action:action.id};
          });
          this.$(container).append(button);this.actionButtons.set(action.id,button);
        }
      }
    }
    for(const [id,button] of this.actionButtons||[]) {button.classList.toggle('selected',state.action===id);button.setAttribute('aria-pressed',String(state.action===id));}
    this.$('simulation-action-name').textContent=this.dynamic?'DynamicMode':[...(state.actions||[]),...(state.experiments||[])].find((action)=>action.id===state.action)?.label||'Walking';
    this.action=state.action;
    const policy=this.$('simulation-policy'), teacher=state.teacher, unified=state.unified, dynamic=state.action==='dynamic_mode'?state.dynamic:null;
    policy.textContent=teacher ? `실험 교사 · ${teacher.label}` : unified ? `통합 정책(실험) · ${unified.label}` : dynamic ? `DynamicMode · ${dynamic.label}` : state.action==='dynamic_mode' ? 'DynamicMode 를 준비하고 있습니다.' : state.policy ? `평가 모델 · ${state.policy.label}` : state.action==='walking' ? '제조사 Walking' : '걷기 중에도 액션을 전환할 수 있습니다.';
    policy.title=teacher?.run_id||unified?.run_id||dynamic?.run_id||state.policy?.run_id||'';
    this.$('simulation-experiment-note').hidden=!teacher&&!unified&&!dynamic;
    if(teacher) this.$('simulation-experiment-note').textContent=teacherNote(teacher);
    else if(unified) this.$('simulation-experiment-note').textContent=unifiedNote(unified);
    else if(dynamic) this.$('simulation-experiment-note').textContent=dynamic.model==='b0'?b0Note(dynamic):String(dynamic.model||'').startsWith('sonic')?sonicNote(dynamic):dynamicNote(dynamic);
    this.$('simulation-dynamic').hidden=!this.dynamic && state.action!=='dynamic_mode';
    if(dynamic) {
      this.$('simulation-instruction-status').textContent=instructionNote(dynamic.text);
      this.noLanguage=dynamic.text?.state==='none';
      this.$('simulation-instruction').disabled=this.noLanguage;
      if(document.activeElement!==this.$('simulation-dynamic-speed')) this.$('simulation-dynamic-speed').value=dynamic.speed;
      for(const button of this.$('simulation-dynamic-posture').querySelectorAll('button')) {
        const selected=button.dataset.posture===dynamic.posture;
        button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));
      }
      if(this.dynamic) {
        this.$('simulation-goal').textContent=dynamic.goal;
        this.$('simulation-source').textContent=dynamic.source==='text'?'문장 명령':'직접 조작';
        this.$('simulation-realtime').textContent=`${dynamic.realtime.toFixed(2)}×`;
        this.$('simulation-model').textContent=dynamic.label;
        this.$('simulation-selection-note').textContent=dynamic.selection_note||'';
      }
    }
    const models=this.$('simulation-model-select');
    if(this.dynamic && models && state.dynamic_models && document.activeElement!==models) {
      for(const item of state.dynamic_models) { const option=models.querySelector(`option[value="${item.id}"]`); if(option) option.textContent=item.label; }
      if(state.dynamic_model) models.value=state.dynamic_model;
    }
    this.paintControls();
    this.$('simulation-time').textContent=`${(state.time||0).toFixed(1)}s`;
    this.$('simulation-speed').textContent=`${Math.hypot(...(state.velocity||[0,0]).slice(0,2)).toFixed(2)} m/s`;
    if(state.environment && this.environment!==state.environment) {
      this.environment=state.environment;
      this.$('simulation-environment').value=state.environment;
      this.$('simulation-terrain-note').textContent=terrainNotes[state.environment];
      this.$('simulation-world-label').textContent=state.environments.find((item)=>item.id===state.environment)?.label;
    }
  }

  destroy() {this.stopped=true;clearTimeout(this.timer);this.setVisible(false);this.view?.destroy();}
}
