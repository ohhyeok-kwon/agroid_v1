import { MujocoView } from './mujoco.js';

const $ = (id) => document.getElementById(id);
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

const instructionNote = (text) => {
  if (!text || text.state==='none') return '이 구성에는 목표 모델이 없어 문장 지시를 쓸 수 없습니다.';
  if (text.state==='loading') return `목표 모델을 불러오는 중입니다(수십 초). 그동안 키 조작은 됩니다.${text.words?`\n대기 중인 문장: “${text.words}”`:''}`;
  if (text.state==='error') return '목표 모델을 쓸 수 없습니다. 키·버튼 조작은 그대로 됩니다.';
  if (text.awaiting) return `“${text.words}” 읽는 중…`;
  if (text.read) return `“${text.words}” → ${text.read}${text.seconds?` (읽는 데 ${Math.round(text.seconds*1000)} ms)`:''}\n다음 키 입력·방향·자세 버튼이나 새 문장까지 유지됩니다.`;
  return '문장을 보내면 목표 모델이 머리 카메라와 함께 읽어 지시로 바꿉니다. 처음 한 번은 모델을 불러옵니다.';
};

export class SimulationPanel {
  constructor(openLibrary) {
    this.visible = false; this.keys = new Set(); this.stopped = false;
    this.launches = []; this.physicsState = 'stopped';
    this.pending = { action:'walking', environment:'flat' };
    $('simulation-library').addEventListener('click', openLibrary);
    // DynamicMode 에서는 초기화·지형 변경 뒤에도 그 모드에 머문다(다른 액션은 Walking 으로 돌아간다).
    const home = () => this.action==='dynamic_mode' ? { action:'dynamic_mode', speed:$('simulation-dynamic-speed').value } : { action:'walking' };
    const reset = () => { this.keys.clear(); this.launches=[]; this.paintKeys(); this.pending = { ...home(), reset:true }; };
    $('simulation-reset').addEventListener('click', reset);
    $('simulation-reset-view').addEventListener('click', reset);
    $('simulation-environment').addEventListener('change', () => {
      this.keys.clear(); this.launches=[]; this.pending = { environment:$('simulation-environment').value, ...home(), reset:true };
      $('simulation-loading').hidden = false;
      $('simulation-message').textContent = '새 지형을 준비하고 있습니다.';
    });
    $('simulation-instruction-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const words=$('simulation-instruction').value.trim();
      if (!words || this.action!=='dynamic_mode') return;
      this.pending={...this.pending,instruction:words.slice(0,200)};
      $('simulation-instruction').select();
    });
    $('simulation-dynamic-speed').addEventListener('change', () => { this.pending={...this.pending,speed:$('simulation-dynamic-speed').value}; });
    for (const button of $('simulation-dynamic-posture').querySelectorAll('button')) {
      button.addEventListener('click', () => { this.keys.clear(); this.paintKeys(); this.pending={...this.pending,posture:button.dataset.posture}; });
    }
    window.addEventListener('keydown', (event) => {
      if (!this.visible || event.ctrlKey || event.altKey || event.metaKey || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
      if (['KeyW','KeyA','KeyS','KeyD'].includes(event.code)) { event.preventDefault(); this.keys.add(event.code); this.paintKeys(); }
    });
    window.addEventListener('keyup', (event) => { this.keys.delete(event.code); this.paintKeys(); });
    window.addEventListener('blur', () => { this.keys.clear(); this.paintKeys(); });
    $('simulation-view').addEventListener('contextmenu', (event) => event.preventDefault());
    $('simulation-launch-speed').addEventListener('input',()=>this.paintLaunchSpeed());
    this.paintLaunchSpeed();
    $('simulation-view').addEventListener('pointerdown',(event)=>{
      this.pointerStart = event.button===0 ? {id:event.pointerId,x:event.clientX,y:event.clientY} : null;
    });
    $('simulation-view').addEventListener('pointercancel',()=>{this.pointerStart=null;});
    $('simulation-view').addEventListener('pointerup',(event)=>{
      const start=this.pointerStart; this.pointerStart=null;
      if (!this.visible || this.physicsState!=='running' || !this.view?.loaded || event.button!==0 || event.buttons ||
          !start || start.id!==event.pointerId || Math.hypot(event.clientX-start.x,event.clientY-start.y)>6) return;
      const rect=$('simulation-view').getBoundingClientRect();
      if(event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) return;
      const point=this.view.point(event.clientX,event.clientY); if(!point) return;
      $('simulation-view').focus({preventScroll:true});
      this.launches.push({point,speed:Number($('simulation-launch-speed').value)});
    });
  }

  paintKeys() { document.querySelectorAll('[data-key]').forEach((key)=>key.classList.toggle('pressed',this.keys.has(key.dataset.key))); }

  paintLaunchSpeed() {
    const speed=Number($('simulation-launch-speed').value);
    $('simulation-launch-speed-value').textContent=`${speed} m/s`;
    $('simulation-launch-speed').setAttribute('aria-valuetext',`${speed} 미터 매 초`);
    $('simulation-launch-momentum').textContent=`${this.launcher?.mass||1} kg 공 · 운동량 ${(speed*(this.launcher?.mass||1)).toFixed(0)} kg·m/s`;
  }

  async initialize() {
    if (!this.view) {
      this.view=new MujocoView($('simulation-view'),camera=>{this.pending.camera=camera;},heading=>{this.pending.heading=heading;});
    }
    this.view.setVisible(true);
  }

  setVisible(visible) {
    const wasVisible=this.visible;
    this.visible=visible; this.keys.clear(); this.launches=[]; this.pointerStart=null; this.paintKeys(); clearTimeout(this.timer);
    if (visible) {
      this.initialize().then(()=>{ if (this.visible) this.poll(); }).catch((error)=>{ $('simulation-message').textContent=error.message; });
    } else if (wasVisible) {
      this.view?.setVisible(false);
      fetch('/api/simulation/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({active:false,axes:[0,0]})}).catch(()=>{});
    }
  }

  async poll() {
    if (!this.visible || this.stopped || this.inflight) return;
    this.inflight=true;
    try {
      let forward=Number(this.keys.has('KeyW'))-Number(this.keys.has('KeyS'));
      let side=Number(this.keys.has('KeyA'))-Number(this.keys.has('KeyD'));
      const length=Math.hypot(forward,side); if(length>1){forward/=length;side/=length;}
      const payload={...this.pending,active:true,axes:[forward,side]};
      if(this.launches.length) payload.launch=this.launches.shift();
      this.pending={};
      const response=await fetch('/api/simulation/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const state=await response.json(); if(!response.ok) throw new Error(state.error||'시뮬레이션 연결 실패');
      this.paint(state);
    } catch(error) {
      $('simulation-alert').hidden=false; $('simulation-alert').textContent=error.message;
    } finally {
      this.inflight=false;
      if(this.visible&&!this.stopped) this.timer=setTimeout(()=>this.poll(),16);
    }
  }

  paint(state) {
    this.physicsState=state.state;
    if(state.launcher && !this.launcher) {
      this.launcher=state.launcher;
      $('simulation-launch-speed').min=state.launcher.min_speed;
      $('simulation-launch-speed').max=state.launcher.max_speed;
      this.paintLaunchSpeed();
    }
    const shots=state.launcher?.shots||0, hits=state.launcher?.hits||0;
    $('simulation-shot-status').textContent=`발사 ${shots} · 적중 ${hits}`;
    $('simulation-shot-status').classList.toggle('hit',!!hits);
    $('simulation-launch-status').textContent=shots ? `최근 발사 ${state.launcher.last.speed} m/s · ${state.launcher.last.hit?'로봇 적중':'발사 완료'}` : '로봇을 향해 왼쪽 클릭으로 발사하세요.';
    const loading=['loading','stopped'].includes(state.state) || (state.state==='running'&&!this.view?.loaded);
    $('simulation-loading').hidden=!loading;
    $('simulation-message').textContent=state.message||'물리 모델과 Walking 정책을 불러오는 중입니다.';
    $('simulation-badge').className=`badge ${state.state==='running'?'live':''}`;
    $('simulation-badge').textContent={loading:'준비 중',stopped:'준비',running:'시뮬레이션 중',paused:'일시정지',error:'실행 오류'}[state.state]||state.state;
    $('simulation-alert').hidden=!state.fallen&&!state.message;
    $('simulation-alert').textContent=state.message||(state.fallen?'로봇이 넘어졌습니다. 초기화로 다시 시작하세요.':'');
    if(state.actions && !this.actionButtons) {
      this.actionButtons=new Map();
      for(const [container,actions] of [['simulation-actions',state.actions],['simulation-experiments',state.experiments||[]]]) {
        for(const action of actions) {
          const button=document.createElement('button'); button.type='button'; button.textContent=action.label; button.dataset.action=action.id;
          button.addEventListener('click',()=>{
            this.keys.clear();this.paintKeys();this.pending={...this.pending,action:action.id};
            if(action.id==='dynamic_mode') this.pending.speed=$('simulation-dynamic-speed').value;
          });
          $(container).append(button);this.actionButtons.set(action.id,button);
        }
      }
    }
    for(const [id,button] of this.actionButtons||[]) {button.classList.toggle('selected',state.action===id);button.setAttribute('aria-pressed',String(state.action===id));}
    $('simulation-action-name').textContent=[...(state.actions||[]),...(state.experiments||[])].find((action)=>action.id===state.action)?.label||'Walking';
    this.action=state.action;
    const policy=$('simulation-policy'), teacher=state.teacher, unified=state.unified, dynamic=state.action==='dynamic_mode'?state.dynamic:null;
    policy.textContent=teacher ? `실험 교사 · ${teacher.label}` : unified ? `통합 정책(실험) · ${unified.label}` : dynamic ? `DynamicMode · ${dynamic.label}` : state.action==='dynamic_mode' ? 'DynamicMode 를 준비하고 있습니다.' : state.policy ? `평가 모델 · ${state.policy.label}` : state.action==='walking' ? '제조사 Walking' : '걷기 중에도 액션을 전환할 수 있습니다.';
    policy.title=teacher?.run_id||unified?.run_id||dynamic?.run_id||state.policy?.run_id||'';
    $('simulation-experiment-note').hidden=!teacher&&!unified&&!dynamic;
    if(teacher) $('simulation-experiment-note').textContent=teacherNote(teacher);
    else if(unified) $('simulation-experiment-note').textContent=unifiedNote(unified);
    else if(dynamic) $('simulation-experiment-note').textContent=dynamicNote(dynamic);
    $('simulation-dynamic').hidden=state.action!=='dynamic_mode';
    if(dynamic) {
      $('simulation-instruction-status').textContent=instructionNote(dynamic.text);
      $('simulation-instruction').disabled=dynamic.text?.state==='none';
      if(document.activeElement!==$('simulation-dynamic-speed')) $('simulation-dynamic-speed').value=dynamic.speed;
      for(const button of $('simulation-dynamic-posture').querySelectorAll('button')) {
        const selected=button.dataset.posture===dynamic.posture;
        button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));
      }
    }
    $('simulation-time').textContent=`${(state.time||0).toFixed(1)}s`;
    $('simulation-speed').textContent=`${Math.hypot(...(state.velocity||[0,0]).slice(0,2)).toFixed(2)} m/s`;
    if(state.environment && this.environment!==state.environment) {
      this.environment=state.environment;
      $('simulation-environment').value=state.environment;
      $('simulation-terrain-note').textContent=terrainNotes[state.environment];
      $('simulation-world-label').textContent=state.environments.find((item)=>item.id===state.environment)?.label;
    }
  }

  destroy() {this.stopped=true;clearTimeout(this.timer);this.setVisible(false);this.view?.destroy();}
}
