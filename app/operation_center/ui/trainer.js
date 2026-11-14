const $ = (id) => document.getElementById(id);
const titles = { careful_walk: '조심 보행', safe_crouch: '안전 웅크리기' };
const states = { starting: '초기화 중', running: '학습 중', completed: '완료', stopped: '종료 · 완료 미확인', failed: '오류', unknown: '상태 미확인' };
const kinds = { policy: 'ONNX 정책', checkpoint: '학습 체크포인트', video: '결과 영상', evaluation: '평가·재현 정보', config: '설정·환경', motion: '모션 데이터', log: '로그' };
const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const scoreLabel = (row) => row?.statistics_version === 5
  ? `${Number.isFinite(row.score) ? `안정성 ${number(row.score)}점` : row.evaluation_scope === 'preview' ? '안정성 예비평가' : '안정성 미산출'} · 목적성 미검증`
  : Number.isFinite(row?.score) ? `${number(row.score)}점` : '점수 미산출';
const size = (bytes) => bytes >= 1048576 ? `${number(bytes / 1048576)} MB` : `${number(bytes / 1024)} KB`;
const stamp = (value) => /^\d{8}-\d{4}$/.test(value || '') ? value.replace(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/, '$1-$2-$3 $4:$5') : value || '—';
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}

export class TrainerPanel {
  constructor() {
    this.runs = []; this.files = []; this.visible = false; this.stopped = false;
    this.selection = ''; this.mediaKey = ''; this.artifactKey = '';
    for (const id of ['trainer-action', 'trainer-state-filter', 'trainer-search']) {
      $(id).addEventListener(id === 'trainer-search' ? 'input' : 'change', () => this.paint());
    }
    $('trainer-checkpoint').addEventListener('change', () => this.showMedia(true));
    $('trainer-auto').addEventListener('change', () => this.showMedia(true));
    $('trainer-artifact-kind').addEventListener('change', () => this.paintArtifacts());
    this.poll();
  }

  async poll() {
    try {
      const response = await fetch('/api/trainer/runs', { cache: 'no-store' });
      if (!response.ok) throw new Error('학습 목록을 불러오지 못했습니다.');
      const data = await response.json();
      this.runs = data.runs;
      const action = $('trainer-action'), selected = action.value;
      action.replaceChildren();
      for (const [value, label] of [['', '전체 액션'], ...[...new Set(this.runs.map(run => run.action))].sort().map(value => [value, titles[value] || value])]) {
        const option = element('option', label); option.value = value; action.append(option);
      }
      action.value = [...action.options].some(option => option.value === selected) ? selected : '';
      const server = data.server || {}, renderer = data.renderer || {};
      $('trainer-total-count').textContent = this.runs.length;
      $('trainer-running-count').textContent = server.connected ? server.processes.length : '미확인';
      $('trainer-completed-count').textContent = this.runs.filter((run) => run.state === 'completed').length;
      $('trainer-output-count').textContent = this.runs.filter((run) => run.checkpoint_count || run.evaluations.length).length;
      const activeRuns = this.runs.filter((run) => ['starting', 'running'].includes(run.state)).length;
      $('trainer-server-status').textContent = server.connected ? `GPU 서버 확인 ${new Date(server.checked * 1000).toLocaleTimeString('ko-KR')} · 학습 ${activeRuns}개 · 프로세스 ${server.processes.length}개` : server.message || 'GPU 서버 상태 확인 중';
      const busy = renderer.active && ['evaluating', 'evaluating_projectiles', 'evaluating_gait', 'evaluating_hard_cases', 'rendering'].includes(renderer.state);
      $('trainer-service-badge').className = `badge ${renderer.active ? 'live' : ''}`;
      $('trainer-service-badge').textContent = !renderer.active ? '자동 생성 꺼짐' : busy ? '산출물 생성 중' : '자동 생성 켜짐';
      const stage = renderer.state === 'rendering' ? '렌더링' : renderer.state === 'evaluating_projectiles' ? '실제 공 평가' : renderer.state === 'evaluating_gait' ? '제조사 보행 비교' : renderer.state === 'evaluating_hard_cases' ? '고정 고난도·목적 구간 평가' : '평가';
      $('trainer-next').textContent = !renderer.active ? '자동 평가·영상 생성이 꺼져 있습니다.' : busy ? `${renderer.iteration ?? ''} 반복 ${stage} 중` : renderer.next_run ? `다음 확인 ${new Date(renderer.next_run * 1000).toLocaleTimeString('ko-KR')}` : '새 체크포인트를 기다립니다.';
      this.paint();
    } catch (error) { $('trainer-list-message').textContent = error.message; }
    if (!this.stopped) this.timer = setTimeout(() => this.poll(), 5000);
  }

  paint() {
    const query = $('trainer-search').value.trim().toLowerCase();
    const runs = this.runs.filter((run) => (!$('trainer-action').value || run.action === $('trainer-action').value)
      && (!$('trainer-state-filter').value || run.state === $('trainer-state-filter').value)
      && `${run.id} ${run.tags} ${run.description || ''} ${titles[run.action] || run.action}`.toLowerCase().includes(query));
    if (!runs.some((run) => run.id === this.selection)) this.selection = runs[0]?.id || '';
    const list = $('trainer-runs'); list.replaceChildren();
    for (const run of runs) {
      const button = element('button', '', 'trainer-run'); button.type = 'button';
      button.classList.toggle('selected', run.id === this.selection);
      button.setAttribute('aria-pressed', String(run.id === this.selection));
      button.append(element('strong', `${titles[run.action] || run.action} · ${run.tags || '기본 설정'}`),
        element('span', `${states[run.state] || '상태 미확인'} · ${run.iteration ?? run.checkpoint ?? '—'} 반복`),
        element('small', stamp(run.started)));
      button.addEventListener('click', () => { this.selection = run.id; this.paint(); });
      list.append(button);
    }
    $('trainer-list-message').textContent = runs.length ? `${runs.length}개 학습 표시` : '조건에 맞는 학습 기록이 없습니다.';
    this.run = runs.find((run) => run.id === this.selection);
    $('trainer-details').hidden = !this.run;
    if (!this.run) {
      $('trainer-title').textContent = '학습을 선택하세요';
      $('trainer-id').textContent = $('trainer-run-state').textContent = $('trainer-run-meta').textContent = $('trainer-objective').textContent = '';
      $('trainer-video').pause(); this.mediaKey = ''; this.shown = null; return;
    }
    const run = this.run, changed = this.previousRun !== run.id;
    $('trainer-title').textContent = `${titles[run.action] || run.action} · ${run.tags || '기본 설정'}`;
    $('trainer-id').textContent = run.id;
    $('trainer-objective').textContent = run.description || '';
    $('trainer-run-state').textContent = states[run.state] || '상태 미확인';
    $('trainer-run-state').className = `badge ${['starting', 'running'].includes(run.state) ? 'live' : ''}`;
    $('trainer-run-meta').textContent = `시작 ${stamp(run.started)} · 병렬 환경 ${run.num_envs?.toLocaleString() || '—'}개 · ONNX 체크포인트 ${run.checkpoint_count ?? '—'}개 · ${run.auto_render ? '자동 영상 생성 대상' : '영상 자동 생성 대상 아님'}`;
    $('trainer-run-error').hidden = !run.error;
    $('trainer-run-error').textContent = run.error || '';
    $('trainer-progress').value = run.progress ?? 0;
    $('trainer-progress-label').textContent = run.progress == null ? '진행률 기록 없음' : `${number(run.progress * 100)}%${run.eta ? ` · 남은 시간 ${run.eta}` : ''}`;
    $('trainer-iteration').textContent = `${run.iteration ?? run.checkpoint ?? '—'}${run.total ? ` / ${run.total}` : ''}`;
    $('trainer-best-score').textContent = run.best?.statistics_version === 5 ? scoreLabel(run.best)
      : run.evaluations.at(-1)?.statistics_version === 5 ? '안정성 미산출 · 목적성 미검증' : number(run.best?.score);
    this.chart(run.evaluations);
    const selector = $('trainer-checkpoint'), selected = changed ? 'latest' : selector.value;
    selector.replaceChildren();
    for (const [value, label] of [['latest', '최신 영상'], ['best', '최고 점수 영상'], ...run.evaluations.filter((row) => row.video).slice().reverse().map((row) => [String(row.iteration), `${row.iteration} 반복 · ${scoreLabel(row)}`])]) {
      const option = element('option', label); option.value = value; selector.append(option);
    }
    selector.value = [...selector.options].some((option) => option.value === selected) ? selected : 'latest';
    this.previousRun = run.id;
    this.showMedia(changed);
    this.loadArtifacts();
  }

  async loadArtifacts() {
    const run = this.run, key = `${run.id}/${run.updated}/${run.evaluations.length}/${Math.floor(Date.now() / 15000)}`;
    if (this.artifactKey === key) return;
    this.artifactKey = key;
    if (this.artifactRun !== run.id) { this.files = []; this.paintArtifacts(); }
    this.artifactRun = run.id;
    $('trainer-artifact-summary').textContent = '산출물 목록을 불러오는 중입니다.';
    try {
      const response = await fetch(`/api/trainer/artifacts?run=${encodeURIComponent(run.id)}`, { cache: 'no-store' });
      if (!response.ok) throw new Error('산출물 목록을 불러오지 못했습니다.');
      const data = await response.json();
      if (this.run?.id !== run.id || this.artifactKey !== key) return;
      this.files = data.files; this.paintArtifacts();
    } catch (error) {
      if (this.run?.id === run.id) { this.artifactKey = ''; $('trainer-artifact-summary').textContent = error.message; }
    }
  }

  paintArtifacts() {
    const rows = $('trainer-artifact-rows'); rows.replaceChildren();
    const files = this.files.filter((file) => !$('trainer-artifact-kind').value || file.kind === $('trainer-artifact-kind').value);
    $('trainer-artifact-summary').textContent = `${files.length}개 파일 · ${size(files.reduce((sum, file) => sum + file.size, 0))}`;
    for (const file of files) {
      const row = document.createElement('tr');
      row.append(element('td', file.name, 'trainer-file-name'), element('td', kinds[file.kind] || file.kind),
        element('td', file.iteration ?? '—'), element('td', size(file.size)));
      const cell = document.createElement('td'), link = element('a', '다운로드');
      link.href = file.url; link.download = ''; link.setAttribute('aria-label', `${file.name} 다운로드`);
      cell.append(link); row.append(cell); rows.append(row);
    }
    if (!files.length) { const row = document.createElement('tr'), cell = element('td', '이 종류의 산출물이 없습니다.'); cell.colSpan = 5; row.append(cell); rows.append(row); }
  }

  showMedia(force = false) {
    const run = this.run;
    if (!run) return;
    const selected = $('trainer-checkpoint').value;
    let row = selected === 'latest' ? run.latest : selected === 'best' ? run.best : run.evaluations.find((item) => String(item.iteration) === selected);
    if (!force && !$('trainer-auto').checked && this.shown?.runId === run.id) row = this.shown.row;
    const balls = row?.statistics_version === 4;
    const fall = balls ? row?.metrics?.ball_fall : row?.metrics?.fall;
    $('trainer-fall').textContent = fall == null ? '—' : `${number(fall * 100)}%`;
    const trials = row?.statistics_version === 5 ? `고정 사례 ${row.metrics.safety_cases ?? '—'}개 · ${row.evaluation_scope === 'full' ? '전체' : '예비'} 안정성 평가`
      : balls ? (fall == null ? '실제 공 평가 미완료' : `공 연사 ${Math.round(fall * row.metrics.ball_cases)}/${row.metrics.ball_cases}사례 넘어짐`) : '저장된 외란 평가 영상';
    $('trainer-preview-caption').textContent = row ? `${row.iteration} 반복 · ${scoreLabel(row)} · ${row.evaluated || ''} · ${trials}${balls ? ' · 공 충돌 데모 영상' : ''}` : '이 학습에는 저장된 결과 영상이 없습니다. 체크포인트와 설정은 산출물 목록에서 확인하세요.';
    const key = `${run.id}/${row?.video}`;
    if (key === this.mediaKey) return;
    const video = $('trainer-video'), playing = !video.paused;
    this.mediaKey = key; this.shown = { runId: run.id, row };
    video.pause(); video.removeAttribute('src'); video.hidden = true;
    $('trainer-empty').hidden = false; $('trainer-empty').textContent = '저장된 결과 영상이 없습니다.';
    if (row?.video) {
      video.src = row.video; video.hidden = false; $('trainer-empty').hidden = true;
      if (this.visible && playing) video.play().catch(() => {});
    }
  }

  chart(rows) {
    const svg = $('trainer-chart'); svg.replaceChildren();
    const version = rows.at(-1)?.statistics_version ?? 1;
    const points = rows.filter((row) => Number.isFinite(row.score) && (row.statistics_version ?? 1) === version);
    if (!points.length) {
      const label = document.createElementNS(svg.namespaceURI, 'text'); label.setAttribute('x', '25'); label.setAttribute('y', '65'); label.textContent = '저장된 평가 결과가 없습니다.'; svg.append(label); return;
    }
    const x0 = points[0].iteration, x1 = points.at(-1).iteration;
    const low = Math.min(...points.map((row) => row.score)) - 1, high = Math.max(...points.map((row) => row.score)) + 1;
    const coordinates = points.map((row) => [35 + (row.iteration - x0) / Math.max(1, x1 - x0) * 615, 105 - (row.score - low) / (high - low) * 85]);
    const line = document.createElementNS(svg.namespaceURI, 'polyline');
    line.setAttribute('points', coordinates.map((point) => point.join(',')).join(' ')); svg.append(line);
    points.forEach((row, i) => {
      const dot = document.createElementNS(svg.namespaceURI, 'circle');
      dot.setAttribute('cx', coordinates[i][0]); dot.setAttribute('cy', coordinates[i][1]); dot.setAttribute('r', '3');
      const title = document.createElementNS(svg.namespaceURI, 'title'); title.textContent = `${row.iteration} 반복 · ${scoreLabel(row)}`; dot.append(title); svg.append(dot);
    });
    for (const [x, text] of [[35, `${x0} 반복`], [550, `${x1} 반복`]]) {
      const label = document.createElementNS(svg.namespaceURI, 'text'); label.setAttribute('x', x); label.setAttribute('y', '125'); label.textContent = text; svg.append(label);
    }
  }

  setVisible(visible) { this.visible = visible; if (!visible) $('trainer-video').pause(); }
  destroy() { this.stopped = true; clearTimeout(this.timer); $('trainer-video').pause(); }
}
