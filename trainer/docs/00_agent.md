# 트레이너 작업 규칙

- 문서 기준은 [agroid 규칙](../../docs/00_agent.md)을 따른다(파일당 최대 50줄, 현재 상태만 기록).
- 코드: `agroid_v1/trainer/`에는 `actions/`(배포용 ONNX·설정), `docs/`, `scripts/`(공용 도구), `src/`(학습 코드·필수 로봇 자산)만 둔다.
- 작업 영역 `<work>` = `agroid_v1/.agent/trainer/`: `pyproject.toml`·`uv.lock`·`.venv`·`runs/`·`data/`·`tools/`·`eval/`·`result/`.
- 영상·평가·학습 데이터·테스트·캐시는 `.agent/`에 둔다. Python 검증은 `-B` 또는 `PYTHONDONTWRITEBYTECODE=1`로 소스 옆 캐시 생성을 막는다.

## 학습은 서버, 추론은 로컬
- 모션 요청의 해석·자료 확보·환경 상호작용·설계/분석 기준은 [공통 원칙](../../docs/12_train.md)을 먼저 읽는다.
- **학습은 GPU 서버로 오프로딩한다.** 접속은 `ssh super-via-ailab`(컨테이너, root, `hs-ailab` 경유; `RL_REMOTE`로 변경), 작업 폴더는 **`/rl-agent`** 다.
- 서버 배치: `/rl-agent/{actions,docs,scripts,src}`는 `trainer/`, `/rl-agent/.agent/`는 `<work>` 중 학습 입출력에 대응한다.
- **추론(평가·렌더링)은 이 PC에서 한다.** 결과 영상은 로컬 전용(`<work>/result/`, 동기화 제외)이다.
- 공유 환경인 GPU 서버에는 **`trainer/` 폴더만 항상 양방향 동기화**한다: `trainer/scripts/00_initialize.sh`, 사용자 서비스 `trainer-sync`.
  - 새것(mtime)이 이긴다. 한쪽에서 지운 파일은 반대쪽도 지운다. 한 번에 300개 넘게 지워야 하면 멈춘다.
  - `<work>`는 학습 실행이 읽고 쓰는 것만 오간다: `pyproject.toml`·`uv.lock`·`runs/`와 스크립트 `WORK_SCOPE`에 적은 학습 입력.
  - 인수인계·도구·평가·영상·그 밖의 데이터와 프로젝트의 다른 부분은 서버에 보내지 않는다. `--check`가 서버의 범위 밖 파일을 보고한다.
  - 새 학습 입력 파일은 `WORK_SCOPE`에 추가하고 `--install-service`로 반영한다. 빠지면 서버 학습이 파일을 찾지 못한다.
- 트레이너는 GPU 서버에만 동기화한다. R1에는 PC 프로젝트 루트의 `scripts/03_deploy.sh`로 로봇 런타임만 배포한다([안내](../../docs/01_initial_setup.md)).
- 서버 환경은 `/rl-agent/.agent/.venv`다. `pyproject.toml`·`uv.lock`이 바뀌면 동기화 때 다시 만든다.
- 서버 드라이버는 575(CUDA 12.9)라서 Warp는 CUDA 12 빌드로 고정한다. 서버 NVIDIA 패키지는 건드리지 않는다.
- R1 학습에는 자기충돌이 필수다. 실행기가 로봇 충돌 마스크를 검사하며, 꺼진 모델은 PPO 실행 전에 중단한다. 평가도 자기충돌을 포함한다.
- 로컬 평가는 MuJoCo EGL 렌더링을 사용한다. 서버 평가(`06_evaluate.sh`)의 상태 보관본은 PC로 가져온 뒤 서버에서 지우며, 영상 렌더링은 PC에서만 한다.

## 학습 ID와 결과
- ID = `<액션>_<날짜-시간>_<주요설정>_n<env수>[_from-<출발 run>@<반복>]`. `r1lab.train`이 붙인다.
  - 공통 실행기가 GPU 초기화 전에 ID·목적·설정을 `run_info.json`에 등록한다. 등록 실패 또는 표준 경로 밖 `--log-root`는 실행을 차단한다.
  - 시작·완료·중단·실패 상태와 `train.log`를 각 run 폴더에 남긴다. 직접 `python -m r1lab.train`을 실행해도 동일하다.
  - 새 태스크는 `r1lab.registry.TASK_SPECS`와 `r1lab.tasks.TASKS`를 함께 등록한다. 불일치하면 학습이 중단된다.
- 데스크톱 Operation Center가 켜져 있을 때 앱 소유 프로세스가 10분마다 새 체크포인트를 로컬 평가·렌더링한다. 창을 닫으면 자식 작업도 종료하며 상시 서비스는 등록하지 않는다.
- 모든 평가 영상은 정책·환경·실패 사례별로 로컬 `<work>/result/video/<학습 ID>/`에 촬영 시각 순으로 저장한다. 디스크 부족은 사유를 기록하며 기존 영상 자동 삭제·서버 전송은 하지 않는다.
- 전체 모션·환경/물리 모델·초기 상태·명령/외란·정책/모델 해시·seed·버전을 같은 시간축의 재현 메타데이터로 연결한다.
  - 등록 학습은 앱이 꺼진 사이 완료돼도 다음 실행 때 최종 체크포인트를 처리한다. 미등록 과거 학습은 `--id <ID>`로 지정한다.
  - 영상과 평지 3D 궤적을 보존한다. 최신 영상과 최고 점수 영상은 별도다.
- 오퍼레이션 센터 `Trainer` 탭은 전체 학습 이력·상태·산출물 관리에 집중한다. ID·설정 검색과 액션·상태 필터, 체크포인트·설정·로그·평가·영상·궤적 다운로드를 제공한다.
  - 목록은 5초마다 갱신된다. ‘새 영상 자동 표시’를 켜면 새 체크포인트 영상을 자동으로 표시한다.
  - 영상 미리보기는 외란 조건의 저장된 산출물이다. 실시간 조작은 `Simulation(임시)`에서 한다. 결과 목록은 `<work>/result/index.tsv`다.

| 스크립트 | 용도 |
|---|---|
| `scripts/00_initialize.sh` | 서버 준비(uv·EGL)·상시 동기화. `--once`·`--check`·`--dry-run`·`--install-service` |
| `scripts/01_train.sh` | 서버 학습. `--gpus 4`는 한 정책을 네 GPU로 공동 학습; 로그 `<work>/runs/` |
| `scripts/02_connect_ssh.sh` | 서버 셸(`/rl-agent`, 렌더링 환경 포함) |
| `scripts/03_status.sh` | 서버 GPU·실행 중 학습·최근 로그·동기화 상태 |
| `scripts/04_remote.sh` | 서버에서 임의 명령 실행. `-d <이름>`은 백그라운드 실행 |
| `scripts/05_results.sh` | 앱이 관리하는 로컬 평가·영상 생성. 수동 `--once`·`--id <ID>` |
| `scripts/06_evaluate.sh` | 서버 평가 묶음(`suite react burst heldout …`, `s2`: [System 2](02_system2.md)). 결과는 PC로 회수 |

- 평가·데이터 생성 도구는 `<work>/tools/`에 있으며 PC에서만 실행한다(서버에 보내지 않는다).
- 원본 모션 데이터(NAS `cinamon`)는 읽기 전용이다. 추출·리타게팅본은 `<work>/data/`에 둔다.
- 학습 절차·태스크·결과는 [01_training.md](01_training.md), 정책 계약·배포는 [12_train.md](../../docs/12_train.md)에 있다.
