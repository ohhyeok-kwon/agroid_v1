# 학습
mjlab 1.6(MuJoCo Warp) + rsl_rl 5.4 기반으로 서버에서 학습한다([규칙](00_agent.md)). 모든 모션의 목적·환경 대응·설계 기준은 [공통 원칙](../../docs/12_train.md)을 따른다.
## 배포 계약 (agroid `walking_action_manager`와 동일)
- 기존 정책: ONNX `810→24`, 정규화기 내장, opset 11. S8의 `927→24`와 참조 조건 교사의 `233→24`는 시뮬레이션 전용이다. 교사는 참조·위상·미래 목표가 필요하며 정규화는 끈다.
- 관측: 81차원 프레임 10개(오래된→최신) = `[자이로, 중력, 명령(3), q−q0, dq, 이전 action]`. 순서는 정책(BFS) 순서다.
- 게인·기본자세·action 크기는 제조사 값이다(`src/r1lab/contract.py`).

## 태스크
| 태스크 | 내용 |
|---|---|
| `careful_walk_s1` | 1단계: 평지, 외란 없음, 넓은 속도 명령. 걷기 습득 |
| `careful_walk_s2` | 2단계: 거친 지형·밀기 0.4 m/s·타격 ±60 N·조용한 착지·조심 속도 범위를 커리큘럼으로 켬 |
| `careful_walk_s3`/`_s4` | 미세조정: 느린 제자리 회전 강화 / 발 들림 9 cm(장애물) |
| `careful_walk_s5`/`_s5p` | 장애물: 발 걸림 벌점, 계단·돌무더기 비중·높이 상향 / + critic 지형 스캔 |
| `careful_walk_s6a`~`_s6d` / `_s7a`·`_s7b` | 회복 대조 / 자기충돌·실제 구체 1kg, 회복 비용 25%·5% |
| `careful_walk_s8` / `_s8r` | 한 정책/4GPU, 전신 판별기·생성 환경·위험 critic / 10%를 사람 모션 상태에서 시작하는 탐색; 기준 평가는 기립 시작 |
| `careful_walk_teacher` / `_precise` / `_coordinated` | 무작위 교사 / 양발 궤적 / 발 추종을 유지한 팔 위치 보완. 한 정책/4GPU, 평지·자기충돌·50Hz. 제조사는 평가 전용 |
| `safe_crouch` | 높이 명령 `[h_cmd−H0, h_target−H0, rate]`로 천천히 웅크리기. 초기 교란·밀기·타격·약한 모터 |
| `wb_task_teacher_v12h0_head_control` / `wb_task_teacher_v12h_head_perception` | 머리 2축 교사: 참조24·행동26·actor417·critic553. H0/H는 시야·카메라 움직임·머리 노력 보상의 가중치만 다름 |

보행·웅크리기 랜덤화(교사 첫 실험은 물리 랜덤화 없이 참조 초기 상태에 작은 오차만 적용):
- IMU·관절: 자이로 노이즈·바이어스·스파이크, 자세 바이어스·드리프트, 엔코더 노이즈·바이어스
- 지연: 관측 0~20 ms, 명령 0~20 ms
- 같은 정책 step의 센서 노이즈·지연은 한 번만 진행하며 부분 reset은 해당 환경만 갱신한다. 물리: PD ±15%, 토크 한계, armature, 마찰 0.25~1.4, 하중 0~3 kg, 무게중심 오프셋.

## 명령
```bash
scripts/01_train.sh careful_walk_teacher --gpus 4 --num-envs 4096 --iters 500 --description '무작위 초기화 전신 교사'
scripts/01_train.sh wb_task_teacher_v12h_head_perception --gpus 4 --num-envs 4096 --resume '.agent/runs/r1_wholebody/<v10s-ID>/model_2748.pt' --resume-next-iteration --reward-ledger --iters 500 --description '머리 시야·카메라 안정화 보상 비교'
scripts/03_status.sh
```
- 재개는 `--resume <model.pt>`, 커리큘럼 기준은 `--t0`다. 기존 AMP만 `--amp-disc`를 쓰며 S8 판별기·환경 에이전트는 checkpoint 안에 저장한다.
- 교사 첫 실험은 무작위 초기화한다. 후속은 원본까지 계보·해시를 검증한 자체 교사 최종 checkpoint의 actor·critic·Adam·학습률을 복원한다. 참조와 기립 비교를 분리하며 기립 참조 XY/yaw는 시작점에 한 번만 정렬한다.
- 사용자는 “○○를 보완한 △△ 학습 실행해줘”라고 요청하면 된다. 에이전트가 구성·등록·서버 실행·PC 동기화·Trainer API 조회 확인까지 수행한다([지침](../../.agent/AGENTS.md)).
- `--description`·`--run-name`으로 검색한다. GPU 초기화 전 등록하며 `run_info.json`·`train.log`에 기록한다. 공식 실행기는 rank 실패를 전체에 반영하고, 모든 rank 완료·launcher 정상 종료 뒤 `completed`로 확정한다.
- `01_train.sh`는 `trainer-sync`가 꺼져 있으면 실행을 막는다. `--gpus N`은 공유 정책, `--num-envs`는 GPU당 환경 수다. 전신 계열(r1wb)은 `GPU=0,1,2,3 … --gpus 4 --num-envs 4096`(합 16,384)이 기준이며, 입력 확장 과제의 `--resume`은 정책·평가자·최적화 상태를 넓혀 이어받는다.
- 전신 재개에서 `--resume-next-iteration`은 저장 회차의 다음 번호부터 시작한다. `--global-statistics`는 다중 GPU의 관측·advantage 통계를 합산한다. 둘 다 기본 꺼짐이며 통계 방식 변경은 별도 대조한다. `--iters`는 추가 갱신 수다.
- 전신 진단 `--failure-replay-buffer`는 기본 꺼짐이다. 최대512MiB의 전체 환경 직전 물리 입력을 보관해 실패 때 저장한다. `r1wb.eval.replay_physics`로 비교하며, 입력 저장 성공과 실제 오류 재현 성공은 구분한다.
- `--reward-ledger`는 보상·종료·외력·critic을 기록한다. H0/H는 별도 `r1_head26.xml`과 같은 v10s 부모를 쓰며 머리 관측·Adam 상태를 확장한다. gzip128MiB/rank·전체 단계 검증 필수이며 S1·S2는 타격 후 시간 유예를 거부한다. 머리 PD는 실측 보정 전이다.
- 전신 학습은 `--execution optimized`(읽기 캐시·초기화·기준 모션 사전 계산·형상별 충돌 묶음·지형 셀 분할)가 기본이다. `compacted`는 충돌 종류별 분류까지, `cached`는 읽기 캐시와 통계 전송만, `standard`는 원래 경로다. 셀 분할은 검증된 커널에서만 쓰며 용량·접촉 경고 시 원래 계산으로 돌아간다. 지형·학습법·정밀도는 유지하고 `performance.jsonl`에 수집·갱신·회차 간 경과 시간을 남긴다.
- 기존 태스크의 critic 입력 변경은 `--actor-only`와 별도 예열을 쓴다. S8는 810→927 초기 출력을 보존하며, 최소64회·고정 환경 그룹별200개 유효 episode를 확보할 때까지 actor를 고정한다.
- S8 무진척 판정은 명령 방향의 새 누적 이동을 센다. 왕복 흔들림은 진척으로 인정하지 않고 회복 중 정지는 허용한다. 판정 변경 시 기존 무진척 통계만 폐기하고 그룹별 새200 episode로 다시 검증한다.
- `--freeze-actor`는 actor 가중치·관측 통계를 함께 고정한다. `--freeze-normalizer`는 관측 통계만 고정하며, 이어서 실행할 때도 다시 지정한다.
- 출력: `<work>/runs/<실험>/<ID>/`에 모델·ONNX·등록 정보를 둔다. S8는 critic·판별기·환경 에이전트·optimizer도 보존하며, 갱신 후 허용 KL을 넘으면 모델과 optimizer를 함께 복원한다.
- 모든 평가의 영상·전체 모션·환경/물리 모델·초기 상태·명령/외란·정책/모델 해시·seed를 로컬 `<work>/result/video/<학습 ID>/`에 시간순 보존한다. 재현 범위와 디스크 부족 사유를 남기며, 서버 전송·이전 영상 자동 삭제는 하지 않는다.
## 현재 확정 정책 (`actions/careful_walk/`, `actions/safe_crouch/`)
- 확정 정책의 과거 평가표는 `<work>/result/render/<학습 ID>/legacy_actions/eval/`, 영상은 `result/video/<학습 ID>/legacy_actions/`에 있다. 실기 검증은 아직 하지 않았다.
- s2f/3300은 실제 연속 투사체 충돌에서 제조사보다 약하다. 채택 전 Simulation과 같은 물리·명령 조건에서 정지·보행 중 충돌 회복을 별도로 비교한다.
- S7 v4는 실제 공24사례, S8 v5는 정상/어려운 조건의 안정성을 별도로 평가한다. 목적성은 전신 동작 구간·영상으로 별도 확인하며 예비/전체·다른 버전 점수를 섞지 않는다.
- 시뮬레이션의 접촉·착지 지표는 실제 음향 측정과 구분한다. 개선 실험의 근거·조건은 [설계](../../.agent/analysis/careful_walk_plan_20261108/plan.md)와 [문헌](../../.agent/analysis/careful_walk_plan_20261108/research.md)에 있다.
