# system 2 (지시 → 의도 → 명령 → 운동 토큰) — 실험 단계

지시문 + 머리 카메라 → VLM(의도: 속도 3·높이·높이 사용) → **느린 고리**(`r1wb.s2.steer`: 실현된 진행을 보고 0.32초마다 명령을 고침, RL) → 운동 토큰 머리(연속 명령) → 토큰기 복호 → 참조 30차원 → System 1(RL 추종 정책, `r1wb.teacher_task`).
코드는 `src/r1wb/s2/`, 자료는 PC `<work>/data/s2/`·서버 `/rl-agent/.agent/data/s2/`. 판정 기록은 PC `.agent/analysis/`에 있다.

## 고정된 규칙
- 경로 생성기 설정(녹화·System 1 학습·평가 공통): `match_pose=reference`, `match_commit`, `match_height=target`, `match_stop=arrive`, `travel_window=32`, `match_noise=0`. System 1 과제는 `decisive_course=True`.
- 토큰기는 인과적(`--causal`), 머리는 `--joint`. VLM 의 의도는 지시 어휘 27개 중 가장 가까운 것으로 읽고, 그 아래 명령은 연속값이다.
- 느린 고리: 가다/서다는 지시 그대로, 고리가 고르는 것은 보정뿐(±0.25 m/s, ±0.6 rad/s). 구조상 좌우 대칭(`--mirror`)이고, 보상은 타격으로 밀린 거리를 되갚을 진행으로 세지 않는다(`--leak 3 --forgive 2 --remember`). 지시를 몸 방향으로 돌리는 변환·학습되는 탐색 잡음·결정마다 뽑는 이산 선택은 쓰지 않는다(각각 실패 원인이었다).
- 라이브러리는 균형 라이브러리 `corebal`이다(`python -m r1wb.balance --subset core --out corebal`, 서버에서도 같은 도구로 생성): 생성기가 멈춰 설 수 있는 프레임의 무게중심을, 발은 둔 채 그 발 자세의 최대 정적 안정 여유 지점으로 옮긴다. 녹화·학습·평가 도구 모두 `--library corebal`.
- 토큰기는 조향된 경로까지 넣어 학습하고(`v2d`), 머리는 거울 쌍을 넣어 학습한다(`small.py --mirror`).
- 새 보상은 학습 전에 `python -m r1wb.eval.steer_reward_check <S1> <과제>`로 손 변형(지시대로·되먹임·항상 움직임·느리면 서기)과 비교한다.
- System 1 의 자세 가격은 포화시킨다(`course_cap=0.12`): 단발 반응은 값이 매겨지고 연타에서 버티는 값은 넘어짐 값보다 싸다.
- System 1 은 좌우 대칭 구조다: 정책 = ½(망(o) + 거울(망(거울 o)))(`r1wb.symmetric_model`, 거울 사상 `r1wb.symmetry`). 새 학습은 과제 `…_cap12_sym*`, 기존 체크포인트는 `python -m r1wb.symmetrize <실행 폴더> <회차> <이름>`으로 대칭화 실행을 만든다(체크포인트·ONNX 가 전체 함수를 담는다). 거울 손실은 쓰지 않는다(정책이 무너짐). 관측 항을 바꾸면 `python -m r1wb.eval.mirror_check <과제> <체크포인트>`로 사상을 다시 검증한다.
- System 1 을 고리가 조향하는 장면에서 이어 학습하는 것(과제 `…_cap12_steer`)은 쓰지 않는다: 그 조건 안에서는 같고, 머리가 쓴 계획에서는 멈춘 뒤 제자리걸음이 늘었다.

## 만드는 순서
```bash
P=".agent/.venv/bin/python -B"; D=.agent/data/s2; L=.agent/data/wb/library_v2_corebal.npz   # 서버 /rl-agent (PC 는 .agent/trainer/…)
# 1) 느린 고리(RL, 4,096환경 × 16결정 × 200회 약 40분; S1 의 학습 장면 그대로, 타격 포함). 이어 학습은 --start <고리>
$P -m r1wb.s2.steer train --system1 <S1 실행>/model_1499.pt --task <S1 과제> --see progress --gate told --mirror --library corebal --leak 3 --forgive 2 --remember --out $D/steer_n18b.pt
# 2) 경로 녹화(각 45.5시간, 약 100초): 지시 어휘 판, 무작위 조향 판(--steer random), 고리가 조향한 판(--steer <고리>)
$P -m r1wb.s2.collect --run <S1 실행> --checkpoint model_1499.pt --out $D/courses_b_v1.npz --envs 4096 --seconds 40 --seed 31 --course-set travel_window=32 --library corebal
# 3) 토큰기(12,000스텝 3분)와 머리(30,000스텝 약 20분). 조향 녹화는 기록된 명령을, 나머지는 지시의 목표를 조건으로 쓴다
$P -m r1wb.s2.tokenizer --courses $D/courses_b_{v1,r1,n1,n2}.npz --library $L --out $D/tokenizer_v2d.pt --causal --steps 12000
$P -m r1wb.s2.small --courses $D/courses_b_v{1,2,3,4,5}.npz $D/courses_b_r{1,2}.npz $D/courses_b_n{1,2,3}.npz --library $L --tokenizer $D/tokenizer_v2d.pt --out $D/small_h12.pt --condition goal --joint --change-weight 8 --steps 30000 --mirror
```
목표 모델(8B 교사 `goal8b_v0`, FSDP 두 장)과 체크포인트 옮기기(`r1wb.s2.delta`)는 이전과 같다(`train_goal`).

## 평가
- PC: `.agent/tests/r1wb/scripts/loop_eval.sh <S1 꼬리표> 1499 <머리> <토큰기> <고리> corebal`(추종·정지·걷는 중 연타), `LIBRARY=corebal TOKENS=v2d STEER=<고리> final_pc.sh <S1 꼬리표> 1499 <머리> react burst heldout wide student`.
- 정지는 걸음의 여러 순간에서 잰다(`eval_stream --stagger 0.32 --repeats 12`, 정지 276건). 연타 한계는 타격 시점 40가지(`impulse_probe --repeats 40`).
- 서버(교사 8B 폐루프): `GPU=3 S2PUSH=1 STEER=n18b HEAD=h12 TOKENS=v2d LIBRARY=corebal STAGGER=0.32 REPEATS=12 GOAL_MODEL=goal8b_v0 BACKBONE=Qwen3-VL-8B-Instruct scripts/06_evaluate.sh <S1 실행> 1499 s2 react wide burst heldout`. 목표 모델 인자를 빼면 표 목표(기계 차이용). 같은 실행 이름의 배터리는 하나씩만 돌린다.
- 타격 반응은 쌍둥이 대비 수치(걷는 중에는 위상에 흔들림)와 함께 `impulse_probe` 끝의 "against its own plan"(자기 계획 대비 이탈·팔 속도)을 본다.
- 계획의 물리 감사: `eval_stream --dump-plan <파일>` → PC `.agent/tests/r1wb/s2_plan_audit.py <파일>`, 정지별 디딤은 `stop_events.py`.

## 카메라 과제(보류)
- 장면 `scene.py`, 목표 수준 PPO `look.py`, VLM 학습에 섞기 `train_goal --look`. 교사가 기준을 통과한 뒤 다시 연다.
