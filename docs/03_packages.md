# 패키지

실행 노드 11개는 Rust ament_cargo다. humanoid_controller_msg는 메시지 생성용 ament_cmake다.

| 패키지 | 역할 |
|---|---|
| boot_manager | 시스템 6개·controller·remote_controller 상태를 순차 점검 |
| humanoid_controller_msg | 제조사 필드 기반 LowCmd·LowState·MotorCmd·MotorState·IMUState 정의 |
| humanoid_controller | 단독 USB 소유, 초기화·수신, 명령 변환·주기 송신 |
| remote_controller | MCU 리모컨 해석, `/joy`·`/remote_status` 발행 |
| safty_manager | 모션 안전 검사, 위험 명령 차단, `/safty_status` 발행 |
| state_manager | 관절 환산·TF, 상태 선택·전환, `/lowcmd` 발행 |
| zero_moment_action_manager | ZeroTorque(0): 목표·속도·토크·이득 0 |
| damping_action_manager | Damping(1): kp=0, kd=4; 머리 kd=0.1 |
| locked_standing_action_manager | MoveTo(6) → Stance(4), 최대 0.1rad/s로 기립·유지 |
| walking_action_manager | AmpLocomotion(811): 제조사 모델·IMU·조이스틱으로 50Hz 관절 목표 계산 |
| careful_walk_action_manager | CarefulWalk(812): 학습 보행. 추종기 R14@100(`r14s100_tracker.onnx`, 입력 1320)과 구운 참조 묶음(`calm43s_bank.bin`, 13 명령)을 바이너리에 내장한다. 스틱 한 방향이 묶음 명령 하나(속도 단계는 출발 순간에 정해져 유지), 놓으면 0.9~1.8초 뒤 서기. 진입 뒤와 리모컨이 끊긴 뒤에는 스틱을 0.2초 가운데 둔 다음에만 방향을 받고, 리모컨이 0.5초 끊기면 바로 서기로 간다. CPU 6 고정(고정에 실패하면 준비 안 됨으로 남아 진입이 막힌다), `--self-test`로 추론·틱 시간 점검 |
| safe_crouch_action_manager | SafeCrouch(813): 확정 웅크리기 정책, 높이 명령을 천천히 낮춰 유지 |

모션 경로: action → state_manager(`/lowcmd`) → safty_manager(`/safty_manager/lowcmd`) → controller → USB.
통과한 LowCmd의 필드·CRC는 바꾸지 않으며 과거 명령을 재송신하지 않는다.
동작 계산은 action, 선택은 state_manager, 안전 판정은 safty_manager, 장치 제어는 controller가 담당한다.
동작 계약은 `state_manager/src/motion/`를 참조한다. 별도 common 패키지는 없다.
safty_manager의 `layers/`는 command_format·joint_range·sudden_motion·speed·boot_status·feedback으로 나뉜다.
현재 모든 안전 레이어 OFF: 명령을 그대로 전달한다. 활성화 시 201은 동작 유지, 202는 차단·기립 요청이다. [판정 기준](05_dtc.md).
state_manager는 최신 완전한 관절 측정으로 기립을 생성하고, safty_manager는 해당 목표 ID·controller 세션을 확인한다.
센서·통신 오류가 남아 있어도 유효한 기립 명령은 전달한다. 범위·급변·속도 검사는 기립 명령에도 적용한다.
복구 후에도 기립을 유지한다. L2+B 감쇠는 우선하며, 전원·USB·모터 장애에서는 자세 유지를 보장하지 않는다.
설정: [safty_manager/config/default.yaml](../src/safty_manager/config/default.yaml). 각 검사와 warning/fault의 `enable`을 지정한다.
설정은 PC 원본 수정 후 다시 배포한다. 다른 설정 파일은 `AGROID_SAFTY_CONFIG`로 지정한다.
느린 기립 속도: [stance.json](../src/motion_manager/locked_standing_action_manager/config/stance.json)의 `joint_velocity_limit_rad_s`.

| controller 모듈 | 역할 |
|---|---|
| usb·tx | 장치 수명·입출력·F1 프레임 송신 |
| motor_init·head_init | 제조사 초기화·머리 기준점 탐색·검증·해제 |
| lowcmd·joint_wire | CRC·35슬롯 검증, 관절→26개 모터 변환 |
| motion_tx | USB 주기 실행, 제조사 380ms 명령 미수신 처리 |
| spsc·telemetry | 크기가 제한된 큐로 USB와 ROS 사이 데이터 전달 |
| rt·clock | FIFO 80·950µs 절대 주기·통계·단조 시계 |
| mcu·motor·bms·imu | MCU·모터·배터리·IMU 수신 해석 |
| motor_exchange·motor_wire·r1_profile | 제조사 교환·인코딩·기체별 주소와 단위 |

부팅은 zero_moment다. 기립 완료 후 R2+A는 제조사 보행, R2+X는 학습 보행이다. 스틱 중립(학습 보행은 서기로 돌아온 뒤) 후 L2+↑로 기립 복귀한다. 학습 동작(812·813)은 리모컨으로만 선택되고 요청 토픽은 거절한다(813은 리모컨 키가 없어 지금은 선택할 수 없다).
보행 오류·원본 기울기 초과·명령 중단의 기존 감쇠 동작은 유지한다. DDS는 [통신 설정](04_dds.md)을 따른다.
PC에서 `./scripts/03_deploy.sh`로 `src/`의 패키지를 자동 탐색해 모두 ARM64 Release 빌드한다. [옵션·검증·백업](01_initial_setup.md).
새 패키지는 배포 스크립트 수정 없이 포함된다. 새 노드의 자동 실행은 해당 런치 파일을 부팅 런치에 연결해야 한다.
R1 런타임은 `/root/workspace/agroid_v1/current/install/`이다. 바이너리·필수 설정만 배포하며 소스는 두지 않는다.
파일 내용·권한이 달라진 경우만 `boot_manager.service`를 재시작한다. 같은 배포는 기존 프로세스를 유지한다.
크로스 도구·R1 sysroot는 `.agent/ros2/lyrical-cross/`, 빌드·배포 묶음은 `.agent/deploy/`에 둔다.
