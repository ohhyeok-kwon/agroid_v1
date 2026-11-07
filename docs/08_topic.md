# 토픽

토픽·타입은 현재 소스 기준. R1 연결 중단으로 최신 실행 상태는 미확인이다.
`ROS_DOMAIN_ID=0`, Fast DDS. 실행 노드 9개. controller는 humanoid_controller, humanoid_controller_msg는 메시지 패키지다.

| 토픽 | 타입 | 발행자 → 구독자 | 주기 |
|---|---|---|---|
| `/bms_status` | std_msgs/String JSON | controller → boot_manager | 1Hz |
| `/motor_status` | std_msgs/String JSON | controller → boot_manager·state_manager | 10Hz |
| `/humanoid_status` | std_msgs/Int32MultiArray | boot_manager → state_manager·safty_manager | 1Hz |
| `/joint_states` | sensor_msgs/JointState | state_manager → 외부 | 10Hz |
| `/tf` | tf2_msgs/TFMessage | state_manager → Operation Center | 10Hz |
| `/tf_static` | tf2_msgs/TFMessage | state_manager → Operation Center | 시작 시 |
| `/robot_description` | std_msgs/String URDF | state_manager → 외부 | 시작 시 |
| `/mcu_status` | std_msgs/String JSON | controller → remote_controller | RF 변경 시·약 200Hz |
| `/joy` | sensor_msgs/Joy | remote_controller → state_manager·walking | 약 50Hz |
| `/remote_status` | std_msgs/String JSON | remote_controller → boot_manager·state_manager | 10Hz |
| `/humanoid_state/request` | std_msgs/String JSON | 외부 → state_manager | 요청 시 |
| `/humanoid_state` | std_msgs/String JSON | state_manager → safty_manager·외부 | 10Hz·안전 전환 시 |
| `/motion_manager/goal` | std_msgs/String JSON | state_manager → action 4개 | 10Hz |
| `/motion_manager/result` | std_msgs/String JSON | 선택된 action → state_manager | 50Hz |
| `/lowcmd` | humanoid_controller_msg/LowCmd | state_manager → safty_manager | 50Hz |
| `/safty_manager/lowcmd` | humanoid_controller_msg/LowCmd | safty_manager → controller | 수신 즉시 |
| `/lowstate` | humanoid_controller_msg/LowState | controller → walking·safty_manager·외부 | 100Hz |
| `/safty_status` | std_msgs/String JSON | safty_manager → state_manager·외부 | 10Hz·상태 변경 시 |
| `/walking_status` | std_msgs/String JSON | walking → state_manager | 10Hz |
| `/rosout` | rcl_interfaces/Log | 운영 노드 → 외부 | 이벤트 |
| `/parameter_events` | rcl_interfaces/ParameterEvent | 앱·ROS 도구 → 외부 | 이벤트 |

`/humanoid_status` 순서: CPU·온도·메모리·디스크·네트워크·배터리·humanoid_controller·remote_controller(8개).
`/lowcmd`·`/safty_manager/lowcmd`·`/lowstate`: Best Effort / Volatile / Keep Last 1. 제조사 필드·35슬롯을 유지한다.
LowCmd는 mode_pr=0(PR)·mode_machine=1(R1), CRC 필수. 구성되지 않은 9슬롯에는 명령을 넣지 않는다.
state_manager의 reserve는 [epoch 하위32, 상위32, 단조 ms, 목표 ID]다. 외부 LowCmd는 이 값을 필수로 요구하지 않는다.
명령 미수신 약 380ms는 controller가 실제 수신 시각으로 판단하고 정상 명령이 들어오면 자동 복귀한다.
`/lowstate`는 26모터 위치·속도·토크·온도·전압·오류와 유효 리모컨 패킷을 전달한다.
LowState는 몸체 IMU(wxyz·rad/s·m/s²·rad·℃)와 발목 속도 환산을 포함한다. ddq·sensor는 0이다.
`/motor_status.execution`: 활성·목표·끊김 원인·수신/거부 수. telemetry_dropped는 ROS 전달 손실 누적이다.
`sent_zero_frames`·`sent_damping_frames`·`sent_position_frames`는 USB 쓰기 성공 횟수다.
상태·정적 TF·URDF: Reliable / Transient Local / 1. `/joint_states`: Reliable / Volatile / 10, `/tf`: Reliable / Volatile / 100.
`/mcu_status`: Reliable / Volatile / 256. `/joy`: Best Effort / Volatile / 10. goal·request: Reliable / Volatile / 1, result: /10.
`/motor_status`·`/mcu_status`의 received_mono_ms는 나이·순서, received_unix_ms는 표시용이다.
`/safty_status`: 현재 전 레이어 OFF로 판정하지 않는다. 활성화 시 201 경고·202 fault와 원인을 발행하며 202는 기립을 요청한다.
`/joy.axes`=[lx,ly,rx,ry], buttons=R1,L1,start,select,R2,L2,F1,F2,A,B,X,Y,up,right,down,left.
요청값: zero_moment·damping·locked_standing·walking. 보행은 기립 완료·모델·IMU 준비가 필요하다.
L2를 먼저 누른 채 B→damping, damping에서 Y→zero_moment, zero/damping에서 ↑→locked_standing.
기립에서 R2+A→보행, 스틱 중립 후 L2+↑→기립. 보행 중 L2+B→즉시 감쇠.
`/walking_status`에는 준비·활성·선택 모델·목표 속도·계산 시간이 포함된다. 리모컨 미연결은 boot_manager에서 202다.
TF 루트는 pelvis_link. map·odom·월드 자세는 미발행이다. 정적 변환은 14개, 관절은 최대 26개다.
USB 단절·3초 초과·유효 관절 0개면 TF를 갱신하지 않는다. 나머지는 유효 관절만 보낸다.
재확인: `ros2 topic list -t`, `ros2 topic info -v /safty_manager/lowcmd`. [패키지](03_packages.md)·[DTC](05_dtc.md).
