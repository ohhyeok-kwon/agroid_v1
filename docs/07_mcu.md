# MCU ↔ Linux

MCU 응답 하나에 BMS·모터·IMU 등이 함께 실린다. [BMS](06_bms.md)·[모듈](03_packages.md)·[바이트 규격](10_motor_protocol.md).
| USB 항목 | 값 |
|---|---|
| 장치·인터페이스 | `3c7b:4d8e`, `USB Quad_Serial`, interface 0 |
| IN | `0x81`, 4,768바이트, `FC FD`, 읽기 버퍼 6,000바이트 |
| OUT | `0x01`, 5,236바이트, `FB FD` |
| timeout | 초기화 각 1,000ms, 준비 중 각 100ms, Ready 이후 각 1ms |

`humanoid_controller`가 USB를 단독 소유하고 시작 시 아래 초기화를 한 번 수행한다.

| 단계 | 구현 |
|---|---|
| 기체 확인 | `sect=10,type=1,peripheral=0`, 기체·모터·보호 설정의 미지원 override 거부 |
| UART 설정 | E2의 baud·enable·timeout 전체 echo 검사 → 500ms |
| 초기 요청 | 동일 모터 payload의 F1 응답 검사 → 500ms·5초 |
| ReadingSpecs | 2,000주기 모터 명령/응답 교환; RX 정보 필드를 일반 측정값과 분리 |
| SpecZero | 250주기 조회 초기화 |
| WaitingFeedback | 250주기 26개 모터·머리 초기 위치 확인 |
| CenteringPitch | 상하축 29를 800주기에 걸쳐 0rad, Kp=30·Kd=2; 총 1,000주기 |
| SeekingYaw | 좌우축 30에 한 방향 +0.45N·m, 상하축은 중앙 유지 |
| Releasing | mode 0·목표/이득 0으로 3,000주기 해제 확인 |
| Ready | 기준점·피드백 확인 완료. 허용된 LowCmd를 기존 F1의 모터 슬롯에 기록 |

- 주기 950µs 절대 마감(`clock_nanosleep` ABSTIME, 원본 timerfd). 초과 시 몰아 실행하지 않는다. 통계는 `/motor_status.timing`.
- UART는 6Mbps×8·921600bps×1, 모터 26개와 보조 IMU `bus=6,id=0`이다.
- USB 작업·libusb 스레드는 원본처럼 격리 CPU 7(실패 시 경고 후 계속). 작업 스레드만 SCHED_FIFO 80, `mlockall`, 로그는 비차단 채널.
- MCU IRQ·CPU 주파수·idle 설정은 `scripts/daemon/04_realtime_tuning.sh`가 서비스 기동 전에 한다.
- 첫 yaw `mode=6,tau_raw=-1280,dq_raw=955`는 특수 요청이다. 실제 탐색은 `mode=1,tau_raw=-537`이다.
- yaw는 251주기마다 위치 변화 <0.05°를 5회 연속 확인하고 정착 위치를 **+115°**로 대응시킨다.
- 추가 검증: FOC 응답·지속적인 측정 부하·살아 있는 응답. 시작부터 끝점인 경우도 부하를 검증한다.
- 원본과 달리 정착 검증 후 즉시 해제한다. 탐색 한도 9초/9,000주기이며 실패를 완료로 처리하지 않는다.
- 구동 중 100ms 미수신·250ms 동일 응답, 범위·방향·정렬 오류는 해제 후 202. 종료·USB 오류 때도 해제를 시도한다.
- 실패·단절 후에는 자동 재탐색하지 않는다. 재초기화는 프로세스 재시작이 필요하다. [DTC](05_dtc.md).
- 원본의 5A/B6 주변장치 명령·flash 전신 보정값 처리·전신 좌표 변환은 미구현이다. IMU는 확인된 영점 [0,0,0] 기준이다.
- 동작 계산은 action 담당이다. controller는 단위 변환·주기 송신·끊김 감시를 맡는다. 초기화 후 모션 송신은 상시 동작한다.
- 근거: 원본 `0x101964e8`·`0x101979a0`·`0x10171628`, [오프라인 비교](../.agent/tmp/motor-init-comparison/native-comparison.json).
BMS·모터는 독립 모듈에서 해석하며 JSON String을 각각 1Hz·10Hz 발행한다. [토픽](08_topic.md).
몸체 IMU는 UART 6/id 0, FD EE·32바이트+CRC16(0xab12/0x8408)이다. 메인보드 IMU와 구분한다.
`imu.rs`가 제조사 축·단위로 환산하며 `/lowstate`에 100Hz 포함한다.
`/mcu_status`는 USB `0x138..0x1d8`의 160바이트와 `received_unix_ms`를 JSON으로 전달한다.
CRC 통과 후 RF 변경마다 전달한다. USB→ROS는 고정 크기 큐이며 JSON 해석·발행은 ROS 쪽에서 수행한다.
RF 조각은 USB `0x1a0..0x1ab`: 데이터 10바이트·조각 번호 1바이트. `remote_controller`가 40바이트로 조립한다.
원본 `0x101918c0`의 RF 헤더는 `55 51`, 마지막 2바이트는 `sum(packet[2:38])`의 big-endian 값이다.
`data`는 기대 모터 26·저장 슬롯 72, CRC·오류·원본 슬롯을 보존한다. Spec 중에는 `specification`에 저장한다.
`auxiliary_imu`·`unconfigured_slots`는 원본 보존 대상이며 모터 판정에서 제외한다.
`initialization.head`는 위치·토크·이동량·정착 횟수·yaw 원시 끝점·라디안 오프셋을 제공한다.
`head_reference_verified`와 `motion_ready`는 모든 준비·해제가 끝난 후에만 참이며 오류 시 해제된다.
현재 서비스 초기화: Ready·모터 26/26 정상. [검증 결과](../.agent/tmp/motion-execution/restart-verified.json).
