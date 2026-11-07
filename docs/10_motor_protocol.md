# 모터 프로토콜

근거: [제조사 SDK](https://github.com/unitreerobotics/r1-7a_sdk/blob/0803d033b530fcfa0e6fa9236de68c2773fc9402/thirdparty/motor_controller/src/MotorProtocol.cpp)와 백업의 `0x10184ff0`(TX)·`0x101854f8`(RX).
아래는 `FE EE`/`FC EE` 계열 명세다. 모든 정수는 little-endian; 다른 모터 계열과 혼용하지 않는다.
R1 USB 봉투는 [MCU](07_mcu.md) 참조. OUT 슬롯은 `0x138+bus×0x1f8+id×0x38`, +1 길이, +2부터 패킷이다.
IN 슬롯은 `0x1d8+bus×0x1f8+id×0x38`, +2부터 패킷이다. bus/id와 관절 번호는 별개다.

| TX 오프셋 | 크기/형식 | 의미 |
|---|---|---|
| 0 | 2/u8 | `FE EE` |
| 2 | 1/u8 | bit0~3 ID, bit4~6 mode, bit7 timeout 보호 요청 |
| 3 | 1/u8 | 제조사 명령/응답 교환 바이트; `motor_exchange` 처리 |
| 4 | 2/i16 | 목표 토크 |
| 6 | 2/i16 | 목표 속도 |
| 8 | 4/i32 | 목표 위치 |
| 12 | 2/i16 | Kp |
| 14 | 2/i16 | Kd |
| 16 | 4/u32 | 앞 16바이트 CRC32; 패킷 총 20바이트 |

| RX 오프셋 | 크기/형식 | 의미 |
|---|---|---|
| 0 | 2/u8 | `FC EE` |
| 2 | 1/u8 | bit0~3 ID, bit4~6 mode, bit7 timeout 상태 |
| 3, 4 | 1/i8, 1/u8 | 드라이버·권선 온도 °C |
| 5 | 1/u8 | 전압: 원시값÷2 V |
| 6, 8 | 각각 2/i16 | 토크·속도 원시값 |
| 10 | 4/i32 | 위치 원시값 |
| 14 | 4/u32 | 오류 비트맵 |
| 18 | 2/u16 | bit13~15 경고 코드; 나머지 예약 |
| 20, 21 | 각각 1/u8 | 제조사 부가 상태·명령/응답 교환 값. `reserve[]`·`exchange_byte`로 보존 |
| 22 | 4/u32 | 오프셋 2~21의 CRC32; 총 26바이트 |

- CRC: 초기 `FFFFFFFF`, 다항식 `04C11DB7`, LE u32마다 MSB부터 32회 처리, 최종 XOR 없음.
- 일반 mode 0 정지·1 FOC. 6/7은 모델별 특수 명령이며 일괄 오류 해제용으로 취급하지 않는다. ID는 0~14다.
- 위치·속도·토크는 별도 mode 번호가 아닌 FOC에서 목표/이득 조합으로 선택한다. [제어 범위](09_motor.md).
- `R=확인된 감속비`, 위치 rad·속도 rad/s·토크 N·m의 출력축 기준 송신 스케일:
  `qraw=round(q×R×32768/(2π))`, `dqraw=round(dq×R×64/(2π))`, `τraw=round(τ×2560/R)`.
  `Kpraw=round(Kp×12800/R²)`, `Kdraw=round(Kd×51200/R²)`; 수신은 역변환한다.
- SDK의 숫자 제한은 부호 있는 정수 인코딩 범위이며 관절의 허용 토크·속도·위치 한계가 아니다.
- 관절 환산은 `state_manager`: 제조사 R1 감속비·방향, 머리 yaw 오프셋, 원본 발목 삼각 보간을 적용한다.
- 머리 30: R=214.34805·방향=-1, q=5215.189, dq=0.4074367, τ=256000, Kp=1280000, Kd=128000000; 원본 정수 변환은 반올림 없이 절삭한다.
- 오류 예: bit0 과전류, 1/2 과전압, 3 저전압, 4/5/6 열, 9 권선, 10/12 엔코더, 22 하드웨어 과전류.
- 경고 bit0은 저전압 전원 이상; 미정의 비트도 보존한다. [전체 제조사 비트 해석](https://github.com/unitreerobotics/r1-7a_sdk/blob/0803d033b530fcfa0e6fa9236de68c2773fc9402/thirdparty/motor_controller/src/MotorProtocol.cpp#L425).
- 원본 `0x10171628`과 같이 6주기마다 RX[21] 토글·니블·난수 bit4로 TX[3]을 갱신한다. 32바이트 중 앞 16바이트를 AES-128로 변환한다.
- `/lowcmd` → safty_manager → `/safty_manager/lowcmd`는 humanoid_controller_msg/LowCmd(제조사 필드·35슬롯·CRC)를 그대로 전달한다.
- controller의 joint_wire가 26개 실제 모터로 변환하고 motion_tx가 F1에 기록한다.
- 제조사 첫 E2/F1: 일반 축 `mode=0,tau_raw=-128`; 머리 `bus=2,id=0`은 `mode=6,tau_raw=-1280,dq_raw=955`.
- 첫 E2/F1 값은 특수 요청이다. 후속 yaw 탐색은 mode 1·τraw=-537·q/dq/Kp/Kd=0·timeout 보호 요청을 주기 송신한다. [원본 비교](07_mcu.md).
- 머리 초기화는 `motor_init`·`head_init`에서 수행한다. Spec 단계 RX[3..14]는 식별/설정 원본이며 일반 측정값으로 발행하지 않는다.
- 보조 IMU는 `data.auxiliary_imu`, 미구성 주소는 `data.unconfigured_slots`에 보존한다. 모터 판정은 내장표의 26개 주소만 사용한다.
