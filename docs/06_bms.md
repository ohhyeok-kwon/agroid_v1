# BMS

`BMS → MCU → USB → humanoid_controller/bms → /bms_status → boot_manager`.
토픽은 `std_msgs/msg/String`의 JSON이며 수신 가능한 BMS 값과 원본 168바이트를 담는다.

| 상태 필드 | 의미 |
|---|---|
| `status_code`, `message` | 200/201/202 판정과 사유 |
| `usb_connected` | USB 인터페이스 열기·점유 성공 여부, BMS 응답 여부와 별개 |
| `received_unix_ms` | 마지막 정상 BMS 프레임 수신 시각(ms), 없으면 null |
| `age_ms` | 마지막 정상 프레임 경과 시간(ms), 없으면 null |
| `data` | 아래 데이터 객체. 정상 프레임을 한 번도 받지 못하면 null |

| `data` 필드 | 타입·개수 | 해석 |
|---|---|---|
| `version_high`, `version_low` | u8 각각 | 버전 바이트: 원본 [7], [6] |
| `fn` | u8 | 원본 [2], 지원 값 0x0f·0x78. 비트 의미 미확정 |
| `soc`, `soh` | u8 각각 | 잔량·건강도 %, 원본 u16÷10 후 정수화 |
| `cell_vol` | u16 ×28 | 확인된 셀 전압 슬롯, mV. 유효 셀 수와 배열 길이는 다름 |
| `bmsvoltage` | u32 ×2 | 원본 [16], [14]의 u16×10. 채널 의미·단위 미확정 |
| `current` | i32 | 원본 [132], 전류 스케일·충방전 부호 미확정 |
| `temperature` | i16 ×12 | 원본 [76..100]÷10 후 정수화. 센서 위치 미확정 |
| `cycle` | u16 | 원본 [18], 사이클 수 |
| `bmsstate` | 5슬롯 | [0]=u16@[4], [2]=u32@[152], [4]=u16@[8]. [1],[3]은 null |
| `manufacturer_date`, `reserve` | null | 유효한 수신 위치 미확인. 값으로 만들어 내지 않음 |
| `raw_frame` | u8 ×168 | 미해석 영역·CRC를 포함한 전체 BMS 원본 |

- 숫자 위치는 BMS 원본의 0부터 시작하는 바이트 오프셋이며 다중 바이트 정수는 little-endian이다.
- 지원 헤더는 `55 51 0F A8`·실측 `FE EF 78 A8`이며, 두 형식 모두 168바이트·같은 CRC 검사를 사용한다.
- 원본 제조사 DDS는 셀 40슬롯·전압 3슬롯이다. 현재 168바이트 프레임에서 해석하는 슬롯은 28·2개다.
- R1 제조사 guard 설정의 감시 셀은 9개다. 나머지 슬롯을 실제 센서 측정으로 단정하지 않는다.
- 제조사 파서는 제조일에 0을 넣는다. 현재 구현은 미확인 값에 null을 사용한다.
- SOC는 Ah·Wh 용량·남은 운전 시간이 아니다. 상태 비트·MCU 내부 배선 규격은 미확정이다.
- 데이터가 오래돼도 마지막 값을 남기되 `age_ms`와 202로 표시한다. 판정 기준은 [DTC](05_dtc.md).
- Linux 수신 구조·조각·CRC는 [MCU](07_mcu.md), 현재 실측 상태는 [토픽](08_topic.md) 참조.

근거: [제조사 필드 정의](../.agent/backup/20261101-r1-full-system/files/rootfs/usr/local/include/unitree/idl/hg/BmsState_.hpp),
[필드 변환 디컴파일](../.agent/analysis/20261101-r1-rea/exports/repaired-basic_service/derived_basic_service--ac61561df4d8/source/functions/0x10173130.c).
