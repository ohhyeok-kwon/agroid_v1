# R1 센서

IMU는 **골반 1개·상체 1개, 총 2개**다. 머리에는 좌우 카메라 2개가 있다.
위치는 제조사 URDF/STL, IMU 역할은 원본 `basic_service`와 [공식 R1 SDK](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/r1/low_level/r1_ankle_swing_example.cpp)를 대조했다.
SDK는 각각 `IMU.pelvis`·`IMU.torso`로 구분한다. 저장된 같은 USB 프레임에서도 두 입력의 CRC와 서로 다른 자세값을 확인했다.

## 장착 위치

아래는 제조사 모델 좌표이며 단위는 mm, X 전방·Y 좌측·Z 위다.

| 센서 | 기준 링크 | X, Y, Z |
|---|---|---|
| 골반 IMU | `pelvis_link` | 76.202482, 0.021991, −88.431464 |
| 왼쪽 카메라 렌즈 중심 | `head_yaw_link` | 78.059356, 30.000000, −25.672875 |
| 오른쪽 카메라 렌즈 중심 | `head_yaw_link` | 78.059356, −30.000000, −25.672875 |
| 상체 IMU | 상체 | XYZ 미확정: 확인한 공개 모델·원본 설정에 장착 좌표 없음 |

카메라 STL의 두 렌즈 중심은 `camera_link` 기준 `(11.8, ±30, 0)`mm, 간격은 60mm다.
`camera_joint`의 `(66.971, 0, −21.637)`mm와 Y축 회전 0.34907rad를 적용해 위 좌표를 계산했다.
골반 IMU는 `imu_in_pelvis_joint` 원점이다. 모델: [Unitree R1 URDF](https://github.com/unitreerobotics/unitree_ros/blob/3ae893d905d11795ddf1075825ee5010364a1109/robots/r1_description/R1.urdf).
상체 IMU의 상체 장착은 공식 SDK의 `IMU.torso`로 확인했다. 칩 중심의 mm 좌표는 확보하지 못했다.
공식 R1 URDF·[MJCF](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_r1/xmls/r1.xml)에 상체 IMU 프레임이 없고, 상체 STL에서도 칩을 식별하지 못했다.
정확한 좌표 확정에는 `waist_yaw_link` 기준 센서 중심 XYZ가 있는 제조사 도면 또는 실물 측정이 필요하다.
다른 기종의 좌표나 상체 링크 중심을 실제 센서 위치로 대입하지 않는다.

## IMU 수신

| 항목 | 골반 IMU | 상체 IMU |
|---|---|---|
| USB 위치 | UART `bus=6,id=0`, 슬롯 `0xda8` | MCU 블록 `0x138..0x1d8` |
| 원본 역할 | `LowState.imu_state`, 보행 입력 | `secondary_imu`, 상체 자세 |
| 검증 | `FD EE`, 32바이트+CRC16 | `FE FD`, 160바이트·CRC32 |
| 현재 제공 | `/lowstate.imu_state` 100Hz | `/mcu_status.data` 원시 블록 |

골반 데이터는 `0xdaa`부터: 0..3 헤더·온도·설정, 4..9 가속도, 10..15 각속도, 16..31 wxyz, 32..33 CRC16이다.
CRC16은 초기값 `0xab12`·반전 다항식 `0x8408`. quaternion은 float32, 가속도·각속도는 int16이다.
설정 `0x15`: ±3g·±2000°/s. 가속도 `[-z,-y,-x]×3/32768×9.80665` m/s²,
각속도 `[-z,-y,-x]×2000/32768×π/180` rad/s로 환산한다. 구형 설정 `0x00`도 지원한다.
상체 블록 offset 4/20/32/68은 wxyz/가속도/각속도/온도다. 원시 각속도 단위는 °/s다.
`imu.rs`가 해석하고 `walking_action_manager`가 골반 각속도·중력 방향을 사용한다. 두 IMU를 섞지 않는다.
원본 `imu_offset_json`·`secondaryimu_offset_json`의 `[0,0,0]`은 도(°) 단위 각도 보정이며 장착 XYZ가 아니다.
센서 칩 제조사·모델은 원본에 명시돼 있지 않다.

## 화면

Operation Center의 ‘관절 위치’가 관절·골반 IMU·좌우 카메라 위치를 함께 켜고 끈다.
상체 IMU는 위치를 임의로 찍지 않고 별도 안내한다. [MCU](07_mcu.md)·[토픽](08_topic.md)·[패키지](03_packages.md).
