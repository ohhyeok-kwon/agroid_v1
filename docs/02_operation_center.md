# Operation Center

```bash
./scripts/01_operation_center.sh                        # 데스크톱
./scripts/01_operation_center.sh --headless --port 8797 # 브라우저
./scripts/01_operation_center.sh --build-only           # 빌드만
```

- PC의 Node.js·npm이 필요하고, 데스크톱은 Rust·WebKitGTK 4.1 개발 환경도 필요하다.
- 메뉴 아이콘은 `scripts/01_operation_center.sh`를 도메인 `0`으로 실행한다.
- 앱은 R1 패키지를 PC에서 빌드하지 않는다. ROS가 없어도 3D 모델 화면은 실행된다.
- ROS 기본 경로는 `/opt/ros/jazzy/setup.bash`이며 `AGROID_ROS_SETUP`으로 지정한다.
- 빌드·캐시는 `.agent/operation_center/`, PC ROS 결과는 `.agent/ros2/`에 둔다.
- ‘관절 위치’를 켜면 관절·골반 IMU·좌우 카메라 위치가 함께 표시된다.
- 좌표가 없는 메인보드 IMU는 위치 미확인으로 안내한다. [센서](11_sensors.md).
- 3D 화면은 R1 모델을 표시한다. 드래그 회전·휠 확대·우클릭 이동을 지원한다.
- 시작 창·카메라 값은 `app/operation_center/startup.json`, 모델 출처는 `assets/robot/source.json`에 둔다.
- 자세는 로컬 ROS의 `/tf`·`/tf_static`으로 수신한다. `/tf` 메시지 1개를 한 측정으로 보고 받은 관절을 갱신한다.
- R1이 뺀 관절은 마지막 각도를 유지하고 각도 칸 `—`, 상태 문구에 미수신 관절을 표시한다.
- 3D 모델은 TF의 관절 이동·회전을 직접 적용한다. 숫자 각도도 같은 TF에서 계산한다.
- 기준 좌표계는 `pelvis_link`다. 전신 위치·기울기와 IMU는 미측정이며 자세 각도 칸은 `—`로 표시한다.
- R1·PC 시스템 시각을 맞춰야 한다. 3초 이상 지난 TF는 사용하지 않고 수신 중단 시 마지막 자세를 표시한다.
- 같은 시각·1초 미만 역행 TF는 건너뛰고, 1초 이상 역행은 R1 시계 조정으로 보고 계속 표시한다.
- 실제 R1 발행 목록은 [토픽](08_topic.md)을 기준으로 한다.
