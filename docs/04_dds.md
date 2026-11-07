# DDS

- R1(Lyrical)·PC(Jazzy)는 Fast DDS(`rmw_fastrtps_cpp`), `ROS_DOMAIN_ID=0`을 유지한다.
- PC 기본값은 `~/.bashrc`, Operation Center 기본값도 `0`이다. 명시한 환경변수는 우선한다.

```bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic list
```

**네트워크**

- R1은 자체 AP(`wlan1`, `192.168.12.1`)로만 PC와 통신한다. 주소는 고정이다.
- PC는 R1 직결 인터페이스 `wlr1`로만 DDS를 쓴다. 구성은 [연결](01_initial_setup.md)을 따른다.
- PC 설정 `~/.ros/fastdds_lan_only.xml`(화이트리스트 `wlr1`)은 `scripts/00_initialize.sh`가 관리한다.
- `~/.bashrc`의 `FASTRTPS_DEFAULT_PROFILES_FILE`이 이 파일을 지정한다. `wlr1`이 없으면 PC는 R1과 DDS 통신을 하지 않는다.
- Fast DDS는 노드 시작 시점의 인터페이스로 고정된다. R1 bring-up은 `wlan1` IPv4를 최대 30초 기다린 뒤 노드를 띄운다.
- 30초 안에 주소가 없으면 경고 후 시작하며, 이때는 서비스 재시작 전까지 PC와 통신하지 않는다.
- [토픽](08_topic.md)·[실행](03_packages.md).
