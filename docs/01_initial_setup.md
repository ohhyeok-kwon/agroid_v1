# R1 연결·실행

사전 조건: AP 모드로 부팅한 R1(기본), 공유기 Wi-Fi에 연결된 Linux PC.
PC에는 Python 3·NetworkManager·iw·OpenSSH·rsync·busctl(Ubuntu 기본 제공)과 Wi-Fi 연결 2개를 지원하는 무선 칩이 필요하다.

```bash
./scripts/00_initialize.sh           # R1 직결 구성 후 양방향 동기화 유지
./scripts/00_initialize.sh --check   # 현재 직결·원격 도구·경로 점검, 변경 없음
./scripts/00_initialize.sh --dry-run # 동기화 계획 확인, 변경 없음
./scripts/02_connect_ssh.sh          # R1 셸
```

- PC는 공유기 Wi-Fi를 유지한 채 같은 무선 칩의 가상 인터페이스 `wlr1`로 R1 AP(`192.168.12.1`)에 붙는다.
- R1 설정은 바꾸지 않는다. R1은 공유기에 붙지 않으며 인터넷·NTP가 없다.
- 반복 실행해도 `wlr1`·메모리 전용 NetworkManager 프로필 `agroid-r1-ap`은 하나만 유지한다. 처음 생성할 때만 sudo가 필요하다.
- 저지연 설정: R1 프로필은 AP BSSID·채널 고정(백그라운드 스캔 없음)·절전 끔·기본 경로 미사용이다.
- 공유기 프로필은 PC 재부팅 전까지 현재 AP·채널에 고정한다. R1 AP SSID로 저장된 다른 프로필은 자동 연결을 끈다.
- 공유기 Wi-Fi와 R1 AP는 같은 채널(현재 5GHz 149)이어야 한다. 다르면 경고하며 지연이 커진다.
- PC 재부팅 후에는 터미널에서 다시 실행한다. R1 재부팅 후에는 실행 중인 동기화가 `wlr1`을 다시 연결한다.
- R1 전원이 꺼지면 동기화는 대기하며, 검색 완료 후 30초 간격으로 AP를 다시 확인한다. 새로 감지된 R1 AP에만 연결해, 꺼진 R1에 접속을 반복하며 Ubuntu `Connection failed` 알림을 만드는 일을 방지한다. 연결 시도가 시간 초과되면 남은 시도도 취소한다.
- GNOME Wi-Fi 목록을 열면 약 10초 동안 지연이 커진다.

R1 셸에서 실행한다.

```bash
cd /root/workspace/agroid_v1
./scripts/03_build.sh
./scripts/daemon/00_add_humanoid_service.sh
./scripts/daemon/03_restart_service.sh
systemctl status boot_manager.service
```

- 서비스는 기동 전 `scripts/daemon/04_realtime_tuning.sh`(MCU IRQ·CPU 7 설정)를 실행한다. 스크립트가 바뀌면 서비스를 다시 등록한다.
- 접속 설정·호스트 키는 `.agent/.r1-setup/`에서 관리한다.
- 비밀번호는 `R1_CONNECTION_INFO.local.json`의 `wifi.password`·`ssh.root_password`에 저장한다.
- 환경변수 `AGROID_R1_AP_PASSWORD`·`AGROID_R1_SSH_PASSWORD`로 덮어쓸 수 있다. 자격 증명은 Git에 넣지 않는다.
- 동기화 대상은 `src/`, `app/`, `scripts/`, `docs/` 등이며 제외 기준은 `.gitignore`다.
- 빌드·실행은 [패키지](03_packages.md), 통신 설정은 [DDS](04_dds.md)를 따른다.
