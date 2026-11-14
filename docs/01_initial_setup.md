# R1 연결·빌드·배포

사전 조건: AP 모드로 부팅한 R1(기본), 공유기 Wi-Fi에 연결된 Linux PC.
PC 프로젝트 루트에서 실행한다. 연결 도구는 Python 3·NetworkManager·iw·OpenSSH·busctl, 배포에는 rsync가 필요하다.

```bash
./scripts/00_initialize.sh           # R1 직결 구성·연결 유지
./scripts/00_initialize.sh --check   # 직결·R1 식별 정보 점검, 변경 없음
./scripts/00_initialize.sh --once    # 한 번 연결하고 종료
./scripts/02_connect_ssh.sh          # R1의 current 배포본에서 셸 열기
./scripts/03_deploy.sh --build-only  # PC에서 ARM64 Release 빌드·배포 묶음 검증
./scripts/03_deploy.sh --dry-run     # 빌드·R1 연결·배포 계획 점검, R1 변경 없음
./scripts/03_deploy.sh               # PC 빌드 → 변경된 결과물 배포 → 서비스 재시작
```

- PC는 공유기 Wi-Fi를 유지한 채 같은 무선 칩의 가상 인터페이스 `wlr1`로 R1 AP(`192.168.12.1`)에 붙는다.
- R1 네트워크 설정은 바꾸지 않는다. R1은 공유기에 붙지 않으며 인터넷·NTP가 없다.
- 반복 실행해도 `wlr1`·메모리 전용 NetworkManager 프로필 `agroid-r1-ap`은 하나만 유지한다. 최초 생성 시 sudo가 필요하다.
- R1 연결은 AP BSSID·채널 고정, 절전 끔, 기본 경로 미사용이다. 공유기 연결도 PC 재부팅 전까지 현재 AP·채널에 고정한다.
- 공유기 Wi-Fi와 R1 AP는 같은 채널이어야 한다. 다르면 지연이 커진다. GNOME Wi-Fi 목록을 열면 약 10초 지연이 커진다.
- PC 재부팅 후 연결 스크립트를 다시 실행한다. 실행 중 R1이 재부팅되면 자동으로 다시 연결한다.
- R1 전원이 꺼지면 검색 완료 후 30초 간격으로 새 AP를 확인한다. 시간 초과된 연결 시도는 취소한다.
- 상시 연결은 사용자 서비스 `agroid-r1-link`가 관리한다. 상태: `systemctl --user status agroid-r1-link`.
- `00_initialize.sh`는 파일을 전송하지 않는다. `trainer/`는 `trainer/scripts/00_initialize.sh`로 GPU 서버에만 동기화한다.

배포는 기존 R1용 GCC·ROS·sysroot(`.agent/ros2/lyrical-cross/`)로 PC에서 수행한다.
- `src/**/package.xml`의 `ament_cargo`·`ament_cmake` 패키지를 모두 자동 탐색한다. 수동 패키지 목록은 없다.
- 소스·빌드 대상·설치본의 목록이 다르거나 실행 파일이 빠지면 배포를 중단한다. Rust의 암시적 실행 파일도 검사한다.
- 컴파일 캐시는 유지하고 설치 폴더는 매번 다시 만들어 삭제·이름 변경된 패키지와 이전 실행 파일이 남지 않게 한다.
- 결과물은 `.agent/deploy/`에 생기며 실행 파일의 ARM64·glibc 2.31 호환성과 패키지 누락을 검사한다.
- R1에는 `/root/workspace/agroid_v1/.releases/<ID>/`의 실행 파일·런타임 설정·서비스 스크립트만 전송한다.
- `.incoming/<ID>/` 전송 완료·체크섬·실행 환경 검증 후 서비스를 적용하고 `current`를 갱신한다.
- 소스·trainer·빌드 도구·생성된 C/Rust 소스는 전송하지 않는다. 설정 변경도 PC 원본 수정 후 다시 배포한다.
- R1과 파일 내용·권한을 비교해 바뀐 경우만 서비스와 머리 초기화를 다시 시작한다. 동일하면 전송·재시작을 생략한다.
- `--no-restart`는 현재 프로세스를 유지하고 다음 기동부터 새 설치본을 쓴다.
- `--jobs N`으로 PC 컴파일 병렬 수를 정한다(기본 8). R1에서 빌드하거나 의존성을 설치하지 않는다.
- 서비스는 기동 전 `scripts/daemon/04_realtime_tuning.sh`로 MCU IRQ·CPU 7을 설정한다.
- 접속 설정·고정 호스트 키·자격 증명은 `.agent/.r1-setup/`에서 관리한다.
- 비밀번호는 `R1_CONNECTION_INFO.local.json`의 `wifi.password`·`ssh.root_password` 또는 환경변수 `AGROID_R1_AP_PASSWORD`·`AGROID_R1_SSH_PASSWORD`를 쓴다.
- 패키지 구조는 [패키지](03_packages.md), 통신 설정은 [DDS](04_dds.md)를 따른다.

배포 검증 기준: 2026-11-08 11:12 KST.
- R1 프로젝트 루트에는 `.releases/`, `current`, `.deploy.lock`만 남아 있으며 용량은 약 44MiB다.
- 소스 없는 상태의 재기동에서 노드 9개·모터/관절 TF 26개, 부팅 상태 8항목 모두 200을 확인했다.
- 동일 파일 재배포 시 서비스 PID 유지도 확인했다. 근거: [검증 기록](../.agent/backup/20261108-r1-runtime-migration/verification.json).
- 기존 설치·소스·서비스 백업은 PC의 [백업 폴더](../.agent/backup/20261108-r1-runtime-migration/)에 있다. 재생성 가능한 build·log는 백업에서 제외했다.
