#!/usr/bin/env bash
# Join the R1 AP through a second laptop Wi-Fi link, then reconcile project sources in both directions.
set -euo pipefail

r1_scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r1_agent_tmp="$(cd -- "$r1_scripts_dir/.." && pwd)/.agent/tmp"
mkdir -p -- "$r1_agent_tmp"
export TMPDIR="$r1_agent_tmp" TMP="$r1_agent_tmp" TEMP="$r1_agent_tmp"
export PYTHONDONTWRITEBYTECODE=1

if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3가 필요합니다. Ubuntu: sudo apt install python3\n' >&2
    exit 1
fi

# Keep standard input available for password prompts and interactive SSH.
exec python3 -c '
import os
import sys

script = sys.argv.pop(1)
sys.argv[0] = script
sys.path[0] = os.path.dirname(os.path.realpath(script))
with os.fdopen(3) as source:
    code = compile(source.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
' "$r1_scripts_dir/00_initialize.sh" "$@" 3<<'R1_INITIALIZE_PYTHON'
"""노트북 이중 Wi-Fi로 R1 AP에 직결하고 PC ↔ R1 프로젝트 소스를 동기화합니다.

노트북의 공유기 Wi-Fi 연결은 유지하고, 같은 무선 칩에 고정 이름의 가상 인터페이스를
하나 만들어 R1 AP에 붙습니다. R1의 네트워크·AP 설정은 변경하지 않습니다.
여러 번 실행해도 가상 인터페이스·NetworkManager 프로필·설정 파일은 하나로 유지됩니다.
Python 표준 라이브러리만 사용합니다.
"""

import argparse
import contextlib
import math
import re
import stat
from contextlib import contextmanager
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".agent" / ".r1-setup"
AGENT_TMP = ROOT / ".agent" / "tmp"
LAST = STATE / ".r1_wifi.last.local.json"
KNOWN_HOSTS = STATE / ".r1_known_hosts"
SSH_CONFIG = STATE / "ssh_config"
AP_IP = "192.168.12.1"
AP_NETWORK = ipaddress.IPv4Network("192.168.12.0/24")
AP_SSID = "R1_03238_492213bc"
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF+jU1uAy8hj10hw7osT3KvhgVFwzhicQTiCX0a56UIB"
ROBOT_MAC = "94:ba:06:fc:1b:82"
ROBOT_MODEL = "Unitree G1 Board"
# Exactly one virtual interface and one in-memory NetworkManager profile, both
# with fixed identities: repeated runs reuse them and never add more.
LINK_INTERFACE = "wlr1"
LINK_PROFILE = "agroid-r1-ap"
LINK_UUID = "2e991939-9e5a-587b-b2cf-a8879418d368"
LINK_RETRY = 30
AP_CACHE_MAX_AGE = 15
CLOCK_TOLERANCE = 1.0
DDS_PROFILE = Path.home() / ".ros" / "fastdds_lan_only.xml"
DDS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!-- scripts/00_initialize.sh가 관리합니다. PC ROS 2(Fast DDS)는 R1 직결 인터페이스로만 통신합니다. -->
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>lan_udp</transport_id>
        <type>UDPv4</type>
        <!-- Fast DDS uses the smallest transport limit for local SHM too. -->
        <maxMessageSize>65500</maxMessageSize>
        <interfaceWhiteList>
          <address>{interface}</address>
        </interfaceWhiteList>
        <!-- 큰 메시지 버스트에 대비한 수신 버퍼(커널 net.core.rmem_max 이하로 자동 제한) -->
        <receiveBufferSize>33554432</receiveBufferSize>
        <sendBufferSize>4194304</sendBufferSize>
      </transport_descriptor>
      <transport_descriptor>
        <transport_id>local_shm</transport_id>
        <type>SHM</type>
        <segment_size>134217728</segment_size>
        <maxMessageSize>33554432</maxMessageSize>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="lan_only" is_default_profile="true">
      <rtps>
        <userTransports>
          <transport_id>lan_udp</transport_id>
          <transport_id>local_shm</transport_id>
        </userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
"""


def run(argv, *, payload=None, timeout=20, check=True, env=None):
    result = subprocess.run(argv, input=payload, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, **(env or {}), "LC_ALL": "C"})
    if check and result.returncode:
        raise RuntimeError(f"{argv[0]} 실패: {result.stderr.strip()[-700:]}")
    return result.stdout.strip()


def write_if_changed(path, content):
    try:
        if path.read_text() == content:
            return False
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".r1-write-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def save(path, value):
    return write_if_changed(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def prepare_state():
    """Keep cache and pinned SSH identity outside the archived backup."""
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not KNOWN_HOSTS.exists():
        with KNOWN_HOSTS.open("x") as stream:
            os.chmod(KNOWN_HOSTS, 0o600)
            stream.write(f"{AP_IP} {HOST_KEY}\n")


def require_tools(names):
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise RuntimeError("필요한 PC 명령이 없습니다: " + ", ".join(missing)
                           + ". Ubuntu: sudo apt install network-manager iproute2 iw openssh-client rsync systemd")


def sudo(*argv, timeout=20):
    """Only iw changes need root. Ask once in a terminal; never block a background run."""
    require_tools(("sudo",))
    cached = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=15)
    if cached.returncode:
        if not sys.stdin.isatty():
            raise RuntimeError("노트북 무선 설정에 sudo 인증이 필요합니다. 터미널에서 이 스크립트를 한 번 실행하세요.")
        print("노트북 무선 설정 변경에 sudo 인증이 필요합니다.", flush=True)
        if subprocess.run(["sudo", "-v"], timeout=120).returncode:
            raise RuntimeError("sudo 인증에 실패했습니다. 노트북 무선 설정을 바꾸지 않았습니다.")
    return run(["sudo", "-n", *argv], timeout=timeout)


def nm(*args, wait=30):
    return run(["nmcli", "--wait", str(wait), *args], timeout=wait + 5)


def field(uuid, name, secret=False):
    args = ["--show-secrets"] if secret else []
    value = nm(*args, "--escape", "no", "--get-values", name, "connection", "show", "uuid", uuid)
    return "" if value == "--" else value


def wifi_devices():
    """Wi-Fi network interfaces mapped to the radio (phy) they run on."""
    devices = {}
    for path in Path("/sys/class/net").iterdir():
        phy = path / "phy80211" / "name"
        if phy.is_file():
            devices[path.name] = phy.read_text().strip()
    return devices


def interface_name(explicit):
    """The laptop's router Wi-Fi: the one Wi-Fi interface that is not the R1 link."""
    if explicit:
        if explicit == LINK_INTERFACE:
            raise RuntimeError(LINK_INTERFACE + "는 R1 직결용입니다. 공유기 Wi-Fi 인터페이스를 지정하세요.")
        return explicit
    devices = sorted(name for name in wifi_devices() if name != LINK_INTERFACE)
    if len(devices) != 1:
        raise RuntimeError("공유기용 Wi-Fi 인터페이스를 하나로 정할 수 없습니다. --wifi-interface로 지정하세요.")
    return devices[0]


def station(interface):
    """Live association of a station interface, or None when it is not associated."""
    link = run(["iw", "dev", interface, "link"], check=False)
    bssid = re.search(r"^Connected to ([0-9a-fA-F:]{17})", link, re.M)
    channel = re.search(r"channel (\d+) \((\d+) MHz\)", run(["iw", "dev", interface, "info"], check=False))
    if not bssid or not channel:
        return None
    ssid = re.search(r"^\s*SSID: (.*)$", link, re.M)
    return {"bssid": bssid.group(1).upper(), "ssid": ssid.group(1) if ssid else "",
            "channel": int(channel.group(1)), "freq": int(channel.group(2))}


def device_connection(interface):
    """(NetworkManager device state, active profile UUID); state is None for an unknown device."""
    try:
        values = nm("--get-values", "GENERAL.STATE,GENERAL.CON-UUID", "device", "show", interface).splitlines()
    except RuntimeError:
        return None, ""
    values += ["", ""]
    state = values[0].split()[0] if values[0] else ""
    return (int(state) if state.isdigit() else None), ("" if values[1] in ("", "--") else values[1])


def direct_route(host, interface):
    ip = ipaddress.IPv4Address(host)
    route = json.loads(run(["ip", "-j", "-4", "route", "get", str(ip)]))[0]
    if route.get("dev") != interface or route.get("gateway"):
        raise RuntimeError(f"R1({ip})으로 가는 {interface} 직접 경로를 확인하지 못했습니다.")


def link_ready():
    """Read-only: the R1 profile is active on the link with an R1 AP address and a direct route."""
    state, uuid = device_connection(LINK_INTERFACE)
    if state != 100 or uuid != LINK_UUID or station(LINK_INTERFACE) is None:
        return False
    try:
        rows = json.loads(run(["ip", "-j", "-4", "address", "show", "dev", LINK_INTERFACE]))
        if not any(ipaddress.IPv4Address(a["local"]) in AP_NETWORK
                   for row in rows for a in row.get("addr_info", [])):
            return False
        direct_route(AP_IP, LINK_INTERFACE)
    except (RuntimeError, ValueError, KeyError, IndexError):
        return False
    return True


def interface_mac(interface):
    return Path("/sys/class/net", interface, "address").read_text().strip()


def local_mac(mac):
    """A stable locally administered address derived from the router Wi-Fi MAC."""
    first = int(mac[:2], 16)
    first = first ^ 0x04 if first & 0x02 else first | 0x02
    return f"{first:02x}{mac[2:]}".lower()


def ensure_interface(router):
    """Create the R1 link interface once; an existing one on the same radio is reused."""
    devices = wifi_devices()
    phy = devices.get(router)
    if phy is None:
        raise RuntimeError(router + "는 Wi-Fi 인터페이스가 아닙니다.")
    existing = devices.get(LINK_INTERFACE)
    if existing is not None:
        if existing != phy:
            raise RuntimeError(f"{LINK_INTERFACE}가 다른 무선 칩({existing})에 있습니다. 정리한 뒤 다시 실행하세요.")
        if not re.search(r"^\s*type managed$", run(["iw", "dev", LINK_INTERFACE, "info"]), re.M):
            raise RuntimeError(LINK_INTERFACE + "가 station(managed) 모드가 아닙니다.")
        return False
    if not re.search(r"#\{[^}]*\bmanaged\b[^}]*\} <= ([2-9]|\d{2,})", run(["iw", "phy", phy, "info"])):
        raise RuntimeError(f"{phy} 무선 칩은 Wi-Fi 연결 2개를 동시에 지원하지 않습니다.")
    mac = local_mac(interface_mac(router))
    print(f"R1 직결용 가상 Wi-Fi {LINK_INTERFACE}를 만듭니다 ({phy}, {mac}).", flush=True)
    sudo("iw", "phy", phy, "interface", "add", LINK_INTERFACE, "type", "managed", "addr", mac)
    return True


def ensure_nm_device():
    for _ in range(40):
        if device_connection(LINK_INTERFACE)[0] is not None:
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("NetworkManager가 " + LINK_INTERFACE + "를 인식하지 못했습니다.")
    # Only this script connects the R1 link; NetworkManager must never put another
    # saved Wi-Fi on it (that could pull the radio onto a second channel).
    try:
        nm("device", "set", LINK_INTERFACE, "managed", "yes", "autoconnect", "no")
    except RuntimeError:
        if not link_ready():  # a background run may lack permission; a live link was set before
            raise


def scan_list(interface, rescan):
    # NM retains old APs and may even add a synthetic AP while connecting.
    # Only an actual, recent sighting is evidence that the powered-off R1 is back.
    since = int(time.clock_gettime(time.CLOCK_BOOTTIME))
    if rescan != "yes":
        since -= AP_CACHE_MAX_AGE
    rows = nm("--terse", "--escape", "yes", "--fields", "SSID,BSSID,CHAN,FREQ,SIGNAL,DBUS-PATH", "device", "wifi",
              "list", "ifname", interface, "--rescan", rescan, wait=30)
    found = []
    for line in rows.splitlines():
        values = [v.replace("\\:", ":").replace("\\\\", "\\") for v in re.split(r"(?<!\\):", line)]
        if (len(values) != 6 or values[0] != AP_SSID or not values[2].isdigit()
                or not re.fullmatch(r"[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}", values[1])):
            continue
        try:
            seen = run(["busctl", "--system", "get-property", "org.freedesktop.NetworkManager", values[5],
                        "org.freedesktop.NetworkManager.AccessPoint", "LastSeen"], timeout=5)
            if int(seen.split()[1]) < max(0, since):
                continue
            found.append({"bssid": values[1].upper(), "channel": int(values[2]),
                          "freq": int(values[3].split()[0]), "signal": int(values[4] or 0)})
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError, IndexError):
            continue  # APs can disappear between the list and the LastSeen query.
    return found


def robot_ap(router):
    """Use recent scan results first; scan once when there is no fresh R1 sighting."""
    for interface, rescan in ((router, "no"), (LINK_INTERFACE, "no"), (LINK_INTERFACE, "yes")):
        if rescan == "yes":
            print("R1 AP를 찾기 위해 한 번 스캔합니다 (약 10초 동안 무선 지연이 커집니다).", flush=True)
        try:
            found = scan_list(interface, rescan)
        except RuntimeError:
            continue
        if found:
            return max(found, key=lambda item: item["signal"])
    raise RuntimeError(f"R1 AP({AP_SSID})가 보이지 않습니다. R1 전원과 AP 부팅을 확인하세요.")


def link_settings(ap):
    return {
        "connection.id": LINK_PROFILE, "connection.interface-name": LINK_INTERFACE,
        "connection.autoconnect": "no",
        "802-11-wireless.ssid": AP_SSID, "802-11-wireless.mode": "infrastructure",
        # Locking BSSID and channel keeps activation scans on one channel and makes
        # NetworkManager omit wpa_supplicant background scans for this link.
        "802-11-wireless.bssid": ap["bssid"], "802-11-wireless.band": "a" if ap["freq"] >= 4900 else "bg",
        "802-11-wireless.channel": str(ap["channel"]), "802-11-wireless.cloned-mac-address": "preserve",
        "802-11-wireless.powersave": "disable",
        "802-11-wireless-security.key-mgmt": "wpa-psk", "802-11-wireless-security.psk": connection_password(
            "AGROID_R1_AP_PASSWORD", "wifi", "password"),
        "ipv4.method": "auto", "ipv4.never-default": "yes", "ipv4.ignore-auto-dns": "yes",
        "ipv4.ignore-auto-routes": "yes", "ipv6.method": "disabled",
    }


def disable_other_robot_profiles():
    """A saved R1 AP profile must not autoconnect: it would move the router Wi-Fi onto the R1 AP."""
    for line in nm("--terse", "--fields", "UUID,TYPE,AUTOCONNECT", "connection", "show").splitlines():
        uuid, kind, autoconnect = (line.split(":") + ["", ""])[:3]
        if (uuid != LINK_UUID and kind == "802-11-wireless" and autoconnect == "yes"
                and field(uuid, "802-11-wireless.ssid") == AP_SSID):
            nm("connection", "modify", "uuid", uuid, "connection.autoconnect", "no")
            print(f"저장된 R1 AP 프로필({field(uuid, 'connection.id')})의 자동 연결을 껐습니다.", flush=True)


def ensure_profile(ap):
    """Keep exactly one in-memory R1 profile; True when it was created or changed."""
    desired = link_settings(ap)
    profiles = []
    for line in nm("--terse", "--escape", "yes", "--fields", "NAME,UUID", "connection", "show").splitlines():
        name, _, uuid = line.rpartition(":")
        profiles.append(uuid)
        if name.replace("\\:", ":") == LINK_PROFILE and uuid != LINK_UUID:
            nm("connection", "delete", "uuid", uuid)
    if LINK_UUID not in profiles:
        nm("connection", "add", "type", "wifi", "ifname", LINK_INTERFACE, "con-name", LINK_PROFILE,
           "ssid", AP_SSID, "save", "no", "connection.uuid", LINK_UUID,
           *[item for pair in desired.items() for item in pair])
        return True
    changed = []
    for key, value in desired.items():
        current = field(LINK_UUID, key, secret=key.endswith(".psk"))
        if (current.upper() if key.endswith(".bssid") else current) != value:
            changed += [key, value]
    if changed:
        nm("connection", "modify", "--temporary", "uuid", LINK_UUID, *changed)
    return bool(changed)


def cancel_pending_link():
    """Cancel only our unfinished activation; nmcli timing out does not cancel NM."""
    state, uuid = device_connection(LINK_INTERFACE)
    if uuid == LINK_UUID and state is not None and 40 <= state < 100:
        nm("device", "disconnect", LINK_INTERFACE, wait=5)


def activate_link(wait=45, quiet=False):
    if not quiet:
        print(f"{LINK_INTERFACE}를 R1 AP({AP_SSID})에 연결합니다.", flush=True)
    try:
        nm("connection", "up", "uuid", LINK_UUID, "ifname", LINK_INTERFACE, wait=wait)
        deadline = time.monotonic() + 10
        while not link_ready():  # the DHCP address from the R1 AP can trail activation
            if time.monotonic() >= deadline:
                raise RuntimeError(f"R1 AP 연결 또는 {AP_NETWORK} 주소를 확인하지 못했습니다.")
            time.sleep(0.5)
    except BaseException:
        with contextlib.suppress(RuntimeError, OSError, subprocess.SubprocessError):
            cancel_pending_link()
        raise


def tune_router_link(router):
    """Lock the router profile to the current AP until NetworkManager restarts.

    NetworkManager omits wpa_supplicant background scans for a BSSID-locked
    profile; those periodic full-band scans stall both links on this one radio.
    """
    current = station(router)
    _, uuid = device_connection(router)
    if current is None or not uuid:
        print("주의: 공유기 Wi-Fi가 연결되어 있지 않습니다. R1 직결만 구성합니다.", flush=True)
        return None
    wanted = [("802-11-wireless.bssid", current["bssid"])]
    if 2400 <= current["freq"] < 5900:
        wanted += [("802-11-wireless.band", "a" if current["freq"] >= 4900 else "bg"),
                   ("802-11-wireless.channel", str(current["channel"]))]
    changed = [item for key, value in wanted
               if (field(uuid, key).upper() if key.endswith(".bssid") else field(uuid, key)) != value
               for item in (key, value)]
    if not changed:
        return current
    print("공유기 Wi-Fi를 현재 AP·채널에 고정합니다 (재부팅 전까지, 백그라운드 스캔 끔). "
          "몇 초간 끊겼다 다시 연결됩니다.", flush=True)
    nm("connection", "modify", "--temporary", "uuid", uuid, *changed)
    nm("connection", "up", "uuid", uuid, "ifname", router, wait=45)
    current = station(router)
    if current is None:
        raise RuntimeError("공유기 Wi-Fi 재연결을 확인하지 못했습니다.")
    return current


def power_save_off(interface):
    if "Power save: on" in run(["iw", "dev", interface, "get", "power_save"], check=False):
        sudo("iw", "dev", interface, "set", "power_save", "off")
        return True
    return False


def ensure_dds_profile():
    DDS_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    return write_if_changed(DDS_PROFILE, DDS_TEMPLATE.format(interface=LINK_INTERFACE))


def ssh_path(path):
    """Quote a path for OpenSSH's config parser, including literal percent signs."""
    value = str(path).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return '"' + value + '"'


def ssh_argv(args, host, *, allow_multiplex=False):
    # The R1 is reachable only over the direct link; bind to it explicitly.
    return ["ssh", "-F", "/dev/null", "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=password", "-o", "NumberOfPasswordPrompts=1",
            "-o", "BatchMode=no", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + ssh_path(KNOWN_HOSTS),
            "-o", "HostKeyAlias=" + AP_IP, "-o", "ConnectTimeout=6",
            "-o", "BindInterface=" + LINK_INTERFACE] + (
                [] if allow_multiplex else ["-o", "ControlMaster=no", "-o", "ControlPath=none"]) + [
                "root@" + str(ipaddress.IPv4Address(host))]


def connection_password(environment, section, key):
    password = os.environ.get(environment, "")
    credentials = STATE / "R1_CONNECTION_INFO.local.json"
    if not password and credentials.exists():
        password = json.loads(credentials.read_text()).get(section, {}).get(key, "")
    if not isinstance(password, str) or not password:
        raise RuntimeError(f"R1 비밀번호가 없습니다. {credentials}의 {section}.{key} 또는 {environment}를 설정하세요.")
    return password


def ssh_password():
    return connection_password("AGROID_R1_SSH_PASSWORD", "ssh", "root_password")


def ssh_run(args, host, command, *, payload=None, timeout=35):
    # This helper is on the laptop only. No authorized_keys or SSH config is
    # installed on the robot. OpenSSH receives the password through ASKPASS.
    password = ssh_password()
    AGENT_TMP.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agroid-r1-askpass-", dir=AGENT_TMP) as folder:
        helper = Path(folder) / "askpass"
        helper.write_text('#!/bin/sh\nprintf "%s\\n" "$AGROID_R1_SSH_PASSWORD"\n')
        helper.chmod(0o700)
        return run([*ssh_argv(args, host), command], payload=payload, timeout=timeout,
                   env={"SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
                        "DISPLAY": "agroid:0", "AGROID_R1_SSH_PASSWORD": password})


def verify_robot(args):
    """Read-only identity check over SSH pinned to the R1 host key."""
    lines = ssh_run(args, AP_IP, "id -u; cat /sys/class/net/wlan0/address; "
                    "tr -d '\\0' < /proc/device-tree/model; echo; "
                    "cat /proc/sys/kernel/random/boot_id; date +%s.%N").splitlines()
    if len(lines) != 5 or lines[0] != "0" or lines[1].lower() != ROBOT_MAC or lines[2] != ROBOT_MODEL:
        raise RuntimeError("R1 식별 정보(root·MAC·모델)가 일치하지 않습니다.")
    return {"boot_id": lines[3], "clock_offset": round(float(lines[4]) - time.time(), 3)}


def find_existing_connection(args=None):
    """Read-only check of the live R1 link for helper tools; None when it is not up."""
    if not link_ready():
        return None
    return {"ip": AP_IP, "interface": LINK_INTERFACE, **verify_robot(args)}


def reconnect_link():
    """Wait for a fresh R1 sighting before asking NM to activate the link again."""
    if LINK_INTERFACE not in wifi_devices():
        raise RuntimeError(LINK_INTERFACE + "가 없습니다. ./scripts/00_initialize.sh를 다시 실행하세요.")
    if link_ready():
        return
    cancel_pending_link()
    # If association was lost before NM noticed, stop its recovery attempt too.
    state, uuid = device_connection(LINK_INTERFACE)
    if state == 100 and uuid == LINK_UUID and station(LINK_INTERFACE) is None:
        nm("device", "disconnect", LINK_INTERFACE, wait=5)
    found = scan_list(LINK_INTERFACE, "yes")
    if not found:
        raise RuntimeError("R1 AP가 보이지 않습니다. 전원이 켜지면 자동으로 다시 연결합니다.")
    ensure_profile(max(found, key=lambda item: item["signal"]))
    activate_link(wait=20, quiet=True)


def connect(args):
    """Idempotent setup: every step checks the live state and changes only what differs."""
    router = args.wifi_interface
    disable_other_robot_profiles()
    router_link = tune_router_link(router)
    ensure_interface(router)
    ensure_nm_device()
    live = station(LINK_INTERFACE) if link_ready() else None
    ap = live or robot_ap(router)
    if ensure_profile(ap) or live is None:
        activate_link()
        live = station(LINK_INTERFACE)
        if live is None:
            raise RuntimeError(LINK_INTERFACE + " 연결 정보를 확인하지 못했습니다.")
    for interface in (router, LINK_INTERFACE):
        if power_save_off(interface):
            print(interface + " Wi-Fi 절전을 껐습니다.", flush=True)
    if ensure_dds_profile():
        print(f"PC ROS 2(Fast DDS)를 {LINK_INTERFACE} 전용으로 설정했습니다: {DDS_PROFILE}", flush=True)
    identity = verify_robot(args)
    if router_link and router_link["channel"] != live["channel"]:
        print(f"주의: 공유기 Wi-Fi(채널 {router_link['channel']})와 R1 AP(채널 {live['channel']})가 달라 "
              "무선 칩이 두 채널을 번갈아 씁니다 (지연 p90 약 100ms). "
              f"공유기 Wi-Fi를 채널 {live['channel']}에 맞추세요.", flush=True)
    if abs(identity["clock_offset"]) > CLOCK_TOLERANCE:
        print(f"주의: R1 시계가 PC보다 {identity['clock_offset']:+.1f}초 다릅니다. "
              "Operation Center는 3초 넘게 지난 TF를 표시하지 않습니다.", flush=True)
    value = {"ip": AP_IP, "interface": LINK_INTERFACE, "router_interface": router, "ssid": AP_SSID,
             "bssid": live["bssid"], "channel": live["channel"], "mac": ROBOT_MAC, "model": ROBOT_MODEL,
             "boot_id": identity["boot_id"]}
    save(LAST, value)
    return value


def finish_setup(args, value):
    config = ("Host r1\n"
              f"    HostName {AP_IP}\n"
              "    User root\n"
              "    Port 22\n"
              f"    BindInterface {LINK_INTERFACE}\n"
              "    PreferredAuthentications password\n"
              "    PubkeyAuthentication no\n"
              "    StrictHostKeyChecking yes\n"
              f"    UserKnownHostsFile {ssh_path(KNOWN_HOSTS)}\n"
              f"    HostKeyAlias {AP_IP}\n"
              "    ConnectTimeout 8\n")
    write_if_changed(SSH_CONFIG, config)
    command = ["ssh", "-F", str(SSH_CONFIG), "r1"]
    print(f"\nR1 직결 확인 완료: {LINK_INTERFACE} → {AP_SSID} (채널 {value['channel']}), R1 {AP_IP}", flush=True)
    print("SSH 접속: " + shlex.join(command))
    return command


# Source reconciliation ported from GreenFarm scripts/01_sync_remote.sh.
DEFAULT_REMOTE_DIR = "/root/workspace/agroid_v1"


def validate_remote_directory(value):
    path = Path(value)
    if (not path.is_absolute() or len(path.parts) < 4 or ".." in path.parts
            or str(path) != value or any(ord(c) < 32 for c in value)
            or path.parts[1] in ("etc", "usr", "bin", "sbin", "lib", "lib64", "proc", "sys", "dev", "run", "unitree")):
        raise SyncError("R1의 전용 프로젝트 절대 경로를 지정하세요: " + DEFAULT_REMOTE_DIR)
    return value


REMOTE_SYNC_PROJECT = r'''
import json, os, shutil, sys
from pathlib import Path
path, create = Path(sys.argv[-2]), sys.argv[-1] == '1'
for program in ('rsync', 'python3', 'flock', 'pgrep'):
    if not shutil.which(program):
        raise RuntimeError('R1에 필요한 도구가 없습니다: ' + program)
descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
try:
    for part in path.parts[1:]:
        try:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        except FileNotFoundError:
            if not create:
                print(json.dumps({'exists': False}))
                sys.exit(0)
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    os.fchdir(descriptor)
    if not os.access('.', os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError('R1 프로젝트 폴더 읽기/쓰기 권한이 필요합니다.')
    print(json.dumps({'exists': True}))
finally:
    os.close(descriptor)
'''


@contextmanager
def sync_authentication():
    """Use the same password provider as setup; never place a secret in argv."""
    with tempfile.TemporaryDirectory(prefix="r1-sync-auth-", dir=temporary_root(ROOT)) as directory:
        helper = Path(directory) / "askpass"
        helper.write_text('#!/bin/sh\nprintf "%s\\n" "$AGROID_R1_SSH_PASSWORD"\n')
        helper.chmod(0o700)
        environment = {"SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
                       "DISPLAY": "agroid:0", "AGROID_R1_SSH_PASSWORD": ssh_password(), "LC_ALL": "C"}
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@contextmanager
def preview_identity():
    """A fresh checkout can inspect the pinned robot without saving connection state."""
    global KNOWN_HOSTS
    if KNOWN_HOSTS.is_file():
        yield
        return
    with tempfile.TemporaryDirectory(prefix="r1-preview-", dir=temporary_root(ROOT)) as directory:
        original = KNOWN_HOSTS
        KNOWN_HOSTS = Path(directory) / "known_hosts"
        KNOWN_HOSTS.write_text(f"{AP_IP} {HOST_KEY}\n")
        KNOWN_HOSTS.chmod(0o600)
        try:
            yield
        finally:
            KNOWN_HOSTS = original


def run_sync(args):
    initial = Sync(args)
    with ssh_session(args, initial):
        if args.ssh or args.check or args.dry_run:
            Sync(args).synchronize()
        else:
            watch(args, initial)


def preview(args):
    """Read-only: no interface, profile, scan or source change."""
    router = station(args.wifi_interface)
    print("공유기 Wi-Fi: " + (f"{router['ssid']} (채널 {router['channel']})" if router else "연결 안 됨"))
    with preview_identity():
        value = find_existing_connection(args)
        if value is None:
            print(f"R1 직결({LINK_INTERFACE})이 연결되어 있지 않습니다.")
            print(f"실행 계획: {LINK_INTERFACE} 확인·생성 → R1 AP 연결·저지연 설정 → 동기화")
            print("동기화 대상: " + str(ROOT) + " ↔ " + args.remote_dir)
            print("제외 규칙: 프로젝트 .gitignore. 네트워크·소스·기준 기록은 변경하지 않았습니다.")
            if args.check:
                raise SyncError("R1 직결 사전 점검 미완료. 일반 실행으로 연결하세요.")
            return
        live = station(LINK_INTERFACE)
        print(f"R1 직결: {LINK_INTERFACE} → {AP_IP}" + (f" (채널 {live['channel']})" if live else ""))
        args.host = AP_IP
        run_sync(args)


# Only shared project paths and file fingerprints are recorded on R1.
# Never follow symlink parents when removing newly excluded shared files.
REMOTE_METADATA = r'''
import errno, json, os, re, stat, sys
from pathlib import Path
os.chdir(sys.argv[-1])
request = json.load(sys.stdin)
directory = Path('.sync-r1')
ledger = directory / 'shared.json'
if directory.is_symlink() or ledger.is_symlink():
    raise RuntimeError('동기화 기록 경로에 심볼릭 링크를 사용할 수 없습니다.')
operation = request['operation']


def targets():
    """Deepest first: (path, its last part, its parent opened without following symlinks)."""
    for name in sorted(request['paths'], key=lambda name: (name.count('/'), name), reverse=True):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
            raise RuntimeError('잘못된 정리 대상 경로: ' + repr(name))
        descriptor = os.open('.', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for parent in path.parts[:-1]:
                try:
                    child = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                except OSError as error:
                    if error.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                        raise
                    break
                os.close(descriptor)
                descriptor = child
            else:
                yield name, path.name, descriptor
        finally:
            os.close(descriptor)


def flush_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if operation == 'read':
    print(ledger.read_text() if ledger.exists() else '{}')
elif operation == 'write':
    directory.mkdir(exist_ok=True)
    temporary = directory / 'shared.tmp'
    if temporary.is_symlink():
        raise RuntimeError('동기화 기록 임시 경로에 심볼릭 링크를 사용할 수 없습니다.')
    # fsync: a ledger emptied by a power cut would stop every later pass.
    with temporary.open('w') as stream:
        stream.write(json.dumps(request['state'], ensure_ascii=True))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    temporary.replace(ledger)
    for path in (directory, '.'):
        flush_directory(path)
    print('{}')
elif operation == 'flush':
    # What this pass wrote must survive a power cut before it counts as
    # delivered. Report which of the given folders still exist.
    os.sync()
    present = []
    for path, name, descriptor in targets():
        try:
            if not stat.S_ISDIR(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                continue  # replaced by the delivered file or link
        except FileNotFoundError:
            continue
        except OSError:
            pass
        present.append(path)
    print(json.dumps({'present': present}))
elif operation == 'clean':
    for _, name, descriptor in targets():
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode):
            # Preserve any unrecorded personal files inside a directory.
            try:
                os.rmdir(name, dir_fd=descriptor)
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
        else:
            os.unlink(name, dir_fd=descriptor)
    print('{}')
elif operation == 'strip':
    # rsync --delete cannot remove a folder that still holds excluded files.
    # In folders this pass removes or replaces, delete Python caches first:
    # only regular files (*.pyc in __pycache__) and folders, never through a
    # symlink or a mount. Other excluded files stay; the final check names them.
    caches = ('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache')
    project = os.path.realpath('.')
    try:
        with open('/proc/self/mountinfo') as stream:
            mounts = [re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), line.split()[4])
                      for line in stream]
        mounts = [mount for mount in mounts if mount.startswith(project + '/')]
    except OSError:
        mounts = None  # Without a mount list, delete nothing.

    def drop(folder, pyc_only):
        entries = list(os.scandir(folder))
        if not all(not pyc_only if entry.is_dir(follow_symlinks=False)
                   else entry.is_file(follow_symlinks=False) and (not pyc_only or '.pyc' in entry.name)
                   for entry in entries):
            raise OSError('Python 캐시가 아닌 항목이 있습니다.')
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder)
                try:
                    drop(child, pyc_only)
                finally:
                    os.close(child)
                os.rmdir(entry.name, dir_fd=folder)
            else:
                os.unlink(entry.name, dir_fd=folder)

    for _, name, descriptor in targets():
        try:
            folder = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        except OSError:
            continue
        try:
            # A mount inside the project at, below or above the folder may hold anything.
            real = os.path.realpath('/proc/self/fd/%d' % folder)
            if mounts is None or any(mount == real or mount.startswith(real + '/') or real.startswith(mount + '/')
                                     for mount in mounts):
                continue
            times = os.stat(folder)
            if times.st_uid != os.geteuid():
                continue  # its mtime could not be restored
            try:
                for entry in list(os.scandir(folder)):
                    if entry.is_dir(follow_symlinks=False) and entry.name in caches:
                        cache = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder)
                        try:
                            drop(cache, entry.name == '__pycache__')
                        finally:
                            os.close(cache)
                        os.rmdir(entry.name, dir_fd=folder)
                    elif entry.is_file(follow_symlinks=False) and entry.name.endswith(('.pyc', '.pyo')):
                        os.unlink(entry.name, dir_fd=folder)
            finally:
                # If the folder survives, plan() must still see its old mtime.
                os.utime(folder, ns=(times.st_atime_ns, times.st_mtime_ns))
        except OSError:
            pass  # What stays is named by the final check.
        finally:
            os.close(folder)
    print('{}')
else:
    raise RuntimeError('알 수 없는 동기화 기록 작업')
'''


class SyncError(Exception):
    pass


def sync_run(command, *, input=None, timeout=None, **kwargs):
    """Stop an active SSH/rsync child as well when the watcher is interrupted."""
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    process = subprocess.Popen(command, **kwargs)
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def manifest(root):
    """Ignore mtimes, but compare file content, permissions and symlinks."""
    result = {}

    def visit(directory):
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            name = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                result[name] = ["link", os.readlink(path)]
            elif stat.S_ISDIR(info.st_mode):
                result[name] = ["dir", mode]
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                result[name] = ["file", mode, digest.hexdigest()]
            else:
                raise SyncError(f"일반 파일·폴더·심볼릭 링크만 동기화합니다: {name!r}")

    visit(root)
    return result


def modification_times(root, entries):
    return {name: (root / name).lstat().st_mtime_ns for name in entries}


def flush_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def settle(stored, pending, local, remote):
    """Baseline after a pass that stopped between its first write and its final
    record. A planned value counts as agreed when no receiver is left for it
    (both sides had it, or delivered() saw the transfer finish) or the receiver
    has it now; otherwise that pass's own baseline applies.
    Without this, a file pushed by such a pass and later deleted on the PC
    would look newly added on R1 and come back."""
    if pending is None:
        return stored
    planned, base, receivers = pending["files"], pending["base"], pending["receivers"]
    settled = {}
    for name in planned.keys() | base.keys():
        side = receivers.get(name)
        received = side is None or (local if side == "local" else remote).get(name) == planned.get(name)
        value = planned.get(name) if received else base.get(name)
        if value is not None:
            settled[name] = value
    return settled


def plan(local, remote, baseline, conflicts=None, local_times=None, remote_times=None):
    desired, origins = {}, {}
    if conflicts is None:
        conflicts = set()
    local_times = local_times or {}
    remote_times = remote_times or {}

    def newer(name, subtree=False):
        prefix = name + "/"
        left_time = max((value for path, value in local_times.items()
                         if path == name or (subtree and path.startswith(prefix))), default=0)
        right_time = max((value for path, value in remote_times.items()
                          if path == name or (subtree and path.startswith(prefix))), default=0)
        if left_time == right_time:
            raise SyncError(f"수정 시각이 같은 충돌 파일: {name!r}. 내용을 같게 맞추거나 유지할 파일을 다시 저장하세요.")
        return "local" if left_time > right_time else "remote"

    for name in sorted(local.keys() | remote.keys() | baseline.keys()):
        left, right, old = local.get(name), remote.get(name), baseline.get(name)
        if left == right:
            value, origin = left, "local"
        elif left is not None and right is not None:
            origin = newer(name, subtree=left[0] == "dir" or right[0] == "dir")
            value = left if origin == "local" else right
            loser = right if origin == "local" else left
            if loser != old:
                conflicts.add(name)
        elif left == old:
            value, origin = right, "remote"
        elif right == old:
            value, origin = left, "local"
        else:
            conflicts.add(name)
            # Deletions have no comparable mtime. Preserve a file modified on
            # the other side when a deletion and an edit occur together.
            value, origin = (left, "local") if left is not None else (right, "remote")
        if value is not None:
            desired[name], origins[name] = value, origin
    # If a directory deletion/type change conflicts with changes below it,
    # resolve the subtree using the side with the latest modification inside it.
    while True:
        root = None
        for name in sorted(desired):
            for parent in reversed(Path(name).parents):
                if parent == Path("."):
                    continue
                if desired.get(parent.as_posix(), [None])[0] != "dir":
                    root = parent.as_posix()
                    break
            if root is not None:
                break
        if root is None:
            break
        conflicts.add(root)
        origin = newer(root, subtree=True)
        source = local if origin == "local" else remote
        prefix = root + "/"
        for name in list(desired):
            if name == root or name.startswith(prefix):
                del desired[name]
                del origins[name]
        for name, value in source.items():
            if name == root or name.startswith(prefix):
                desired[name], origins[name] = value, origin
    return desired, origins


def materialize(target, desired, origins, snapshots):
    for name, value in sorted(desired.items(), key=lambda item: (item[0].count("/"), item[0])):
        destination = target / name
        source = snapshots[origins[name]] / name
        if value[0] == "dir":
            destination.mkdir()
        elif value[0] == "link":
            destination.symlink_to(value[1])
        else:
            shutil.copy2(source, destination)
    # Apply directory permissions after writing their children.
    for name, value in desired.items():
        if value[0] == "dir":
            (target / name).chmod(value[1])


def validate_policy(policy):
    lines = [line.lstrip("/") for line in policy.splitlines()]
    if any(line.startswith("!") for line in lines):
        raise SyncError(".gitignore의 ! 재포함 패턴은 지원하지 않습니다.")
    for required in (".agent/", ".git/", ".sync-r1/", ".sync-r1.lock/"):
        if required not in lines:
            raise SyncError(f".gitignore에 필수 제외 규칙 {required}를 넣으세요.")


def private_directory(path):
    """Create or reuse a folder that only this user can open."""
    with contextlib.suppress(FileExistsError):
        path.mkdir(mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise SyncError(f"개인 동기화 폴더를 사용할 수 없습니다: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        path.chmod(0o700)
    return path


def temporary_root(project):
    # Project policy: snapshots, sockets and helpers stay in .agent/.
    path = project / ".agent" / "tmp"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def runtime_directory(project):
    return private_directory(temporary_root(project) / "sync-sockets")


def state_home():
    return ROOT / ".agent" / "r1-sync"


def sync_state_directory(project):
    git_dir = project / ".git"
    if git_dir.is_symlink() or git_dir.is_file():
        raise SyncError("git worktree 또는 심볼릭 링크 .git은 지원하지 않습니다.")
    # Durable deletion history survives removal of disposable .agent/ files.
    return git_dir / "r1-sync" if git_dir.is_dir() else project / ".agent" / "r1-sync"


def hold_lock(path, message):
    """flock is released by the kernel on exit or crash; no stale lock remains."""
    stream = open(path, "a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise SyncError(message) from None
    except BaseException:
        stream.close()
        raise
    return stream


def reject_git_metadata(*manifests):
    """rsync skips only .git folders; a .git file (worktree or submodule pointer)
    would overwrite the other machine's pointer, so refuse to sync it."""
    for entries in manifests:
        for name in entries:
            if Path(name).name == ".git":
                raise SyncError(f"git 메타데이터 파일은 동기화할 수 없습니다: {name!r}. "
                                ".gitignore에 추가하거나 옮기세요.")


def control_path_for_cycle(args, project):
    """Return this cycle's SSH ControlPath, or None for plain SSH. Without a
    socket there is no reachable master, so start in a fresh folder: a master
    whose socket was deleted still unlinks its old pathname when it exits."""
    if not getattr(args, "multiplex", False):
        return None
    directory = args.control_dir
    if directory is not None:
        if (directory / "s").is_socket():
            return str(directory / "s")
        shutil.rmtree(directory, ignore_errors=True)
        args.control_dir = None
    try:
        directory = Path(tempfile.mkdtemp(prefix="ssh-", dir=runtime_directory(project)))
    except (OSError, SyncError):
        return None
    # ssh first binds "<path>.<16 random chars>"; sun_path holds 103 bytes on
    # macOS and 107 on Linux. Use plain SSH for paths that would not fit.
    if len(os.fsencode(str(directory / "s"))) + 17 > 103:
        directory.rmdir()
        return None
    args.control_dir = directory
    return str(directory / "s")


class Sync:
    def __init__(self, args):
        self.args = args
        self.project = ROOT.resolve()
        self.policy = self.project / ".gitignore"
        self.host = str(ipaddress.IPv4Address(args.host))
        self.user, self.port = "root", "22"
        self.remote = validate_remote_directory(args.remote_dir)
        for program in ("ssh", "rsync"):
            if not shutil.which(program):
                raise SyncError(f"로컬 PC에 {program}가 필요합니다.")
        if not self.policy.is_file() or self.policy.is_symlink() or self.project == Path("/"):
            raise SyncError("로컬 프로젝트 위치 또는 .gitignore를 확인하세요.")
        self.policy_bytes = self.policy.read_bytes()
        validate_policy(self.policy_bytes.decode())
        self.endpoint = f"{self.user}@{self.host}"
        git_dir = self.project / ".git"
        if git_dir.is_file():
            # rsync excludes only a .git folder; a worktree's .git file would be synced.
            raise SyncError("git worktree·submodule 폴더는 동기화할 수 없습니다. 원래 클론에서 실행하세요.")
        self.ssh_base = ssh_argv(args, self.host, allow_multiplex=True)[:-1] + [
            "-T", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes"]
        control_path = control_path_for_cycle(args, self.project)
        if control_path:
            # Only preflight, the first command of each pass, may start a master;
            # later commands never recreate a socket that vanished mid-pass.
            control = self.ssh_base + ["-o", "ControlPath=" + control_path]
            self.ssh_master = control + ["-o", "ControlMaster=auto", "-o", "ControlPersist=30"]
            self.ssh = control + ["-o", "ControlMaster=no"]
        else:
            # Also override ~/.ssh/config, whose ControlPath could name a removed folder.
            self.ssh = self.ssh_base + ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
            self.ssh_master = self.ssh
        # Keep the baseline in .git: if git clean -fdx removed it, the next pass
        # would be a first sync and copy locally deleted files back from R1.
        self.state_dir = sync_state_directory(self.project)
        identity = json.dumps([HOST_KEY, ROBOT_MAC, self.user, self.port, self.remote], ensure_ascii=True)
        self.identity = identity
        self.state_file = self.state_dir / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
        self.key = hashlib.sha256(os.fsencode(str(self.project))).hexdigest()[:16]

    def make_state_dir(self):
        try:
            self.state_dir.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            return
        flush_directory(self.state_dir.parent)  # its entry in .git must survive too

    def lock(self, name, message):
        # Next to the baseline: git clean cannot remove it, and passes for any
        # target never write this local tree at the same time.
        self.make_state_dir()
        return hold_lock(self.state_dir / f"{name}.lock", message)

    def preflight(self, quiet=False):
        # Only normal first initialization may create the dedicated project folder.
        create = not (self.args.check or self.args.dry_run or self.state_file.exists())
        command = shlex.join(["python3", "-c", REMOTE_SYNC_PROJECT, self.remote, str(int(create))])
        result = sync_run(self.ssh_master + [self.endpoint, command],
                          text=True, stdout=subprocess.PIPE, timeout=30)
        info = json.loads(result.stdout)
        self.remote_exists = info["exists"]
        if not self.remote_exists and (self.args.check or self.state_file.exists()):
            raise SyncError("R1 프로젝트 폴더가 없습니다: " + self.remote +
                            ". 기존 동기화 기준이 있으면 폴더를 복구한 뒤 재실행하세요.")
        if not quiet:
            print("R1 SSH·필수 도구·프로젝트 경로 확인 완료" if self.remote_exists else
                  "R1 프로젝트 폴더 생성 예정: " + self.remote, flush=True)

    def remote_lock(self):
        command = "set -eu\ncd -- " + shlex.quote(self.remote) + "\n" + r'''
for program in flock pgrep; do
    command -v "$program" >/dev/null 2>&1 || { printf '오류: R1에 %s가 필요합니다.\n' "$program" >&2; exit 20; }
done
# The kernel drops a flock when its holder exits, is killed or loses power.
exec 9<.
if ! flock -n 9; then
    printf '오류: R1에서 다른 동기화가 실행 중입니다.\n' >&2
    exit 26
fi
# Older versions lock with this folder only, which a power loss leaves behind.
# Under the flock, only an older version's lock shell in this folder holds it.
old_holder() {
    # Hidden processes (hidepid) or an unreadable cwd count as a holder.
    [ -r /proc/1/cmdline ] || return 0
    pids=$(pgrep -f 'Keep the lock for the lifetime of the local SS[H]') || [ $? -eq 1 ] || return 0
    for pid in $pids; do
        [ "$(stat -L -c %d:%i "/proc/$pid/cwd" 2>/dev/null || stat -c %d:%i .)" != "$(stat -c %d:%i .)" ] || return 0
    done
    return 1
}
fail() {
    printf '오류: R1에서 이전 버전 동기화가 실행 중이거나 잠금을 정리할 수 없습니다.\n' >&2
    exit 26
}
# The owner file keeps the folder non-empty, so an older client's EXIT trap
# (a plain rmdir) cannot remove a folder this version made.
take() { mkdir -m 775 -- .sync-r1.lock 2>/dev/null && { printf '%s\n' "$$" > .sync-r1.lock/owner; } 2>/dev/null; }
if ! take; then
    if [ -L .sync-r1.lock ] || [ ! -d .sync-r1.lock ] || old_holder \
            || ! { rm -f -- .sync-r1.lock/owner; rmdir -- .sync-r1.lock 2>/dev/null; } || ! take; then
        fail
    fi
    # An older client that made the folder between the check and rmdir is
    # still running; hand the emptied folder over to it (no EXIT trap yet).
    if old_holder; then
        rm -f -- .sync-r1.lock/owner
        fail
    fi
    printf 'R1에 남아 있던 이전 동기화 잠금을 정리했습니다.\n' >&2
fi
trap 'rm -f -- .sync-r1.lock/owner; rmdir -- .sync-r1.lock 2>/dev/null || true' EXIT
trap 'exit 1' HUP INT TERM
printf 'SYNC_LOCK_READY\n'
# Both locks last until the local sync closes this input stream.
cat >/dev/null
'''
        process = subprocess.Popen(self.ssh + [self.endpoint, command],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline()
                if line == "SYNC_LOCK_READY\n":
                    return process
                if not line:
                    raise SyncError("R1의 동기화 잠금을 얻지 못했습니다.")
                print(line, end="", flush=True)
        except BaseException:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            raise

    def rsync(self, source, destination, delete=False, announce=False):
        if isinstance(source, Path) and not source.is_dir():
            raise SyncError("전송할 스냅샷이 사라졌습니다. .agent 정리를 멈추고 다시 실행하세요.")
        options = ["rsync", "--archive", "--compress", "--checksum", "--no-owner", "--no-group",
                   "--human-readable", "--modify-window=-1",
                   "--exclude-from=" + str(self.policy), "--rsh=" + shlex.join(self.ssh)]
        if announce:
            options.append("--itemize-changes")
        if delete:
            options += ["--delete-delay", "--delay-updates"]
        # All exclusions come from .gitignore. Never use --delete-excluded.
        sync_run(options + ["--", str(source).rstrip("/") + "/", str(destination).rstrip("/") + "/"])

    def snapshot(self, root):
        local, remote = root / "local", root / "remote"
        local.mkdir()
        remote.mkdir()
        self.rsync(self.project, local)
        if self.remote_exists:
            self.rsync(f"{self.endpoint}:{self.remote}", remote)
        return {"local": local, "remote": remote}, manifest(local), manifest(remote)

    def metadata(self, operation, side="remote", **values):
        """Run REMOTE_METADATA on R1, or on this PC for side="local"."""
        if side == "local":
            command = [sys.executable, "-c", REMOTE_METADATA, "--", str(self.project)]
        else:
            command = self.ssh + [self.endpoint, shlex.join(["python3", "-c", REMOTE_METADATA, "--", self.remote])]
        result = sync_run(command, input=json.dumps(dict(operation=operation, **values)),
                     text=True, stdout=subprocess.PIPE)
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise SyncError("R1의 공용 동기화 기록을 읽을 수 없습니다.") from error

    def strip(self, side, before, desired):
        """Delete Python caches in the folders this pass removes or replaces on
        one side (REMOTE_METADATA strip) and return those folders."""
        names = [name for name, value in before.items()
                 if value[0] == "dir" and desired.get(name, [None])[0] != "dir"]
        if names:
            self.metadata("strip", side=side, paths=names)
        return names

    def delivered(self, stored, record, side, folders):
        """rsync to `side` succeeded. After flushing that side to disk (the
        robot can lose power any time), every change for it counts as arrived
        except folders from strip() that still exist. Later edits there are
        then edits, not a missed delivery (see settle())."""
        present = set(self.metadata("flush", side=side, paths=folders)["present"])
        record["receivers"] = {name: receiver for name, receiver in record["receivers"].items()
                               if receiver != side or name in present}
        self.write_state(stored, record)

    def write_state(self, files, pending=None):
        # Version 2 marks a pending record so older versions refuse the file
        # instead of ignoring the record. fsync: a pending record must survive
        # a PC power loss once the first write has reached R1.
        state = {"version": 1 if pending is None else 2, "endpoint": self.identity, "files": files}
        if pending is not None:
            state["pending"] = pending
        self.make_state_dir()
        temporary = self.state_file.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(state, ensure_ascii=True))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        temporary.replace(self.state_file)
        flush_directory(self.state_dir)

    def newly_excluded(self, root, recorded):
        """Let rsync itself apply the exact same exclusions to recorded paths."""
        source, filtered = root / "recorded", root / "filtered"
        source.mkdir()
        filtered.mkdir()
        for name, value in recorded.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
                raise SyncError(f"잘못된 동기화 기록 경로: {name!r}")
            if not isinstance(value, list) or not value or value[0] not in ("dir", "file", "link"):
                raise SyncError(f"잘못된 동기화 기록 항목: {name!r}")
            (source / path).parent.mkdir(parents=True, exist_ok=True)
        for name, value in recorded.items():
            if value[0] == "dir":
                (source / name).mkdir(parents=True, exist_ok=True)
        for name, value in recorded.items():
            path = Path(name)
            target = source / path
            if not target.is_dir():
                # Content is unnecessary; only path and directory matching matter.
                target.touch()
        self.rsync(source, filtered)
        return sorted(recorded.keys() - manifest(filtered).keys())

    def backup_conflicts(self, conflicts, local, remote, desired, snapshots):
        copies = {side: [name for name in entries if entries[name] != desired.get(name)
                        and any(name == root or name.startswith(root + "/") for root in conflicts)]
                  for side, entries in (("local", local), ("r1", remote))}
        if not any(copies.values()):
            return
        # Backups are local private data, excluded from source synchronization.
        directory = state_home() / "backups" / f"{self.project.name}-{self.key}"
        directory.mkdir(parents=True, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=directory))
        for side, names in copies.items():
            entries = local if side == "local" else remote
            snapshot = snapshots["local" if side == "local" else "remote"]
            for name in sorted(names, key=lambda path: (path.count("/"), path)):
                target = backup / side / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if entries[name][0] == "dir":
                    target.mkdir(exist_ok=True)
                elif entries[name][0] == "link":
                    target.symlink_to(entries[name][1])
                else:
                    shutil.copy2(snapshot / name, target)
        print(f"덮어쓸 충돌 파일 보관: {backup}", flush=True)

    def synchronize(self, quiet=False):
        if not (self.args.dry_run or self.args.check):
            prepare_state()
        with sync_authentication():
            return self._synchronize(quiet)

    def _synchronize(self, quiet=False):
        if not quiet:
            print(f"원격: {self.endpoint}:{self.remote}\n로컬: {self.project}", flush=True)
        self.preflight(quiet=quiet)
        if self.args.check:
            print("사전 점검 완료: 소스 파일은 변경하지 않았습니다.")
            return
        local_lock = None if self.args.dry_run else self.lock("sync", "로컬에서 다른 동기화가 실행 중입니다.")
        remote_lock = None
        try:
            if not self.args.dry_run:
                remote_lock = self.remote_lock()
            # Excluded .agent/tmp snapshots must remain present during each transfer.
            with tempfile.TemporaryDirectory(prefix="agroid-r1-snapshot-",
                                             dir=temporary_root(self.project)) as directory:
                root = Path(directory)
                stored, pending = {}, None
                state_file = self.state_file if self.state_file.exists() else None
                if state_file:
                    try:
                        state = json.loads(state_file.read_text())
                        if state["version"] not in (1, 2) or state["endpoint"] != self.identity:
                            raise ValueError("invalid state identity")
                        stored, pending = state["files"], state.get("pending")
                        if not isinstance(stored, dict) or not (pending is None or (
                                all(isinstance(pending.get(key), dict) for key in ("base", "files", "receivers"))
                                and set(pending["receivers"].values()) <= {"local", "remote"})):
                            raise ValueError("invalid state manifest")
                    except (ValueError, KeyError, TypeError, AttributeError) as error:
                        raise SyncError(f"동기화 기준 기록을 읽을 수 없습니다: {state_file}") from error
                if pending is not None and not self.args.dry_run:
                    # settle() trusts what it sees on each side; an interrupted
                    # pass may have left it unflushed, so flush both first.
                    for side in ("remote", "local"):
                        self.metadata("flush", side=side, paths=[])
                snapshots, local, remote = self.snapshot(root)
                baseline = settle(stored, pending, local, remote)
                # Resolve the shared .gitignore first. Incoming new exclusions must
                # preserve the local private file during this very invocation.
                conflicts = set()
                local_times = modification_times(snapshots["local"], local)
                remote_times = modification_times(snapshots["remote"], remote)
                policy_plan, policy_origins = plan(
                    {".gitignore": local[".gitignore"]} if ".gitignore" in local else {},
                    {".gitignore": remote[".gitignore"]} if ".gitignore" in remote else {},
                    {".gitignore": baseline[".gitignore"]} if ".gitignore" in baseline else {},
                    conflicts, local_times, remote_times)
                if ".gitignore" not in policy_plan:
                    raise SyncError("공통 제외 목록 .gitignore는 삭제하거나 제외할 수 없습니다.")
                chosen_policy = (snapshots[policy_origins[".gitignore"]] / ".gitignore").read_bytes()
                validate_policy(chosen_policy.decode())
                # Freeze the chosen policy for all transfers including .gitignore.
                frozen_policy = root / "excludes"
                frozen_policy.write_bytes(chosen_policy)
                self.policy = frozen_policy
                if chosen_policy != self.policy_bytes:
                    rescan = root / "policy-rescan"
                    rescan.mkdir()
                    snapshots, local, remote = self.snapshot(rescan)
                    local_times = modification_times(snapshots["local"], local)
                    remote_times = modification_times(snapshots["remote"], remote)
                    baseline = settle(stored, pending, local, remote)
                reject_git_metadata(local, remote)
                shared = self.metadata("read") if self.remote_exists else {}
                if not isinstance(shared, dict) or (shared and (shared.get("version") != 1 or not isinstance(shared.get("files"), dict))):
                    raise SyncError("R1의 공용 동기화 기록 형식이 올바르지 않습니다.")
                recorded = dict(shared.get("files", {}))
                recorded.update(baseline)
                clean_paths = [name for name in self.newly_excluded(root, recorded) if name not in remote]
                desired, origins = plan(local, remote, baseline, conflicts, local_times, remote_times)
                changes = []
                for name in clean_paths:
                    changes.append(f"R1 제외 파일 정리: {name!r} (로컬 보존)")
                for side, current in (("로컬", local), ("R1", remote)):
                    for name in sorted(current.keys() | desired.keys()):
                        if current.get(name) != desired.get(name):
                            action = "삭제" if name not in desired else "갱신"
                            changes.append(f"{side} {action}: {name!r}")
                if conflicts:
                    print("최신 수정본 우선: " + ", ".join(
                        f"{name!r} ← {'로컬' if origins.get(name) == 'local' else 'R1'}"
                        for name in sorted(conflicts)), flush=True)
                if changes or not quiet:
                    print("\n".join(changes) if changes else "변경 사항이 없습니다.", flush=True)
                if self.args.dry_run:
                    print("미리보기 완료: 소스와 동기화 기준 기록은 변경하지 않았습니다.")
                    return
                if (not changes and state_file is not None and pending is None and baseline == desired
                        and shared.get("files") == desired):
                    return
                merged = root / "merged"
                merged.mkdir()
                materialize(merged, desired, origins, snapshots)
                if not (merged / ".gitignore").is_file():
                    raise SyncError("공통 제외 목록 .gitignore는 삭제할 수 없습니다.")
                validate_policy((merged / ".gitignore").read_text())
                # Detect edits made while scanning/planning before any source write.
                verification = root / "before"
                verification.mkdir()
                checked, now_local, now_remote = self.snapshot(verification)
                if (now_local != local or now_remote != remote
                        or modification_times(checked["local"], now_local) != local_times
                        or modification_times(checked["remote"], now_remote) != remote_times
                        or (self.project / ".gitignore").read_bytes() != self.policy_bytes):
                    raise SyncError("점검 중 소스 또는 제외 규칙이 변경되었습니다. 다시 실행하세요. 소스는 변경하지 않았습니다.")
                if remote_lock.poll() is not None:
                    raise SyncError("R1 동기화 잠금 연결이 끊겼습니다. 다시 실행하세요.")
                self.backup_conflicts(conflicts, local, remote, desired, snapshots)
                # Record the plan before the first write; settle() reads it if
                # this pass stops before the final record below.
                record = {"base": baseline, "files": desired,
                          "receivers": {name: "remote" if remote.get(name) != desired.get(name) else "local"
                                        for name in local.keys() | remote.keys()
                                        if local.get(name) != remote.get(name)}}
                self.write_state(stored, record)
                if clean_paths:
                    self.metadata("clean", paths=clean_paths)
                # Record paths before a transfer can partially succeed. This
                # lets a later ignore rule clean an uploaded file even if the
                # interrupted cycle never reached final verification.
                if remote != desired or local != desired:
                    managed = dict(shared.get("files", {}))
                    managed.update(remote)
                    managed.update(desired)
                    for name in clean_paths:
                        managed.pop(name, None)
                    self.metadata("write", state={"version": 1, "files": managed})
                stripped = {"R1": [], "로컬": []}
                if remote != desired:
                    stripped["R1"] = self.strip("remote", remote, desired)
                    merged.chmod(stat.S_IMODE(snapshots["remote"].stat().st_mode))
                    self.rsync(merged, f"{self.endpoint}:{self.remote}", delete=True, announce=True)
                    self.delivered(stored, record, "remote", stripped["R1"])
                if remote_lock.poll() is not None:
                    raise SyncError("R1 동기화 잠금 연결이 끊겼습니다. 기준 기록은 갱신하지 않았습니다.")
                if local != desired:
                    stripped["로컬"] = self.strip("local", local, desired)
                    merged.chmod(stat.S_IMODE(snapshots["local"].stat().st_mode))
                    self.rsync(merged, self.project, delete=True, announce=True)
                    self.delivered(stored, record, "local", stripped["로컬"])
                verification = root / "after"
                verification.mkdir()
                _, final_local, final_remote = self.snapshot(verification)
                if final_local != desired or final_remote != desired:
                    kept = [f"{side} {name!r}" for side, final in (("R1", final_remote), ("로컬", final_local))
                            for name in stripped[side] if name in final and name not in desired]
                    raise SyncError("전송 후 양쪽 소스가 일치하지 않습니다. " + (
                        "동기화 제외 파일이 남아 지우지 못한 폴더: " + ", ".join(kept)
                        + ". 그 안의 파일을 옮기거나 지우세요. " if kept else "")
                        + "기준 기록은 갱신하지 않았습니다. 다시 실행하세요.")
                self.metadata("write", state={"version": 1, "files": desired})
                self.write_state(desired)
                if changes or not quiet:
                    print("동기화 완료: 개인 설정을 제외한 로컬과 R1의 프로젝트 파일이 일치합니다.", flush=True)
        finally:
            try:
                if remote_lock is not None:
                    try:
                        remote_lock.stdin.close()
                    except BrokenPipeError:
                        pass
                    try:
                        remote_lock.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        remote_lock.terminate()
                        try:
                            remote_lock.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            remote_lock.kill()
                            remote_lock.wait()
                    remote_lock.stdout.close()
            finally:
                if local_lock is not None:
                    local_lock.close()


@contextlib.contextmanager
def ssh_session(args, initial):
    """Reuse local SSH authentication during polling; stop the master on exit.
    Each cycle recreates any missing private socket in .agent/tmp/."""
    args.multiplex, args.control_dir = True, None
    try:
        yield
    finally:
        args.multiplex = False
        directory, args.control_dir = args.control_dir, None
        if directory is not None:
            if (directory / "s").is_socket():
                command = initial.ssh_base + ["-o", "ControlMaster=no",
                                              "-o", "ControlPath=" + str(directory / "s"),
                                              "-O", "exit", initial.endpoint]
                try:
                    sync_run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass
            shutil.rmtree(directory, ignore_errors=True)


def watch(args, initial):
    with initial.lock("watch", "이 로컬 프로젝트의 지속 동기화가 이미 실행 중입니다."):
        print(f"지속 동기화 시작: {initial.project} ↔ {initial.endpoint}:{initial.remote}\n"
              f"각 동기화 후 {args.interval:g}초 간격으로 확인합니다. 종료: Ctrl+C", flush=True)
        previous_error = None
        relinked, link_error = -math.inf, None
        while True:
            try:
                if not link_ready():
                    if time.monotonic() - relinked >= LINK_RETRY:
                        try:
                            reconnect_link()
                            link_error = None
                        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                            link_error = str(exc)
                        finally:
                            # Leave a full quiet interval after a scan/activation finishes.
                            relinked = time.monotonic()
                    if not link_ready():
                        raise SyncError("R1 직결 재연결 대기: " + (link_error or "연결 확인 중"))
                # Re-read .gitignore and state every time, including a policy
                # downloaded by the previous pass or edited locally while idle.
                Sync(args).synchronize(quiet=True)
            except (SyncError, OSError, subprocess.SubprocessError) as error:
                message = (f"{Path(error.cmd[0]).name} 실행 실패 (종료 코드 {error.returncode})"
                           if isinstance(error, subprocess.CalledProcessError) else str(error))
                if message != previous_error:
                    print(f"동기화 대기: {message}\n{args.interval:g}초 후 재시도합니다.",
                          file=sys.stderr, flush=True)
                previous_error = message
            else:
                if previous_error is not None:
                    print("동기화 연결 및 상태가 정상화되었습니다.", flush=True)
                previous_error = None
            time.sleep(args.interval)


def polling_interval(value):
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("간격은 양수여야 합니다.") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("간격은 유한한 양수여야 합니다.")
    return seconds


def stop_requested(signum, frame):
    raise KeyboardInterrupt



def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="00_initialize.sh",
        description="노트북 공유기 Wi-Fi를 유지한 채 가상 Wi-Fi로 R1 AP에 직결하고 프로젝트를 양방향 동기화합니다.",
        epilog=f"기본: 5초 간격 동기화, Ctrl+C 종료. R1: {AP_IP} ({LINK_INTERFACE}). 제외: .gitignore. "
               "원격 기본 경로: /root/workspace/agroid_v1. 여러 번 실행해도 같은 구성을 재사용합니다.")
    parser.add_argument("--wifi-interface", "-i", help="공유기에 연결된 노트북 무선 인터페이스; 하나면 자동 선택")
    parser.add_argument("--remote-dir", default=os.environ.get("AGROID_R1_PROJECT_DIR") or DEFAULT_REMOTE_DIR,
                        help="R1 전용 프로젝트 절대 경로")
    parser.add_argument("--ssh", action="store_true", help="연결·1회 동기화 성공 후 SSH 셸 열기")
    parser.add_argument("--interval", type=polling_interval, default=5, metavar="SECONDS",
                        help="동기화 완료 후 다음 확인까지의 초 (기본: 5)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="현재 R1 직결·원격 도구·경로만 점검; 변경 없음")
    mode.add_argument("--dry-run", action="store_true", help="현재 R1 직결에서 동기화 계획 확인; 변경 없음")
    args = parser.parse_args(argv)
    if args.ssh and (args.check or args.dry_run):
        parser.error("--ssh는 --check, --dry-run과 함께 사용할 수 없습니다.")
    previous = {s: signal.signal(s, stop_requested) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    command = None
    try:
        require_tools(("nmcli", "ip", "iw", "ssh", "rsync", "busctl"))
        validate_remote_directory(args.remote_dir)
        policy = ROOT / ".gitignore"
        if policy.is_symlink():
            raise SyncError(".gitignore는 일반 파일이어야 합니다.")
        validate_policy(policy.read_text())
        args.wifi_interface = interface_name(args.wifi_interface)
        if args.check or args.dry_run:
            preview(args)
            return 0
        if os.geteuid() == 0:
            raise RuntimeError("sudo 없이 PC의 일반 사용자로 실행하세요. 필요한 sudo 인증은 도중에 요청합니다.")
        directory = sync_state_directory(ROOT)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with hold_lock(directory / "initialize.lock", "이 프로젝트의 초기화·동기화가 이미 실행 중입니다."):
            prepare_state()
            value = connect(args)
            command = finish_setup(args, value)
            args.host = AP_IP
            run_sync(args)
    except BlockingIOError:
        print("다른 R1 설정 작업이 실행 중입니다. 해당 작업이 끝난 뒤 다시 실행하세요.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\n초기화·동기화를 종료했습니다. R1 직결({LINK_INTERFACE})은 유지됩니다.", flush=True)
        return 130
    except Exception as exc:
        print("미완료: " + str(exc), file=sys.stderr)
        return 1
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    if args.ssh and command:
        try:
            os.execvp(command[0], command)
        except OSError as exc:
            print("SSH 실행 실패: " + str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

R1_INITIALIZE_PYTHON
