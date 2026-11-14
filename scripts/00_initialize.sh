#!/usr/bin/env bash
# Join the R1 AP through a second laptop Wi-Fi link and keep the connection available.
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
"""노트북 이중 Wi-Fi로 R1 AP에 직결하고 연결을 유지합니다. 파일은 전송하지 않습니다.

노트북의 공유기 Wi-Fi 연결은 유지하고, 같은 무선 칩에 고정 이름의 가상 인터페이스를
하나 만들어 R1 AP에 붙습니다. R1의 네트워크·AP 설정은 변경하지 않습니다.
여러 번 실행해도 가상 인터페이스·NetworkManager 프로필·설정 파일은 하나로 유지됩니다.
Python 표준 라이브러리만 사용합니다.
"""

import argparse
import contextlib
import math
import re
from contextlib import contextmanager
import fcntl
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
                           + ". Ubuntu: sudo apt install network-manager iproute2 iw openssh-client systemd")


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


@contextmanager
def preview_identity():
    """Read the pinned robot identity without creating persistent connection state."""
    global KNOWN_HOSTS
    if KNOWN_HOSTS.is_file():
        yield
        return
    AGENT_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r1-preview-", dir=AGENT_TMP) as directory:
        original = KNOWN_HOSTS
        KNOWN_HOSTS = Path(directory) / "known_hosts"
        KNOWN_HOSTS.write_text(f"{AP_IP} {HOST_KEY}\n")
        KNOWN_HOSTS.chmod(0o600)
        try:
            yield
        finally:
            KNOWN_HOSTS = original


@contextmanager
def hold_lock(path, message):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(message) from error
        yield


def preview(args):
    router = station(args.wifi_interface)
    print("공유기 Wi-Fi: " + (f"{router['ssid']} (채널 {router['channel']})" if router else "연결 안 됨"))
    with preview_identity():
        value = find_existing_connection(args)
    if value is None:
        print(f"R1 직결({LINK_INTERFACE})이 연결되어 있지 않습니다.")
        print(f"실행 계획: {LINK_INTERFACE} 확인·생성 → R1 AP 연결·저지연 설정 → 연결 유지")
        if args.check:
            raise RuntimeError("R1 직결 사전 점검 미완료. 일반 실행으로 연결하세요.")
        return
    print(f"R1 직결 확인: {LINK_INTERFACE} → {AP_IP}, 모델 {ROBOT_MODEL}")
    print("파일 전송 없음. PC 크로스 빌드·배포: scripts/03_deploy.sh")


def watch(args):
    print(f"R1 연결 유지: {LINK_INTERFACE}, {args.interval:g}초 간격 확인. 종료: Ctrl+C", flush=True)
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
                        relinked = time.monotonic()
                if not link_ready():
                    raise RuntimeError("R1 직결 재연결 대기: " + (link_error or "연결 확인 중"))
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            message = str(error)
            if message != previous_error:
                print("연결 대기: " + message, file=sys.stderr, flush=True)
            previous_error = message
        else:
            if previous_error is not None:
                print("R1 직결이 복구되었습니다.", flush=True)
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
        description="노트북 공유기 Wi-Fi를 유지한 채 가상 Wi-Fi로 R1 AP에 직결합니다. 파일은 전송하지 않습니다.",
        epilog=f"기본: 5초 간격 연결 유지, Ctrl+C 종료. R1: {AP_IP} ({LINK_INTERFACE}). "
               "빌드·배포는 PC에서 scripts/03_deploy.sh로 실행합니다.")
    parser.add_argument("--wifi-interface", "-i", help="공유기에 연결된 노트북 무선 인터페이스; 하나면 자동 선택")
    parser.add_argument("--ssh", action="store_true", help="연결 확인 후 SSH 셸 열기")
    parser.add_argument("--interval", type=polling_interval, default=5, metavar="SECONDS",
                        help="연결 확인 간격 (기본: 5초)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="한 번 연결하고 종료")
    mode.add_argument("--check", action="store_true", help="현재 R1 직결·식별 정보만 점검; 변경 없음")
    mode.add_argument("--dry-run", action="store_true", help="현재 연결·연결 계획 확인; 변경 없음")
    args = parser.parse_args(argv)
    if args.ssh and (args.check or args.dry_run):
        parser.error("--ssh는 --check, --dry-run과 함께 사용할 수 없습니다.")
    previous = {s: signal.signal(s, stop_requested) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    command = None
    try:
        require_tools(("nmcli", "ip", "iw", "ssh", "busctl"))
        args.wifi_interface = interface_name(args.wifi_interface)
        if args.check or args.dry_run:
            preview(args)
            return 0
        if os.geteuid() == 0:
            raise RuntimeError("sudo 없이 PC의 일반 사용자로 실행하세요. 필요한 sudo 인증은 도중에 요청합니다.")
        prepare_state()
        with hold_lock(STATE / "initialize.lock", "이 프로젝트의 R1 연결 관리가 이미 실행 중입니다."):
            value = connect(args)
            command = finish_setup(args, value)
            if not (args.once or args.ssh):
                watch(args)
    except KeyboardInterrupt:
        print(f"\n연결 관리를 종료했습니다. R1 직결({LINK_INTERFACE})은 유지됩니다.", flush=True)
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
