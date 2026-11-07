#!/usr/bin/env bash
# R1 실시간 경로 설정. boot_manager.service의 ExecStartPre로 매 기동 전에 실행한다(멱등, 실패해도 기동은 계속).
#   1) MCU USB(3c7b:4d8e) 호스트 IRQ를 격리 CPU 7로 옮기고 IRQ 스레드를 SCHED_FIFO 90으로 둔다
#      (controller의 USB 스레드 FIFO 80보다 높게: 전송 완료 처리가 USB 루프를 기다리지 않게).
#   2) CPU 7 주파수 정책(CPU 6·7 공유)의 최저 주파수를 AGROID_RT_MIN_KHZ 이상 가용 주파수로 올린다.
#      기본 1416000kHz, 0이면 바꾸지 않는다. 전력·발열과 맞바꾸므로 /motor_status timing으로 확인해 조정한다.
#   3) CPU 7의 깊은 idle 상태(state1 이상)를 끈다. 복귀 지연이 USB 주기를 흔들지 않게 한다.
set -u

# OS 설정을 바꾸기 전에 실제 R1 하드웨어인지 확인합니다.
require_r1() {
    local model device_tree
    [[ $(/usr/bin/uname -m) == aarch64 ]] || return 1
    for device_tree in /sys/firmware/devicetree/base /proc/device-tree; do
        [[ -r $device_tree/model ]] || continue
        model=
        IFS= read -r -d "" model < "$device_tree/model" || true
        [[ $model == "Unitree G1 Board" ]] && return 0
    done
    return 1
}
log() { printf 'realtime_tuning: %s\n' "$*"; }
require_r1 || { log "R1이 아니므로 변경하지 않습니다."; exit 0; }

# 1) MCU USB IRQ -> CPU 7, IRQ 스레드 FIFO 90
bus=
for device in /sys/bus/usb/devices/*; do
    [[ -r $device/idVendor && -r $device/idProduct && -r $device/busnum ]] || continue
    if [[ $(<"$device/idVendor") == 3c7b && $(<"$device/idProduct") == 4d8e ]]; then
        bus=$(<"$device/busnum")
        break
    fi
done
if [[ -z $bus ]]; then
    log "MCU USB 장치를 찾지 못해 IRQ는 그대로 둡니다."
else
    irqs=$(grep -E "[[:alnum:]_-]+:usb${bus}(,|[[:space:]]|$)" /proc/interrupts | awk '{ sub(":", "", $1); print $1 }')
    for irq in $irqs; do
        if echo 7 > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null; then
            log "IRQ $irq(usb$bus) -> CPU 7"
        else
            log "IRQ $irq(usb$bus) affinity 변경 실패"
        fi
        for pid in $(ps -eo pid=,comm= | awk -v prefix="irq/$irq-" 'index($2, prefix) == 1 { print $1 }'); do
            chrt -f -p 90 "$pid" && log "IRQ $irq 스레드 $pid -> FIFO 90"
        done
    done
fi

# 2) CPU 7 정책 최저 주파수
min_khz=${AGROID_RT_MIN_KHZ:-1416000}
policy=/sys/devices/system/cpu/cpu7/cpufreq
if [[ $min_khz != 0 && -w $policy/scaling_min_freq ]]; then
    target=$min_khz
    if [[ -r $policy/scaling_available_frequencies ]]; then
        target=$(tr ' ' '\n' < "$policy/scaling_available_frequencies" | awk -v want="$min_khz" 'NF && $1 >= want' | sort -n | head -n 1)
        [[ -n $target ]] || target=$(<"$policy/scaling_max_freq")
    fi
    if echo "$target" > "$policy/scaling_min_freq" 2>/dev/null; then
        log "CPU 6·7 최저 주파수 ${target}kHz"
    else
        log "CPU 6·7 최저 주파수 변경 실패"
    fi
fi

# 3) CPU 7 깊은 idle 끄기
for state in /sys/devices/system/cpu/cpu7/cpuidle/state*; do
    [[ -w $state/disable && ${state##*state} -ge 1 ]] || continue
    echo 1 > "$state/disable" 2>/dev/null && log "CPU 7 idle $(<"$state/name") 끔"
done
exit 0
