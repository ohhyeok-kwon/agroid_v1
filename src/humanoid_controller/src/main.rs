//! Periodic MCU status exchange -> independent BMS/motor decoders -> ROS.
use humanoid_controller::{
    bms::BmsMonitor,
    mcu::McuReceiver,
    mcu_telemetry::McuTelemetry,
    motor::MotorMonitor,
    motor_init::{InitializationStatus, MotorInitializer},
    r1_profile::R1Profile,
    rt,
    tx::{Transmitter, REQUEST_PERIOD},
    usb::UsbTransport,
};
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::humanoid_controller_msg::msg::{LowCmd, LowState};
use ros_env::std_msgs::msg::String as RosString;
use std::{
    io,
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Monitors {
    bms: BmsMonitor,
    motor: MotorMonitor,
    feedback: humanoid_controller::motion_tx::Feedback,
    yaw_offset: Option<f32>,
    remote: humanoid_controller::emergency::Receiver,
    imu: Option<humanoid_controller::imu::Imu>,
}

impl Default for Monitors {
    fn default() -> Self {
        let mut motor = MotorMonitor::default();
        motor.set_initialization(&InitializationStatus::default());
        Self {
            bms: BmsMonitor::default(),
            motor,
            feedback: Default::default(),
            yaw_offset: None,
            remote: Default::default(),
            imu: None,
        }
    }
}

impl Monitors {
    fn connection(&mut self, connected: bool, error: Option<String>) {
        self.bms.connection(connected, error.clone());
        self.motor.connection(connected, error);
    }
}

struct ReceiverTask {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

fn publish_snapshot(
    output: &mut humanoid_controller::spsc::Producer<humanoid_controller::telemetry::Snapshot, 128>,
    initializer: &InitializationStatus,
    frame: Option<&[u8; humanoid_controller::mcu::FRAME_SIZE]>,
    connected: bool,
    timing: Option<rt::CycleTiming>,
    execution: humanoid_controller::motion_tx::ExecutionStatus,
    dropped: &mut u64,
) {
    use humanoid_controller::{clock, telemetry};
    let packet = telemetry::Snapshot {
        connected,
        frame: frame.copied(),
        mono_ms: clock::mono_ms(),
        unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        initialization: telemetry::Initialization::capture(initializer),
        timing,
        execution,
        dropped: *dropped,
    };
    if !output.push(packet) {
        *dropped += 1;
    }
}

impl ReceiverTask {
    fn start(
        mut output: humanoid_controller::spsc::Producer<
            humanoid_controller::telemetry::Snapshot,
            128,
        >,
        mut commands: humanoid_controller::spsc::Consumer<
            humanoid_controller::motion_command::Command,
            8,
        >,
        log: SyncSender<String>,
        memory_locked: bool,
        epoch: u64,
    ) -> io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("usb_status".into())
            .spawn(move || {
                use humanoid_controller::{clock, emergency, motion_tx};
                rt::prefault_stack();
                let note = |text: String| {
                    let _ = log.try_send(text);
                };
                let mut frames = McuReceiver::default();
                let mut feedback = motion_tx::Feedback::default();
                let mut estop = emergency::Receiver::default();
                let mut execution = motion_tx::Executor::new(epoch);
                let mut initializer = MotorInitializer::default();
                let mut buffer = [0u8; 6000];
                let mut dropped = 0;
                while !stopping.load(Ordering::Relaxed) {
                    let _ = rt::set_normal();
                    let result = UsbTransport::open().and_then(|mut usb| {
                        let realtime = rt::set_fifo(rt::USB_PRIORITY).is_ok();
                        frames.reset();
                        feedback = motion_tx::Feedback::default();
                        let profile = if !initializer.attempted() {
                            match R1Profile::load() {
                                Ok(p) => Some(p),
                                Err(e) => {
                                    initializer.reject_profile(&e);
                                    return Err(e);
                                }
                            }
                        } else {
                            None
                        };
                        if let Some(profile) = &profile {
                            initializer.initialize(
                                profile,
                                &mut usb,
                                |delay| interruptible_wait(delay, &stopping),
                                |status, bytes| {
                                    note(format!(
                                        "humanoid_controller: {:?}: {}",
                                        status.phase, status.message
                                    ));
                                    publish_snapshot(
                                        &mut output,
                                        status,
                                        None,
                                        true,
                                        None,
                                        execution.status(clock::mono_us()),
                                        &mut dropped,
                                    );
                                    if let Some(bytes) = bytes {
                                        frames.receive_each(bytes, |frame| {
                                            publish_snapshot(
                                                &mut output,
                                                status,
                                                Some(frame),
                                                true,
                                                None,
                                                execution.status(clock::mono_us()),
                                                &mut dropped,
                                            )
                                        });
                                    }
                                },
                            )?;
                        }
                        let mut transmitter = match profile {
                            Some(p) if initializer.poll_motors() => {
                                Transmitter::with_motor_polling(p)
                            }
                            _ => Transmitter::default(),
                        };
                        let result = (|| -> io::Result<()> {
                            let mut period =
                                rt::Periodic::new(REQUEST_PERIOD, Duration::from_secs(1));
                            let mut timing = None;
                            let mut cycles = 0u64;
                            while !stopping.load(Ordering::Relaxed) {
                                period.begin();
                                let phase = initializer.status().phase;
                                initializer.tick(Instant::now());
                                let ready = initializer.status().phase
                                    == humanoid_controller::motor_init::Phase::Ready;
                                if ready {
                                    usb.set_periodic();
                                }
                                let now_us = clock::mono_us();
                                let targets = execution.tick(
                                    now_us,
                                    ready,
                                    &feedback,
                                    commands.latest(),
                                    estop.active(),
                                );
                                transmitter.update_initialization(initializer.status());
                                let applied = execution.applies() && ready;
                                transmitter.send_control(
                                    &mut usb,
                                    &mut initializer,
                                    applied.then_some(&targets),
                                )?;
                                if applied {
                                    execution.record_applied();
                                }
                                let count = usb.read(&mut buffer)?;
                                frames.receive_each(&buffer[..count], |frame| {
                                    feedback.receive(frame, clock::mono_us());
                                    estop.receive(frame);
                                    if initializer.status().phase
                                        == humanoid_controller::motor_init::Phase::Ready
                                    {
                                        initializer.observe_ready(frame);
                                    } else if let Ok(data) =
                                        humanoid_controller::motor::MotorData::decode_with_context(
                                            frame,
                                            initializer.status().phase.is_specification(),
                                        )
                                    {
                                        initializer.observe(&data);
                                    }
                                    publish_snapshot(
                                        &mut output,
                                        initializer.status(),
                                        Some(frame),
                                        true,
                                        timing,
                                        execution.status(clock::mono_us()),
                                        &mut dropped,
                                    );
                                });
                                initializer.tick(Instant::now());
                                if phase != initializer.status().phase {
                                    note(format!(
                                        "humanoid_controller: {:?}: {}",
                                        initializer.status().phase,
                                        initializer.status().message
                                    ));
                                }
                                if let Some(value) = period.end(realtime, memory_locked) {
                                    timing = Some(value);
                                }
                                cycles += 1;
                                if count == 0 && cycles % 100 == 0 {
                                    publish_snapshot(
                                        &mut output,
                                        initializer.status(),
                                        None,
                                        true,
                                        timing,
                                        execution.status(clock::mono_us()),
                                        &mut dropped,
                                    );
                                }
                                period.sleep();
                            }
                            Ok(())
                        })();
                        initializer.stop();
                        for _ in 0..3 {
                            if transmitter.release(&mut usb).is_err() {
                                break;
                            }
                        }
                        result
                    });
                    if let Err(error) = result {
                        initializer.disconnected(&error);
                        note(format!("humanoid_controller: {error}; reconnecting"));
                    }
                    publish_snapshot(
                        &mut output,
                        initializer.status(),
                        None,
                        false,
                        None,
                        execution.status(clock::mono_us()),
                        &mut dropped,
                    );
                    if interruptible_wait(Duration::from_secs(1), &stopping).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

fn interruptible_wait(duration: Duration, stop: &AtomicBool) -> io::Result<()> {
    let deadline = Instant::now() + duration;
    loop {
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "controller is stopping",
            ));
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Ok(());
        };
        thread::sleep(remaining.min(Duration::from_millis(50)));
    }
}

impl Drop for ReceiverTask {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct ShutdownSignals(libc::sigset_t);

impl ShutdownSignals {
    fn block() -> io::Result<Self> {
        // SAFETY: initialize the set and block signals before creating any threads.
        unsafe {
            let mut set = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGTERM);
            libc::sigaddset(&mut set, libc::SIGINT);
            let result = libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
            if result != 0 {
                return Err(io::Error::from_raw_os_error(result));
            }
            Ok(Self(set))
        }
    }

    fn requested(&self) -> io::Result<bool> {
        let timeout = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: initialized signal set and timeout; siginfo is optional.
        if unsafe { libc::sigtimedwait(&self.0, std::ptr::null_mut(), &timeout) } >= 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EAGAIN | libc::EINTR) => Ok(false),
            _ => Err(error),
        }
    }
}

/// `/motor_status` also feeds the joint display (Operation Center polls every 100 ms);
/// `/bms_status` and the DDS interface check stay at 1 s.
const MOTOR_STATUS_PERIOD: Duration = Duration::from_millis(100);

// Copy under the lock, serialize after it: the USB thread takes this lock every cycle.
fn bms_message(monitor: &Mutex<Monitors>) -> Result<RosString, Box<dyn std::error::Error>> {
    let bms = monitor
        .lock()
        .map_err(|_| io::Error::other("MCU receiver stopped unexpectedly"))?
        .bms
        .clone();
    Ok(RosString {
        data: serde_json::to_string(&bms.status())?,
    })
}

fn motor_message(monitor: &Mutex<Monitors>) -> Result<RosString, Box<dyn std::error::Error>> {
    let motor = monitor
        .lock()
        .map_err(|_| io::Error::other("MCU receiver stopped unexpectedly"))?
        .motor
        .clone();
    Ok(RosString {
        data: serde_json::to_string(&motor.status())?,
    })
}

/// Identifies this controller process (one motor-initialization session) to consumers.
fn new_epoch() -> u64 {
    let mut bytes = [0u8; 8];
    let random = std::fs::File::open("/dev/urandom")
        .and_then(|mut file| io::Read::read_exact(&mut file, &mut bytes))
        .is_ok();
    if random && bytes != [0; 8] {
        return u64::from_le_bytes(bytes);
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |time| time.as_nanos() as u64)
}

fn lowstate_message(
    state: &Monitors,
    encoder: &humanoid_controller::joint_wire::Encoder,
    epoch: u64,
) -> Option<LowState> {
    use humanoid_controller::{
        checksum::crc32_words,
        r1_profile::{JOINT_MOTORS, NATIVE_INDICES},
    };
    if !state.feedback.available() {
        return None;
    }
    let positions = encoder.positions(state.feedback.q, state.yaw_offset)?;
    let data = state.motor.latest_data()?;
    let mut msg = LowState::default();
    msg.mode_machine = 1;
    msg.tick = (state.feedback.received_us / 1000) as u32;
    msg.reserve = [epoch as u32, (epoch >> 32) as u32, 0, 0];
    if let Some(packet) = state.remote.packet() {
        msg.wireless_remote = packet;
    }
    for (i, (address, direction, ratio)) in JOINT_MOTORS.iter().enumerate() {
        let slot = data
            .motors
            .iter()
            .find(|m| m.bus == address.bus && m.id == address.id)?;
        let sample = slot.data.as_ref()?;
        let motor = &mut msg.motor_state[NATIVE_INDICES[i]];
        motor.mode = sample.mode;
        motor.q = positions[i];
        let (dq_scale, tau_scale) = if i == 25 {
            (0.40743667, 256000.)
        } else {
            (10.185916, 2560.)
        };
        motor.dq = f32::from(sample.dq_raw) * direction / ratio / dq_scale;
        motor.tau_est = f32::from(sample.tau_raw) * direction * ratio / tau_scale;
        motor.temperature = sample.temperature;
        motor.vol = sample.voltage_v;
        motor.motorstate = sample.error_code | if sample.timeout { 1 << 30 } else { 0 };
    }
    let speeds = encoder.velocities(
        state.feedback.q,
        std::array::from_fn(|i| msg.motor_state[NATIVE_INDICES[i]].dq),
    )?;
    for (i, speed) in speeds.into_iter().enumerate() {
        msg.motor_state[NATIVE_INDICES[i]].dq = speed;
    }
    if let Some(imu) = state.imu {
        msg.imu_state.quaternion = imu.quaternion;
        msg.imu_state.gyroscope = imu.gyroscope;
        msg.imu_state.accelerometer = imu.accelerometer;
        msg.imu_state.rpy = imu.rpy;
        msg.imu_state.temperature = imu.temperature;
    }
    // Serialize the manufacturer's plain little-endian struct (including zero padding)
    // for its application CRC, not the DDS/CDR envelope.
    let mut bytes = Vec::with_capacity(2088);
    for value in msg.version {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend([msg.mode_pr, msg.mode_machine, 0, 0]);
    bytes.extend_from_slice(&msg.tick.to_le_bytes());
    for v in msg
        .imu_state
        .quaternion
        .into_iter()
        .chain(msg.imu_state.gyroscope)
        .chain(msg.imu_state.accelerometer)
        .chain(msg.imu_state.rpy)
    {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&msg.imu_state.temperature.to_le_bytes());
    bytes.extend([0, 0]);
    for m in &msg.motor_state {
        bytes.extend([m.mode, 0, 0, 0]);
        for v in [m.q, m.dq, m.ddq, m.tau_est] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in m.temperature {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&m.vol.to_le_bytes());
        for v in m.sensor {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&m.motorstate.to_le_bytes());
        for v in m.reserve {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes.extend(msg.wireless_remote);
    for v in msg.reserve {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), 2088);
    msg.crc = crc32_words(&bytes);
    Some(msg)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Before any thread starts: later stacks and heap are locked when first touched.
    let memory_locked = match rt::lock_memory() {
        Ok(()) => true,
        Err(error) => {
            eprintln!("humanoid_controller: mlockall failed: {error}; memory is not locked");
            false
        }
    };
    let signals = ShutdownSignals::block()?;
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("humanoid_controller")?;
    let bms_publisher =
        node.create_publisher::<RosString>("bms_status".keep_last(1).reliable().transient_local())?;
    let motor_publisher = node
        .create_publisher::<RosString>("motor_status".keep_last(1).reliable().transient_local())?;
    let monitor = Arc::new(Mutex::new(Monitors::default()));
    let epoch = new_epoch();
    monitor
        .lock()
        .map_err(|_| io::Error::other("MCU receiver stopped unexpectedly"))?
        .motor
        .set_epoch(epoch);
    bms_publisher.publish(bms_message(&monitor)?)?;
    motor_publisher.publish(motor_message(&monitor)?)?;
    let raw_publisher =
        node.create_publisher::<RosString>("mcu_status".keep_last(256).reliable())?;
    let (raw_sender, raw_receiver) = mpsc::sync_channel(256);
    let (log_sender, log_receiver) = mpsc::sync_channel::<String>(64);
    let (snapshots, mut snapshot_rx) =
        humanoid_controller::spsc::channel::<humanoid_controller::telemetry::Snapshot, 128>();
    let (target_tx, target_rx) =
        humanoid_controller::spsc::channel::<humanoid_controller::motion_command::Command, 8>();
    let target_tx = Mutex::new(target_tx);
    let command_monitor = Arc::clone(&monitor);
    let encoder = humanoid_controller::joint_wire::Encoder::default();
    let rejected = std::sync::atomic::AtomicU64::new(0);
    let _commands = node.create_subscription::<LowCmd, _>(
        humanoid_controller::motion_command::TOPIC
            .keep_last(1)
            .best_effort(),
        move |message: LowCmd| {
            use humanoid_controller::{
                clock,
                lowcmd::{LowCommand, Motor},
            };
            let low = LowCommand {
                mode_pr: message.mode_pr,
                mode_machine: message.mode_machine,
                motors: std::array::from_fn(|i| {
                    let m = &message.motor_cmd[i];
                    Motor {
                        mode: m.mode,
                        q: m.q,
                        dq: m.dq,
                        tau: m.tau,
                        kp: m.kp,
                        kd: m.kd,
                        reserve: m.reserve,
                    }
                }),
                reserve: message.reserve,
                crc: message.crc,
            };
            let state = command_monitor.lock().unwrap();
            let command = if state.feedback.available() {
                encoder.encode(&low, state.feedback.q, state.yaw_offset, clock::mono_ms())
            } else {
                Err("motor feedback required for LowCmd")
            };
            drop(state);
            match command {
                Ok(mut c) => {
                    c.epoch = epoch;
                    let _ = target_tx.lock().unwrap().push(c);
                }
                Err(error) => {
                    let count = rejected.fetch_add(1, Ordering::Relaxed) + 1;
                    if count == 1 || count % 500 == 0 {
                        eprintln!("humanoid_controller: rejected LowCmd ({count}): {error}");
                    }
                }
            }
        },
    )?;
    let _receiver = ReceiverTask::start(snapshots, target_rx, log_sender, memory_locked, epoch)?;
    let mut telemetry = McuTelemetry::new(raw_sender);
    let receiver_monitor = Arc::clone(&monitor);
    let _raw_timer = node.create_timer_repeating(Duration::from_millis(5), move || {
        for _ in 0..127 {
            let Some(packet) = snapshot_rx.pop() else {
                break;
            };
            let mut state = receiver_monitor.lock().unwrap();
            state.connection(packet.connected, None);
            let initialization = packet.initialization.restore();
            state.yaw_offset = initialization.head.and_then(|h| h.yaw_offset_rad);
            state.motor.set_initialization(&initialization);
            state.motor.set_execution(packet.execution, packet.dropped);
            if let Some(t) = packet.timing {
                state.motor.set_timing(t);
            }
            if let Some(bytes) = packet.frame {
                state.feedback.receive(&bytes, packet.mono_ms * 1000);
                state.remote.receive(&bytes);
                if let Some(imu) = humanoid_controller::imu::Imu::decode(&bytes) {
                    state.imu = Some(imu);
                }
                let frame = humanoid_controller::mcu::McuFrame { bytes };
                state.bms.receive(frame.bms_fragment());
                state
                    .motor
                    .receive_at(&frame.bytes, packet.unix_ms, packet.mono_ms);
                telemetry.receive_at(&frame, packet.unix_ms, packet.mono_ms);
            }
        }
        for sample in raw_receiver.try_iter().take(256) {
            if let Err(error) = serde_json::to_string(&sample)
                .map_err(|e| e.to_string())
                .and_then(|data| {
                    raw_publisher
                        .publish(RosString { data })
                        .map_err(|e| e.to_string())
                })
            {
                eprintln!("humanoid_controller: mcu_status publish failed: {error}");
            }
        }
    })?;
    let motor_monitor = Arc::clone(&monitor);
    let _motor_timer = node.create_timer_repeating(MOTOR_STATUS_PERIOD, move || {
        for line in log_receiver.try_iter() {
            eprintln!("{line}");
        }
        if let Err(error) = motor_message(&motor_monitor)
            .and_then(|motor| motor_publisher.publish(motor).map_err(Into::into))
        {
            eprintln!("humanoid_controller: motor_status publish failed: {error}");
        }
    })?;
    // Native policy runs at 50 Hz: publish the latest fixed-layout feedback at 100 Hz.
    let lowstate = node.create_publisher::<LowState>("/lowstate".keep_last(1).best_effort())?;
    let state_encoder = humanoid_controller::joint_wire::Encoder::default();
    let feedback_monitor = Arc::clone(&monitor);
    let _feedback_timer = node.create_timer_repeating(Duration::from_millis(10), move || {
        let state = feedback_monitor.lock().unwrap();
        if let Some(message) = lowstate_message(&state, &state_encoder, epoch) {
            if let Err(e) = lowstate.publish(message) {
                eprintln!("humanoid_controller: lowstate publish failed: {e}");
            }
        }
    })?;
    let _timer = node.create_timer_repeating(Duration::from_secs(1), move || {
        if let Err(error) =
            bms_message(&monitor).and_then(|bms| bms_publisher.publish(bms).map_err(Into::into))
        {
            eprintln!("humanoid_controller: bms_status publish failed: {error}");
        }
    })?;
    while !signals.requested()? {
        executor
            .spin(SpinOptions::default().timeout(Duration::from_millis(100)))
            .timeout_ok()
            .first_error()?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("humanoid_controller: {error}");
            ExitCode::FAILURE
        }
    }
}
