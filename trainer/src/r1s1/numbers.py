"""Every hand-chosen number of S1, registered once with its reason (not swept). The README table is generated from
REASONS; changing a value means a new registered version (objective.md section 8)."""
from __future__ import annotations

import math

from r1lab import contract as C

VERSION = "s1-v2"

# --- time ------------------------------------------------------------------------------------------------------
DT = 0.02                     # policy step (50 Hz, deploy contract); physics 200 Hz (4 substeps)
EPISODE_S = 20.0

# --- goals -----------------------------------------------------------------------------------------------------
GOAL_SHARE = {"hold": 0.25, "locomotion": 0.45, "crouch": 0.30}
LOCO_DIST_M = (0.0, 2.0)
LOCO_TURN_RAD = (-math.pi, math.pi)
PACE_MPS = (0.2, 0.6)
TURN_RATE_RPS = (0.4, 1.0)
SETTLE_S = 0.5
STAND_H = C.STAND_HEIGHT       # 0.7313 m, default pose with flat feet (contract)
CROUCH_H_M = (0.40, STAND_H)
CROUCH_RATE_MPS = (0.1, 0.3)
HOLD_AFTER_S = (1.0, 4.0)      # the goal stays in effect this long after its deadline
HOLD_ONLY_S = (2.0, 6.0)       # duration of a hold (balance) goal, deadline at issue

# --- success, rest, potential ----------------------------------------------------------------------------------
TOL = (0.10, 0.15, 0.04)       # planar m, heading rad, height m
REST = (0.05, 0.20)            # pelvis |v| m/s, |omega| rad/s (all three axes)
LAMBDA = (1.0, math.pi / 2.0, 0.15)
W_SUCCESS = 1.0
W_PROGRESS = 2.0
GAMMA = 0.995

# --- effort (closed form: largest price with r >= 0 at saturation whenever S = 1, including the potential's drain
#     w_progress * (1 - gamma) * Phi per step at a stationary state) ----------------------------------------------
TAU_MAX = tuple(C.GROUPS[C._group_of(n)]["effort"] for n in C.ENTITY_JOINTS)
# joint velocity limits of the robot's own URDF (src/state_manager/config/r1.urdf <limit velocity>), entity order
QDOT_MAX = {
  "left_hip_pitch_joint": 18.8, "left_hip_roll_joint": 18.8, "left_hip_yaw_joint": 18.8, "left_knee_joint": 18.8,
  "left_ankle_pitch_joint": 30.0, "left_ankle_roll_joint": 30.0,
  "right_hip_pitch_joint": 18.8, "right_hip_roll_joint": 18.8, "right_hip_yaw_joint": 18.8, "right_knee_joint": 18.8,
  "right_ankle_pitch_joint": 30.0, "right_ankle_roll_joint": 30.0,
  "waist_roll_joint": 18.8, "waist_yaw_joint": 18.7,
  "left_shoulder_pitch_joint": 18.7, "left_shoulder_roll_joint": 18.7, "left_shoulder_yaw_joint": 33.4,
  "left_elbow_joint": 33.4, "left_wrist_roll_joint": 33.4,
  "right_shoulder_pitch_joint": 18.7, "right_shoulder_roll_joint": 18.7, "right_shoulder_yaw_joint": 33.4,
  "right_elbow_joint": 33.4, "right_wrist_roll_joint": 33.4,
}
QDOT_MAX_T = tuple(QDOT_MAX[n] for n in C.ENTITY_JOINTS)
E_JOULE_SAT = float(len(TAU_MAX))                                           # 24
E_POWER_SAT = float(sum(t * v for t, v in zip(TAU_MAX, QDOT_MAX_T)))        # 28,375.2 W
DRAIN = W_PROGRESS * (1.0 - GAMMA) / DT                                     # 0.5 per second at Phi = 1
C_JOULE = (W_SUCCESS - DRAIN) / E_JOULE_SAT                                 # 1/48
C_POWER = (W_SUCCESS - DRAIN) / E_POWER_SAT                                 # 1.762e-5 / W

# --- terminations ------------------------------------------------------------------------------------------------
GROUND_FORCE_N = 1.0
SELF_FORCE_N = 10.0

# --- physics randomisation (startup draws: each env keeps its body; the policy never sees an env id) ---------------
ACT_LAG_STEPS = 4              # physics steps of 5 ms: 0..20 ms, drawn per physics step with hold 0.9
ACT_HOLD = 0.9
OBS_LAG_STEPS = 1              # policy steps: 0 or 20 ms on the whole sensor frame, hold 0.9
OBS_HOLD = 0.9
FOOT_MU = (0.3, 1.2)
TORSO_COM_M = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
TORSO_MASS_ADD_KG = (-1.0, 2.0)
HAND_LOAD_KG = (0.0, 3.0)
HAND_COM_M = 0.03
PD_SCALE = (0.85, 1.15)
ENCODER_BIAS_RAD = 0.01
IMPULSE_NS = (5.0, 25.0)
IMPULSE_EVERY_S = (4.0, 8.0)
IMPULSE_HEIGHT_M = (-0.35, 0.40)
IMPULSE_SIDE_M = (-0.2, 0.2)

# --- observation -------------------------------------------------------------------------------------------------
HISTORY = 25                   # policy steps = 0.5 s
NOISE = {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 0.5}
GYRO_SCALE = math.sqrt(9.81 / STAND_H)            # 3.66 rad/s: pendulum rate of the body's own size
TAU_SCALE_S = 4.0
RESET_JOINT_RAD = 0.05
RESET_VEL = 0.1

# --- learning ------------------------------------------------------------------------------------------------------
HIDDEN = (512, 256, 128)
LEARNING_RATE = 1.0e-3
DESIRED_KL = 0.01
CLIP = 0.2
EPOCHS = 5
MINI_BATCHES = 4
LAM = 0.95
ENTROPY = 0.001
INIT_STD = 1.0
STEPS_PER_ENV = 24
SAVE_INTERVAL = 50
ITERATIONS = 6000              # measured 1.85-1.89 s/update: 2 seeds x 6,000 = 6.2 GPU-h (README)

# --- R12 S0b (positive 22, 2026-11-20; coordinator 04:1x-05:2x; negative 18 addenda 31-34) ------------------------
# S0b runner = S0 unchanged (objective v2, entropy 0.001, every number above, rsl_rl's schedule) + records + in-run
# gates (r1s1.learning.S1PPO). The fix slot is open (coordinator 05:2x: not chosen yet).
LR_FLOOR = 1.0e-5              # = rsl_rl's own lower bound (S0's schedule); 1e-7 of 04:4x withdrawn at 05:2x
GATE_STREAK = 100              # in-run gate S1 (floor): consecutive updates; the KL streak is recorded only
GATE_WINDOW = 500              # gate S2: floor share window, read at the readout updates
GATE_SHARE = 0.5               # gate S2: share that stops the run
READOUTS_FIRST = (1000, 2000)  # readout updates with gates (500 is record only) ...
READOUT_EVERY = 2000           # ... then every 2,000
SIGMA_PAIR_RATIO_MAX = 2.0     # symmetry gate: largest left/right mirror-pair std ratio
SIGMA_COLLAPSE = 0.1           # sigma collapse: min std < 0.1 x median std of the 24 joints
SIGMA_PARAMS = ("scalar", "log", "tied", "tied_log", "fixed")   # policy-std parametrisations; "scalar" = S0
# S0b-fixed policy std (coordinator 07:5x / 08:0x): the recorded vector FIXED_STD (sigma_500, not swept) times the
# one registered scalar SIGMA_FIXED_SCALE (reasons below; order ENTITY_JOINTS)
SIGMA_FIXED_SCALE = 0.5
FIXED_STD = (0.632652, 0.641591, 0.844348, 0.470685, 0.253064, 0.558918,     # left leg
             0.632652, 0.641591, 0.844348, 0.470685, 0.253064, 0.558918,     # right leg (= left, mirror pairs)
             0.336179, 0.62315,                                              # waist roll, yaw
             0.926197, 0.829115, 0.753655, 0.816444, 0.948703,               # left arm
             0.926197, 0.829115, 0.753655, 0.816444, 0.948703)               # right arm

REASONS = {
  "EPISODE_S": "2-3 goals per episode (a move of at most ~8 s incl. settle + 1-4 s hold); the parent's 10 s held one and a half",
  "GOAL_SHARE": "locomotion has the largest goal space (planar 2 + heading 1); hold is the push-recovery cell",
  "LOCO_DIST_M": "0 = turn in place; 2 m = several gait cycles; beyond that the goal is System 2's job",
  "PACE_MPS": "covers the scorecard's commanded speeds (0.2, 0.35 m/s) with a slow end; not faster (calm first)",
  "TURN_RATE_RPS": "covers the scorecard's turn command (0.6 rad/s)",
  "SETTLE_S": "time to come to rest added to every move budget",
  "CROUCH_H_M": "low end below the parent's learned 0.50 m (R11-1: 0.54 m reached, 0.32-0.35 m not); checked for heels-down kinematic reach on the CPU before launch",
  "CROUCH_RATE_MPS": "slow to moderate pelvis descent; a squat in 1-3 s",
  "HOLD_AFTER_S": "success is only collected after the deadline: the hold is where it is earned",
  "TOL": "planar 0.10 m < one step (~0.25 m), so 'arrived' needs no further step; 0.15 rad ~ 9 deg facing; 0.04 m ~ 5 % of standing pelvis height",
  "REST": "about 5x human quiet-standing CoM speed (~0.01 m/s) and sway rate; excludes stepping (pelvis > 0.1 m/s)",
  "LAMBDA": "the goal distribution's median sizes (1 m, pi/2, 0.15 m), so the potential has slope at the farthest start",
  "W_PROGRESS": "2: the potential's per-step drain w(1-gamma)Phi stays at half the success rate (v1's 4 drained all of it, so the realised in-goal step reward was -effort); policy-invariant anyway",
  "GAMMA": "horizon 1/(1-gamma) = 200 steps = 4 s ~ the median deadline budget: paying effort now for success at the deadline is not discounted below 1/e",
  "C_JOULE": "largest price with r >= 0 whenever S = 1 at a stationary state, at full saturation of all 24 joints: (w_s - drain) / 24 = 1/48",
  "C_POWER": "the same with all joints at torque and velocity limits (E_sat = 28,375 W): 0.5 / 28,375.2 W",
  "GROUND_FORCE_N": "contact detection floor of the parent's trunk_ground (solver noise below)",
  "SELF_FORCE_N": "mjlab / parent self-collision detection threshold (grazing below)",
  "ACT_LAG_STEPS": "deployment chain policy -> 3 DDS hops -> USB 950 us -> motor; p99 sum ~13 ms (+ laptop Wi-Fi one way p99 2.2 ms) rounded up to the 5 ms physics quantum with margin for the unmeasured inference/MCU parts: 0-20 ms (README latency table)",
  "OBS_LAG_STEPS": "sensor -> /lowstate 100 Hz hold (0-10 ms) + DDS (p99 7-8 ms) + Wi-Fi; mjlab delays observations in whole policy steps: lag 0 or 20 ms brackets 1-21 ms",
  "FOOT_MU": "rubber sole on polished/dusty floors (0.3) to carpet/rubber (1.2); mjlab tracking default",
  "TORSO_COM_M": "parent / mjlab model-error bracket",
  "TORSO_MASS_ADD_KG": "parent bracket (cables, battery variants)",
  "HAND_LOAD_KG": "the user's carry range: 0-3 kg per hand (R11 load cell)",
  "HAND_COM_M": "load centre anywhere within a fist-sized volume",
  "PD_SCALE": "project walking standard (trainer/docs/01_training.md: PD +-15 %)",
  "IMPULSE_NS": "the evaluation's push range (scorecard standing push 5-25 N s)",
  "IMPULSE_EVERY_S": "separate events, most time calm (parent spacing)",
  "HISTORY": "one Froude-scaled step period (~0.47 s for the R1's leg): every window holds at least one touchdown, the event that reveals friction, mass and latency",
  "NOISE": "the parent's sensor noise levels (gyro, gravity, encoder, joint velocity)",
  "GYRO_SCALE": "sqrt(g/L) of the standing pelvis height",
  "HIDDEN": "mjlab's from-scratch humanoid velocity recipe (G1)",
  "LEARNING_RATE": "mjlab's from-scratch recipe: 1e-3 with the adaptive KL schedule (desired KL 0.01)",
  "ENTROPY": "the Joule price's pull on log std at std 1 is ~1e-3-1e-2 per joint in normalised-advantage units; the bonus stays below it (R4-R10 runaway when nothing priced the noise)",
  "INIT_STD": "framework default; learned per joint, a rise is logged as the saturation signal",
  "ITERATIONS": "fixed from the measured seconds per update so that two seeds fit the 8 GPU-h budget (README)",
  "LR_FLOOR": "rsl_rl's own constant, so S0b optimises exactly as S0. History: 1e-7 was registered at 04:4x from the R12 KL measurement and withdrawn at 05:2x (coordinator): 99.6-99.7 % of the measured KL came from one joint whose std had collapsed (right ankle pitch, 0.006), and in the 200-update resume smoke the rate followed that std down (README R12-G.4 / G.5)",
  "GATE_STREAK": "negative 18 addendum 33 (freeze 58): S0 would have rung at 938 / 812 with 100 (its longest excursions off the floor were 12 / 22 updates). The KL streak (2 x desired KL) is counted and logged, not a stop (coordinator 05:2x)",
  "GATE_WINDOW": "negative 18 addendum 33 (freeze 58): the floor share of the previous 500 updates at each readout",
  "GATE_SHARE": "negative 18 addendum 33 (freeze 58): S0's 500-1,000 shares were 0.38 / 0.62",
  "READOUTS_FIRST": "negative 18 addendum 33 (freeze 58): readouts 500 (record only), 1,000, 2,000, then every 2,000; 6,000 is not promised",
  "READOUT_EVERY": "negative 18 addendum 33 (freeze 58)",
  "SIGMA_COLLAPSE": "negative 18 addendum 37 (freeze 63): chosen before the replay from {0.1, 0.2, 0.3} as the largest value that rings within +-150 updates of S0's first floor touch and never in r1adv_hold runs from scratch; replay: 0.1 rings at 750 / 650 in S0 (floor 759 / 618) and in none of 59 r1adv runs",
  "FIXED_STD": "coordinator 07:5x (S0b = S0 with the std not learned): the per-joint std_param of the four S0-setting runs at update 500 (S0 seeds 1, 2: model_500 sha256 7f5fb91acb915bb2..., 187b1825d99de059...; seeds 3, 4: fabdc31b2044e83f..., 1d2cbd8d8c45dbab...; full hashes in README R12-G.27), averaged over the runs, then each left/right mirror pair averaged; the waist joints keep their run average. Read once from the records, not swept. min / median 0.36",
  "SIGMA_FIXED_SCALE": "coordinator 08:0x: the fixed std is FIXED_STD x this scalar. By negative 18's pre-value expectation the v2 success term is near 0 at sigma 0.7 in gate 3, so sigma_500 as is would close the B outcome by construction; 0.5 is the single re-pick value negative 18 froze (not swept)",
  "SIGMA_PARAMS": "prepared, not chosen (coordinator 06:1x; negative 18 addendum 37 arms): scalar = rsl_rl GaussianDistribution as S0; log = arm (ㄷ), the same class with std_type log (Adam steps act on log std, i.e. relative changes); tied = arm (ㄱ), r1s1.distribution.MirrorTiedGaussian, one std per orbit of the actor's own mirror map (11 pairs + 2 waist = 13), so the policy distribution is mirror-symmetric like its mean; tied_log = both. Initial std 1.0 in all four (INIT_STD)",
  "SIGMA_PAIR_RATIO_MAX": "a joint's KL weight is 1/std^2, so a pair ratio of 2 is a factor 4 between mirror twins; in S0 the largest pair ratio was 1.20 at update 500 (both seeds) and 37-59 from update 1,000 on (right / left ankle pitch); 2 separates the two with room (the coordinator asked the symmetry gate to look at std, 05:2x)",
}
