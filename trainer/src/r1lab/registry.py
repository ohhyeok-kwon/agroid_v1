"""Lightweight task identity, loaded before GPU libraries and environment setup."""

TASK_SPECS = {
  "wb_task_teacher_v12h_calm10": {"action":"wholebody","tags":"p1t12h-calm10","runner":"r1wb"},
  "wb_task_teacher_v12h_calm01": {"action":"wholebody","tags":"p1t12h-calm01","runner":"r1wb"},
  "wb_task_teacher_v12h_calm11": {"action":"wholebody","tags":"p1t12h-calm11","runner":"r1wb"},
  "wb_task_teacher_v12h0_head_control_v11c": {"action":"wholebody","tags":"p1t12h0c-head-control","runner":"r1wb"},
  "wb_task_teacher_v12h_head_perception_v11c": {"action":"wholebody","tags":"p1t12hc-head-perception","runner":"r1wb"},
  "wb_task_teacher_v12h0_head_control": {"action":"wholebody","tags":"p1t12h0-head-control","runner":"r1wb"},
  "wb_task_teacher_v12h_head_perception": {"action":"wholebody","tags":"p1t12h-head-perception","runner":"r1wb"},
  "wb_task_teacher_v12h0_head_control_v11a": {"action":"wholebody","tags":"p1t12h0a-head-control","runner":"r1wb"},
  "wb_task_teacher_v12h_head_perception_v11a": {"action":"wholebody","tags":"p1t12ha-head-perception","runner":"r1wb"},
  "careful_walk": {"action": "careful_walk", "tags": "s2-rough-quiet"},
  "careful_walk_flat": {"action": "careful_walk", "tags": "s2-flat"},
  "careful_walk_s1": {"action": "careful_walk", "tags": "s1-flat-wide"},
  "careful_walk_s2": {"action": "careful_walk", "tags": "s2-rough-quiet"},
  "careful_walk_s3": {"action": "careful_walk", "tags": "s3-slowturn"},
  "careful_walk_s4": {"action": "careful_walk", "tags": "s4-slowturn-clr9"},
  "careful_walk_s5": {"action": "careful_walk", "tags": "s5-stumble-hard-clr9"},
  "careful_walk_s5p": {"action": "careful_walk", "tags": "s5p-stumble-hard-clr9-scan"},
  "careful_walk_s6a": {"action": "careful_walk", "tags": "s6a-recovery-control"},
  "careful_walk_s6b": {"action": "careful_walk", "tags": "s6b-impulse-burst"},
  "careful_walk_s6c": {"action": "careful_walk", "tags": "s6c-recovery-gate"},
  "careful_walk_s6d": {"action": "careful_walk", "tags": "s6d-burst-gate"},
  "careful_walk_s7a": {"action": "careful_walk", "tags": "s7-ball-self-q25"},
  "careful_walk_s7b": {"action": "careful_walk", "tags": "s7-ball-self-q05"},
  "careful_walk_s8": {"action": "careful_walk", "tags": "s8-scan-human-k4-anchor",
                       "evaluation_profile": "careful_adversarial_v1"},
  "careful_walk_s8r": {"action": "careful_walk", "tags": "s8r-scan-human-rsi10-k4-anchor",
                        "evaluation_profile": "careful_adversarial_v1"},
  "careful_walk_teacher": {"action": "careful_walk", "tags": "t1-fullbody-reference-scratch",
                            "evaluation_profile": "reference_teacher_v1"},
  "careful_walk_teacher_precise": {"action": "careful_walk", "tags": "t2-feet-world-scratch-lineage",
                                    "evaluation_profile": "reference_teacher_v1",
                                    "continue_random_teacher": True, "precise_foot_tracking": True},
  "careful_walk_teacher_coordinated": {"action": "careful_walk", "tags": "t3-arm-points-scratch-lineage",
                                        "evaluation_profile": "reference_teacher_v1",
                                        "continue_random_teacher": True, "precise_foot_tracking": True,
                                        "arm_position_tracking": True},
  "careful_walk_teacher_fullcapture": {"action": "careful_walk", "tags": "t4-fullcapture-startstop-lineage",
                                        "evaluation_profile": "reference_teacher_v1",
                                        "continue_random_teacher": True, "precise_foot_tracking": True,
                                        "reference_transition": True, "reference_contract": "full_capture"},
  "careful_walk_teacher_phasefree": {"action": "careful_walk", "tags": "t5-phasefree-feet-lineage",
                                       "evaluation_profile": "reference_teacher_v1",
                                       "continue_random_teacher": True, "precise_foot_tracking": True,
                                       "phase_mode": "zero"},
  "careful_walk_teacher_library": {"action": "careful_walk", "tags": "t6-library-phasefree-lineage",
                                     "evaluation_profile": "reference_teacher_v1",
                                     "continue_random_teacher": True, "precise_foot_tracking": True,
                                     "reference_transition": True, "phase_mode": "zero",
                                     "reference_contract": "library"},
  "careful_walk_teacher_library_xy": {"action": "careful_walk", "tags": "t6x-library-xyterm-lineage",
                                        "evaluation_profile": "reference_teacher_v1",
                                        "continue_random_teacher": True, "precise_foot_tracking": True,
                                        "reference_transition": True, "phase_mode": "zero",
                                        "reference_contract": "library", "xy_termination_m": 0.5},
  "careful_walk_teacher_matching": {"action": "careful_walk", "tags": "t7-mm-command-lineage",
                                      "evaluation_profile": "reference_teacher_v1",
                                      "continue_random_teacher": True, "precise_foot_tracking": True,
                                      "reference_transition": True, "phase_mode": "zero",
                                      "reference_contract": "matching"},
  "careful_walk_teacher_matching_push": {"action": "careful_walk", "tags": "t8-mm-push-lineage",
                                           "evaluation_profile": "reference_teacher_v1",
                                           "continue_random_teacher": True, "precise_foot_tracking": True,
                                           "reference_transition": True, "phase_mode": "zero",
                                           "reference_contract": "matching"},
  "careful_walk_teacher_matching_quiet": {"action": "careful_walk", "tags": "t9-mm-push-slip-lineage",
                                            "evaluation_profile": "reference_teacher_v1",
                                            "continue_random_teacher": True, "precise_foot_tracking": True,
                                            "reference_transition": True, "phase_mode": "zero",
                                            "reference_contract": "matching"},
  "careful_walk_teacher_matching_projectile": {"action": "careful_walk", "tags": "t10-mm-push-slip-ball-lineage",
                                                 "evaluation_profile": "reference_teacher_v1",
                                                 "continue_random_teacher": True, "precise_foot_tracking": True,
                                                 "reference_transition": True, "phase_mode": "zero",
                                                 "reference_contract": "matching"},
  "careful_walk_teacher_matching_soft": {"action": "careful_walk", "tags": "t11-mm-push-slip-ball-land-lineage",
                                           "evaluation_profile": "reference_teacher_v1",
                                           "continue_random_teacher": True, "precise_foot_tracking": True,
                                           "reference_transition": True, "phase_mode": "zero",
                                           "reference_contract": "matching"},
  "careful_walk_teacher_matching_v4": {"action": "careful_walk", "tags": "t12-mm-v4-land-ball-lineage",
                                         "evaluation_profile": "reference_teacher_v1",
                                         "continue_random_teacher": True, "precise_foot_tracking": True,
                                         "reference_transition": True, "phase_mode": "zero",
                                         "reference_contract": "matching"},
  "careful_walk_teacher_matching_recover": {"action": "careful_walk", "tags": "t13-mm-v4-recover-lineage",
                                              "evaluation_profile": "reference_teacher_v1",
                                              "continue_random_teacher": True, "precise_foot_tracking": True,
                                              "reference_transition": True, "phase_mode": "zero",
                                              "reference_contract": "matching"},
  "careful_walk_teacher_matching_v4_ball": {"action": "careful_walk", "tags": "t12b-mm-v4-ball-lineage",
                                              "evaluation_profile": "reference_teacher_v1",
                                              "continue_random_teacher": True, "precise_foot_tracking": True,
                                              "reference_transition": True, "phase_mode": "zero",
                                              "reference_contract": "matching"},
  "careful_walk_teacher_matching_still": {"action": "careful_walk", "tags": "t14-mm-v4-still-lineage",
                                            "evaluation_profile": "reference_teacher_v1",
                                            "continue_random_teacher": True, "precise_foot_tracking": True,
                                            "reference_transition": True, "phase_mode": "zero",
                                            "reference_contract": "matching"},
  # 810-input student distilled (DAgger) from a completed matching teacher of teacher_task.
  "careful_walk_student_matching": {"action": "careful_walk", "tags": "d1-dagger810-mm-calm",
                                     "evaluation_profile": "command_student_v1", "distillation": True,
                                     "teacher_task": "careful_walk_teacher_matching_still"},
  "careful_walk_teacher_matching_arms": {"action": "careful_walk", "tags": "t15-mm-v4-arms-lineage",
                                           "evaluation_profile": "reference_teacher_v1",
                                           "continue_random_teacher": True, "precise_foot_tracking": True,
                                           "reference_transition": True, "phase_mode": "zero",
                                           "reference_contract": "matching"},
  "careful_walk_teacher_matching_lateral": {"action": "careful_walk", "tags": "t16-mm-v5-lateral-lineage",
                                              "evaluation_profile": "reference_teacher_v1",
                                              "continue_random_teacher": True, "precise_foot_tracking": True,
                                              "reference_transition": True, "phase_mode": "zero",
                                              "reference_contract": "matching"},
  "careful_walk_student_matching_crisis_lateral": {"action": "careful_walk", "tags": "d5-dagger810-mm-v5-lateral-push-ball",
                                                   "evaluation_profile": "command_student_v1", "distillation": True,
                                                   "teacher_task": "careful_walk_teacher_matching_lateral"},
  "careful_walk_student_matching_crisis_arms": {"action": "careful_walk", "tags": "d4-dagger810-mm-arms-push-ball",
                                                "evaluation_profile": "command_student_v1", "distillation": True,
                                                "teacher_task": "careful_walk_teacher_matching_arms"},
  "careful_walk_student_matching_crisis": {"action": "careful_walk", "tags": "d3-dagger810-mm-push-ball",
                                           "evaluation_profile": "command_student_v1", "distillation": True,
                                           "teacher_task": "careful_walk_teacher_matching_still"},
  # PPO fine-tune of a completed distilled student (student_task), anchored to the teacher it was distilled from.
  "careful_walk_student_rl_lateral": {"action": "careful_walk", "tags": "d6-ppo810-bc-mm-v5-push-ball",
                                      "evaluation_profile": "command_student_v1", "student_rl": True,
                                      "teacher_task": "careful_walk_teacher_matching_lateral",
                                      "student_task": "careful_walk_student_matching_crisis_lateral"},
  # Terrain PPO fine-tune of a completed distilled student anchored to itself (careful_walk terrain task).
  "careful_walk_student_terrain": {"action": "careful_walk", "tags": "d7t-ppo810-self-terrain",
                                   "evaluation_profile": "command_student_v1", "student_rl": True,
                                   "self_anchor": True, "teacher_task": "careful_walk_teacher_matching_lateral",
                                   "student_task": "careful_walk_student_matching_crisis_lateral"},
  # D7T-b: the same, on obstacle-dense terrain at fixed difficulty with moving commands and a stumble penalty.
  "careful_walk_student_terrain_dense": {"action": "careful_walk", "tags": "d7tb-ppo810-self-terrain-dense",
                                         "evaluation_profile": "command_student_v1", "student_rl": True,
                                         "self_anchor": True, "teacher_task": "careful_walk_teacher_matching_lateral",
                                         "student_task": "careful_walk_student_matching_crisis_lateral"},
  "safe_crouch": {"action": "safe_crouch", "tags": "push0.3-hit45-weak0.6"},
  # Whole-body model family (package r1wb, design .agent/analysis/unified_policy_20261111/design.md).
  # P1: one privileged tracking teacher over the motion library; trained by r1wb.train.
  "wb_track_teacher": {"action": "wholebody", "tags": "p1v2-track-library-v2core", "runner": "r1wb"},
  # P1c: the P1 teacher continued with the contact-match and slip-excess terms, under actuator delay and gain /
  # mass variation (the student's conditions).
  "wb_track_teacher_dr": {"action": "wholebody", "tags": "p1c-track-contactmatch-hardened", "runner": "r1wb"},
  # P1d: cause check of the teacher's disturbance response (stock-strength kicks / force impulses at the torso).
  "wb_track_teacher_kick": {"action": "wholebody", "tags": "p1d-track-strongkick-drift1m", "runner": "r1wb"},
  "wb_track_teacher_impulse": {"action": "wholebody", "tags": "p1d-track-impulse5to25-drift1m", "runner": "r1wb"},
  # P1e: impulse-trained teacher + uniform L1 price on the joint distance to the course.
  "wb_track_teacher_course": {"action": "wholebody", "tags": "p1e-track-impulse-jointcourse5", "runner": "r1wb"},
  # P1g: price teacher + bursts (4 to 12 balls of 8 to 16 N s) in half of the disturbance events.
  "wb_track_teacher_bursts": {"action": "wholebody", "tags": "p1g-track-impulse-bursts-jointcourse5", "runner": "r1wb"},
  # P1h: tracking terminations suspended for 2 s after a hit (physical fall always ends), single hits / bursts.
  "wb_track_teacher_course_grace": {"action": "wholebody", "tags": "p1h-track-impulse-jointcourse5-grace2s", "runner": "r1wb"},
  "wb_track_teacher_bursts_grace": {"action": "wholebody", "tags": "p1h-track-bursts-jointcourse5-grace2s", "runner": "r1wb"},
  # P1f: one-stage teacher from scratch (kernels + joint-course price + impulses 20 ms to 0.4 s), 2048-2048-1024-512.
  "wb_track_teacher_xl": {"action": "wholebody", "tags": "p1f-track-xl-impulsemix-jointcourse5", "runner": "r1wb"},
  # P1t: the teacher on the task itself (student's environment, teacher's observation + goal and goal lag).
  "wb_task_teacher": {"action": "wholebody", "tags": "p1t-task-teacher-events", "runner": "r1wb"},
  "wb_task_teacher_full": {"action": "wholebody", "tags": "p1t-task-teacher-events-bursts-anywhere", "runner": "r1wb"},
  "wb_task_teacher_survive": {"action": "wholebody", "tags": "p1t-task-teacher-events-bursts-anywhere-failurecost1000", "runner": "r1wb"},
  "wb_task_teacher_v2s": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-singles", "runner": "r1wb"},
  "wb_task_teacher_v2": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-fullrange", "runner": "r1wb"},
  "wb_task_teacher_v2_anywhere": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-singles-anywhere", "runner": "r1wb"},
  "wb_task_teacher_v2_rows": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-rows-trunk", "runner": "r1wb"},
  "wb_task_teacher_v2_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-fullrange-rows12", "runner": "r1wb"},
  "wb_task_teacher_v2_rows25": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-fullrange-rows25", "runner": "r1wb"},
  "wb_task_teacher_v2_price15_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price15", "runner": "r1wb"},
  "wb_task_teacher_v2_price15_anywhere": {"action": "wholebody", "tags": "p1t2-task-teacher-singles-anywhere-price15", "runner": "r1wb"},
  "wb_task_teacher_v2_price45_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price45", "runner": "r1wb"},
  "wb_task_teacher_v2_price45_rows25": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows25-price45", "runner": "r1wb"},
  "wb_task_teacher_v2_price135_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price135", "runner": "r1wb"},
  "wb_task_teacher_v2_price405_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price405", "runner": "r1wb"},
  "wb_task_teacher_v2_price135_fall3000_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price135-fall3000", "runner": "r1wb"},
  "wb_task_teacher_v2_price405_fall9000_rows12": {"action": "wholebody", "tags": "p1t2-task-teacher-fullrange-rows12-price405-fall9000", "runner": "r1wb"},
  "wb_task_teacher_v3_price45_rows12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price45", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_rows12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000", "runner": "r1wb"},
  "wb_task_teacher_v3_price405_fall18000_rows12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price405-fall18000", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_travel27": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-travel27", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap05": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap05", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_steer": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-steered", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_sym": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-sym", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_motion": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-motion", "runner": "r1wb"},
  "wb_task_teacher_v5_recovery_control": {"action": "wholebody", "tags": "p1t5-recovery-control", "runner": "r1wb"},
  "wb_task_teacher_v5_recovery_risk": {"action": "wholebody", "tags": "p1t5-recovery-risk2", "runner": "r1wb"},
  "wb_task_teacher_v6_exposure_control": {"action": "wholebody", "tags": "p1t6-exposure-control", "runner": "r1wb"},
  "wb_task_teacher_v6_exposure_pose": {"action": "wholebody", "tags": "p1t6-exposure-pose2", "runner": "r1wb"},
  "wb_task_teacher_v6_exposure_predict": {"action": "wholebody", "tags": "p1t6-exposure-predict2", "runner": "r1wb"},
  "wb_task_teacher_v7_no_hit_grace_control": {"action": "wholebody", "tags": "p1t7-no-grace-control", "runner": "r1wb"},
  "wb_task_teacher_v7_no_hit_grace_pose": {"action": "wholebody", "tags": "p1t7-no-grace-pose2", "runner": "r1wb"},
  "wb_task_teacher_v7_no_hit_grace_predict": {"action": "wholebody", "tags": "p1t7-no-grace-predict2", "runner": "r1wb"},
  "wb_task_teacher_v8_physical_ends_control": {"action": "wholebody", "tags": "p1t8-physical-ends-control", "runner": "r1wb"},
  "wb_task_teacher_v8b_plan_free_control": {"action": "wholebody", "tags": "p1t8b-plan-free-control", "runner": "r1wb"},
  "wb_task_teacher_v9a_arm_smooth_control": {"action": "wholebody", "tags": "p1t9a-arm-smooth-control", "runner": "r1wb"},
  "wb_task_teacher_v10s_scan": {"action": "wholebody", "tags": "p1t10s-scan", "runner": "r1wb"},
  "wb_task_teacher_v11a_scan_upper_course": {"action": "wholebody", "tags": "p1t11a-scan-upper-course", "runner": "r1wb"},
  "wb_task_teacher_v11c_scan_noise_consistency": {"action": "wholebody", "tags": "p1t11c-scan-noise-consistency", "runner": "r1wb"},
  "wb_task_teacher_v9c_heading_return_control": {"action": "wholebody", "tags": "p1t9c-heading-return-control", "runner": "r1wb"},
  "wb_task_teacher_v9d_heading_arm_smooth": {"action": "wholebody", "tags": "p1t9d-heading-arm-smooth", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_senses": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-senses", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_history": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-history", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_fall12000": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-fall12000", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_free_fall12000": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-free-fall12000", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_free": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-free", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_feet": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-feet", "runner": "r1wb"},
  "wb_task_teacher_v4_ground_sym_bal": {"action": "wholebody", "tags": "p1t4-task-teacher-ground-sym-bal", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_sym_rest50": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-sym-rest50", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_mirror": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-mirror", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_calmrest50": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-calmrest50", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_room100": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-room100", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall12000_cap12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall12000-cap12", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_rest12": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-rest12", "runner": "r1wb"},
  "wb_task_teacher_v3_price135_fall3000_cap12_rest50": {"action": "wholebody", "tags": "p1t3-task-teacher-decisivecourse-rows12-price135-fall3000-cap12-rest50", "runner": "r1wb"},
  "wb_task_teacher_v2_steps": {"action": "wholebody", "tags": "p1t2-task-teacher-failurecost-coherentcourse-fullrange-stepcost25", "runner": "r1wb"},
  # P2: reference-free sparse-goal student (System 1 first version), imitation of the P1 teacher + PPO.
  "wb_goal_student": {"action": "wholebody", "tags": "p2v2-goal-student-reanchored", "runner": "r1wb"},
  # P2p: reference-free student under the teacher's recipe (impulses, joint-course price, grace), with / without labels.
  "wb_goal_student_price": {"action": "wholebody", "tags": "p2p-goal-student-impulse-jointcourse5-labels", "runner": "r1wb"},
  "wb_goal_student_price_nolabel": {"action": "wholebody", "tags": "p2p-goal-student-impulse-jointcourse5-nolabels", "runner": "r1wb"},
  # P2q: student recipe with half of the environments on command streams; stream labels and price only while calm / never.
  "wb_goal_student_mix_price_calm": {"action": "wholebody", "tags": "p2q-student-mix-impulse-jointcourse5-streamcalm", "runner": "r1wb"},
  "wb_goal_student_mix_price_rl": {"action": "wholebody", "tags": "p2q-student-mix-impulse-jointcourse5-streamrl", "runner": "r1wb"},
  "wb_goal_student_mix_price_held": {"action": "wholebody", "tags": "p2r-student-mix-impulse-jointcourse5-streamheld", "runner": "r1wb"},
  "wb_goal_student_mix_price_nohit": {"action": "wholebody", "tags": "p2r-student-mix-impulse-jointcourse5-streamnohit", "runner": "r1wb"},
  # P2b: the student with half of its environments on command streams whose reference is matched to the command
  # and to the robot's own state inside the environment.
  "wb_goal_student_mix": {"action": "wholebody", "tags": "p2b-goal-student-matched-streams", "runner": "r1wb"},
  # P3: correction on the frozen S1-a under impulses, with deviation prices relieved by the balance-urgency gate.
  "wb_harden": {"action": "wholebody", "tags": "p3-harden-residual-impulses", "runner": "r1wb"},
  "wb_harden_norelief": {"action": "wholebody", "tags": "p3-harden-residual-impulses-norelief", "runner": "r1wb"},
  "wb_harden_restrained": {"action": "wholebody", "tags": "p3-harden-armrestraint-statenoise", "runner": "r1wb"},
  "wb_harden_focus": {"action": "wholebody", "tags": "p3-harden-armrestraint-statenoise-backfocus", "runner": "r1wb"},
  "wb_harden_burst": {"action": "wholebody", "tags": "p3-harden-armrestraint-statenoise-bursts", "runner": "r1wb"},
  # P3, sixth round: a single network continued from S1-a; the model learns its own reaction to hits.
  "wb_anchor": {"action": "wholebody", "tags": "p3-anchored-selflearn-spikes-bursts", "runner": "r1wb"},
  "wb_anchor_learned": {"action": "wholebody", "tags": "p3-anchored-selflearn-learnednoise", "runner": "r1wb"},
  "wb_anchor_extreme": {"action": "wholebody", "tags": "p3-bracket-s1a-extreme", "runner": "r1wb"},
  "wb_anchor_scratch": {"action": "wholebody", "tags": "p3-bracket-scratch-extreme", "runner": "r1wb"},
  "wb_anchor_free": {"action": "wholebody", "tags": "p3-bracket-s1a-extreme-noanchor", "runner": "r1wb"},
  "wb_walker": {"action": "wholebody", "tags": "p3-bracket-stock-walker-extreme", "runner": "r1wb"},
  "wb_anchor_quiet": {"action": "wholebody", "tags": "p3-bracket-s1a-extreme-quietreaction", "runner": "r1wb"},
  "wb_anchor_beyond": {"action": "wholebody", "tags": "p3-bracket-s1a-beyond-anywhere", "runner": "r1wb"},
  "wb_anchor_react": {"action": "wholebody", "tags": "p3-bracket-reaction-gentle-sharpcosts", "runner": "r1wb"},
  "wb_anchor_react_free": {"action": "wholebody", "tags": "p3-bracket-reaction-noanchor", "runner": "r1wb"},
  "wb_anchor_react_calm": {"action": "wholebody", "tags": "p3-bracket-reaction-calmanchor", "runner": "r1wb"},
  "wb_anchor_react_nominal": {"action": "wholebody", "tags": "p3-bisect-reaction-nominalanchor", "runner": "r1wb"},
  "wb_anchor_lateral": {"action": "wholebody", "tags": "p3-bracket-lateral-rewardsonly", "runner": "r1wb"},
  "wb_anchor_combo": {"action": "wholebody", "tags": "p3-combo-extreme-anywhere-lateral", "runner": "r1wb"},
  "wb_anchor_course": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-armsoncourse", "runner": "r1wb"},
  "wb_anchor_lateral_course": {"action": "wholebody", "tags": "p3-extreme-lateral-armsoncourse", "runner": "r1wb"},
  "wb_anchor_lateral_course_hard": {"action": "wholebody", "tags": "p3-extreme-lateral-armsoncourse-hardbursts", "runner": "r1wb"},
  "wb_anchor_lateral_course_g40": {"action": "wholebody", "tags": "p3-extreme-lateral-armsoncourse-gentle40", "runner": "r1wb"},
  "wb_anchor_course_first": {"action": "wholebody", "tags": "p3-armcourse-firsthit-weight5", "runner": "r1wb"},
  "wb_anchor_course_first_repeat": {"action": "wholebody", "tags": "p3-armcourse-firsthit-weight5-repeatanchor", "runner": "r1wb"},
  "wb_anchor_lateral_course_g40w5": {"action": "wholebody", "tags": "p3-extreme-lateral-armsoncourse-gentle40-weight5", "runner": "r1wb"},
  "wb_anchor_combo2": {"action": "wholebody", "tags": "p3-combo2-anywhere-lateral-armsoncourse", "runner": "r1wb"},
  "wb_anchor_react2_loose": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-anchor2", "runner": "r1wb"},
  "wb_anchor_react3": {"action": "wholebody", "tags": "p3-reaction3-armcommandprice-classadv", "runner": "r1wb"},
  "wb_anchor_react2_scaled": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-classadv", "runner": "r1wb"},
  "wb_anchor_react3_quietnoise": {"action": "wholebody", "tags": "p3-reaction3-armcommandprice-classadv-uppernoise0.3", "runner": "r1wb"},
  "wb_anchor_react2_quietnoise": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-uppernoise0.3", "runner": "r1wb"},
  "wb_anchor_react2_plain": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-settledanchor", "runner": "r1wb"},
  "wb_anchor_react2": {"action": "wholebody", "tags": "p3-reaction2-gentleprices-nominalanchor-estimate", "runner": "r1wb"},
  "wb_anchor_react_estimate": {"action": "wholebody", "tags": "p3-bisect-reaction-nominalanchor-estimate", "runner": "r1wb"},
  "wb_anchor_reexplore": {"action": "wholebody", "tags": "p3-bisect-s1aweights-scratchsettings-extreme", "runner": "r1wb"},
  # Adversarial co-evolution program, positive agent (r1adv.positive, .agent/adversarial/rounds/R01/positive): R1 Hold
  # bracket, fixed objective r = L + beta * D; beta 0 / 1 x actor with / without the previous action (810 / 570).
  "adv_hold_b0_prev": {"action": "r1adv_hold", "tags": "r01-hold-b0-prev810"},
  "adv_hold_b0_noprev": {"action": "r1adv_hold", "tags": "r01-hold-b0-noprev570"},
  "adv_hold_b1_prev": {"action": "r1adv_hold", "tags": "r01-hold-b1-prev810"},
  "adv_hold_b1_noprev": {"action": "r1adv_hold", "tags": "r01-hold-b1-noprev570"},
  # Adversarial co-evolution program, branch W (r1adv.wam, .agent/adversarial/rounds/R01/wam/e1_plan.md): W1-J
  # tracker E1 bracket. A = SONIC as published, B = reference-relative prices + residual actions + augmented references.
  "adv_wam_e1a": {"action": "r1adv_wam_track", "tags": "w1j-e1a-sonic"},
  "adv_wam_e1b": {"action": "r1adv_wam_track", "tags": "w1j-e1b-identity"},
  # Adversarial co-evolution program, positive agent (r1adv.positive.hold_r2, .agent/adversarial/rounds/R02/
  # coordinator_lock_r2.md): R2 Hold teachers of wave 1 -- C0 = the R1 recipe on the R2 mission (starts + hits),
  # T = the same with the actor reading the noise-free 570 frames.
  "adv_hold_r2_c0": {"action": "r1adv_hold", "tags": "r2-c0-570"},
  "adv_hold_r2_t": {"action": "r1adv_hold", "tags": "r2-t-570clean"},
  "adv_hold_r2_v": {"action": "r1adv_hold", "tags": "r2-v-330novel"},
  "adv_hold_r2_a": {"action": "r1adv_hold", "tags": "r2-a-ar1tau05"},
  # R2 one-factor runs (coordinator 12:18): T form, seed 1, 500 updates; N / S / P / P-low / P-e0.
  "adv_hold_r2_of_n": {"action": "r1adv_hold", "tags": "r2-of-n"},
  "adv_hold_r2_of_s": {"action": "r1adv_hold", "tags": "r2-of-s"},
  "adv_hold_r2_of_p": {"action": "r1adv_hold", "tags": "r2-of-p"},
  "adv_hold_r2_of_plow": {"action": "r1adv_hold", "tags": "r2-of-plow"},
  "adv_hold_r2_of_pe0": {"action": "r1adv_hold", "tags": "r2-of-pe0"},
  # R2c (coordinator 13:0x): curriculum teacher (level feed of the negative), T form.
  "adv_hold_r2c": {"action": "r1adv_hold", "tags": "r2c-curriculum"},
  # R2d (coordinator 15:3x): curriculum + actuator PD randomisation / without joint velocity; 330 evaluation env.
  "adv_hold_r2d_pd": {"action": "r1adv_hold", "tags": "r2d-pd-curriculum"},
  "adv_hold_r2d_v": {"action": "r1adv_hold", "tags": "r2d-v330-curriculum"},
  "adv_hold_r2_of_nv": {"action": "r1adv_hold", "tags": "r2-of-nv330"},
  # R2e (coordinator 17:17): curriculum arm (base = R2 uniform hits, levels 0.3 fixed, entropy 0, default starts).
  "adv_hold_r2e_cur": {"action": "r1adv_hold", "tags": "r2e-cur-e0"},
  # R3 wave 1 (coordinator 21:41, rounds/R03/coordinator_decisions_r3_wave1.md): foot-lead input b x posture L (point /
  # human distribution), from scratch, P-e0 recipe otherwise; the tag carries 'r3w1' and the cell (C0 / B1 / L1 / BL).
  "adv_hold_w1_c0": {"action": "r1adv_hold", "tags": "r3w1-C0"},
  "adv_hold_w1_b1": {"action": "r1adv_hold", "tags": "r3w1-B1"},
  "adv_hold_w1_l1": {"action": "r1adv_hold", "tags": "r3w1-L1"},
  "adv_hold_w1_bl": {"action": "r1adv_hold", "tags": "r3w1-BL"},
  # R4 wave 2 (coordinator 2026-11-18 01:4x, rounds/R04/04_r4_state.md 3): training-side arms on the W1 recipe (objective
  # unchanged): XA = B1 + stepped starts + std floor, XB = C0 with a plain MLP actor, XC = C0 + azimuth curriculum; XAs /
  # XAf / XAn = one-factor and AR(1) follow-ups of XA (not in the early verdict). The tag carries 'r4w2' and the cell.
  "adv_hold_w2_xa": {"action": "r1adv_hold", "tags": "r4w2-XA"},
  "adv_hold_w2_xb": {"action": "r1adv_hold", "tags": "r4w2-XB"},
  "adv_hold_w2_xc": {"action": "r1adv_hold", "tags": "r4w2-XC"},
  "adv_hold_w2_xas": {"action": "r1adv_hold", "tags": "r4w2-XAs"},
  "adv_hold_w2_xaf": {"action": "r1adv_hold", "tags": "r4w2-XAf"},
  "adv_hold_w2_xan": {"action": "r1adv_hold", "tags": "r4w2-XAn"},
  # R5 wave 3 (coordinator 2026-11-18 07:1x, rounds/R04/06_wave3_plan.md; world spec rounds/R04/r5w3_world_spec.md
  # dea59d5a): the friction-world far end, exploratory. XBF = XB + friction cells, XBN = XB-null (same draws, cell mu =
  # the floor's), C0F = C0 + friction cells. The tag carries 'r5w3' and the cell.
  "adv_hold_w3_xbf": {"action": "r1adv_hold", "tags": "r5w3-XBF"},
  "adv_hold_w3_xbn": {"action": "r1adv_hold", "tags": "r5w3-XBN"},
  "adv_hold_w3_c0f": {"action": "r1adv_hold", "tags": "r5w3-C0F"},
  # R6 wave-4 option S (coordinator 2026-11-18 11:0x, rounds/R04/06B_r6_synthesis.md lever S): human-push disturbance
  # (half-sine 0.1-0.8 s, impulse share 0.3, same hit stream) on XB; a building block, not a launched cell yet.
  "adv_hold_w4_xbs": {"action": "r1adv_hold", "tags": "r6w4-XBS"},
  # R7 wave-4 branch B (positive 10, 2026-11-18; r1adv.positive.hold_w4amp, rounds/R04/07B_r7_synthesis.md 2 / 5): XBS with
  # the D slot replaced by the online phi discriminator D_online (panel design). Launch only after gate T1 passes.
  # Runs carry --run-name r7b-DON_s<seed> ('r7a-' is branch A's token).
  "adv_hold_w4_xbs_amp": {"action": "r1adv_hold", "tags": "r7b-DON"},
  # Adversarial co-evolution program, branch W (r1adv.wam, .agent/adversarial/rounds/R01/wam/e1_plan.md): W1-J
  # tracker E1 relaunch with initial action std 1.0 (first launch stalled at std 0.05; e1_progress.md 12:22).
  "adv_wam_e1a_s1": {"action": "r1adv_wam_track", "tags": "w1j-e1a-sonic-std1"},
  "adv_wam_e1b_s1": {"action": "r1adv_wam_track", "tags": "w1j-e1b-identity-std1"},
  # R6 diagnosis arm r6d (coordinator 2026-11-18 10:5x, rounds/R04/04_r4_state.md 10:58; r1adv.positive.s2f_r6d): s2f/3300
  # continued with its own careful_walk objective, hits only changed: ctl / imp (Hold hits) / long (imp + option S).
  # Runs carry the negative watcher's token in --run-name (r6d-S2F<cell>_s<seed>).
  "careful_walk_r6d_ctl": {"action": "careful_walk", "tags": "r6d-ctl"},
  "careful_walk_r6d_imp": {"action": "careful_walk", "tags": "r6d-imp"},
  "careful_walk_r6d_long": {"action": "careful_walk", "tags": "r6d-long"},
  # R7 branch A control cell A-GMM (coordinator 2026-11-18 20:2x, rounds/R04/07B_r7_synthesis.md 5.1; r1adv.positive.s2f_r7a):
  # r6d long parents with the Hold objective (L + D, arm target r7a), standing commands, flat floor, no curricula.
  # Main runs carry the negative watcher's token in --run-name (r7a-GMM_<parent>); critic warm-up runs do not.
  "careful_walk_r7a_gmm": {"action": "careful_walk", "tags": "r7a-hold"},
  # R9 S0 scene sentinel (positive 14, coordinator 2026-11-19 03:xx, rounds/R04/09B_r9_synthesis.md 2/4): B0 (A-GMM on
  # the O6 body) + scenes (careful_walk_r8_scene_o6, flat 0.30, S-U, judge v3). Runs carry r9-S0_p<k>s<j>.
  "careful_walk_r9_s0": {"action": "careful_walk", "tags": "r9-s0"},
  # R10 / W7 candidates (positive 16, coordinator 2026-11-19 09:xx, rounds/R04/10_r10_facts.md): S0 + one change each.
  # Main runs carry r10-<cell>_p<k>s<j> in --run-name; critic warm-up runs carry r10W-<cell>_p<k>s<j>.
  "careful_walk_r10_s0e0": {"action": "careful_walk", "tags": "r10-s0e0"},
  "careful_walk_r10_s0sfix": {"action": "careful_walk", "tags": "r10-s0sfix"},
  "careful_walk_r10_stands": {"action": "careful_walk", "tags": "r10-stands"},
  "careful_walk_r10_s0cont": {"action": "careful_walk", "tags": "r10-s0cont"},
  "careful_walk_r10_s0contlr": {"action": "careful_walk", "tags": "r10-s0contlr"},
  # R10 b' / f (positive 16, coordinator 09:2x): S0 with the std set to B0's trajectory; SHA-s = that + L + walking AMP.
  "careful_walk_r10_s0ssch": {"action": "careful_walk", "tags": "r10-s0ssch"},
  "careful_walk_r10_shas": {"action": "careful_walk", "tags": "r10-shas"},
  # R10 contrast C2-flat (positive 16, coordinator 09:5x): B0 + D_AMP with the std schedule, no scene.
  "careful_walk_r10_shaflat": {"action": "careful_walk", "tags": "r10-shaflat"},
  # A-GMM-O6 (positive 14, coordinator 2026-11-19 02:0x): A-GMM on the unified O6 body (r1lab.robot.use_o6_hands,
  # signature 15446e6f), nothing else. Main runs carry r9-GMMo6_p<parent>s<seed> in --run-name; warm-ups do not.
  "careful_walk_r7a_gmm_o6": {"action": "careful_walk", "tags": "r7a-hold-o6"},
  # D_shape (positive 15, coordinator 2026-11-19 02:4x): B0 = A-GMM-O6 with D = the online discriminator v1 on phi +
  # arm / trunk shape channels (r1adv.positive_shape.dshape). Main runs carry r9-DSHAPE_p<parent>s<seed>; warm-ups do not.
  "careful_walk_r9_dshape_o6": {"action": "careful_walk", "tags": "r9-dshape-o6"},
  # ARMLOCK (positive 15, coordinator 2026-11-19 02:4x): B0 = A-GMM-O6 with the arms locked (r1lab.armlock), diagnosis
  # only. Main runs carry r9-ARMLOCK_p<parent>s<seed> in --run-name; warm-ups (r9W-ARMLOCK_...) do not.
  "careful_walk_r9_armlock_o6": {"action": "careful_walk", "tags": "r9-armlock-o6"},
  # R8 diagnosis cell A-STAND (positive 14, coordinator 2026-11-19 00:1x, rounds/R04/08B_r8_synthesis.md 6): A-GMM with
  # the parents' own s2f objective (31 careful_walk terms + AMP style, parent weights) instead of the Hold objective.
  # Main runs carry r8-ASTAND_p<parent>s<seed> in --run-name; critic warm-up runs do not.
  "careful_walk_r8_astand": {"action": "careful_walk", "tags": "r8-astand"},
  # R8 scratch probe (coordinator 2026-11-18 22:5x, rounds/R04/08B_r8_synthesis.md 2): A-GMM's task from scratch,
  # checkpoints every 50 updates; runs carry r8-scratch_s<k> in --run-name.
  "careful_walk_r7a_gmm_scratch": {"action": "careful_walk", "tags": "r8-scratch-hold"},
  # A-GMMe0 (coordinator 2026-11-18 21:2x): A-GMM with entropy_coef 0.0 (the Hold runs' value), nothing else.
  "careful_walk_r7a_gmm_e0": {"action": "careful_walk", "tags": "r7a-hold-e0"},
  # R8 scene task (positive 13, 2026-11-18 23:0x; r1adv.positive_scene.scene_r8, rounds/R04/positive/r8_scene/README.md):
  # A-GMM + indoor scene patches (library v2.1 train split, both sources), judge collision ends the episode like a fall,
  # scene observations.
  "careful_walk_r8_scene": {"action": "careful_walk", "tags": "r8-scene"},
  # the same with one library source of v2.1 only: ProcTHOR (K 176) / ScanNet++ vobb (K 1,536)
  "careful_walk_r8_scene_pt": {"action": "careful_walk", "tags": "r8-scene-pt"},
  "careful_walk_r8_scene_sn": {"action": "careful_walk", "tags": "r8-scene-sn"},
  # the three on the O6-hand body (coordinator 2026-11-19: one body = r1wb; r1lab.robot.use_o6_hands, hand_load -0.15..+0.37 kg)
  "careful_walk_r8_scene_o6": {"action": "careful_walk", "tags": "r8-scene-o6"},
  "careful_walk_r8_scene_pt_o6": {"action": "careful_walk", "tags": "r8-scene-pt-o6"},
  "careful_walk_r8_scene_sn_o6": {"action": "careful_walk", "tags": "r8-scene-sn-o6"},
}
for spec in TASK_SPECS.values():
  spec["experiment"] = "r1_" + spec["action"]


# R12 S1 (positive 22, 2026-11-19; package r1s1): outcome-goal System 1 from random weights, one effort price each.
TASK_SPECS.update({
  "wb_s1_s0_joule": {"action": "wholebody", "tags": "s1v1-s0-outcome-goal-joule-scratch", "runner": "r1wb",
                     "experiment": "r1_wholebody"},
  "wb_s1_s0_power": {"action": "wholebody", "tags": "s1v1-s0-outcome-goal-power-scratch", "runner": "r1wb",
                     "experiment": "r1_wholebody"},
})


# R12 S1 S0b (positive 22, 2026-11-20): S0 run by r1s1.learning.S1PPO (records + in-run gates; fix slot open).
TASK_SPECS.update({
  "wb_s1_s0b_power": {"action": "wholebody", "tags": "s1-s0b-scratch", "runner": "r1wb",
                      "experiment": "r1_wholebody"},
})


# R12 S1 S0b sigma arms (positive 22, 2026-11-20 06:1x): mirror-tied / log / tied+log policy std (prepared, not chosen).
TASK_SPECS.update({
  "wb_s1_s0b_power_tied": {"action": "wholebody", "tags": "s1-s0b-tied-scratch", "runner": "r1wb",
                           "experiment": "r1_wholebody"},
  "wb_s1_s0b_power_log": {"action": "wholebody", "tags": "s1-s0b-log-scratch", "runner": "r1wb",
                          "experiment": "r1_wholebody"},
  "wb_s1_s0b_power_tied_log": {"action": "wholebody", "tags": "s1-s0b-tied-log-scratch", "runner": "r1wb",
                               "experiment": "r1_wholebody"},
})


# R12 S1 S0b fixed sigma (positive 22, 2026-11-20 07:5x): S0 with a constant per-joint policy std (not learned).
TASK_SPECS.update({
  "wb_s1_s0b_power_fixed": {"action": "wholebody", "tags": "s1-s0b-fixed-sigma-scratch", "runner": "r1wb",
                            "experiment": "r1_wholebody"},
})


# R13 S1 positive control (positive 23, 2026-11-20): S0b-fixed + careful_walk stage-1 rewards (diagnostic, not adopted).
TASK_SPECS.update({
  "wb_s1_posctl_walk": {"action": "wholebody", "tags": "s1-posctl-velwalk-diagnostic-scratch", "runner": "r1wb",
                        "experiment": "r1_wholebody"},
})


# R13 S1 dense base (positive 23, 2026-11-20 10:1x): objective v2 + dense walk / crouch tracking (diagnostic, not adopted).
TASK_SPECS.update({
  "wb_s1_posctl_dense": {"action": "wholebody", "tags": "s1-posctl-dense-walk-crouch-scratch", "runner": "r1wb",
                         "experiment": "r1_wholebody"},
})


# R13 anchor (positive 23, 2026-11-20 14:4x): careful_walk stage 1 from random weights, recorded tracking terms.
TASK_SPECS.update({
  "careful_walk_s1_repro": {"action": "careful_walk", "tags": "r13-s1-repro-scratch", "experiment": "r1_careful_walk"},
})


# E1 c13 (positive 23, 2026-11-20 16:0x): anchor + S1 environment (r1) / representation-optimisation (r2) leaves.
TASK_SPECS.update({
  "careful_walk_c13_r1": {"action": "careful_walk", "tags": "e1-c13-r1-s1env", "experiment": "r1_careful_walk"},
  "careful_walk_c13_r2": {"action": "careful_walk", "tags": "e1-c13-r2-s1repr", "experiment": "r1_careful_walk"},
})


# B0 (positive 23, 2026-11-20 18:2x): S1 objective v2 + the anchor's representation (E1 R1 setting).
TASK_SPECS.update({
  "careful_walk_b0": {"action": "careful_walk", "tags": "b0-s1objective-anchorrepr", "experiment": "r1_careful_walk"},
})


# B0+AMP (positive 23, 2026-11-20 18:4x): B0 + the anchor's AMP style term.
TASK_SPECS.update({
  "careful_walk_b0amp": {"action": "careful_walk", "tags": "b0amp-s1objective-anchorrepr-amp", "experiment": "r1_careful_walk"},
})


# E3 (positive 23, 2026-11-20 22:0x): L = B0 + sigma 0.34 fixed; a04 = B0 + push boundary d.
TASK_SPECS.update({
  "careful_walk_e3l": {"action": "careful_walk", "tags": "e3-l-b0-sigmafixed", "experiment": "r1_careful_walk"},
  "careful_walk_e3a04": {"action": "careful_walk", "tags": "e3-a04-b0-pushboundary", "experiment": "r1_careful_walk"},
})


# E3X (positive 23, 2026-11-20 23:0x): cap / capa04 / col arms on B0.
TASK_SPECS.update({
  "careful_walk_e3cap": {"action": "careful_walk", "tags": "e3-cap-b0-sigmacap08", "experiment": "r1_careful_walk"},
  "careful_walk_e3capa04": {"action": "careful_walk", "tags": "e3-capa04-b0-sigmacap08-pushboundary", "experiment": "r1_careful_walk"},
  "careful_walk_e3col": {"action": "careful_walk", "tags": "e3-col-b0-colorednoise", "experiment": "r1_careful_walk"},
})


# E4M (positive 25, 2026-11-20 23:0x-23:2x): board-2 method arms on B0 (m1, m2p, m4, mx, jl; _cap, _jl).
TASK_SPECS.update({
  "careful_walk_e4m1": {"action": "careful_walk", "tags": "e4-m1-b0-successrest", "experiment": "r1_careful_walk"},
  "careful_walk_e4m1_jl": {"action": "careful_walk", "tags": "e4-m1-b0-successrest-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4m1_cap": {"action": "careful_walk", "tags": "e4-m1-b0-successrest-cap08", "experiment": "r1_careful_walk"},
  "careful_walk_e4m1_cap_jl": {"action": "careful_walk", "tags": "e4-m1-b0-successrest-cap08-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4m2p": {"action": "careful_walk", "tags": "e4-m2p-b0-gaitprior", "experiment": "r1_careful_walk"},
  "careful_walk_e4m2p_jl": {"action": "careful_walk", "tags": "e4-m2p-b0-gaitprior-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4m2p_cap": {"action": "careful_walk", "tags": "e4-m2p-b0-gaitprior-cap08", "experiment": "r1_careful_walk"},
  "careful_walk_e4m2p_cap_jl": {"action": "careful_walk", "tags": "e4-m2p-b0-gaitprior-cap08-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4m4": {"action": "careful_walk", "tags": "e4-m4-b0-actuatornoise", "experiment": "r1_careful_walk"},
  "careful_walk_e4m4_jl": {"action": "careful_walk", "tags": "e4-m4-b0-actuatornoise-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4m4_cap": {"action": "careful_walk", "tags": "e4-m4-b0-actuatornoise-cap08", "experiment": "r1_careful_walk"},
  "careful_walk_e4m4_cap_jl": {"action": "careful_walk", "tags": "e4-m4-b0-actuatornoise-cap08-jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4mx": {"action": "careful_walk", "tags": "e4-mx-b0-m1m2pm4jl", "experiment": "r1_careful_walk"},
  "careful_walk_e4mx_cap": {"action": "careful_walk", "tags": "e4-mx-b0-m1m2pm4jl-cap08", "experiment": "r1_careful_walk"},
  "careful_walk_e4jl": {"action": "careful_walk", "tags": "e4-jl-b0-jointrange", "experiment": "r1_careful_walk"},
  "careful_walk_e4jl_cap": {"action": "careful_walk", "tags": "e4-jl-b0-jointrange-cap08", "experiment": "r1_careful_walk"},
})


# E4N (positive 25, 2026-11-20 23:5x): board-3 arms on B0 (m1m3, m5, m6, m7).
TASK_SPECS.update({
  "careful_walk_e4m1m3": {"action": "careful_walk", "tags": "e4-m1m3-b0-successrest-nofall", "experiment": "r1_careful_walk"},
  "careful_walk_e4m5": {"action": "careful_walk", "tags": "e4-m5-b0-vib", "experiment": "r1_careful_walk"},
  "careful_walk_e4m6": {"action": "careful_walk", "tags": "e4-m6-b0-smoothrest", "experiment": "r1_careful_walk"},
  "careful_walk_e4m7": {"action": "careful_walk", "tags": "e4-m7-b0-holdmix", "experiment": "r1_careful_walk"},
})
