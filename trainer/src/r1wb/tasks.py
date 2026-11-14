"""Tasks of the whole-body model family (registered in r1lab.registry so the launcher and Trainer see them)."""
from r1wb.student import env_cfg as student
from r1wb.tracking import env_cfg as tracking

from r1wb import anchored, harden, teacher_task, walker  # noqa: E402
from r1wb.student import stream  # noqa: E402

def _student_rl(bc: float):
  """The student's learning settings with a constant imitation weight (0 = the labels are not used)."""
  cfg = student.student_rl_cfg()
  cfg.algorithm.bc_coef, cfg.algorithm.bc_floor = bc, bc
  cfg.algorithm.ramp_learning_rate = bc == 0.0
  return cfg


TASKS = {
  "wb_track_teacher": dict(env=tracking.tracking_env_cfg, rl=tracking.tracking_rl_cfg, stage="P1"),
  # resume a finished P1 teacher under the conditions the student trains in (needs --resume)
  "wb_track_teacher_dr": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                play, first_clip, hardened=True, slip_excess=True, contact_match=True, **kw),
                              rl=tracking.tracking_rl_cfg, stage="P1c"),
  # P1d (cause check, needs --resume <P1c model>): the P1c teacher continued under a stronger disturbance, planar drift
  # limit 1 m (a 25 N s hit moves the robot further than the 0.5 m of the second version). One differs from P1c in the
  # kick strength only (the framework's stock kicks), the other replaces kicks by force impulses at the torso.
  "wb_track_teacher_kick": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                  play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                  disturbance="strong_kick", drift_limit=1.0, **kw),
                                rl=tracking.tracking_rl_cfg, stage="P1d"),
  "wb_track_teacher_impulse": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                     play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                     disturbance="impulse", drift_limit=1.0, **kw),
                                   rl=tracking.tracking_rl_cfg, stage="P1d"),
  # P1e: the impulse-trained teacher with a price on the intervention itself (5 per rad of mean joint distance to
  # the course, every joint alike)
  "wb_track_teacher_course": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                    play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                    disturbance="impulse", drift_limit=1.0, joint_course=5.0, **kw),
                                  rl=tracking.tracking_rl_cfg, stage="P1e"),
  # P1g: the price teacher with bursts in its disturbance distribution (half of the events)
  "wb_track_teacher_bursts": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                    play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                    disturbance="impulse_bursts", drift_limit=1.0, joint_course=5.0, **kw),
                                  rl=tracking.tracking_rl_cfg, stage="P1g"),
  # P1h: after a hit only a physical fall ends the episode for 2 s (tracking terminations suspended): the same with
  # single hits (control) and with bursts in half of the events
  "wb_track_teacher_course_grace": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                          play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                          disturbance="impulse", drift_limit=1.0, joint_course=5.0, hit_grace=2.0, **kw),
                                        rl=tracking.tracking_rl_cfg, stage="P1h"),
  "wb_track_teacher_bursts_grace": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                          play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                          disturbance="impulse_bursts", drift_limit=1.0, joint_course=5.0, hit_grace=2.0, **kw),
                                        rl=tracking.tracking_rl_cfg, stage="P1h"),
  # P1f: the whole recipe in one stage from scratch, larger network, meant for two GPUs: tracking kernels + uniform
  # joint-course price + impulses of 5 to 25 N s lasting 20 ms to 0.4 s (opened over the first 500 iterations)
  "wb_track_teacher_xl": dict(env=lambda play=False, first_clip=0, **kw: tracking.tracking_env_cfg(
                                play, first_clip, hardened=True, slip_excess=True, contact_match=True,
                                disturbance="impulse_mix", drift_limit=1.0, joint_course=5.0, joint_course_cap=0.3, **kw),
                              rl=lambda: tracking.tracking_rl_cfg(hidden=(2048, 2048, 1024, 512)), stage="P1f"),
  # P1t: the teacher on the task itself (r1wb.teacher_task): the student's environment with the teacher's observation
  # plus the goal and the goal lag. Started with --start-from <tracking teacher> (its function at the start; new
  # inputs padded). Disturbance: separate events of single hits at the trunk / the evaluation's full range (single
  # balls and rows of balls anywhere between thigh and head, also off-centre).
  "wb_task_teacher": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                            play=play, disturbance="impulse_events", **kw),
                          rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_full": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                 play=play, disturbance="impulse_events_full", **kw),
                               rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # ... with a cost on failing that outweighs what a row of balls costs while it lands (E76 fell more and more)
  "wb_task_teacher_survive": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                    play=play, disturbance="impulse_events_full", failure_cost=1000.0, **kw),
                                  rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # P1t second version: the failure cost and the stream's course played as it was scored (whole-step pieces): single
  # hits / the full range. Differs from wb_task_teacher_survive in the course only.
  "wb_task_teacher_v2s": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                play=play, disturbance="impulse_events", failure_cost=1000.0, coherent_course=True, **kw),
                              rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v2": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                               play=play, disturbance="impulse_events_full", failure_cost=1000.0, coherent_course=True, **kw),
                             rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # ... the two halves of the full range alone (which one makes the reaction to a single trunk hit large)
  "wb_task_teacher_v2_anywhere": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                        play=play, disturbance="impulse_events_anywhere", failure_cost=1000.0,
                                        coherent_course=True, **kw),
                                      rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v2_rows": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                    play=play, disturbance="impulse_events_rows", failure_cost=1000.0,
                                    coherent_course=True, **kw),
                                  rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # ... the full range with rows as one event in eight / one in four (between "never", quiet, and "half", whole-body)
  "wb_task_teacher_v2_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                      play=play, disturbance="impulse_events_full", failure_cost=1000.0,
                                      coherent_course=True, row_share=0.125, **kw),
                                    rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v2_rows25": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                      play=play, disturbance="impulse_events_full", failure_cost=1000.0,
                                      coherent_course=True, row_share=0.25, **kw),
                                    rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # ... the price on leaving the course's posture three times as high (a fall is then worth about 6 s of whole-body
  # reaction instead of 19 s): rows as one event in eight / no rows
  "wb_task_teacher_v2_price15_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                              play=play, disturbance="impulse_events_full", joint_course=15.0,
                                              failure_cost=1000.0, coherent_course=True, row_share=0.125, **kw),
                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                            pad_inputs=True),
  "wb_task_teacher_v2_price15_anywhere": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                play=play, disturbance="impulse_events_anywhere", joint_course=15.0,
                                                failure_cost=1000.0, coherent_course=True, **kw),
                                              rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                              pad_inputs=True),
  # ... three times as high again (E85 / E86 at 15: rows 1/8 gave eight-hit limits at the manufacturer's level with
  # 5 to 9 arm joints moving on a single hit; no rows gave 0 to 6 arm joints and weaker limits): rows 1/8 and 1/4
  "wb_task_teacher_v2_price45_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                              play=play, disturbance="impulse_events_full", joint_course=45.0,
                                              failure_cost=1000.0, coherent_course=True, row_share=0.125, **kw),
                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                            pad_inputs=True),
  "wb_task_teacher_v2_price45_rows25": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                              play=play, disturbance="impulse_events_full", joint_course=45.0,
                                              failure_cost=1000.0, coherent_course=True, row_share=0.25, **kw),
                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                            pad_inputs=True),
  # ... and the far end of the price (E87 at 45: the eight-hit limits held and the sweep had the fewest falls, a single
  # 12 N s hit from the front or the back still moved 14 / 18 joints against the manufacturer's 12 / 9): 135 and 405.
  # A fall (1000) is then worth about 0.7 s / 0.2 s of whole-body reaction.
  "wb_task_teacher_v2_price135_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                               play=play, disturbance="impulse_events_full", joint_course=135.0,
                                               failure_cost=1000.0, coherent_course=True, row_share=0.125, **kw),
                                             rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                             pad_inputs=True),
  "wb_task_teacher_v2_price405_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                               play=play, disturbance="impulse_events_full", joint_course=405.0,
                                               failure_cost=1000.0, coherent_course=True, row_share=0.125, **kw),
                                             rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                             pad_inputs=True),
  # ... the same prices with the cost of a fall raised in proportion (the ratio of E87, 45 : 1000). E89 (135 : 1000)
  # reacted to a single hit as little as the manufacturer's policy and lost the eight-hit limits while walking and
  # the dense-burst probes; E90 (405 : 1000) gave up on every hard hit. If the size of a reaction to a moderate hit
  # follows the price itself and the willingness to fight a hard one follows price : fall, these keep both.
  "wb_task_teacher_v2_price135_fall3000_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                        play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                        failure_cost=3000.0, coherent_course=True, row_share=0.125, **kw),
                                                      rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                      init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v2_price405_fall9000_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                        play=play, disturbance="impulse_events_full", joint_course=405.0,
                                                        failure_cost=9000.0, coherent_course=True, row_share=0.125, **kw),
                                                      rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                      init_from_teacher=True, pad_inputs=True),
  # v3: the recipe of E87 (45 : 1000, rows 1/8) on the corrected course generator (teacher_task decisive_course)
  "wb_task_teacher_v3_price45_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                              play=play, disturbance="impulse_events_full", joint_course=45.0,
                                              failure_cost=1000.0, coherent_course=True, row_share=0.125,
                                              decisive_course=True, **kw),
                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True,
                                            pad_inputs=True),
  "wb_task_teacher_v3_price135_fall3000_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                        play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                        failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                        decisive_course=True, **kw),
                                                      rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                      init_from_teacher=True, pad_inputs=True),
  # ... E92 (405 : 9000, old generator) moved fewer joints than the manufacturer's policy on a single hit from every
  # side and held the eight-hit limits one size lower while walking: the price at 405 with the fall twice as dear.
  "wb_task_teacher_v3_price405_fall18000_rows12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                         play=play, disturbance="impulse_events_full", joint_course=405.0,
                                                         failure_cost=18000.0, coherent_course=True, row_share=0.125,
                                                         decisive_course=True, **kw),
                                                       rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                       init_from_teacher=True, pad_inputs=True),
  # ... E94 (v3, 135 : 3000) reacted and withstood like the manufacturer's policy and lagged sideways courses (0.67 to
  # 0.75 of the course's own distance, heading 40 to 50 degrees off): the same with the travel priced as well
  # (27 per m/s = one fifth of the posture price per rad).
  "wb_task_teacher_v3_price135_fall3000_travel27": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                          play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                          failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                          decisive_course=True, course_travel=27.0, **kw),
                                                        rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                        init_from_teacher=True, pad_inputs=True),
  # ... E94 again with the posture price saturating at 0.05 / 0.12 rad of mean joint distance (teacher_task course_cap)
  "wb_task_teacher_v3_price135_fall3000_cap05": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                       play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                       failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                       decisive_course=True, course_cap=0.05, **kw),
                                                     rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                     init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v3_price135_fall3000_cap12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                       play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                       failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                       decisive_course=True, course_cap=0.12, **kw),
                                                     rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                     init_from_teacher=True, pad_inputs=True),
  # ... E99 (cap 0.12) with the third line of the price list where it is exactly defined: a step taken while the
  # course stands still (teacher_task step_rest). 12.5 / 50 = 0.25 / 1.0 in the learner's units per landing
  "wb_task_teacher_v3_price135_fall3000_cap12_rest12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                              play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                              failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                              decisive_course=True, course_cap=0.12, step_cost=12.5,
                                                              step_rest=True, **kw),
                                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                            init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v3_price135_fall3000_cap12_rest50": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                              play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                              failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                              decisive_course=True, course_cap=0.12, step_cost=50.0,
                                                              step_rest=True, **kw),
                                                            rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                            init_from_teacher=True, pad_inputs=True),
  # ... E99's task with a policy that is left-right symmetric by construction (r1wb.symmetric_model), plain and with
  # the step price at rest (1.0 per landing)
  "wb_task_teacher_v3_price135_fall3000_cap12_sym": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                           play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                           failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                           decisive_course=True, course_cap=0.12, **kw),
                                                         rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                                         init_from_teacher=True, pad_inputs=True),
  # ... the symmetric E99 task on a field of shapes with every height taken above the ground under the feet
  # (teacher_task ground; gates 11-14 02:3x, hypothesis G1)
  "wb_task_teacher_v4_ground_sym": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                          play=play, disturbance="impulse_events_full", joint_course=135.0,
                                          failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                          decisive_course=True, course_cap=0.12, ground=True, **kw),
                                        rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                        init_from_teacher=True, pad_inputs=True),
  # ... where the policy also reads the last 0.2 s of its own body and the ground around it (r1wb.senses; the other
  # end of the design's first experiment: gates 11-14 05:xx, hypothesis S1), and the history alone
  "wb_task_teacher_v4_ground_sym_senses": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                 play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                 failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                 decisive_course=True, course_cap=0.12, ground=True, senses="both", **kw),
                                               rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                               init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v4_ground_sym_history": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                  play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                  failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                  decisive_course=True, course_cap=0.12, ground=True, senses="history", **kw),
                                                rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                                init_from_teacher=True, pad_inputs=True),
  # ... with the fall priced four times higher: on shaped ground the posture price runs for the whole episode (up to
  # 162 at the cap) against 60 for a fall (gates 11-14 03:2x, hypothesis G4)
  "wb_task_teacher_v4_ground_sym_fall12000": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                    play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                    failure_cost=12000.0, coherent_course=True, row_share=0.125,
                                                    decisive_course=True, course_cap=0.12, ground=True, **kw),
                                                  rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                                  init_from_teacher=True, pad_inputs=True),
  # ... both: the fall priced four times higher and no ending on the course's heights (the 2 x 2 with the three above)
  "wb_task_teacher_v4_ground_sym_free_fall12000": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                         play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                         failure_cost=12000.0, coherent_course=True, row_share=0.125,
                                                         decisive_course=True, course_cap=0.12, ground=True, plan_ends=False, **kw),
                                                       rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                                       init_from_teacher=True, pad_inputs=True),
  # ... where leaving the course's heights or tilt does not end an episode (teacher_task plan_ends=False; gates
  # 11-14 03:1x, hypothesis G3)
  "wb_task_teacher_v4_ground_sym_free": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                               play=play, disturbance="impulse_events_full", joint_course=135.0,
                                               failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                               decisive_course=True, course_cap=0.12, ground=True, plan_ends=False, **kw),
                                             rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                             init_from_teacher=True, pad_inputs=True),
  # ... with each foot's height taken above the ground under that foot (teacher_task ground="feet"; gates 11-14 03:0x)
  "wb_task_teacher_v4_ground_sym_feet": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                               play=play, disturbance="impulse_events_full", joint_course=135.0,
                                               failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                               decisive_course=True, course_cap=0.12, ground="feet", **kw),
                                             rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                             init_from_teacher=True, pad_inputs=True),
  # ... and trained on the library whose rest poses are balanced (r1wb.balance), the plan the stack gives it
  # (gates 11-14 02:5x, hypothesis G2)
  "wb_task_teacher_v4_ground_sym_bal": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                              play=play, disturbance="impulse_events_full", joint_course=135.0,
                                              failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                              decisive_course=True, course_cap=0.12, ground=True,
                                              **{"subset": "corebal", **kw}),
                                            rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True), stage="P1t",
                                            init_from_teacher=True, pad_inputs=True),
  "wb_task_teacher_v3_price135_fall3000_cap12_sym_rest50": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                                  play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                                  failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                                  decisive_course=True, course_cap=0.12, step_cost=50.0,
                                                                  step_rest=True, **kw),
                                                                rl=lambda: teacher_task.teacher_task_rl_cfg(symmetric=True),
                                                                stage="P1t", init_from_teacher=True, pad_inputs=True),
  # ... E99's task with the left-right mirror loss in the learner (teacher_task_rl_cfg mirror; r1wb.symmetry)
  "wb_task_teacher_v3_price135_fall3000_cap12_mirror": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                              play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                              failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                              decisive_course=True, course_cap=0.12, **kw),
                                                            rl=lambda: teacher_task.teacher_task_rl_cfg(mirror=4.0), stage="P1t",
                                                            init_from_teacher=True, pad_inputs=True),
  # ... the step price at rest only where nothing hit the robot in the last 2 s (1.0 per landing)
  "wb_task_teacher_v3_price135_fall3000_cap12_calmrest50": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                                  play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                                  failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                                  decisive_course=True, course_cap=0.12, step_cost=50.0,
                                                                  step_rest=True, step_calm=2.0, **kw),
                                                                rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                                init_from_teacher=True, pad_inputs=True),
  # ... E99 with the condition priced instead of the symptom: while the course stands still (and nothing hit the
  # robot) the capture point lacking 4 cm of room inside the support costs 100 per metre and second
  "wb_task_teacher_v3_price135_fall3000_cap12_room100": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                               play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                               failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                               decisive_course=True, course_cap=0.12, rest_room=100.0, **kw),
                                                             rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                             init_from_teacher=True, pad_inputs=True),
  # ... E99 again with falling four times as dear. Rows of eight at 9 to 12 N s while walking are all withstood and
  # 6% of them end in a fall 0.1 to 2 s after the last hit, on the way back to the walking course (23 of 400, server
  # lists 11-13). Under 135 : 3000 that is the cheaper choice: a second of not yet rejoining the course costs 10 to
  # 16 (posture price near its cap), a 6% risk of falling 3.6.
  "wb_task_teacher_v3_price135_fall12000_cap12": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                        play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                        failure_cost=12000.0, coherent_course=True, row_share=0.125,
                                                        decisive_course=True, course_cap=0.12, **kw),
                                                      rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                      init_from_teacher=True, pad_inputs=True),
  # ... E99 (cap 0.12) continued with the learned System 2 (steer n16) giving the streams their commands
  "wb_task_teacher_v3_price135_fall3000_cap12_steer": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                                             play=play, disturbance="impulse_events_full", joint_course=135.0,
                                                             failure_cost=3000.0, coherent_course=True, row_share=0.125,
                                                             decisive_course=True, course_cap=0.12, steer="n16", **kw),
                                                           rl=teacher_task.teacher_task_rl_cfg, stage="P1t",
                                                           init_from_teacher=True, pad_inputs=True),
  # ... plus the price on steps the course does not take (tracking.mdp.step_unprompted)
  "wb_task_teacher_v2_steps": dict(env=lambda play=False, **kw: teacher_task.teacher_task_env_cfg(
                                     play=play, disturbance="impulse_events_full", failure_cost=1000.0,
                                     coherent_course=True, step_cost=25.0, **kw),
                                   rl=teacher_task.teacher_task_rl_cfg, stage="P1t", init_from_teacher=True, pad_inputs=True),
  # needs --teacher <P1 model_N.pt>: the frozen tracking teacher that labels the student's actions
  "wb_goal_student": dict(env=student.student_env_cfg, rl=student.student_rl_cfg, stage="P2", teacher=True),
  # P2p (needs --teacher <price teacher> and --start-from <student>: new critic and schedule): the reference-free
  # student under the teacher's
  # recipe (impulses of 5 to 25 N s, uniform joint-course price against the clip, tracking and lag judged only 2 s
  # after a hit), with the teacher's labels at a constant weight / without any labels (is the teacher needed?)
  "wb_goal_student_price": dict(env=lambda play=False, teacher=None, **kw: student.student_env_cfg(
                                  play=play, teacher=teacher, disturbance="impulse", joint_course=5.0, hit_grace=2.0, **kw),
                                rl=lambda: _student_rl(bc=0.5), stage="P2p", teacher=True, init_from_teacher=True),
  "wb_goal_student_price_nolabel": dict(env=lambda play=False, teacher=None, **kw: student.student_env_cfg(
                                          play=play, teacher=teacher, disturbance="impulse", joint_course=5.0, hit_grace=2.0, **kw),
                                        rl=lambda: _student_rl(bc=0.0), stage="P2p", teacher=True, init_from_teacher=True),
  # P2q: the same recipe with half of the environments on command streams. After a hit a stream's re-matched
  # reference is not a course: the labels and the price count there only while calm / never (goal rewards alone)
  "wb_goal_student_mix_price_calm": dict(env=lambda play=False, teacher=None, **kw: stream.mixed_env_cfg(
                                           play=play, teacher=teacher, disturbance="impulse", joint_course=5.0,
                                           hit_grace=2.0, stream_course="calm", **kw),
                                         rl=lambda: _student_rl(bc=0.5), stage="P2q", teacher=True, init_from_teacher=True),
  "wb_goal_student_mix_price_rl": dict(env=lambda play=False, teacher=None, **kw: stream.mixed_env_cfg(
                                         play=play, teacher=teacher, disturbance="impulse", joint_course=5.0,
                                         hit_grace=2.0, stream_course="never", **kw),
                                       rl=lambda: _student_rl(bc=0.5), stage="P2q", teacher=True, init_from_teacher=True),
  # P2r: the stream's reference holds its plan for the 2 s after a hit (it is then a course through the hit, like a
  # clip), so the labels and the price count on the streams all the time
  "wb_goal_student_mix_price_held": dict(env=lambda play=False, teacher=None, **kw: stream.mixed_env_cfg(
                                           play=play, teacher=teacher, disturbance="impulse", joint_course=5.0,
                                           hit_grace=2.0, stream_course="held", **kw),
                                         rl=lambda: _student_rl(bc=0.5), stage="P2r", teacher=True, init_from_teacher=True),
  # ... or the streams are not hit at all: the reaction is learned on the clips (where it can be priced) and the
  # streams, calm, teach the command range. Differs from wb_goal_student_mix_price_calm in the stream hits only.
  "wb_goal_student_mix_price_nohit": dict(env=lambda play=False, teacher=None, **kw: stream.mixed_env_cfg(
                                            play=play, teacher=teacher, disturbance="impulse", joint_course=5.0,
                                            hit_grace=2.0, stream_course="calm", stream_hits=False, **kw),
                                          rl=lambda: _student_rl(bc=0.5), stage="P2r", teacher=True, init_from_teacher=True),
  # P2b: the same student with half of the environments on command streams and a state-matched reference
  # (needs --teacher; usually resumed from a P2 checkpoint with --resume)
  "wb_goal_student_mix": dict(env=stream.mixed_env_cfg, rl=stream.mixed_rl_cfg, stage="P2b", teacher=True),
  # P3: a zero-initialised correction on the frozen S1-a (--teacher <S1-a model_N.pt>), command streams + impulses
  "wb_harden": dict(env=harden.harden_env_cfg, rl=harden.harden_rl_cfg, stage="P3", teacher=True, zero_init=True),
  # variant: only 30% of the base policy's arm output is applied, exploration noise is state-dependent
  "wb_harden_restrained": dict(env=lambda play=False, teacher=None, **kw: harden.harden_env_cfg(play, teacher, arm_base_scale=0.3, **kw),
                               rl=lambda: harden.harden_rl_cfg(state_dependent_noise=True), stage="P3", teacher=True,
                               zero_init=True),
  # the restrained variant with 40% of the impulses drawn as strong pushes toward the back (where the falls are)
  "wb_harden_focus": dict(env=lambda play=False, teacher=None, **kw: harden.harden_env_cfg(
                            play, teacher, arm_base_scale=0.3, focus_share=0.0 if play else 0.4, **kw),
                          rl=lambda: harden.harden_rl_cfg(state_dependent_noise=True), stage="P3", teacher=True,
                          zero_init=True),
  # the restrained variant with bursts: 40% of the impulse events are 2-8 ball-sized hits from one side
  "wb_harden_burst": dict(env=lambda play=False, teacher=None, **kw: harden.harden_env_cfg(
                            play, teacher, arm_base_scale=0.3, burst_prob=0.0 if play else 0.4, **kw),
                          rl=lambda: harden.harden_rl_cfg(state_dependent_noise=True), stage="P3", teacher=True,
                          zero_init=True),
  # P3, sixth round: one network, continued from S1-a (--teacher <S1-a model_N.pt> is both the start weights and
  # the frozen anchor), with the imitation of S1-a released while the robot is unsettled after a hit
  "wb_anchor": dict(env=anchored.anchored_env_cfg, rl=anchored.anchored_rl_cfg, stage="P3", teacher=True,
                    init_from_teacher=True),
  # the same with the exploration noise as a second output of the network (learned per joint and per state)
  "wb_anchor_learned": dict(env=anchored.anchored_env_cfg, rl=lambda: anchored.anchored_rl_cfg(learned_noise=True),
                            stage="P3", teacher=True, init_from_teacher=True),
  # far ends of the design axes (first round of the bracket, see gates.md): extreme disturbances from S1-a ...
  "wb_anchor_extreme": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                              play, teacher, intensity="extreme", **kw),
                            rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # ... the same from random weights (the anchor teaches the calm behaviour, the reaction is learned from nothing) ...
  "wb_anchor_scratch": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                              play, teacher, intensity="extreme", **kw),
                            rl=lambda: anchored.anchored_rl_cfg(scratch=True), stage="P3", teacher=True),
  # far end of the reaction-size axis: extreme disturbances from S1-a, with waist and arm motion and the pelvis
  # turning away priced as high as the pay for staying up allows while unsettled
  "wb_anchor_quiet": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                            play, teacher, intensity="extreme", quiet_reaction=True, **kw),
                          rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # far corner of the reaction axis: gentle environments (single moderate hits) next to the extreme ones, and the
  # reaction priced by its size against the state before the hit (see anchored_env_cfg). Started with --start-from
  # a policy that already stays up, so that the first checkpoints show the reaction and not the relearning.
  "wb_anchor_react": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                            play, teacher, intensity="extreme", reaction="sharp", **kw),
                          rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # anchor axis under the reaction prices. Far end: no anchor at all ...
  "wb_anchor_react_free": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                 play, teacher, intensity="extreme", reaction="sharp", **kw),
                               rl=lambda: anchored.anchored_rl_cfg(anchor=0.0), stage="P3", teacher=True,
                               init_from_teacher=True),
  # ... and the candidate between the ends: the anchor holds only the environments that are never hit
  "wb_anchor_react_calm": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                 play, teacher, intensity="extreme", reaction="sharp", anchor_on="calm", **kw),
                               rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # ... and between those two: the anchor holds the never-hit environments, which run without exploration noise
  "wb_anchor_react_nominal": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                    play, teacher, intensity="extreme", reaction="sharp", anchor_on="nominal", **kw),
                                  rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # the nominal anchor plus an estimation head: the policy learns to tell from its own inputs that it was hit (a
  # feature that separates the reaction from the anchored, undisturbed behaviour)
  "wb_anchor_react_estimate": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                     play, teacher, intensity="extreme", reaction="sharp", anchor_on="nominal",
                                     estimate=True, **kw),
                                   rl=lambda: anchored.anchored_rl_cfg(estimate=True), stage="P3", teacher=True,
                                   init_from_teacher=True),
  # the same with the second form of the reaction prices (see anchored_env_cfg, reaction="gentle")
  "wb_anchor_react2": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                             play, teacher, intensity="extreme", reaction="gentle", anchor_on="nominal",
                             estimate=True, **kw),
                           rl=lambda: anchored.anchored_rl_cfg(estimate=True), stage="P3", teacher=True,
                           init_from_teacher=True),
  # ... and the opposite corner of the structure with the same prices: the anchor on every settled sample (as in
  # E31), no estimation head. Separates "the prices were wrong" from "the structure was needed".
  "wb_anchor_react2_plain": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                   play, teacher, intensity="extreme", reaction="gentle", **kw),
                                 rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # third form of the reaction prices: the arm command itself is priced while reacting (gentle environments); the
  # anchor is off for the whole reaction
  "wb_anchor_react3": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                             play, teacher, intensity="extreme", reaction="hold", anchor_on="not_reacting", **kw),
                           rl=lambda: anchored.anchored_rl_cfg(class_advantages=True), stage="P3", teacher=True,
                           init_from_teacher=True),
  # the second form of the prices with the advantages scaled per environment class (nothing else changed against
  # wb_anchor_react2_plain): separates "the gradient was drowned" from "the price needed another form"
  "wb_anchor_react2_scaled": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                    play, teacher, intensity="extreme", reaction="gentle", **kw),
                                  rl=lambda: anchored.anchored_rl_cfg(class_advantages=True), stage="P3", teacher=True,
                                  init_from_teacher=True),
  # noise axis: the same two with the exploration noise of waist and arms at 0.3 of S1-a's
  "wb_anchor_react3_quietnoise": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                        play, teacher, intensity="extreme", reaction="hold", anchor_on="not_reacting", **kw),
                                      rl=lambda: anchored.anchored_rl_cfg(class_advantages=True), stage="P3",
                                      teacher=True, init_from_teacher=True, upper_noise_scale=0.3),
  "wb_anchor_react2_quietnoise": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                        play, teacher, intensity="extreme", reaction="gentle", **kw),
                                      rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True,
                                      upper_noise_scale=0.3),
  # anchor-strength axis between its ends (20: the reaction does not move, 0: the posture collapses into a crouch):
  # the second form of the prices with the anchor at a tenth
  "wb_anchor_react2_loose": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                   play, teacher, intensity="extreme", reaction="gentle", **kw),
                                 rl=lambda: anchored.anchored_rl_cfg(anchor=2.0), stage="P3", teacher=True,
                                 init_from_teacher=True),
  # the anchor itself says "arms as before" during a moderate hit (training target; see anchored.arm_course)
  "wb_anchor_course": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                             play, teacher, intensity="extreme", reaction="gentle", arms_on_course=True, **kw),
                           rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True), stage="P3", teacher=True,
                           init_from_teacher=True),
  # combination of what the brackets settled so far: E31's setting (S1-a start, extreme disturbances, anchor on the
  # settled samples) + hits spread over the body up to head height + the anchor released for sideways commands
  "wb_anchor_combo": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                            play, teacher, intensity="anywhere", release_lateral=True, **kw),
                          rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # everything the brackets settled in one run: extreme disturbances spread over the body, the anchor released for
  # sideways commands, the reaction prices of the gentle environments and the arms kept on their course there
  "wb_anchor_combo2": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                             play, teacher, intensity="anywhere", reaction="gentle", arms_on_course=True,
                             release_lateral=True, **kw),
                           rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True), stage="P3", teacher=True,
                           init_from_teacher=True),
  # one change against wb_anchor_lateral (E31's rewards, anchor released for sideways commands): the arms' anchor
  # during a moderate hit is "as before the hit"; a quarter of the disturbed environments are gentle
  "wb_anchor_lateral_course": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                     play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True, **kw),
                                   rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True), stage="P3", teacher=True,
                                   init_from_teacher=True),
  # strength of the arm course, bracketed from the quarter-share run: 40% gentle environments, and the same with
  # the course weighted five times in the anchor loss
  "wb_anchor_lateral_course_g40": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                         play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True,
                                         gentle_share=0.5, **kw),
                                       rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True), stage="P3",
                                       teacher=True, init_from_teacher=True),
  "wb_anchor_lateral_course_g40w5": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                           play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True,
                                           gentle_share=0.5, **kw),
                                         rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True, course_weight=5.0),
                                         stage="P3", teacher=True, init_from_teacher=True),
  # arm course by hit count (see anchored.arm_course, mode "first_hit"), weight 5: on the first hit of a reaction
  # (at most 15 N s) the arms keep their course; the pair differs in one thing, the arm anchor once hits repeat
  # (none / the arm output of the checkpoint the run continues from)
  "wb_anchor_course_first": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                   play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True,
                                   course_mode="first_hit", **kw),
                                 rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True, course_weight=5.0),
                                 stage="P3", teacher=True, init_from_teacher=True),
  "wb_anchor_course_first_repeat": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                          play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True,
                                          course_mode="first_hit", **kw),
                                        rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True, course_weight=5.0,
                                                                            repeat_weight=5.0),
                                        stage="P3", teacher=True, init_from_teacher=True),
  # the same with 30% of the bursts long and dense (the eight-ball shape)
  "wb_anchor_lateral_course_hard": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                          play, teacher, intensity="extreme", release_lateral=True, arms_on_course=True,
                                          hard_bursts=0.3, **kw),
                                        rl=lambda: anchored.anchored_rl_cfg(arms_on_course=True), stage="P3",
                                        teacher=True, init_from_teacher=True),
  # command-coverage axis, far end "no data, rewards only": E31's setting with the anchor released while a sideways
  # speed is commanded (S1-a cannot walk sideways; the motion library has 1.2 s of it)
  "wb_anchor_lateral": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                              play, teacher, intensity="extreme", release_lateral=True, **kw),
                            rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # far end of coverage: hits anywhere on the body, up to head height, stronger and denser than "extreme"
  "wb_anchor_beyond": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                             play, teacher, intensity="beyond", **kw),
                           rl=anchored.anchored_rl_cfg, stage="P3", teacher=True, init_from_teacher=True),
  # midpoint of the start axis: S1-a's weights with the learning settings of the run from scratch (exploration
  # noise 1.0 and learnable, no warm-up, the higher learning rate). Separates "the weights" from "how it explores"
  "wb_anchor_reexplore": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                                play, teacher, intensity="extreme", **kw),
                              rl=lambda: anchored.anchored_rl_cfg(scratch=True), stage="P3", teacher=True,
                              init_from_teacher=True, start_noise=1.0),
  # ... and from S1-a with no anchor at all
  "wb_anchor_free": dict(env=lambda play=False, teacher=None, **kw: anchored.anchored_env_cfg(
                           play, teacher, intensity="extreme", **kw),
                         rl=lambda: anchored.anchored_rl_cfg(anchor=0.0), stage="P3", teacher=True, init_from_teacher=True),
  # ceiling test: a walking policy from scratch with mjlab's stock velocity recipe (no human motion, no anchor)
  # under the extreme disturbance set
  "wb_walker": dict(env=walker.walker_env_cfg, rl=walker.walker_rl_cfg, stage="P3"),
  # control for the urgency gate: the same run with the relief switched off
  "wb_harden_norelief": dict(env=lambda play=False, teacher=None, **kw: harden.harden_env_cfg(play, teacher, relief=False, **kw),
                             rl=harden.harden_rl_cfg, stage="P3", teacher=True, zero_init=True),
}
