"""Reaction measured against the un-hit ORBIT instead of the time-indexed twin (walking with a phase-following
reference shifts its gait phase after a hit; a phase shift is not a reaction).
For every time after the hit the twin's state is taken at the time shift (within +-0.6 s) where the twin's LEG joint
angles are closest to the robot's; joints are then compared at that shift. Standing is unaffected (the twin is still).
usage: orbit_distance.py name=<trace npz made with --sync-twins> ..."""
import sys
import numpy as np
print(f"{'policy':12s} {'cell':18s} | time-indexed twin: joints>0.1 (arms) | orbit-aligned: joints>0.1 (legs, waist, arms)  mean |arm| rad  arm speed off orbit rad/s | peak tilt | back within 0.1 rad of the orbit after (s)")
for item in sys.argv[1:]:
    name, _, path = item.partition("=")
    d = np.load(path, allow_pickle=True); dt = float(d["dt"]); T = d["joint_pos"].shape[0]
    cases = [str(c) for c in d["cases"]]
    span = int(0.6 / dt)
    cells = {}
    for i, c in enumerate(cases):
        parts = c.split("|"); cells.setdefault((parts[0], (parts[-3] if len(parts) > 3 else parts[1]) + " " + parts[-2]), []).append(i)
    for (ctx, direction), idx in cells.items():
        rows = []
        for i in idx:
            h = int(d["hit_step"][i]); t = int(d["twin_of"][i]); e = min(h + int(2.0 / dt), T - span - 1)
            if int(d["fell_step"][i]) < T or e <= h: continue
            q, qt = d["joint_pos"], d["joint_pos"]
            plain = np.abs(q[h:e, i] - q[h:e, t]).max(0)
            aligned = np.zeros((e - h, 24)); speed = np.zeros(e - h)
            for k, step in enumerate(range(h, e)):
                lo, hi = max(1, step - span), min(T - 2, step + span)
                window = q[lo:hi, t, :12]
                best = lo + int(np.argmin(np.square(window - q[step, i, :12]).sum(1)))
                aligned[k] = np.abs(q[step, i] - q[best, t])
                speed[k] = np.sqrt(np.mean(np.square(d["joint_vel"][step, i, 14:] - d["joint_vel"][best, t, 14:])))
            peak = aligned.max(0)
            inside = (aligned.max(1) < 0.1)
            back = np.nan
            for k in range(len(inside)):
                if inside[k:k + int(0.5 / dt)].all() and k + int(0.5 / dt) <= len(inside): back = k * dt; break
            rows.append([(plain > 0.1).sum(), (plain[14:] > 0.1).sum(), (peak > 0.1).sum(), (peak[:12] > 0.1).sum(), (peak[12:14] > 0.1).sum(), (peak[14:] > 0.1).sum(),
                         aligned[:, 14:].mean(), np.sqrt(np.mean(speed ** 2)), d["tilt"][h:e, i].max(), back])
        if not rows: print(f"{name:12s} {ctx + ' ' + direction:18s} | all fell"); continue
        m = np.nanmedian(np.array(rows, dtype=float), 0)
        print(f"{name:12s} {ctx + ' ' + direction:18s} |        {m[0]:4.1f} ({m[1]:.0f})                 |       {m[2]:4.1f} ({m[3]:.0f}, {m[4]:.0f}, {m[5]:.0f})                  {m[6]:.3f}          {m[7]:.3f}          |  {m[8]:.2f}  | {m[9]:.2f}")
