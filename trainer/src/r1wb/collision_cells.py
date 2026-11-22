# Copyright 2025 The Newton Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split known MuJoCo heightfield CCD kernels into two ordered row bands.

The arithmetic and contact selection come from the installed native builder
(Newton Developers, Apache-2.0). An exact source digest guards the transformation;
other versions use the native path. Overflow and scratch exhaustion also route
back to the original kernel without changing its capacities or terrain.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import linecache

import torch
import warp as wp
from warp._src.context import capture_pause, capture_resume
from mujoco_warp._src import collision_convex as cc
from mujoco_warp._src.types import GeomType, MJ_MAXCONPAIR

SPLIT = 2
_NATIVE_SHA256 = "931629bc41be10a599704cbbcb520e8611cd13be60a9bd31479729e31a609fb9"
_EPA = ("epa_vert_in", "epa_vert_index_in", "epa_face_in", "epa_pr_in", "epa_norm2_in", "epa_horizon_in")

@lru_cache(maxsize=1)
def _builders():
    source = inspect.getsource(cc.ccd_hfield_kernel_builder)
    if hashlib.sha256(source.encode()).hexdigest() != _NATIVE_SHA256:
        return None
    source = source[source.index('def ccd_hfield_kernel_builder'):]
    start = source.index('    # contact 0: minimum distance')
    finish = source.rindex('  return ccd_hfield_kernel')
    tail = source[start:finish]
    header = source[:source.index('    collisionid = wp.tid()')]
    extra = '''    tile_counts: wp.array2d[int],
    tile_dist: wp.array2d[float],
    tile_pos: wp.array2d[wp.vec3],
    tile_normal: wp.array2d[wp.vec3],
'''
    header = header.replace('    overflow_out: wp.array[int],\n', '    overflow_out: wp.array[int],\n'+extra)
    body = source[source.index('    collisionid = wp.tid()'):start]
    worker_body = body.replace('    collisionid = wp.tid()', '    collisionid, lane = wp.tid()\n    if collisionid >= ncollision_in[0]:\n      return\n    tile_counts[collisionid, lane] = 0')
    worker_body = worker_body.replace('ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)', 'ccdid = collisionid * wp.static(SPLIT) + lane')
    worker_body = worker_body.replace('for r in range(rmin, rmax):', 'for r in range(rmin + (rmax-rmin)*lane//wp.static(SPLIT), rmin + (rmax-rmin)*(lane+1)//wp.static(SPLIT)):')
    worker_body = worker_body.replace("    hfield_contact_dist = vec_maxconpair()\n", "")
    worker_body = worker_body.replace("    hfield_contact_pos = mat_maxconpair()\n", "")
    worker_body = worker_body.replace("    hfield_contact_normal = mat_maxconpair()\n", "")
    worker_body = worker_body.replace("hfield_contact_dist[count] = dist", "tile_dist[collisionid, lane * wp.static(MJ_MAXCONPAIR) + count] = dist")
    worker_body = worker_body.replace("          hfield_contact_pos[count, 0] = pos[0]\n          hfield_contact_pos[count, 1] = pos[1]\n          hfield_contact_pos[count, 2] = pos[2]", "          tile_pos[collisionid, lane * wp.static(MJ_MAXCONPAIR) + count] = pos")
    worker_body = worker_body.replace("          hfield_contact_normal[count, 0] = normal[0]\n          hfield_contact_normal[count, 1] = normal[1]\n          hfield_contact_normal[count, 2] = normal[2]", "          tile_normal[collisionid, lane * wp.static(MJ_MAXCONPAIR) + count] = normal")
    worker_end = "    tile_counts[collisionid, lane] = count\n\n  return ccd_hfield_kernel\n"
    worker = (header + worker_body + worker_end).replace('ccd_hfield_kernel_builder', 'cell_worker_builder').replace('def ccd_hfield_kernel(', 'def cell_worker(').replace('return ccd_hfield_kernel','return cell_worker')
    reducer_body = body[:body.index('    geom1, geom2 = geom_collision_pair(')]
    reducer_body += '''    hfield_contact_dist = vec_maxconpair()
    hfield_contact_pos = mat_maxconpair()
    hfield_contact_normal = mat_maxconpair()
    min_dist = float(MJ_MAXVAL)
    min_normal = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
    min_pos = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
    min_id = int(-1)
    collision_pairid = collision_pairid_in[collisionid]
    count = int(0)
    for lane in range(wp.static(SPLIT)):
      for k in range(tile_counts[collisionid, lane]):
        if count >= MJ_MAXCONPAIR:
          wp.atomic_or(overflow_out, worldid, OverflowType.HFIELD)
          continue
        slot = lane * wp.static(MJ_MAXCONPAIR) + k
        dist = tile_dist[collisionid, slot]
        pos = tile_pos[collisionid, slot]
        normal = tile_normal[collisionid, slot]
        hfield_contact_dist[count] = dist
        hfield_contact_pos[count, 0] = pos[0]
        hfield_contact_pos[count, 1] = pos[1]
        hfield_contact_pos[count, 2] = pos[2]
        hfield_contact_normal[count, 0] = normal[0]
        hfield_contact_normal[count, 1] = normal[1]
        hfield_contact_normal[count, 2] = normal[2]
        if dist < min_dist:
          min_dist = dist
          min_normal = normal
          min_pos = pos
          min_id = count
        count += 1
'''
    reducer = (header+reducer_body+tail+'  return ccd_hfield_kernel\n').replace('ccd_hfield_kernel_builder','cell_reduce_builder').replace('def ccd_hfield_kernel(', 'def cell_reduce(').replace('return ccd_hfield_kernel','return cell_reduce')
    scope=dict(vars(cc));scope['SPLIT']=SPLIT
    for name, code in [('cell_workers',worker),('cell_reducer',reducer)]:
        filename='<r1wb_'+name+'.py>'
        linecache.cache[filename]=(len(code),None,code.splitlines(keepends=True),filename)
        exec(compile(code,filename,'exec'),scope)
    return scope['cell_worker_builder'],scope['cell_reduce_builder']



@wp.kernel(module="unique", enable_backward=False)
def _choose_counts(count: wp.array[int], capacity: int, split: wp.array[int], native: wp.array[int]):
    split[0] = wp.where(count[0] <= capacity, count[0], 0)
    native[0] = wp.where(count[0] <= capacity, 0, count[0])


@wp.kernel(module="unique", enable_backward=False)
def _check_overflow(count: wp.array[int], tiles: wp.array2d[int], overflow: wp.array[int], bad: wp.array[int]):
    i = wp.tid()
    if i < overflow.shape[0] and overflow[i] != 0:
        wp.atomic_max(bad, 0, 1)
    if i < count[0] and tiles[i, 0] + tiles[i, 1] >= wp.static(MJ_MAXCONPAIR):
        wp.atomic_max(bad, 0, 1)


@wp.kernel(module="unique", enable_backward=False)
def _fallback(count: wp.array[int], bad: wp.array[int], split: wp.array[int], native: wp.array[int]):
    if bad[0] != 0:
        split[0] = 0
        native[0] = count[0]


class HeightfieldCells:
    """Per-simulator graph workspace; all kernels run on the same capture stream."""

    def __init__(self, sim, max_pairs=131072):
        self.builders = _builders()
        self.kernels = {}
        self.workspaces = {}
        self.max_pairs = max_pairs
        self.budget = 0
        self.conditional = False
        if self.builders is not None:
            free, _ = torch.cuda.mem_get_info(str(sim.wp_data.geom_xpos.device))
            self.budget = free // 8
            try:
                from warp._src.context import assert_conditional_graph_support
                assert_conditional_graph_support()
                self.conditional = True
            except (ImportError, RuntimeError):
                pass

    def _allocate(self, layout, full, ix, pairs):
        capacity = layout[0]
        if layout not in self.workspaces:
            epa = {name: wp.empty((capacity * SPLIT, full[ix[name]].shape[1]),
                                  dtype=full[ix[name]].dtype, device=pairs.device) for name in _EPA}
            extra = [wp.empty((capacity, SPLIT), dtype=int, device=pairs.device),
                     wp.empty((capacity, SPLIT * int(MJ_MAXCONPAIR)), dtype=float, device=pairs.device),
                     wp.empty((capacity, SPLIT * int(MJ_MAXCONPAIR)), dtype=wp.vec3, device=pairs.device),
                     wp.empty((capacity, SPLIT * int(MJ_MAXCONPAIR)), dtype=wp.vec3, device=pairs.device)]
            count = full[ix["ncollision_in"]]
            split, fallback, bad = (wp.zeros_like(count) for _ in range(3))
            overflow = wp.zeros_like(full[ix["overflow_out"]])
            self.workspaces[layout] = epa, extra, split, fallback, bad, overflow

    def launch(self, kernel, ix, closure, inputs, kwargs, native_launch):
        """Return False when unsupported; otherwise enqueue the guarded split."""
        # Capsules dominate this robot's prism work. Splitting the mostly empty
        # sphere/box batches costs more than their narrowphase work saves.
        if self.builders is None or closure["geomtype2"] != int(GeomType.CAPSULE):
            return False
        full = list(inputs) + list(kwargs.get("outputs", ()))
        if len(full) != len(ix) or any(name not in ix for name in _EPA):
            return False
        pairs = full[ix["collision_pair_in"]]
        per_pair = SPLIT * (int(MJ_MAXCONPAIR) * 28 + 4 + sum(full[ix[name]].strides[0] for name in _EPA))
        capacity = min(self.max_pairs, pairs.shape[0], full[ix["naccdmax_in"]], self.budget // per_pair)
        if capacity < 1:
            return False
        layout = (capacity, tuple((full[ix[name]].shape[1], full[ix[name]].dtype) for name in _EPA),
                  full[ix["overflow_out"]].shape[0])
        if layout not in self.workspaces:
            # These arrays must also be valid when forward runs before step.
            stream = kwargs.get("stream") or pairs.device.stream
            graph = capture_pause(stream=stream) if stream in pairs.device.captures else None
            try:
                self._allocate(layout, full, ix, pairs)
            finally:
                if graph is not None:
                    capture_resume(graph, stream=stream)
        epa, extra, split, fallback, bad, overflow = self.workspaces[layout]
        if kernel not in self.kernels:
            parameters = [closure[name] for name in ("geomtype1", "geomtype2", "gjk_iterations", "epa_iterations", "geomgeomid")]
            # Workers write only private overflow flags. Native fallback owns warnings.
            self.kernels[kernel] = (self.builders[0](*parameters, False),
                                    self.builders[1](*parameters, closure["warn_overflow"]))
        worker, reducer = self.kernels[kernel]
        options = {key: value for key, value in kwargs.items() if key not in ("dim", "inputs", "outputs")}
        count = full[ix["ncollision_in"]]
        overflow.zero_()
        bad.zero_()
        native_launch(_choose_counts, dim=1, inputs=[count, capacity, split, fallback], **options)
        worker_inputs = full[:]
        for name, array in epa.items():
            worker_inputs[ix[name]] = array
        worker_inputs[ix["naccdmax_in"]] = capacity * SPLIT
        worker_inputs[ix["ncollision_in"]] = split
        worker_inputs[ix["overflow_out"]] = overflow
        native_launch(worker, dim=(capacity, SPLIT), inputs=worker_inputs + extra, **options)
        native_launch(_check_overflow, dim=max(capacity, overflow.shape[0]), inputs=[split, extra[0], overflow, bad], **options)
        native_launch(_fallback, dim=1, inputs=[count, bad, split, fallback], **options)
        reducer_inputs = full[:]
        reducer_inputs[ix["ncollision_in"]] = split
        native_launch(reducer, dim=capacity, inputs=reducer_inputs + extra, **options)
        fallback_inputs = full[:]
        fallback_inputs[ix["ncollision_in"]] = fallback
        def run_native():
            native_launch(kernel, dim=kwargs["dim"], inputs=fallback_inputs, **options)

        if self.conditional:
            wp.capture_if(fallback, on_true=run_native, stream=options.get("stream"))
        else:
            run_native()
        return True
