"""Batch heightfield candidates without changing contact geometry.

MuJoCo Warp launches a separate heightfield kernel for each shape type over a
mixed candidate list. Pack matching candidates, then group the same geometry
across worlds to keep nearby threads on similar work. Capacities, contact
parameters, narrowphase equations and solver settings remain native.
"""
from __future__ import annotations

import inspect
from threading import RLock

import warp as wp
from warp._src.context import capture_pause, capture_resume


@wp.kernel(module="unique", enable_backward=False)
def _pack_pairs(types: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i), pairids: wp.array(dtype=wp.vec2i),
                worlds: wp.array(dtype=int), count: wp.array(dtype=int), first: int, second: int,
                out_pairs: wp.array(dtype=wp.vec2i), out_ids: wp.array(dtype=wp.vec2i),
                out_worlds: wp.array(dtype=int), out_count: wp.array(dtype=int)):
  i = wp.tid()
  if i >= count[0]:
    return
  pair = pairs[i]
  if types[pair[0]] == first and types[pair[1]] == second:
    j = wp.atomic_add(out_count, 0, 1)
    out_pairs[j] = pair
    out_ids[j] = pairids[i]
    out_worlds[j] = worlds[i]


@wp.kernel(module="unique", enable_backward=False)
def _count_geoms(types: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i), count: wp.array(dtype=int),
                 first: int, second: int, totals: wp.array(dtype=int), out_count: wp.array(dtype=int)):
  i = wp.tid()
  if i >= count[0]:
    return
  pair = pairs[i]
  if types[pair[0]] == first and types[pair[1]] == second:
    wp.atomic_add(totals, pair[1], 1)
    wp.atomic_add(out_count, 0, 1)


@wp.kernel(module="unique", enable_backward=False)
def _prefix_geoms(totals: wp.array(dtype=int), offsets: wp.array(dtype=int)):
  i = wp.tid()
  total = int(0)
  for j in range(i):
    total += totals[j]
  offsets[i] = total


@wp.kernel(module="unique", enable_backward=False)
def _scatter_geoms(types: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i), pairids: wp.array(dtype=wp.vec2i),
                   worlds: wp.array(dtype=int), count: wp.array(dtype=int), first: int, second: int,
                   offsets: wp.array(dtype=int), out_pairs: wp.array(dtype=wp.vec2i),
                   out_ids: wp.array(dtype=wp.vec2i), out_worlds: wp.array(dtype=int)):
  i = wp.tid()
  if i >= count[0]:
    return
  pair = pairs[i]
  if types[pair[0]] == first and types[pair[1]] == second:
    j = wp.atomic_add(offsets, pair[1], 1)
    out_pairs[j] = pair
    out_ids[j] = pairids[i]
    out_worlds[j] = worlds[i]


_capture_lock = RLock()
_fields = ("geom_type", "geom_xpos_in", "collision_pair_in", "collision_pairid_in",
           "collision_worldid_in", "ncollision_in")


def install_collision_batching(sim, *, group_by_geom=True, parallel_cells=False):
  """Adapt this simulator's graph capture, including later graph recreations.

  The temporary launch adapter accepts only this sim's geom_xpos pointer. It is
  removed in finally; unrelated Warp calls and uncaptured stepping stay native.
  Signature checks leave other MuJoCo Warp versions on their original path.
  Above 1024 geoms, use type compaction to avoid the quadratic prefix cost.
  """
  if not sim.use_cuda_graph or hasattr(sim, "_r1wb_collision_workspaces"):
    return
  native_capture = sim.create_graph
  cells = None
  if parallel_cells:
    from r1wb.collision_cells import HeightfieldCells
    cells = HeightfieldCells(sim)

  def create_graph():
    workspaces = {}
    signatures = {}
    with _capture_lock:
      native_launch = wp.launch

      def launch(kernel, *args, **kwargs):
        if "ccd_hfield_kernel" not in kernel.key or "inputs" not in kwargs:
          return native_launch(kernel, *args, **kwargs)
        if kernel not in signatures:
          names = list(inspect.signature(kernel.func).parameters)
          closure = inspect.getclosurevars(kernel.func).nonlocals
          signatures[kernel] = ({name: i for i, name in enumerate(names)}, closure)
        ix, closure = signatures[kernel]
        inputs = list(kwargs["inputs"])
        if (not all(name in ix and ix[name] < len(inputs) for name in _fields)
            or not all(name in closure for name in ("geomtype1", "geomtype2"))
            or inputs[ix["geom_xpos_in"]].ptr != sim.wp_data.geom_xpos.ptr
            or inputs[ix["geom_xpos_in"]].device != sim.wp_data.geom_xpos.device):
          return native_launch(kernel, *args, **kwargs)
        pairs, ids, worlds, count = (inputs[ix[name]] for name in _fields[2:])
        types = inputs[ix["geom_type"]]
        grouped = group_by_geom and types.shape[0] <= 1024
        key = (kernel, pairs.ptr)
        if key not in workspaces:
          stream = kwargs.get("stream") or pairs.device.stream
          graph = capture_pause(stream=stream) if stream in pairs.device.captures else None
          try:
            workspaces[key] = (wp.empty_like(pairs), wp.empty_like(ids), wp.empty_like(worlds), wp.zeros_like(count))
            if grouped:
              totals = wp.zeros_like(types)
              workspaces[key] += (totals, wp.empty_like(totals))
          finally:
            if graph is not None:
              capture_resume(graph, stream=stream)
        out_pairs, out_ids, out_worlds, out_count = workspaces[key][:4]
        out_count.zero_()
        first, second = closure["geomtype1"], closure["geomtype2"]
        if grouped:
          totals, offsets = workspaces[key][4:]
          totals.zero_()
          native_launch(_count_geoms, dim=pairs.shape[0], inputs=[types, pairs, count, first, second],
                        outputs=[totals, out_count], device=pairs.device)
          native_launch(_prefix_geoms, dim=totals.shape[0], inputs=[totals],
                        outputs=[offsets], device=pairs.device)
          native_launch(_scatter_geoms, dim=pairs.shape[0],
                        inputs=[types, pairs, ids, worlds, count, first, second, offsets],
                        outputs=[out_pairs, out_ids, out_worlds], device=pairs.device)
        else:
          native_launch(_pack_pairs, dim=pairs.shape[0], inputs=[types, pairs, ids, worlds, count, first, second],
                        outputs=[out_pairs, out_ids, out_worlds, out_count], device=pairs.device)
        for name, value in zip(_fields[2:], (out_pairs, out_ids, out_worlds, out_count)):
          inputs[ix[name]] = value
        kwargs["inputs"] = inputs
        if cells is not None and not args and cells.launch(kernel, ix, closure, inputs, kwargs, native_launch):
          return
        return native_launch(kernel, *args, **kwargs)

      wp.launch = launch
      try:
        native_capture()
      finally:
        wp.launch = native_launch
    # Keep each captured array alive for as long as these graphs can use it.
    sim._r1wb_collision_workspaces = workspaces
    if cells is not None:
      sim._r1wb_cell_batch = cells

  sim.create_graph = create_graph
  create_graph()
