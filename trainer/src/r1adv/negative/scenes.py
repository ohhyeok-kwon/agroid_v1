"""Indoor scans -> 2.5D dual height fields + tags, cut into 6 x 6 m patches (an atlas). Runs on the PC (NAS, CPU).

Per 5 cm cell, from the mesh's faces subdivided to <= 2.5 cm (deterministic; no random sampling):
  F  collision floor: highest surface below the cut height (1.5 m above the main floor; R1 is about 1.2 m tall), so
     walls, furniture and anything the robot cannot pass under become solid columns; an empty cell (outside the
     scan) is a wall. Heights are relative to the main floor (the largest upward-facing level).
  C  overhead clearance: lowest surface above the walkable floor of that cell (3 m where open); for perception and
     planning, and the second (flipped) height field when a world needs a ceiling.
Scan noise is not difficulty: floor cells (within 25 cm of the main floor) are median-filtered over 25 cm and heights
under 2.5 cm snap to 0 (relief is added only by explicit parameters). Tags per patch from the free mask (floor within
6 cm): open; doorway / corridor (a free region that a 0.3 m-radius body can cross but a 0.45 m one cannot: the narrow
part's area < 0.6 m^2 is a doorway, >= 1.0 m^2 a corridor); threshold (>= 15 floor step cells of 2.5-6 cm); clutter
(>= 3 obstacles 6-90 cm high of 0.02-0.6 m^2); spawn cell (free, clearance >= 0.6 m, nearest the patch centre).
Splits (coordinator, 11-17): HM3D val/minival are sealed for evaluation and are refused here.
usage: python -I -m r1adv.negative.scenes --glb a.glb [...] --out atlas.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

CELL = 0.05
CUT = 1.5
CEILING = 3.0
PATCH = 6.0
STRIDE = 3.0
FREE = 0.06


def sha256(path) -> str:
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
  import trimesh
  if any(part in ("val", "minival") or "-val-" in part or "minival" in part for part in path.parts):
    raise ValueError("HM3D val/minival are sealed for evaluation (coordinator 11-17)")
  mesh = trimesh.load(path, force="mesh", process=False)
  vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
  normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
  area = 0.5 * np.linalg.norm(normals, axis=1)
  unit = normals / np.maximum(2 * area, 1e-12)[:, None]
  # up axis: the axis most of the horizontal area faces along (floors and ceilings); floors face +up
  share = [float(area[np.abs(unit[:, k]) > 0.95].sum()) for k in range(3)]
  up = int(np.argmax(share))
  if up != 2:
    order = [k for k in range(3) if k != up] + [up]
    vertices = vertices[:, order]
    if np.linalg.det(np.eye(3)[order]) < 0:
      vertices[:, 0] *= -1.0
  return vertices, faces, dict(up_axis="xyz"[up], horizontal_share=share)


def samples(vertices: np.ndarray, faces: np.ndarray, edge: float = 0.025):
  """Centroids (x, y, z) and up-components of the normals of the faces subdivided to edges <= `edge`."""
  import trimesh
  v, f = trimesh.remesh.subdivide_to_size(vertices, faces, max_edge=edge, max_iter=12)
  tri = v[f]
  normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  up = normal[:, 2] / np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
  return tri.mean(axis=1), up


def main_floor(points: np.ndarray, up: np.ndarray) -> float:
  """Height of the largest upward-facing level (5 cm histogram)."""
  z = points[up > 0.9, 2]
  bins = np.arange(z.min(), z.max() + 0.05, 0.05)
  counts, edges = np.histogram(z, bins=bins)
  k = int(np.argmax(counts))
  near = z[(z >= edges[k] - 0.05) & (z <= edges[k + 1] + 0.05)]
  return float(np.median(near))


def fields(points: np.ndarray, up: np.ndarray, floor: float):
  rel = points.copy()
  rel[:, 2] -= floor
  low, high = rel[:, :2].min(0) - 0.5, rel[:, :2].max(0) + 0.5
  shape = tuple(np.ceil((high - low) / CELL).astype(int))
  ij = np.floor((rel[:, :2] - low) / CELL).astype(int)
  flat = ij[:, 0] * shape[1] + ij[:, 1]
  n = shape[0] * shape[1]
  column = (rel[:, 2] >= -0.25) & (rel[:, 2] <= CUT)
  F = np.full(n, -np.inf)
  np.maximum.at(F, flat[column], rel[column, 2])
  walk = column & (up > 0.85) & (rel[:, 2] <= 0.25)
  W = np.full(n, -np.inf)
  np.maximum.at(W, flat[walk], rel[walk, 2])
  F = np.where(np.isfinite(F), F, CUT)                       # outside the scan: a wall
  W = np.where(np.isfinite(W), W, np.nan)
  F, W = F.reshape(shape), W.reshape(shape)
  from scipy import ndimage
  floor_band = F <= 0.25
  smooth = ndimage.median_filter(np.where(floor_band, F, 0.0), size=5)
  F = np.where(floor_band, np.where(np.abs(smooth) < 0.025, 0.0, smooth), F)
  W = np.where(np.isfinite(W), np.where(np.abs(W) < 0.025, 0.0, W), W)
  F, W = F.reshape(-1), W.reshape(-1)
  above = rel[:, 2] > np.nan_to_num(W[flat], nan=10.0) + 0.05
  C = np.full(n, np.inf)
  np.minimum.at(C, flat[above], rel[above, 2])
  C = np.where(np.isfinite(C), np.minimum(C, CEILING), CEILING)
  return F.reshape(shape).astype(np.float32), C.reshape(shape).astype(np.float32), W.reshape(shape).astype(np.float32), low


def tags(F: np.ndarray, W: np.ndarray) -> dict:
  from scipy import ndimage
  free = F <= FREE
  clearance = ndimage.distance_transform_edt(free) * CELL
  def regions(radius):
    core = clearance >= radius
    labels, count = ndimage.label(core)
    sizes = ndimage.sum(np.ones_like(clearance), labels, index=np.arange(1, count + 1)) * CELL * CELL if count else np.zeros(0)
    return core, int((sizes >= 1.0).sum())
  wide_core, wide = regions(0.45)
  body_core, body = regions(0.30)
  narrow = body_core & ~ndimage.binary_dilation(wide_core, iterations=int(0.15 / CELL))
  narrow_area = float(narrow.sum()) * CELL * CELL
  split = wide > body
  step = np.zeros_like(F, dtype=bool)
  floor = np.where(free, F, np.nan)
  for axis in (0, 1):
    d = np.abs(np.diff(floor, axis=axis))
    edge = (d >= 0.025) & (d <= 0.06)
    step |= np.pad(edge, [(0, 1), (0, 0)] if axis == 0 else [(0, 0), (0, 1)])
  low_obstacle = ndimage.binary_opening((F > FREE) & (F <= 0.9), structure=np.ones((3, 3)))
  parts, n_parts = ndimage.label(low_obstacle)
  sizes = ndimage.sum(np.ones_like(F), parts, index=np.arange(1, n_parts + 1)) * CELL * CELL if n_parts else np.zeros(0)
  centre = tuple(s // 2 for s in F.shape)
  spawn_ok = free & (clearance >= 0.6)
  spawn = None
  if spawn_ok.any():
    cells = np.argwhere(spawn_ok)
    spawn = cells[np.argmin(((cells - np.asarray(centre)) ** 2).sum(1))].tolist()
  info = dict(free_fraction=float(free.mean()), max_clearance_m=float(clearance.max()), narrow_area_m2=narrow_area,
              doorway=int(split and narrow_area < 0.6), corridor=int(split and narrow_area >= 1.0),
              threshold_cells=int(step.sum()), clutter=int(((sizes >= 0.02) & (sizes <= 0.6)).sum()), spawn_cell=spawn)
  info["tags"] = [name for name, on in (("open", info["max_clearance_m"] >= 1.5 and info["free_fraction"] >= 0.6),
                                        ("doorway", info["doorway"] > 0), ("corridor", info["corridor"] > 0),
                                        ("threshold", info["threshold_cells"] >= 15), ("clutter", info["clutter"] >= 3)) if on]
  return info


def patches(scene_id: str, F, C, W, min_free: float = 0.35):
  size, stride = int(round(PATCH / CELL)), int(round(STRIDE / CELL))
  out = []
  for i in range(0, max(1, F.shape[0] - size + 1), stride):
    for j in range(0, max(1, F.shape[1] - size + 1), stride):
      f, c, w = F[i:i + size, j:j + size], C[i:i + size, j:j + size], W[i:i + size, j:j + size]
      if f.shape != (size, size):
        continue
      info = tags(f, w)
      if info["free_fraction"] < min_free or info["spawn_cell"] is None:
        continue
      out.append(dict(scene=scene_id, window=[i, j], F=f, C=c, **info))
  return out


def convert(paths: list[Path], out: Path) -> dict:
  started, every, sources = time.time(), [], []
  for path in paths:
    t0 = time.time()
    vertices, faces, axis = load(path)
    points, up = samples(vertices, faces)
    floor = main_floor(points, up)
    F, C, W, low = fields(points, up, floor)
    found = patches(path.stem, F, C, W)
    every += found
    sources.append(dict(path=str(path), sha256=sha256(path), faces=int(len(faces)), samples=int(len(points)), floor_z=floor,
                        grid=list(F.shape), origin_xy=low.tolist(), patches=len(found), seconds=round(time.time() - t0, 1), **axis))
  meta = dict(schema="r1adv-scene-atlas-v1", cell_m=CELL, patch_m=PATCH, stride_m=STRIDE, cut_m=CUT, ceiling_m=CEILING,
              free_m=FREE, sources=sources, seconds=round(time.time() - started, 1),
              patches=[{k: v for k, v in p.items() if k not in ("F", "C")} for p in every])
  np.savez_compressed(out, F=np.stack([p["F"] for p in every]), C=np.stack([p["C"] for p in every]),
                      spawn=np.asarray([p["spawn_cell"] for p in every]), meta=json.dumps(meta))
  meta["atlas_sha256"] = sha256(out)
  out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
  return meta


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--glb", type=Path, nargs="+", required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  meta = convert(args.glb, args.out)
  print(json.dumps({k: meta[k] for k in ("seconds", "atlas_sha256")} | {"patches": len(meta["patches"])}))
  for source in meta["sources"]:
    print(json.dumps({k: source[k] for k in ("path", "faces", "samples", "floor_z", "grid", "patches", "seconds", "up_axis")}))


if __name__ == "__main__":
  main()
