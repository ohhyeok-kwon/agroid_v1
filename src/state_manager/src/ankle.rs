//! Same Delaunay triangles and piecewise-linear interpolation as basic_service.
//! Vertices and triangles were extracted offline; see config/source.json.
pub struct AnkleMap {
    points: Vec<[f64; 4]>,
    triangles: Vec<([usize; 3], [f64; 4])>,
    bounds: [f64; 4],
}

impl Default for AnkleMap {
    fn default() -> Self {
        let bytes = include_bytes!("../config/ankle.bin");
        assert_eq!(&bytes[..4], b"R1A1");
        let integer = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let count = integer(4);
        let triangle_count = integer(8);
        let points: Vec<[f64; 4]> = (0..count)
            .map(|i| {
                std::array::from_fn(|j| {
                    let at = 12 + i * 32 + j * 8;
                    f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
                })
            })
            .collect();
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for p in &points {
            bounds[0] = bounds[0].min(p[0]);
            bounds[1] = bounds[1].min(p[1]);
            bounds[2] = bounds[2].max(p[0]);
            bounds[3] = bounds[3].max(p[1]);
        }
        let triangles = (0..triangle_count)
            .map(|i| {
                let ids: [usize; 3] =
                    std::array::from_fn(|j| integer(12 + count * 32 + i * 12 + j * 4));
                let p = ids.map(|i| points[i]);
                let bound = [
                    p.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                    p.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                    p.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
                    p.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
                ];
                (ids, bound)
            })
            .collect();
        Self {
            points,
            triangles,
            bounds,
        }
    }
}

impl AnkleMap {
    /// Forward position and its local Jacobian, used by the native PR -> motor mapping.
    pub fn position_jacobian(
        &self,
        a: f64,
        b: f64,
        left: bool,
    ) -> Option<([f64; 2], [[f64; 2]; 2])> {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let s = if left { 1. } else { -1. };
        let (x, y) = (a * s, b * s);
        // Control inputs are rejected outside the table; do not silently clamp commands.
        if x < self.bounds[0] || x > self.bounds[2] || y < self.bounds[1] || y > self.bounds[3] {
            return None;
        }
        for (ids, bounds) in &self.triangles {
            if x < bounds[0] - 1e-12
                || y < bounds[1] - 1e-12
                || x > bounds[2] + 1e-12
                || y > bounds[3] + 1e-12
            {
                continue;
            }
            let [a, b, c] = ids.map(|i| self.points[i]);
            let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if det.abs() < 1e-16 {
                continue;
            }
            let u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det;
            let v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det;
            if u < -1e-10 || v < -1e-10 || 1. - u - v < -1e-10 {
                continue;
            }
            let mut j = [[0.; 2]; 2];
            for d in 0..2 {
                let z = d + 2;
                let sign = if d == 0 { s } else { 1. };
                j[d][0] =
                    sign * ((a[z] - c[z]) * (b[1] - c[1]) + (b[z] - c[z]) * (c[1] - a[1])) / det;
                j[d][1] =
                    sign * ((a[z] - c[z]) * (c[0] - b[0]) + (b[z] - c[z]) * (a[0] - c[0])) / det;
            }
            return Some((
                [
                    u * a[2] + v * b[2] + (1. - u - v) * c[2],
                    s * (u * a[3] + v * b[3] + (1. - u - v) * c[3]),
                ],
                j,
            ));
        }
        None
    }
    /// Arguments are manufacturer joint slots 4,5 (or 10,11), in radians.
    /// Right-side inputs and roll output are mirrored as in 0x102c5408.
    pub fn position(&self, a: f64, b: f64, left: bool) -> Option<[f64; 2]> {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let sign = if left { 1.0 } else { -1.0 };
        let x = (a * sign).clamp(self.bounds[0], self.bounds[2]);
        let y = (b * sign).clamp(self.bounds[1], self.bounds[3]);
        for (ids, bound) in &self.triangles {
            if x < bound[0] - 1e-12
                || y < bound[1] - 1e-12
                || x > bound[2] + 1e-12
                || y > bound[3] + 1e-12
            {
                continue;
            }
            let [a, b, c] = ids.map(|i| self.points[i]);
            let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if det.abs() < 1e-16 {
                continue;
            }
            let u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det;
            let v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det;
            let w = 1.0 - u - v;
            if u >= -1e-10 && v >= -1e-10 && w >= -1e-10 {
                return Some([
                    u * a[2] + v * b[2] + w * c[2],
                    sign * (u * a[3] + v * b[3] + w * c[3]),
                ]);
            }
        }
        None
    }
}
