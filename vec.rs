#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_norm_sq(a: &[f32]) -> f32 { dot(a, a) }

pub fn normalize(v: &mut [f32]) {
    let n = l2_norm_sq(v).sqrt();
    if n > 1e-12 {
        let inv = 1.0 / n;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

#[inline]
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 { dot(a, b) }

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm_sq(a).sqrt();
    let nb = l2_norm_sq(b).sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    dot(a, b) / (na * nb)
}

pub fn axpy_inplace(dst: &mut [f32], alpha: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) { *d = alpha.mul_add(*s, *d); }
}

pub fn half_life_decay(delta_secs: i64, half_life_hours: f32) -> f32 {
    if delta_secs <= 0 { return 1.0; }
    let hours = delta_secs as f32 / 3600.0;
    0.5f32.powf(hours / half_life_hours.max(1e-3))
}

pub fn skip_curve(position_sec: f32, tau: f32) -> f32 {
    -(-position_sec / tau.max(1e-3)).exp()
}

pub fn camelot_distance(a: u8, b: u8) -> f32 {
    let a = a as i32;
    let b = b as i32;
    let num_dist = ((a / 2 - b / 2).abs()).min(6 - ((a / 2 - b / 2).abs()));
    let mode_diff = (a % 2) ^ (b % 2);
    let dist = num_dist + mode_diff;
    (1.0 - dist as f32 / 6.0).clamp(0.0, 1.0)
}