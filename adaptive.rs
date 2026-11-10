use rand::Rng;
use rand_distr::{Beta, Distribution};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::ids::{ArtistId, GenreId, TrackId};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ArmState {
    pub alpha: f32,
    pub beta:  f32,
    pub last:  i64,
    pub n:     u32,
}

impl Default for ArmState {
    fn default() -> Self { Self { alpha: 1.0, beta: 1.0, last: 0, n: 0 } }
}

impl ArmState {
    pub fn decayed(&self, now: i64, tau_hours: f32) -> Self {
        if self.last == 0 { return *self; }
        let dt = (now - self.last).max(0) as f32 / 3600.0;
        let f  = (-dt / tau_hours.max(1e-3)).exp();
        Self {
            alpha: 1.0 + (self.alpha - 1.0) * f,
            beta:  1.0 + (self.beta  - 1.0) * f,
            last:  self.last,
            n:     self.n,
        }
    }
    pub fn update(&mut self, reward: f32, now: i64) {
        let r = reward.clamp(0.0, 1.0);
        self.alpha += r;
        self.beta  += 1.0 - r;
        self.last   = now;
        self.n     += 1;
    }
}

#[derive(Clone)]
pub struct UserPolicySnapshot {
    pub tracks:    Arc<FxHashMap<TrackId, ArmState>>,
    pub tau_hours: f32,
    pub shrink:    f32,
}

#[derive(Clone)]
pub struct GlobalPolicySnapshot {
    pub artists:   Arc<FxHashMap<ArtistId, ArmState>>,
    pub genres:    Arc<FxHashMap<GenreId, ArmState>>,
    pub global:    ArmState,
    pub tau_hours: f32,
    pub shrink:    f32,
}

pub struct UserPolicy {
    pub tracks:    FxHashMap<TrackId, ArmState>,
    pub tau_hours: f32,
    pub shrink:    f32,
}

impl UserPolicy {
    pub fn new(tau_hours: f32, shrink: f32) -> Self {
        Self { tracks: FxHashMap::default(), tau_hours, shrink }
    }
    pub fn observe(&mut self, track: TrackId, reward: f32, now: i64) {
        self.tracks.entry(track).or_default().update(reward, now);
    }
    pub fn snapshot(&self) -> UserPolicySnapshot {
        UserPolicySnapshot {
            tracks: Arc::new(self.tracks.clone()),
            tau_hours: self.tau_hours,
            shrink: self.shrink,
        }
    }
    pub fn known_tracks(&self) -> usize { self.tracks.len() }
}

pub struct GlobalPolicy {
    pub artists:   FxHashMap<ArtistId, ArmState>,
    pub genres:    FxHashMap<GenreId, ArmState>,
    pub global:    ArmState,
    pub tau_hours: f32,
    pub shrink:    f32,
}

impl GlobalPolicy {
    pub fn new(tau_hours: f32, shrink: f32) -> Self {
        Self {
            artists: FxHashMap::default(),
            genres:  FxHashMap::default(),
            global:  ArmState::default(),
            tau_hours, shrink,
        }
    }
    pub fn observe(&mut self, artist: ArtistId, genre: GenreId, reward: f32, now: i64) {
        self.artists.entry(artist).or_default().update(reward, now);
        self.genres .entry(genre ).or_default().update(reward, now);
        self.global.update(reward, now);
    }
    pub fn snapshot(&self) -> GlobalPolicySnapshot {
        GlobalPolicySnapshot {
            artists: Arc::new(self.artists.clone()),
            genres:  Arc::new(self.genres.clone()),
            global:  self.global,
            tau_hours: self.tau_hours,
            shrink: self.shrink,
        }
    }
}

/// Иерархический prior: user-track получает 2× вес, global — 0.5×.
pub fn prior_dual(
    user:   &UserPolicySnapshot,
    global: &GlobalPolicySnapshot,
    track: TrackId, artist: ArtistId, genre: GenreId, now: i64,
) -> (f32, f32) {
    let t  = user.tracks.get(&track).copied().unwrap_or_default().decayed(now, user.tau_hours);
    let a  = global.artists.get(&artist).copied().unwrap_or_default().decayed(now, global.tau_hours);
    let g  = global.genres .get(&genre ).copied().unwrap_or_default().decayed(now, global.tau_hours);
    let gl = global.global.decayed(now, global.tau_hours);

    let wt  = (t.n  as f32) * 2.0;
    let wa  = a.n  as f32;
    let wg  = g.n  as f32;
    let wgl = (gl.n as f32) * 0.5;
    let shrink = user.shrink.max(global.shrink);
    let total = wt + wa + wg + wgl + shrink;
    if total < 1e-6 { return (1.0, 1.0); }
    let alpha = (t.alpha * wt + a.alpha * wa + g.alpha * wg + gl.alpha * wgl + shrink) / total;
    let beta  = (t.beta  * wt + a.beta  * wa + g.beta  * wg + gl.beta  * wgl + shrink) / total;
    (alpha.max(0.01), beta.max(0.01))
}

pub fn sample_dual<R: Rng>(
    user: &UserPolicySnapshot, global: &GlobalPolicySnapshot,
    track: TrackId, artist: ArtistId, genre: GenreId, now: i64, rng: &mut R,
) -> f32 {
    let (a, b) = prior_dual(user, global, track, artist, genre, now);
    Beta::new(a as f64, b as f64).map(|d| d.sample(rng) as f32).unwrap_or(0.5)
}

pub fn reward_from_listen(played_sec: f32, duration_sec: f32, liked: bool, disliked: bool) -> f32 {
    if disliked { return 0.0; }
    if liked    { return 1.0; }
    let ratio = if duration_sec > 0.0 { played_sec / duration_sec } else { 1.0 };
    if      ratio < 0.05 { 0.0  }
    else if ratio < 0.30 { 0.25 }
    else if ratio < 0.80 { 0.60 }
    else                 { 1.0  }
}