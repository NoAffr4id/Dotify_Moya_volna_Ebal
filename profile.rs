use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use crate::context::{Device, TemporalContext};
use crate::ids::{ArtistId, GenreId, LangId, TrackId, UserId};
use crate::catalog::Track;
use crate::vec::{axpy_inplace, half_life_decay, normalize};

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct ContextStat {
    pub plays:    u32,
    pub skips:    u32,
    pub plays_ma: f32,
    pub skips_ma: f32,
}

impl ContextStat {
    pub fn update(&mut self, positive: bool) {
        if positive { self.plays += 1; } else { self.skips += 1; }
        self.plays_ma = 0.95 * self.plays_ma + 0.05 * if positive { 1.0 } else { 0.0 };
        self.skips_ma = 0.95 * self.skips_ma + 0.05 * if positive { 0.0 } else { 1.0 };
    }
    pub fn quality(&self) -> f32 { (self.plays_ma - 0.5 * self.skips_ma).clamp(-1.0, 1.0) }
}

#[derive(Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id:         UserId,
    pub long_vec:        Vec<f32>,
    pub artist_affinity: FxHashMap<ArtistId, f32>,
    pub genre_affinity:  FxHashMap<GenreId, f32>,
    pub lang_affinity:   FxHashMap<LangId, f32>,
    pub era_ma:          f32,
    pub bpm_ma:          f32,
    pub vocal_ma:        f32,
    pub energy_ma:       SmallVec<[f32; 8]>,
    pub skip_rate:       FxHashMap<TrackId, f32>,
    pub last_played:     FxHashMap<TrackId, i64>,
    pub plays:           FxHashMap<TrackId, u32>,
    pub skips:           FxHashMap<TrackId, u32>,
    pub co_like:         FxHashMap<TrackId, SmallVec<[(TrackId, f32); 32]>>,
    pub co_skip:         FxHashMap<TrackId, SmallVec<[(TrackId, f32); 32]>>,
    pub context_stats:   FxHashMap<(u8, Device), ContextStat>,
    pub context_genre_stats: FxHashMap<(u8, GenreId), ContextStat>,
    pub dim:             usize,
    pub half_life_hours: f32,
    pub cold_play_threshold: u32,
    pub total_weight:    f32,
    pub total_plays:     u32,
}

impl UserProfile {
    pub fn new(user_id: UserId, dim: usize, half_life_hours: f32, cold_play_threshold: u32) -> Self {
        Self {
            user_id,
            long_vec: vec![0.0; dim],
            artist_affinity: FxHashMap::default(),
            genre_affinity: FxHashMap::default(),
            lang_affinity: FxHashMap::default(),
            era_ma: 2000.0,
            bpm_ma: 110.0,
            vocal_ma: 0.5,
            energy_ma: SmallVec::from_elem(0.5, 8),
            skip_rate: FxHashMap::default(),
            last_played: FxHashMap::default(),
            plays: FxHashMap::default(),
            skips: FxHashMap::default(),
            co_like: FxHashMap::default(),
            co_skip: FxHashMap::default(),
            context_stats: FxHashMap::default(),
            context_genre_stats: FxHashMap::default(),
            dim,
            half_life_hours,
            cold_play_threshold,
            total_weight: 0.0,
            total_plays: 0,
        }
    }

    pub fn on_positive(&mut self, track: &Track, now: i64, weight: f32, ctx: TemporalContext) {
        axpy_inplace(&mut self.long_vec, weight, &track.embedding);
        self.total_weight += weight;
        if self.total_weight > 1e-6 {
            let inv = 1.0 / self.total_weight;
            for x in self.long_vec.iter_mut() { *x *= inv; }
            self.total_weight = self.total_weight.min(1.0);
        }
        normalize(&mut self.long_vec);
        *self.artist_affinity.entry(track.artist_id).or_insert(0.0) += weight;
        *self.genre_affinity.entry(track.genre_id).or_insert(0.0)   += weight * 0.5;
        if let Some(l) = track.language { *self.lang_affinity.entry(l).or_insert(0.0) += weight * 0.5; }
        self.era_ma = 0.97 * self.era_ma + 0.03 * track.release_year as f32;
        if let Some(b) = track.bpm { self.bpm_ma = 0.95 * self.bpm_ma + 0.05 * b; }
        if let Some(v) = track.vocal { self.vocal_ma = 0.95 * self.vocal_ma + 0.05 * v; }
        if let Some(e) = track.energy_curve.as_ref() {
            for (slot, val) in self.energy_ma.iter_mut().zip(e.iter()) {
                *slot = 0.95 * *slot + 0.05 * val;
            }
        }
        *self.plays.entry(track.id).or_insert(0) += 1;
        self.last_played.insert(track.id, now);
        self.total_plays += 1;
        self.recompute_skip_rate(track.id);
        self.context_stats.entry(ctx.bucket()).or_default().update(true);
        self.context_genre_stats.entry((ctx.hour, track.genre_id)).or_default().update(true);
    }

    pub fn on_skip(&mut self, track: &Track, now: i64, severe: bool, ctx: TemporalContext) {
        *self.skips.entry(track.id).or_insert(0) += 1;
        self.last_played.insert(track.id, now);
        self.recompute_skip_rate(track.id);
        let decay = if severe { 0.995 } else { 0.998 };
        for x in self.long_vec.iter_mut() { *x *= decay; }
        self.context_stats.entry(ctx.bucket()).or_default().update(false);
        self.context_genre_stats.entry((ctx.hour, track.genre_id)).or_default().update(false);
    }

    pub fn on_neutral(&mut self, track: &Track, now: i64) { self.last_played.insert(track.id, now); }

    pub fn on_seek(&mut self, track: &Track, now: i64, _from_sec: f32, to_sec: f32, duration: f32) {
        if duration > 0.0 && to_sec / duration > 0.9 {
            *self.skips.entry(track.id).or_insert(0) += 1;
            self.recompute_skip_rate(track.id);
        }
        self.last_played.insert(track.id, now);
    }

    pub fn record_co_occurrence(&mut self, a: TrackId, b: TrackId, positive: bool) {
        let map = if positive { &mut self.co_like } else { &mut self.co_skip };
        let alpha = if positive { 0.1 } else { 0.05 };
        Self::bump_co(map, a, b, alpha);
        Self::bump_co(map, b, a, alpha);
    }

    fn bump_co(map: &mut FxHashMap<TrackId, SmallVec<[(TrackId, f32); 32]>>, a: TrackId, b: TrackId, alpha: f32) {
        let entry = map.entry(a).or_default();
        if let Some(slot) = entry.iter_mut().find(|(id, _)| *id == b) {
            slot.1 = (1.0 - alpha) * slot.1 + alpha;
        } else {
            if entry.len() >= 32 {
                if let Some((idx, _)) = entry.iter().enumerate().min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal)) {
                    entry.swap_remove(idx);
                }
            }
            entry.push((b, alpha));
        }
    }

    fn recompute_skip_rate(&mut self, id: TrackId) {
        let p = self.plays.get(&id).copied().unwrap_or(0) as f32;
        let s = self.skips.get(&id).copied().unwrap_or(0) as f32;
        let total = p + s;
        self.skip_rate.insert(id, if total > 0.0 { s / total } else { 0.0 });
    }

    pub fn total_plays(&self) -> u32 { self.total_plays }
}