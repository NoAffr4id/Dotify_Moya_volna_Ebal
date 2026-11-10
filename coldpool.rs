use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use crate::catalog::Catalog;
use crate::config::WaveConfig;
use crate::ids::{GenreId, TrackId};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColdPoolConfig {
    pub enabled:           bool,
    pub max_age_days:      u32,
    pub target_share:      f32,
    pub max_per_genre:     usize,
    pub min_popularity:    f32,
}

impl Default for ColdPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_age_days: 30,
            target_share: 0.15,
            max_per_genre: 2,
            min_popularity: 0.0,
        }
    }
}

struct Inner {
    members: rustc_hash::FxHashMap<TrackId, (i64, GenreId)>,
}

pub struct ColdPool {
    inner: RwLock<Inner>,
    cfg:   ColdPoolConfig,
}

impl ColdPool {
    pub fn new(cfg: ColdPoolConfig) -> Self {
        Self { inner: RwLock::new(Inner { members: rustc_hash::FxHashMap::default() }), cfg }
    }

    pub fn cfg(&self) -> &ColdPoolConfig { &self.cfg }

    pub fn register(&self, track: TrackId, genre: GenreId, added_ts: i64) {
        self.inner.write().members.insert(track, (added_ts, genre));
    }

    pub fn unregister(&self, track: TrackId) {
        self.inner.write().members.remove(&track);
    }

    pub fn len(&self) -> usize { self.inner.read().members.len() }
    pub fn is_empty(&self) -> bool { self.inner.read().members.is_empty() }
    pub fn contains(&self, track: TrackId) -> bool { self.inner.read().members.contains_key(&track) }

    pub fn sample(
        &self,
        catalog: &Catalog,
        exclude: &FxHashSet<TrackId>,
        k: usize,
        now: i64,
        rng: &mut impl rand::Rng,
    ) -> Vec<TrackId> {
        use rand::seq::SliceRandom;
        if k == 0 || !self.cfg.enabled { return Vec::new(); }
        let g = self.inner.read();
        let mut candidates: Vec<(TrackId, GenreId)> = g.members.iter()
            .filter(|(id, (ts, _))| {
                if exclude.contains(id) { return false; }
                let age_days = ((now - *ts).max(0) / 86_400) as u32;
                age_days <= self.cfg.max_age_days
            })
            .filter_map(|(id, (_, genre))| {
                catalog.get(*id).ok()
                    .filter(|t| t.popularity >= self.cfg.min_popularity)
                    .map(|_| (*id, *genre))
            })
            .collect();
        candidates.shuffle(rng);

        let mut out = Vec::with_capacity(k);
        let mut genre_counts: rustc_hash::FxHashMap<GenreId, usize> = rustc_hash::FxHashMap::default();
        for (id, genre) in candidates {
            if out.len() >= k { break; }
            let cnt = genre_counts.entry(genre).or_insert(0);
            if *cnt >= self.cfg.max_per_genre { continue; }
            *cnt += 1;
            out.push(id);
        }
        out
    }

    pub fn maybe_promote(&self, track: &crate::catalog::Track, plays: u32, threshold: u32, now: i64) {
        if plays < threshold {
            self.register(track.id, track.genre_id, now);
        } else {
            self.unregister(track.id);
        }
    }

    pub fn cleanup_old(&self, now: i64) {
        let max_age_secs = (self.cfg.max_age_days as i64) * 86_400;
        self.inner.write().members.retain(|_, (ts, _)| now - *ts <= max_age_secs);
    }

    pub fn target_count(&self, wave_len: usize, cfg: &WaveConfig) -> usize {
        if !self.cfg.enabled { return 0; }
        let share = self.cfg.target_share.clamp(0.0, 1.0);
        ((wave_len as f32) * share).ceil() as usize
            .min(wave_len)
            .max(if cfg.explore_fraction > 0.0 { 1 } else { 0 })
    }
}