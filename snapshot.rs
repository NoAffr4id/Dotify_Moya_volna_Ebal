use std::sync::Arc;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use crate::context::{Device, TemporalContext};
use crate::ids::{ArtistId, GenreId, LangId, TrackId};
use crate::profile::{ContextStat, UserProfile};

pub use crate::session::SessionSnapshot;

#[derive(Clone)]
pub struct ProfileSnapshot {
    pub user_id:         crate::ids::UserId,
    pub long_vec:        Arc<[f32]>,
    pub artist_affinity: Arc<FxHashMap<ArtistId, f32>>,
    pub genre_affinity:  Arc<FxHashMap<GenreId, f32>>,
    pub lang_affinity:   Arc<FxHashMap<LangId, f32>>,
    pub era_ma:          f32,
    pub bpm_ma:          f32,
    pub vocal_ma:        f32,
    pub energy_ma:       SmallVec<[f32; 8]>,
    pub skip_rate:       Arc<FxHashMap<TrackId, f32>>,
    pub last_played:     Arc<FxHashMap<TrackId, i64>>,
    pub plays:           Arc<FxHashMap<TrackId, u32>>,
    pub skips:           Arc<FxHashMap<TrackId, u32>>,
    pub co_like:         Arc<FxHashMap<TrackId, SmallVec<[(TrackId, f32); 32]>>>,
    pub co_skip:         Arc<FxHashMap<TrackId, SmallVec<[(TrackId, f32); 32]>>>,
    pub context_stats:   Arc<FxHashMap<(u8, Device), ContextStat>>,
    pub context_genre_stats: Arc<FxHashMap<(u8, GenreId), ContextStat>>,
    pub total_plays:     u32,
    pub half_life_hours: f32,
    pub cold_play_threshold: u32,
}

impl ProfileSnapshot {
    pub fn freshness(&self, id: TrackId, now: i64) -> f32 {
        match self.last_played.get(&id) {
            None => 1.0,
            Some(&ts) => 1.0 - crate::vec::half_life_decay(now - ts, self.half_life_hours),
        }
    }
    pub fn is_cold(&self, id: TrackId) -> bool {
        self.plays.get(&id).copied().unwrap_or(0) < self.cold_play_threshold
    }
    pub fn co_score(&self, seed: TrackId, candidate: TrackId) -> f32 {
        let pos = self.co_like.get(&seed).and_then(|v| v.iter().find(|(id, _)| *id == candidate).map(|(_, w)| *w)).unwrap_or(0.0);
        let neg = self.co_skip.get(&seed).and_then(|v| v.iter().find(|(id, _)| *id == candidate).map(|(_, w)| *w)).unwrap_or(0.0);
        pos - neg
    }
    pub fn context_quality(&self, ctx: &TemporalContext, genre: GenreId) -> f32 {
        let g = self.context_genre_stats.get(&(ctx.hour, genre)).map(|c| c.quality()).unwrap_or(0.0);
        let u = self.context_stats.get(&ctx.bucket()).map(|c| c.quality()).unwrap_or(0.0);
        (0.7 * g + 0.3 * u).clamp(-1.0, 1.0)
    }
}

impl UserProfile {
    pub fn snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot {
            user_id:         self.user_id,
            long_vec:        Arc::from(self.long_vec.clone().into_boxed_slice()),
            artist_affinity: Arc::new(self.artist_affinity.clone()),
            genre_affinity:  Arc::new(self.genre_affinity.clone()),
            lang_affinity:   Arc::new(self.lang_affinity.clone()),
            era_ma:          self.era_ma,
            bpm_ma:          self.bpm_ma,
            vocal_ma:        self.vocal_ma,
            energy_ma:       self.energy_ma.clone(),
            skip_rate:       Arc::new(self.skip_rate.clone()),
            last_played:     Arc::new(self.last_played.clone()),
            plays:           Arc::new(self.plays.clone()),
            skips:           Arc::new(self.skips.clone()),
            co_like:         Arc::new(self.co_like.clone()),
            co_skip:         Arc::new(self.co_skip.clone()),
            context_stats:   Arc::new(self.context_stats.clone()),
            context_genre_stats: Arc::new(self.context_genre_stats.clone()),
            total_plays:     self.total_plays,
            half_life_hours: self.half_life_hours,
            cold_play_threshold: self.cold_play_threshold,
        }
    }
}

#[derive(Clone)]
pub struct CoocSnapshot {
    pub pmi_track:  Arc<FxHashMap<TrackId, SmallVec<[(TrackId, f32); 16]>>>,
    pub pmi_artist: Arc<FxHashMap<ArtistId, SmallVec<[(ArtistId, f32); 16]>>>,
    pub album_co:   Arc<FxHashMap<crate::ids::AlbumId, SmallVec<[(TrackId, f32); 16]>>>,
}

impl CoocSnapshot {
    pub fn pmi_track_neighbors(&self, id: TrackId) -> &[(TrackId, f32)] {
        self.pmi_track.get(&id).map(SmallVec::as_slice).unwrap_or(&[])
    }
    pub fn pmi_artist_neighbors(&self, id: ArtistId) -> &[(ArtistId, f32)] {
        self.pmi_artist.get(&id).map(SmallVec::as_slice).unwrap_or(&[])
    }
    pub fn album_siblings(&self, album: crate::ids::AlbumId) -> &[(TrackId, f32)] {
        self.album_co.get(&album).map(SmallVec::as_slice).unwrap_or(&[])
    }
    pub fn pmi_score(&self, seed: &[TrackId], candidate: TrackId) -> f32 {
        if seed.is_empty() { return 0.0; }
        let mut s = 0.0f32;
        for &sid in seed {
            for &(n, w) in self.pmi_track_neighbors(sid) {
                if n == candidate { s += w; }
            }
        }
        (s / seed.len() as f32).min(1.0)
    }
}

#[derive(Clone)]
pub struct MarkovSnapshot {
    pub trans: Arc<FxHashMap<TrackId, SmallVec<[(TrackId, u32); 8]>>>,
}

impl MarkovSnapshot {
    pub fn prob(&self, from: TrackId, candidate: TrackId) -> f32 {
        let Some(row) = self.trans.get(&from) else { return 0.0 };
        let total: u32 = row.iter().map(|(_, c)| *c).sum();
        if total == 0 { return 0.0; }
        let c = row.iter().find(|(id, _)| *id == candidate).map(|(_, c)| *c).unwrap_or(0);
        (c as f32 + 0.1) / (total as f32 + 0.1 * row.len() as f32)
    }
    pub fn top_next(&self, from: TrackId, k: usize) -> &[(TrackId, u32)] {
        self.trans.get(&from).map(|row| { let end = row.len().min(k); &row[..end] }).unwrap_or(&[])
    }
}

#[derive(Clone)]
pub struct BanditSnapshot {
    pub state: Arc<FxHashMap<crate::bandit::Arm, (f64, f64)>>,
}

#[derive(Clone)]
pub struct UserKnnSnapshot {
    pub users: Arc<FxHashMap<crate::ids::UserId, FxHashMap<TrackId, f32>>>,
}