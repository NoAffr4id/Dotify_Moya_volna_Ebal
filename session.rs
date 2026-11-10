use parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::Arc;
use crate::context::TemporalContext;
use crate::ids::{AlbumId, ArtistId, GenreId, TrackId, UserId};
use crate::catalog::Track;
use crate::config::WaveConfig;

#[derive(Clone, Copy, Debug)]
pub struct RecentEntry {
    pub track:  TrackId,
    pub artist: ArtistId,
    pub genre:  GenreId,
    pub album:  Option<AlbumId>,
    pub ts:     i64,
}

#[derive(Clone)]
pub struct SessionSnapshot {
    pub user_id:       UserId,
    pub short_vec:     Arc<[f32]>,
    pub long_vec:      Arc<[f32]>,
    pub recent:        SmallVec<[RecentEntry; 64]>,
    pub skip_streak:   u32,
    pub last_event_ts: i64,
    pub anchor:        Option<(TrackId, f32, i64)>,
    pub rest_track:    SmallVec<[(TrackId, i64); 8]>,
    pub rest_genre:    SmallVec<[(GenreId, i64); 8]>,
    pub rest_artist:   SmallVec<[(ArtistId, i64); 8]>,
    pub rest_album:    SmallVec<[(AlbumId, i64); 8]>,
    pub last_ctx:      TemporalContext,
    pub session_start_ts: i64,
    pub vec_at_5ago:   Arc<[f32]>,
    pub anchor_half_life_hours: f32,
}

impl SessionSnapshot {
    pub fn drift(&self) -> f32 {
        1.0 - crate::vec::cosine_normalized(&self.short_vec, &self.vec_at_5ago)
    }
    pub fn anchor_score(&self, id: TrackId, now: i64) -> f32 {
        match self.anchor {
            Some((tid, w, ts)) if tid == id => w * crate::vec::half_life_decay(now - ts, self.anchor_half_life_hours),
            _ => 0.0,
        }
    }
    pub fn is_resting(&self, id: TrackId, now: i64) -> bool {
        self.rest_track.iter().any(|(tid, until)| *tid == id && now < *until)
    }
    pub fn is_resting_genre(&self, id: GenreId, now: i64) -> bool {
        self.rest_genre.iter().any(|(gid, until)| *gid == id && now < *until)
    }
    pub fn is_resting_artist(&self, id: ArtistId, now: i64) -> bool {
        self.rest_artist.iter().any(|(aid, until)| *aid == id && now < *until)
    }
    pub fn is_resting_album(&self, id: AlbumId, now: i64) -> bool {
        self.rest_album.iter().any(|(aid, until)| *aid == id && now < *until)
    }
}

struct Inner {
    user_id:         UserId,
    short_vec:       Vec<f32>,
    long_vec:        Vec<f32>,
    recent:          SmallVec<[RecentEntry; 64]>,
    vec_at_5ago:     Vec<f32>,
    skip_streak:     u32,
    last_event_ts:   i64,
    anchor:          Option<(TrackId, f32, i64)>,
    rest_track:      SmallVec<[(TrackId, i64); 8]>,
    rest_genre:      SmallVec<[(GenreId, i64); 8]>,
    rest_artist:     SmallVec<[(ArtistId, i64); 8]>,
    rest_album:      SmallVec<[(AlbumId, i64); 8]>,
    last_ctx:        TemporalContext,
    session_start_ts: i64,
}

pub struct Session {
    inner: RwLock<Inner>,
    cfg:   Arc<WaveConfig>,
}

impl Session {
    pub fn new(user_id: UserId, initial_short: Vec<f32>, initial_long: Vec<f32>, cfg: Arc<WaveConfig>) -> Self {
        let mut s = initial_short; s.resize(cfg.dim, 0.0); crate::vec::normalize(&mut s);
        let mut l = initial_long;  l.resize(cfg.dim, 0.0); crate::vec::normalize(&mut l);
        Self {
            inner: RwLock::new(Inner {
                user_id,
                short_vec: s.clone(),
                long_vec: l,
                recent: SmallVec::new(),
                vec_at_5ago: s,
                skip_streak: 0,
                last_event_ts: 0,
                anchor: None,
                rest_track: SmallVec::new(),
                rest_genre: SmallVec::new(),
                rest_artist: SmallVec::new(),
                rest_album: SmallVec::new(),
                last_ctx: TemporalContext::default(),
                session_start_ts: 0,
            }),
            cfg,
        }
    }

    pub fn user_id(&self) -> UserId { self.inner.read().user_id }

    pub fn snapshot(&self) -> SessionSnapshot {
        let g = self.inner.read();
        SessionSnapshot {
            user_id: g.user_id,
            short_vec: Arc::from(g.short_vec.clone().into_boxed_slice()),
            long_vec:  Arc::from(g.long_vec.clone().into_boxed_slice()),
            recent: g.recent.clone(),
            skip_streak: g.skip_streak,
            last_event_ts: g.last_event_ts,
            anchor: g.anchor,
            rest_track: g.rest_track.clone(),
            rest_genre: g.rest_genre.clone(),
            rest_artist: g.rest_artist.clone(),
            rest_album: g.rest_album.clone(),
            last_ctx: g.last_ctx,
            session_start_ts: g.session_start_ts,
            vec_at_5ago: Arc::from(g.vec_at_5ago.clone().into_boxed_slice()),
            anchor_half_life_hours: self.cfg.anchor_half_life_hours,
        }
    }

    pub fn apply(&self, track: &Track, reward: f32, negative: bool, ctx: TemporalContext, now: i64) {
        if reward.abs() < f32::EPSILON && !negative { return; }
        let cfg = &self.cfg;
        let mut g = self.inner.write();

        let since = (now - g.last_event_ts) as f32;
        if g.last_event_ts > 0 && ctx.is_new_session(since, cfg.session_break_hours) {
            let long = g.long_vec.clone();
            g.short_vec.copy_from_slice(&long);
            g.skip_streak = 0;
            g.recent.clear();
            g.session_start_ts = now;
        }
        if g.session_start_ts == 0 { g.session_start_ts = now; }

        if reward.abs() > f32::EPSILON {
            for x in g.short_vec.iter_mut() { *x *= cfg.session_decay_short; }
            for x in g.long_vec.iter_mut()  { *x *= cfg.session_decay_long; }
            crate::vec::axpy_inplace(&mut g.short_vec, reward, &track.embedding);
            if reward > 0.0 {
                crate::vec::axpy_inplace(&mut g.long_vec, reward * 0.3, &track.embedding);
            }
            crate::vec::normalize(&mut g.short_vec);
            crate::vec::normalize(&mut g.long_vec);
        }

        if negative {
            g.skip_streak = g.skip_streak.saturating_add(1);
            if g.skip_streak >= cfg.rest_trigger_streak {
                let until_t = now + (cfg.rest_track_hours * 3600.0) as i64;
                let until_g = now + (cfg.rest_genre_hours * 3600.0) as i64;
                if g.rest_track.len()  >= 8 { g.rest_track.remove(0); }
                if g.rest_genre.len()  >= 8 { g.rest_genre.remove(0); }
                if g.rest_artist.len() >= 8 { g.rest_artist.remove(0); }
                if g.rest_album.len()  >= 8 { g.rest_album.remove(0); }
                g.rest_track.push((track.id, until_t));
                g.rest_genre.push((track.genre_id, until_g));
                g.rest_artist.push((track.artist_id, until_t));
                if let Some(a) = track.album_id { g.rest_album.push((a, until_t)); }
            }
        } else {
            g.skip_streak = 0;
            g.recent.push(RecentEntry {
                track: track.id, artist: track.artist_id, genre: track.genre_id,
                album: track.album_id, ts: now,
            });
            if g.recent.len() > cfg.long_window { g.recent.remove(0); }
            if g.recent.len() >= 5 { g.vec_at_5ago = g.short_vec.clone(); }
        }

        g.last_event_ts = now;
        g.last_ctx = ctx;
    }

    pub fn set_anchor(&self, id: TrackId, weight: f32, now: i64) {
        self.inner.write().anchor = Some((id, weight, now));
    }
    pub fn cleanup_rest(&self, now: i64) {
        let mut g = self.inner.write();
        g.rest_track.retain(|(_, u)| now < *u);
        g.rest_genre.retain(|(_, u)| now < *u);
        g.rest_artist.retain(|(_, u)| now < *u);
        g.rest_album.retain(|(_, u)| now < *u);
        if let Some((_, _, ts)) = g.anchor {
            if crate::vec::half_life_decay(now - ts, self.cfg.anchor_half_life_hours) < 0.05 {
                g.anchor = None;
            }
        }
    }
}