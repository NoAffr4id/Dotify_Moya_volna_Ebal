use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use parking_lot::RwLock;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::adaptive::{reward_from_listen, GlobalPolicy, UserPolicy};
use crate::bandit::{Arm, ThompsonBandit};
use crate::candidates::{Candidate, Source, Sources};
use crate::catalog::{Catalog, Track};
use crate::coldpool::ColdPool;
use crate::config::WaveConfig;
use crate::context::TemporalContext;
use crate::cooccurrence::CoOccurrence;
use crate::diversify::{dpp, mmr, Scored};
use crate::error::{Result, WaveError};
use crate::ids::{AlbumId, TrackId, UserId};
use crate::markov::MarkovChain;
use crate::metrics::{TrackOutcome, WaveMetrics};
use crate::persistence::Store;
use crate::profile::UserProfile;
use crate::ranker::{score, RankerCtx};
use crate::session::Session;
use crate::usermap::UserKnn;
use crate::vec::cosine_normalized;

#[derive(Clone, Debug)]
pub enum Event {
    PlayStarted   { track: TrackId, ts: i64 },
    PlayCompleted { track: TrackId, ts: i64, played_sec: f32 },
    Liked         { track: TrackId, ts: i64 },
    Disliked      { track: TrackId, ts: i64 },
    Skipped       { track: TrackId, ts: i64, position_sec: f32 },
    Seeked        { track: TrackId, ts: i64, from_sec: f32, to_sec: f32 },
    SearchQuery   { track: TrackId, ts: i64 },
    PlaylistAdd   { track: TrackId, ts: i64 },
    FocusChange   { track: TrackId, ts: i64, focused: bool },
    AlbumSkip     { album: AlbumId, ts: i64 },
    PlaylistSkip  { ts: i64 },
}

pub struct Engine {
    cfg:      Arc<WaveConfig>,
    catalog:  Arc<Catalog>,
    profile:  RwLock<UserProfile>,
    session:  Arc<Session>,
    cooc:     RwLock<CoOccurrence>,
    markov:   RwLock<MarkovChain>,
    usermap:  RwLock<UserKnn>,
    bandit:   RwLock<ThompsonBandit>,
    user_policy:   RwLock<UserPolicy>,
    global_policy: Arc<RwLock<GlobalPolicy>>,
    coldpool: ColdPool,
    metrics:  Arc<WaveMetrics>,
    store:    Option<Arc<Store>>,
    ingest_n: AtomicU64,
    seed:     u64,
}

impl Engine {
    pub fn new(cfg: Arc<WaveConfig>, catalog: Arc<Catalog>, user_id: UserId) -> Result<Self> {
        let global = Arc::new(RwLock::new(GlobalPolicy::new(cfg.adaptive_tau_hours, cfg.adaptive_shrink)));
        Self::with_global(cfg, catalog, user_id, global)
    }

    pub fn with_global(cfg: Arc<WaveConfig>, catalog: Arc<Catalog>, user_id: UserId, global_policy: Arc<RwLock<GlobalPolicy>>) -> Result<Self> {
        cfg.validate()?;
        let init = catalog.iter().fold(vec![0.0f32; cfg.dim], |mut acc, t| {
            crate::vec::axpy_inplace(&mut acc, t.popularity, &t.embedding);
            acc
        });
        let session = Arc::new(Session::new(user_id, init.clone(), init, cfg.clone()));
        let profile = UserProfile::new(user_id, cfg.dim, cfg.profile_half_life_hours, cfg.cold_play_threshold);
        let seed = 0xdead_beef ^ user_id.raw();
        let store = Store::new(&cfg.persistence).ok().map(Arc::new);
        let coldpool = ColdPool::new(cfg.cold_pool.clone());
        Ok(Self {
            cfg: cfg.clone(),
            catalog,
            profile: RwLock::new(profile),
            session,
            cooc:    RwLock::new(CoOccurrence::new()),
            markov:  RwLock::new(MarkovChain::new()),
            usermap: RwLock::new(UserKnn::new()),
            bandit:  RwLock::new(ThompsonBandit::new(seed)),
            user_policy:   RwLock::new(UserPolicy::new(cfg.adaptive_tau_hours, cfg.adaptive_shrink)),
            global_policy,
            coldpool,
            metrics: Arc::new(WaveMetrics::new()),
            store,
            ingest_n: AtomicU64::new(0),
            seed,
        })
    }

    pub fn with_persistence(mut self, store: Arc<Store>) -> Self { self.store = Some(store); self }
    pub fn with_metrics(mut self, metrics: Arc<WaveMetrics>) -> Self { self.metrics = metrics; self }

    pub fn global_policy(&self) -> &Arc<RwLock<GlobalPolicy>> { &self.global_policy }
    pub fn metrics(&self) -> &Arc<WaveMetrics> { &self.metrics }
    pub fn session(&self) -> &Session { &self.session }
    pub fn cfg(&self) -> &WaveConfig { &self.cfg }
    pub fn store(&self) -> Option<&Arc<Store>> { self.store.as_ref() }
    pub fn coldpool(&self) -> &ColdPool { &self.coldpool }

    pub fn replace_usermap(&self, new_map: UserKnn) { *self.usermap.write() = new_map; }

    pub fn snapshot_plays(&self, track: TrackId) -> u32 {
        self.profile.read().plays.get(&track).copied().unwrap_or(0)
    }

    pub fn load_from_store(&self, user: UserId) -> Result<()> {
        let Some(store) = self.store.as_ref() else { return Ok(()) };
        if let Some(p) = store.load_profile(user)? { *self.profile.write() = p; }
        if let Some(c) = store.load_cooc()?   { *self.cooc.write()   = c; }
        if let Some(m) = store.load_markov()? { *self.markov.write() = m; }
        if let Some(u) = store.load_usermap()?{ *self.usermap.write()= u; }
        if let Some((state, seed)) = store.load_bandit()? {
            *self.bandit.write() = ThompsonBandit::restore(state, seed);
        }
        if let Some(p) = store.load_policy()? {
            let mut up = UserPolicy::new(p.tau_hours, p.shrink);
            up.tracks = p.tracks.into_iter().collect();
            *self.user_policy.write() = up;
        }
        if let Some(gp) = store.load_global_policy()? {
            let mut g = self.global_policy.write();
            g.artists = gp.artists.into_iter().collect();
            g.genres  = gp.genres.into_iter().collect();
            g.global  = gp.global;
            g.tau_hours = gp.tau_hours;
            g.shrink = gp.shrink;
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        let Some(store) = self.store.as_ref() else { return Ok(()) };
        store.save_profile(&self.profile.read())?;
        store.save_cooc(&self.cooc.read())?;
        store.save_markov(&self.markov.read())?;
        store.save_usermap(&self.usermap.read())?;
        {
            let b = self.bandit.read();
            let bstate: Vec<(Arm, (f64, f64))> = b.state().iter().map(|(k, v)| (*k, (v.alpha, v.beta))).collect();
            store.save_bandit(&bstate, self.seed)?;
        }
        {
            let up = self.user_policy.read();
            store.save_policy(&up.tracks, up.tau_hours, up.shrink)?;
        }
        {
            let gp = self.global_policy.read();
            store.save_global_policy(&gp.artists, &gp.genres, &gp.global, gp.tau_hours, gp.shrink)?;
        }
        Ok(())
    }

    pub fn ingest(&self, ev: Event, ctx: TemporalContext) -> Result<()> {
        let bg = if ctx.focused { 1.0 } else { self.cfg.background_weight };
        match ev {
            Event::PlayStarted { track, ts } => {
                let t = self.catalog.get(track)?;
                self.profile.write().on_neutral(&t, ts);
                self.session.apply(&t, 0.0, false, ctx, ts);
            }
            Event::PlayCompleted { track, ts, played_sec } => {
                let t = self.catalog.get(track)?;
                let reward = reward_from_listen(played_sec, t.duration_sec, false, false) * bg;
                self.user_policy  .write().observe(t.id, reward, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, reward, ts);
                self.session.apply(&t, reward * 1.5, false, ctx, ts);
                self.profile.write().on_positive(&t, ts, reward, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), reward);
                self.bandit.write().update(Arm::Artist(t.artist_id.raw()), reward);
                self.session.set_anchor(t.id, 1.0, ts);
                self.metrics.record_outcome(t.id, TrackOutcome::Completed);

                let plays = self.profile.read().plays.get(&t.id).copied().unwrap_or(0);
                self.coldpool.maybe_promote(&t, plays, self.cfg.cold_play_threshold, ts);

                let prev: Vec<TrackId> = self.session.snapshot().recent.iter().rev().take(2).map(|e| e.track).collect();
                if prev.len() >= 2 {
                    let p = prev[1];
                    self.markov.write().observe(p, t.id);
                    self.profile.write().record_co_occurrence(p, t.id, true);
                }
            }
            Event::Liked { track, ts } => {
                let t = self.catalog.get(track)?;
                let reward = 1.0 * bg;
                self.user_policy  .write().observe(t.id, reward, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, reward, ts);
                self.session.apply(&t, 1.8 * bg, false, ctx, ts);
                self.profile.write().on_positive(&t, ts, 2.0 * bg, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), reward);
                self.bandit.write().update(Arm::Artist(t.artist_id.raw()), reward);
                self.session.set_anchor(t.id, 2.0, ts);
                self.metrics.record_outcome(t.id, TrackOutcome::Completed);
                let plays = self.profile.read().plays.get(&t.id).copied().unwrap_or(0);
                self.coldpool.maybe_promote(&t, plays, self.cfg.cold_play_threshold, ts);
            }
            Event::Disliked { track, ts } => {
                let t = self.catalog.get(track)?;
                self.user_policy  .write().observe(t.id, 0.0, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, 0.0, ts);
                self.session.apply(&t, -1.5, true, ctx, ts);
                self.profile.write().on_skip(&t, ts, true, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), 0.0);
                self.metrics.record_outcome(t.id, TrackOutcome::Skipped);
            }
            Event::Skipped { track, ts, position_sec } => {
                let t = self.catalog.get(track)?;
                let reward = reward_from_listen(position_sec, t.duration_sec, false, false);
                let severe = reward < 0.1;
                self.user_policy  .write().observe(t.id, reward, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, reward, ts);
                self.session.apply(&t, reward - 1.0, true, ctx, ts);
                self.profile.write().on_skip(&t, ts, severe, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), reward);
                self.metrics.record_outcome(t.id, TrackOutcome::Skipped);
                if severe { self.markov.write().rest(t.id, ts + 3600); }
                let plays = self.profile.read().plays.get(&t.id).copied().unwrap_or(0);
                self.coldpool.maybe_promote(&t, plays, self.cfg.cold_play_threshold, ts);
            }
            Event::Seeked { track, ts, from_sec, to_sec } => {
                let t = self.catalog.get(track)?;
                self.profile.write().on_seek(&t, ts, from_sec, to_sec, t.duration_sec);
                if t.duration_sec > 0.0 && to_sec / t.duration_sec > 0.9 {
                    self.session.apply(&t, -0.5, true, ctx, ts);
                }
            }
            Event::SearchQuery { track, ts } => {
                let t = self.catalog.get(track)?;
                self.user_policy  .write().observe(t.id, 1.0, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, 1.0, ts);
                self.session.apply(&t, 2.5, false, ctx, ts);
                self.profile.write().on_positive(&t, ts, 3.0, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), 1.0);
                self.session.set_anchor(t.id, 2.5, ts);
            }
            Event::PlaylistAdd { track, ts } => {
                let t = self.catalog.get(track)?;
                self.user_policy  .write().observe(t.id, 1.0, ts);
                self.global_policy.write().observe(t.artist_id, t.genre_id, 1.0, ts);
                self.session.apply(&t, 2.0, false, ctx, ts);
                self.profile.write().on_positive(&t, ts, 2.5, ctx);
                self.bandit.write().update(Arm::Genre(t.genre_id.raw() as u32), 1.0);
                self.session.set_anchor(t.id, 2.0, ts);
            }
            Event::FocusChange { track, ts, focused } => {
                let t = self.catalog.get(track)?;
                let mut c = ctx; c.focused = focused;
                self.profile.write().on_neutral(&t, ts);
                self.session.apply(&t, 0.0, false, c, ts);
            }
            Event::AlbumSkip { album, ts } => {
                let tracks = self.catalog.tracks_by_album(album);
                for &tid in tracks.iter().take(8) {
                    if let Ok(t) = self.catalog.get(tid) {
                        self.session.apply(&t, -0.3, true, ctx, ts);
                        self.profile.write().on_skip(&t, ts, false, ctx);
                    }
                }
            }
            Event::PlaylistSkip { ts: _ } => {
                let recent: Vec<TrackId> = self.session.snapshot().recent.iter().rev().take(5).map(|e| e.track).collect();
                for r in recent {
                    if let Ok(t) = self.catalog.get(r) {
                        self.profile.write().on_skip(&t, 0, false, ctx);
                    }
                }
            }
        }

        let n = self.ingest_n.fetch_add(1, Ordering::Relaxed) + 1;
        let p = &self.cfg.persistence;
        if n % p.profile_every_n == 0 { let _ = self.flush(); }
        Ok(())
    }

    pub fn compact_cooccurrence(&self, top_k: usize) { self.cooc.write().compact(top_k, self.cfg.pmi_smoothing); }
    pub fn cleanup(&self, now: i64) {
        self.session.cleanup_rest(now);
        self.markov.write().cleanup(now);
        self.coldpool.cleanup_old(now);
    }

    pub fn generate(&self, now: i64, k: usize) -> Result<Vec<TrackId>> {
        let started = Instant::now();
        if self.catalog.is_empty() { return Err(WaveError::EmptyCatalog); }
        if self.catalog.needs_rebuild() { self.catalog.rebuild_indices()?; }

        let cfg = &self.cfg;
        let k = k.min(cfg.wave_len).min(self.catalog.len());
        let pool_size = cfg.pool_per_source;

        let sess_snap    = self.session.snapshot();
        let prof_snap    = self.profile.read().snapshot();
        let cooc_snap    = self.cooc.read().snapshot();
        let markov_snap  = self.markov.read().snapshot();
        let usermap_snap = self.usermap.read().snapshot();
        let bandit_snap  = self.bandit.read().snapshot();
        let user_policy_snap   = self.user_policy.read().snapshot();
        let global_policy_snap = self.global_policy.read().snapshot();

        let short_vec = sess_snap.short_vec.clone();
        let long_vec  = sess_snap.long_vec.clone();
        let recent: Vec<TrackId> = sess_snap.recent.iter().rev().take(10).map(|e| e.track).collect();
        let skip_streak = sess_snap.skip_streak;
        let drift = sess_snap.drift();
        let last_ctx = sess_snap.last_ctx;

        let mut pool: Vec<Candidate> = Vec::with_capacity(pool_size * 8);
        let mut seen: FxHashSet<TrackId> = FxHashSet::with_capacity(pool_size * 8);
        let mut rng = SmallRng::seed_from_u64(self.seed ^ now as u64 ^ k as u64);

        {
            let src = Sources {
                catalog:  &self.catalog,
                cooc:     &cooc_snap,
                markov:   &markov_snap,
                usermap:  &usermap_snap,
                coldpool: &self.coldpool,
            };
            let mut take = |v: SmallVec<[Candidate; 64]>, pool: &mut Vec<Candidate>, seen: &mut FxHashSet<TrackId>| {
                for c in v {
                    let Some(c) = src.filter_resting(c, &sess_snap, now) else { continue };
                    if seen.insert(c.track) { pool.push(c); }
                }
            };

            take(src.content(&short_vec, pool_size), &mut pool, &mut seen);
            take(src.content(&long_vec,  pool_size / 2), &mut pool, &mut seen);
            take(src.intro_content(&short_vec, pool_size / 3), &mut pool, &mut seen);
            take(src.alt_content(&short_vec, pool_size / 3), &mut pool, &mut seen);
            take(src.chunk_content(&short_vec, pool_size / 3), &mut pool, &mut seen);
            take(src.from_cooc(&recent, pool_size / 2), &mut pool, &mut seen);
            take(src.from_markov(&recent, pool_size / 4), &mut pool, &mut seen);
            take(src.from_artist(&prof_snap.artist_affinity, pool_size), &mut pool, &mut seen);
            take(src.from_genre(&prof_snap.genre_affinity, pool_size), &mut pool, &mut seen);
            take(src.anchor(&sess_snap, now, 10), &mut pool, &mut seen);

            let mut explore_frac = cfg.explore_fraction;
            if skip_streak >= cfg.skip_streak_threshold { explore_frac += cfg.skip_streak_explore_boost; }
            if drift > cfg.drift_threshold { explore_frac += 0.10; }
            let explore_k = ((k as f32) * explore_frac.min(0.5)).ceil() as usize;
            take(src.explore(explore_k.max(3), &mut rng), &mut pool, &mut seen);

            let cold_target = self.coldpool.target_count(k, cfg);
            if cold_target > 0 {
                let recent_set: FxHashSet<TrackId> = recent.iter().copied().collect();
                take(src.from_coldpool(&recent_set, cold_target * 2, now, &mut rng), &mut pool, &mut seen);
            }

            let user_id = self.session.user_id();
            let recent_clone = recent.clone();
            take(src.from_user_knn(user_id, |t| recent_clone.contains(&t), pool_size / 3), &mut pool, &mut seen);
        }

        let mut rctx = RankerCtx {
            profile:       &prof_snap,
            session:       &sess_snap,
            cooc:          &cooc_snap,
            markov:        &markov_snap,
            bandit:        &bandit_snap,
            user_policy:   &user_policy_snap,
            global_policy: &global_policy_snap,
            cfg,
            now,
            ctx: last_ctx,
            rng: &mut rng,
        };

        let mut scored: Vec<Scored> = Vec::with_capacity(pool.len());
        for c in pool {
            let t = self.catalog.get(c.track)?;
            let s = score(&c, &t, &mut rctx);
            scored.push(Scored { track: t, score: s });
        }
        drop(rctx);

        let out = if cfg.dpp_enabled { dpp(scored, k, cfg.dpp_epsilon) } else { mmr(scored, k, cfg) };

        if out.len() < k {
            self.metrics.record_insufficient();
            tracing::warn!(requested = k, got = out.len(), "insufficient candidates");
        }

        let diversity = {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for i in 0..out.len() {
                for j in (i + 1)..out.len() {
                    sum += 1.0 - cosine_normalized(&out[i].track.embedding, &out[j].track.embedding);
                    n += 1;
                }
            }
            if n > 0 { sum / n as f32 } else { 0.0 }
        };

        let ids: Vec<TrackId> = out.into_iter().map(|s| s.track.id).collect();
        let latency_us = started.elapsed().as_micros() as u64;
        self.metrics.record_wave(now, ids.clone(), diversity, latency_us);

        Ok(ids)
    }
}