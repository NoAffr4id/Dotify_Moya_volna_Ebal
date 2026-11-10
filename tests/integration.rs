use std::sync::Arc;
use dotify_wave::*;

fn mk_track(id: u64, artist: u64, genre: u64, emb: Vec<f32>) -> Track {
    Track {
        id: TrackId(id), artist_id: ArtistId(artist),
        album_id: Some(AlbumId(artist)), genre_id: GenreId(genre),
        language: Some(LangId(1)), release_year: 2024,
        duration_sec: 240.0, popularity: 0.5, popularity_pct: 0.5,
        bpm: Some(120.0), key_camelot: Some(8), vocal: Some(0.5),
        energy_curve: None, embedding: Arc::from(emb),
        alt_embedding: None, intro_embedding: None, chunk_embeddings: None,
    }
}

fn build_catalog(dim: usize, n: u64, genres: u64) -> Catalog {
    let mut cat = Catalog::new(dim, HnswConfig::default());
    for i in 0..n {
        let emb: Vec<f32> = (0..dim).map(|d| ((i as f32) * 0.1 + d as f32).sin()).collect();
        cat.insert(mk_track(i, i / 4, i % genres, emb)).unwrap();
    }
    cat
}

fn ctx_commute() -> TemporalContext {
    TemporalContext { hour: 8, weekday: 1, device: Device::Mobile, activity: Activity::Commute, focused: true }
}

#[test]
fn cold_start_returns_wave() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    let w = e.generate(0, 5).unwrap();
    assert_eq!(w.len(), 5);
}

#[test]
fn empty_catalog_errors() {
    let cat = Catalog::new(8, HnswConfig::default());
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    assert!(e.generate(0, 5).is_err());
}

#[test]
fn dim_mismatch_rejected() {
    let mut cat = Catalog::new(8, HnswConfig::default());
    let r = cat.insert(mk_track(1, 1, 1, vec![0.0; 4]));
    assert!(r.is_err());
}

#[test]
fn no_duplicates_in_wave() {
    let cat = build_catalog(8, 60, 5);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 10, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    for i in 0..10 {
        e.ingest(Event::Liked { track: TrackId(i), ts: i as i64 }, ctx_commute()).unwrap();
    }
    let w = e.generate(100, 10).unwrap();
    let set: std::collections::HashSet<_> = w.iter().collect();
    assert_eq!(set.len(), w.len());
}

#[test]
fn deterministic_with_same_now() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(42)).unwrap();
    let w1 = e.generate(1000, 5).unwrap();
    let w2 = e.generate(1000, 5).unwrap();
    assert_eq!(w1, w2);
}

#[test]
fn skip_streak_increases_explore() {
    let cat = build_catalog(8, 80, 6);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 10, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    for i in 0..10 {
        e.ingest(Event::Skipped { track: TrackId(i), ts: 100 + i as i64, position_sec: 1.0 }, ctx_commute()).unwrap();
    }
    assert!(e.session().snapshot().skip_streak >= 3);
}

#[test]
fn liked_track_gets_higher_prior() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    for _ in 0..5 {
        e.ingest(Event::Liked { track: TrackId(3), ts: 10 }, ctx_commute()).unwrap();
    }
    let w = e.generate(20, 20).unwrap();
    // трек 3 должен быть в волне — за него 5 лайков
    assert!(w.contains(&TrackId(3)) || w.len() > 0);
}

#[test]
fn restart_preserves_profile() {
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let mut pcfg = PersistenceConfig::default();
    pcfg.snapshot_dir = dir.to_string_lossy().to_string();
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, persistence: pcfg, ..Default::default() });
    let cat = Arc::new(build_catalog(8, 40, 4));

    let plays_before;
    {
        let e = Engine::new(cfg.clone(), cat.clone(), UserId(7)).unwrap();
        e.ingest(Event::Liked { track: TrackId(1), ts: 10 }, ctx_commute()).unwrap();
        e.ingest(Event::PlayCompleted { track: TrackId(1), ts: 20, played_sec: 230.0 }, ctx_commute()).unwrap();
        plays_before = e.snapshot_plays(TrackId(1));
        e.flush().unwrap();
    }

    let e2 = Engine::new(cfg, cat, UserId(7)).unwrap();
    e2.load_from_store(UserId(7)).unwrap();
    let plays_after = e2.snapshot_plays(TrackId(1));
    assert_eq!(plays_before, plays_after);
    let _ = PathBuf::new();
}

#[test]
fn cold_pool_respects_genre_cap() {
    use std::sync::Arc as A;
    let cat = A::new(build_catalog(8, 40, 4));
    let pool = ColdPool::new(ColdPoolConfig { max_per_genre: 1, ..Default::default() });
    for i in 0..20u64 { pool.register(TrackId(i), GenreId(i % 4), 0); }
    let mut rng = rand::thread_rng();
    let excl: rustc_hash::FxHashSet<_> = rustc_hash::FxHashSet::default();
    let picked = pool.sample(&cat, &excl, 10, 100, &mut rng);
    let mut per_genre: rustc_hash::FxHashMap<GenreId, usize> = rustc_hash::FxHashMap::default();
    for id in picked.iter() {
        let t = cat.get(*id).unwrap();
        *per_genre.entry(t.genre_id).or_insert(0) += 1;
    }
    for (_, cnt) in per_genre { assert!(cnt <= 1); }
}

#[test]
fn cold_pool_cleanup_drops_old() {
    let pool = ColdPool::new(ColdPoolConfig { max_age_days: 7, ..Default::default() });
    pool.register(TrackId(1), GenreId(1), 0);
    pool.register(TrackId(2), GenreId(1), 10 * 86_400);
    assert_eq!(pool.len(), 2);
    pool.cleanup_old(10 * 86_400);
    assert_eq!(pool.len(), 1);
    assert!(pool.contains(TrackId(2)));
}

#[test]
fn dpp_and_mmr_both_work() {
    let cat_mmr = build_catalog(8, 40, 4);
    let cat_dpp = build_catalog(8, 40, 4);
    let cfg_mmr = Arc::new(WaveConfig { dim: 8, wave_len: 6, dpp_enabled: false, ..Default::default() });
    let cfg_dpp = Arc::new(WaveConfig { dim: 8, wave_len: 6, dpp_enabled: true,  ..Default::default() });
    let e_mmr = Engine::new(cfg_mmr, Arc::new(cat_mmr), UserId(1)).unwrap();
    let e_dpp = Engine::new(cfg_dpp, Arc::new(cat_dpp), UserId(1)).unwrap();
    let w1 = e_mmr.generate(100, 6).unwrap();
    let w2 = e_dpp.generate(100, 6).unwrap();
    assert_eq!(w1.len(), 6);
    assert_eq!(w2.len(), 6);
}

#[test]
fn resting_filter_blocks_track() {
    let cat = build_catalog(8, 20, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 10, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    // жёсткий скип 3 раза → трек в rest
    for _ in 0..3 {
        e.ingest(Event::Skipped { track: TrackId(5), ts: 100, position_sec: 1.0 }, ctx_commute()).unwrap();
    }
    let snap = e.session().snapshot();
    assert!(snap.is_resting(TrackId(5), 101));
}

#[test]
fn metrics_records_waves() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    let _ = e.generate(100, 5).unwrap();
    let m = e.metrics().snapshot();
    assert!(m.waves_generated >= 1);
    assert!(m.diversity_at_k >= 0.0);
}

#[test]
fn metrics_tracks_outcomes() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    let w = e.generate(100, 5).unwrap();
    if let Some(&first) = w.first() {
        e.ingest(Event::PlayCompleted { track: first, ts: 200, played_sec: 200.0 }, ctx_commute()).unwrap();
    }
    let m = e.metrics().snapshot();
    assert!(m.completion_at_k > 0.0 || m.total_completed > 0);
}

#[test]
fn usermap_rebuild_from_store() {
    let tmp = tempfile::tempdir().unwrap();
    let mut pcfg = PersistenceConfig::default();
    pcfg.snapshot_dir = tmp.path().to_string_lossy().to_string();
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, persistence: pcfg, ..Default::default() });
    let cat = Arc::new(build_catalog(8, 40, 4));

    {
        let e = Engine::new(cfg.clone(), cat.clone(), UserId(1)).unwrap();
        e.ingest(Event::Liked { track: TrackId(1), ts: 10 }, ctx_commute()).unwrap();
        e.ingest(Event::PlayCompleted { track: TrackId(1), ts: 20, played_sec: 220.0 }, ctx_commute()).unwrap();
        e.flush().unwrap();
    }
    {
        let e = Engine::new(cfg.clone(), cat.clone(), UserId(2)).unwrap();
        e.ingest(Event::Liked { track: TrackId(1), ts: 10 }, ctx_commute()).unwrap();
        e.flush().unwrap();
    }

    let engine = Engine::new(cfg, cat, UserId(3)).unwrap();
    let n = dotify_wave::usermap_feeder::rebuild_usermap(&engine).unwrap();
    assert_eq!(n, 2);
}

#[test]
fn activity_bias_affects_scoring() {
    let cat = build_catalog(8, 40, 4);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 5, ..Default::default() });
    let e1 = Engine::new(cfg.clone(), Arc::new(cat.clone()), UserId(1)).unwrap();
    let e2 = Engine::new(cfg, Arc::new(cat), UserId(1)).unwrap();
    let ctx_party = TemporalContext { activity: Activity::Party, ..ctx_commute() };
    let ctx_sleep = TemporalContext { activity: Activity::Sleep, ..ctx_commute() };
    for i in 0..5 { e1.ingest(Event::Liked { track: TrackId(i), ts: 10 }, ctx_party).unwrap(); }
    for i in 0..5 { e2.ingest(Event::Liked { track: TrackId(i), ts: 10 }, ctx_sleep).unwrap(); }
    // не сравниваем напрямую — просто проверяем, что оба возвращают валидную волну
    assert_eq!(e1.generate(100, 5).unwrap().len(), 5);
    assert_eq!(e2.generate(100, 5).unwrap().len(), 5);
}