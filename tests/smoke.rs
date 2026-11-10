use std::sync::Arc;
use dotify_wave::*;

fn mk_track(id: u64, artist: u64, genre: u64, emb: Vec<f32>) -> Track {
    Track {
        id: TrackId(id),
        artist_id: ArtistId(artist),
        album_id: Some(AlbumId(artist)),
        genre_id: GenreId(genre),
        language: Some(LangId(1)),
        release_year: 2024,
        duration_sec: 240.0,
        popularity: 0.5,
        popularity_pct: 0.5,
        bpm: Some(120.0),
        key_camelot: Some(8),
        vocal: Some(0.5),
        energy_curve: None,
        embedding: Arc::from(emb),
        alt_embedding: None,
        intro_embedding: None,
        chunk_embeddings: None,
    }
}

#[test]
fn wave_generates_distinct_tracks() {
    let dim = 8;
    let mut cat = Catalog::new(dim);
    for i in 0..80u64 {
        let emb: Vec<f32> = (0..dim).map(|d| ((i + d as u64) as f32).sin()).collect();
        cat.insert(mk_track(i, i / 5, i % 6, emb)).unwrap();
    }
    let cfg = Arc::new(WaveConfig { dim, wave_len: 10, ..Default::default() });
    let engine = Engine::new(cfg, Arc::new(cat), UserId(1));
    let ctx = TemporalContext { hour: 10, weekday: 1, device: Device::Mobile, activity: Activity::Commute, focused: true };

    for i in 0..10 {
        engine.ingest(Event::Liked { track: TrackId(i), ts: 100 + i as i64 }, ctx).unwrap();
    }
    for i in 10..15 {
        engine.ingest(Event::Skipped { track: TrackId(i), ts: 200 + i as i64, position_sec: 2.0 }, ctx).unwrap();
    }

    let wave = engine.generate(300, 10).unwrap();
    assert_eq!(wave.len(), 10);
    let unique: std::collections::HashSet<_> = wave.iter().collect();
    assert_eq!(unique.len(), 10);
}

#[test]
fn dim_mismatch_rejected() {
    let mut cat = Catalog::new(8);
    let r = cat.insert(mk_track(1, 1, 1, vec![0.0; 4]));
    assert!(r.is_err());
}

#[test]
fn empty_catalog_errors() {
    let cat = Catalog::new(8);
    let cfg = Arc::new(WaveConfig { dim: 8, wave_len: 10, ..Default::default() });
    let engine = Engine::new(cfg, Arc::new(cat), UserId(1));
    assert!(engine.generate(0, 10).is_err());
}

#[test]
fn cold_start_returns_something() {
    let dim = 8;
    let mut cat = Catalog::new(dim);
    for i in 0..20u64 {
        let emb: Vec<f32> = (0..dim).map(|d| ((i + d as u64) as f32).cos()).collect();
        cat.insert(mk_track(i, i, i % 3, emb)).unwrap();
    }
    let cfg = Arc::new(WaveConfig { dim, wave_len: 5, ..Default::default() });
    let engine = Engine::new(cfg, Arc::new(cat), UserId(1));
    let wave = engine.generate(0, 5).unwrap();
    assert_eq!(wave.len(), 5);
}

#[test]
fn skip_streak_increases_explore() {
    let dim = 8;
    let mut cat = Catalog::new(dim);
    for i in 0..60u64 {
        let emb: Vec<f32> = (0..dim).map(|d| ((i * d) as f32).sin()).collect();
        cat.insert(mk_track(i, i / 3, i % 4, emb)).unwrap();
    }
    let cfg = Arc::new(WaveConfig { dim, wave_len: 10, ..Default::default() });
    let engine = Engine::new(cfg, Arc::new(cat), UserId(1));
    let ctx = TemporalContext::default();

    engine.ingest(Event::Liked { track: TrackId(0), ts: 10 }, ctx).unwrap();
    let _ = engine.generate(20, 5).unwrap();

    for i in 1..6 {
        engine.ingest(Event::Skipped { track: TrackId(i), ts: 100 + i as i64, position_sec: 1.0 }, ctx).unwrap();
    }
    assert!(engine.session().skip_streak() >= 3);
    let _ = engine.generate(200, 5).unwrap();
}

#[test]
fn dpp_mode_works() {
    let dim = 8;
    let mut cat = Catalog::new(dim);
    for i in 0..40u64 {
        let emb: Vec<f32> = (0..dim).map(|d| ((i + d as u64) as f32).sin()).collect();
        cat.insert(mk_track(i, i / 2, i % 3, emb)).unwrap();
    }
    let cfg = Arc::new(WaveConfig { dim, wave_len: 6, dpp_enabled: true, ..Default::default() });
    let engine = Engine::new(cfg, Arc::new(cat), UserId(1));
    let wave = engine.generate(0, 6).unwrap();
    assert_eq!(wave.len(), 6);
}