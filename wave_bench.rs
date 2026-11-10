use std::sync::Arc;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
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

fn build(dim: usize, n: u64) -> Engine {
    let mut cat = Catalog::new(dim, HnswConfig { max_elements: n as usize + 1, ..Default::default() });
    for i in 0..n {
        let emb: Vec<f32> = (0..dim).map(|d| ((i as f32) * 0.01 + d as f32).sin()).collect();
        cat.insert(mk_track(i, i / 8, i % 16, emb)).unwrap();
    }
    let cat = Arc::new(cat);
    let cfg = Arc::new(WaveConfig { dim, wave_len: 30, ..Default::default() });
    let e = Engine::new(cfg, cat, UserId(1)).unwrap();
    // прогрев
    for i in 0..20 { let _ = e.ingest(Event::Liked { track: TrackId(i), ts: 10 }, TemporalContext::default()); }
    e
}

fn bench_generate(c: &mut Criterion) {
    let mut group = c.benchmark_group("wave_generate");
    for &(label, dim, n) in &[("1k_64d", 64usize, 1_000u64), ("10k_256d", 256, 10_000), ("50k_512d", 512, 50_000)] {
        let e = build(dim, n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::new("generate", label), &e, |b, engine| {
            b.iter(|| {
                let w = engine.generate(black_box(1000), black_box(30)).unwrap();
                black_box(w.len());
            });
        });
    }
    group.finish();
}

fn bench_ingest(c: &mut Criterion) {
    let e = build(64, 5_000);
    c.bench_function("wave_ingest", |b| {
        b.iter(|| {
            let _ = e.ingest(black_box(Event::Skipped { track: TrackId(3), ts: 10, position_sec: 1.0 }), TemporalContext::default());
        });
    });
}

criterion_group!(benches, bench_generate, bench_ingest);
criterion_main!(benches);