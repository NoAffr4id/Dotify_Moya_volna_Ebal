use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use crate::ids::TrackId;

const MAX_WAVES: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrackOutcome { Skipped, Completed }

struct WaveRecord {
    ts:        i64,
    tracks:    Vec<TrackId>,
    status:    Vec<u8>,     // 0=pending, 1=skipped, 2=completed
    diversity: f32,
    latency_us: u64,
}

impl WaveRecord {
    fn mark(&mut self, track: TrackId, status: u8) {
        for (i, t) in self.tracks.iter().enumerate() {
            if *t == track && self.status[i] == 0 {
                self.status[i] = status;
                return;
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    pub waves_generated:       u64,
    pub total_tracks_shown:    u64,
    pub total_skipped:         u64,
    pub total_completed:       u64,
    pub insufficient_candidates: u64,
    pub diversity_ema:         f32,
    pub latency_us_ema:        f32,
    pub skip_at_k:             f32,
    pub completion_at_k:       f32,
    pub diversity_at_k:        f32,
}

pub struct WaveMetrics {
    waves_generated:       AtomicU64,
    total_tracks_shown:    AtomicU64,
    total_skipped:         AtomicU64,
    total_completed:       AtomicU64,
    insufficient_candidates: AtomicU64,
    diversity_ema_bits:    AtomicU64,   // f32 as bits
    latency_us_ema_bits:   AtomicU64,
    recent_waves:          Mutex<VecDeque<WaveRecord>>,
}

impl Default for WaveMetrics {
    fn default() -> Self {
        Self {
            waves_generated:         AtomicU64::new(0),
            total_tracks_shown:      AtomicU64::new(0),
            total_skipped:           AtomicU64::new(0),
            total_completed:         AtomicU64::new(0),
            insufficient_candidates: AtomicU64::new(0),
            diversity_ema_bits:      AtomicU64::new(0f32.to_bits() as u64),
            latency_us_ema_bits:     AtomicU64::new(0f32.to_bits() as u64),
            recent_waves:            Mutex::new(VecDeque::with_capacity(MAX_WAVES)),
        }
    }
}

impl WaveMetrics {
    pub fn new() -> Self { Self::default() }

    fn ema_update(slot: &AtomicU64, new_val: f32, alpha: f32) {
        let old_bits = slot.load(Ordering::Relaxed);
        let old = f32::from_bits(old_bits as u32);
        let merged = if old == 0.0 { new_val } else { alpha * new_val + (1.0 - alpha) * old };
        slot.store(merged.to_bits() as u64, Ordering::Relaxed);
    }

    pub fn record_wave(&self, ts: i64, tracks: Vec<TrackId>, diversity: f32, latency_us: u64) {
        self.waves_generated.fetch_add(1, Ordering::Relaxed);
        self.total_tracks_shown.fetch_add(tracks.len() as u64, Ordering::Relaxed);
        Self::ema_update(&self.diversity_ema_bits,  diversity,     0.05);
        Self::ema_update(&self.latency_us_ema_bits, latency_us as f32, 0.05);

        let mut g = self.recent_waves.lock();
        if g.len() >= MAX_WAVES { g.pop_front(); }
        g.push_back(WaveRecord {
            ts,
            status: vec![0u8; tracks.len()],
            tracks,
            diversity,
            latency_us,
        });
    }

    pub fn record_outcome(&self, track: TrackId, kind: TrackOutcome) {
        match kind {
            TrackOutcome::Skipped   => { self.total_skipped  .fetch_add(1, Ordering::Relaxed); }
            TrackOutcome::Completed => { self.total_completed.fetch_add(1, Ordering::Relaxed); }
        }
        let code = match kind {
            TrackOutcome::Skipped   => 1,
            TrackOutcome::Completed => 2,
        };
        let mut g = self.recent_waves.lock();
        for w in g.iter_mut().rev() {
            w.mark(track, code);
        }
    }

    pub fn record_insufficient(&self) { self.insufficient_candidates.fetch_add(1, Ordering::Relaxed); }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let g = self.recent_waves.lock();
        let mut shown = 0usize;
        let mut skipped = 0usize;
        let mut completed = 0usize;
        let mut div_sum = 0.0f32;
        for w in g.iter() {
            shown += w.tracks.len();
            for s in w.status.iter() {
                if *s == 1 { skipped += 1; }
                else if *s == 2 { completed += 1; }
            }
            div_sum += w.diversity;
        }
        let denom = shown.max(1) as f32;
        MetricsSnapshot {
            waves_generated:         self.waves_generated.load(Ordering::Relaxed),
            total_tracks_shown:      self.total_tracks_shown.load(Ordering::Relaxed),
            total_skipped:           self.total_skipped.load(Ordering::Relaxed),
            total_completed:         self.total_completed.load(Ordering::Relaxed),
            insufficient_candidates: self.insufficient_candidates.load(Ordering::Relaxed),
            diversity_ema:           f32::from_bits(self.diversity_ema_bits.load(Ordering::Relaxed) as u32),
            latency_us_ema:          f32::from_bits(self.latency_us_ema_bits.load(Ordering::Relaxed) as u32),
            skip_at_k:               skipped as f32 / denom,
            completion_at_k:         completed as f32 / denom,
            diversity_at_k:          div_sum / g.len().max(1) as f32,
        }
    }
}