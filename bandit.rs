use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Beta, Distribution};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::context::Device;
use crate::snapshot::BanditSnapshot;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arm {
    Source(u8),
    Genre(u32),
    GenreHour(u32, u8),
    GenreDevice(u32, Device),
    Artist(u64),
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BetaState { pub alpha: f64, pub beta: f64 }

pub struct ThompsonBandit {
    state: FxHashMap<Arm, BetaState>,
    rng:   rand::rngs::SmallRng,
}

impl ThompsonBandit {
    pub fn new(seed: u64) -> Self {
        Self { state: FxHashMap::default(), rng: rand::rngs::SmallRng::seed_from_u64(seed) }
    }
    pub fn snapshot(&self) -> BanditSnapshot {
        BanditSnapshot {
            state: Arc::new(self.state.iter().map(|(k, v)| (*k, (v.alpha, v.beta))).collect()),
        }
    }
    pub fn sample(&mut self, arm: Arm) -> f64 {
        let s = self.state.get(&arm).copied().unwrap_or(BetaState { alpha: 1.0, beta: 1.0 });
        Beta::new(s.alpha, s.beta).map(|d| d.sample(&mut self.rng)).unwrap_or(0.5)
    }
    pub fn update(&mut self, arm: Arm, reward: f32) {
        let r = (reward as f64).clamp(0.0, 1.0);
        let e = self.state.entry(arm).or_insert(BetaState { alpha: 1.0, beta: 1.0 });
        e.alpha += r;
        e.beta  += 1.0 - r;
    }
    pub fn state(&self) -> &FxHashMap<Arm, BetaState> { &self.state }
    pub fn restore(state: FxHashMap<Arm, BetaState>, seed: u64) -> Self {
        Self { state, rng: rand::rngs::SmallRng::seed_from_u64(seed) }
    }
}

pub fn sample_from_snapshot(snap: &BanditSnapshot, arm: Arm, rng: &mut impl Rng) -> f32 {
    let (a, b) = snap.state.get(&arm).copied().unwrap_or((1.0, 1.0));
    Beta::new(a, b).map(|d| d.sample(rng) as f32).unwrap_or(0.5)
}