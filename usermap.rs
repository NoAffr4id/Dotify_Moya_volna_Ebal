use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::ids::{TrackId, UserId};
use crate::snapshot::UserKnnSnapshot;

#[derive(Serialize, Deserialize, Default)]
pub struct UserKnn {
    users: FxHashMap<UserId, FxHashMap<TrackId, f32>>,
}

impl UserKnn {
    pub fn new() -> Self { Self::default() }
    pub fn snapshot(&self) -> UserKnnSnapshot {
        UserKnnSnapshot { users: Arc::new(self.users.clone()) }
    }
    pub fn upsert(&mut self, u: UserId, history: FxHashMap<TrackId, f32>) { self.users.insert(u, history); }
    pub fn remove(&mut self, u: UserId) { self.users.remove(&u); }
    pub fn users(&self) -> &FxHashMap<UserId, FxHashMap<TrackId, f32>> { &self.users }
    pub fn users_iter(&self) -> impl Iterator<Item = (&UserId, &FxHashMap<TrackId, f32>)> {
        self.users.iter()
    }
    pub fn len(&self) -> usize { self.users.len() }
}

pub fn recommend_snapshot(
    snap: &UserKnnSnapshot,
    u: UserId,
    k: usize,
    exclude: impl Fn(TrackId) -> bool,
) -> Vec<(TrackId, f32)> {
    let Some(me) = snap.users.get(&u) else { return Vec::new() };
    let n1: f32 = me.values().map(|w| w * w).sum::<f32>().sqrt();
    if n1 < 1e-6 { return Vec::new(); }

    let mut sims: Vec<(UserId, f32)> = Vec::with_capacity(snap.users.len());
    for (&other, hist) in snap.users.iter() {
        if other == u { continue; }
        let mut dot = 0.0f32;
        let mut n2 = 0.0f32;
        for (_, w) in hist.iter() { n2 += w * w; }
        for (t, w1) in me.iter() {
            if let Some(w2) = hist.get(t) { dot += w1 * w2; }
        }
        n2 = n2.sqrt();
        if n1 * n2 > 1e-6 { sims.push((other, dot / (n1 * n2))); }
    }
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.truncate(20);

    let mut scores: FxHashMap<TrackId, f32> = FxHashMap::default();
    for (other, sim) in sims {
        if let Some(hist) = snap.users.get(&other) {
            for (&t, &w) in hist.iter() {
                if exclude(t) { continue; }
                if me.contains_key(&t) { continue; }
                *scores.entry(t).or_insert(0.0) += sim * w;
            }
        }
    }
    let mut out: Vec<(TrackId, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    out
}