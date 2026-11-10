use std::sync::Arc;
use std::time::Duration;
use rustc_hash::FxHashMap;
use crate::engine::Engine;
use crate::error::Result;
use crate::ids::{TrackId, UserId};
use crate::profile::UserProfile;
use crate::usermap::UserKnn;

pub fn rebuild_usermap(engine: &Engine) -> Result<usize> {
    let Some(store) = engine.store() else { return Ok(0) };
    let profiles = store.load_all_profiles()?;
    let mut new_map = UserKnn::new();
    for p in profiles {
        let mut hist: FxHashMap<TrackId, f32> = FxHashMap::default();
        for (&tid, &plays) in p.plays.iter() {
            if plays > 0 {
                let w = (plays as f32).min(5.0) / 5.0;
                hist.insert(tid, w);
            }
        }
        if !hist.is_empty() {
            new_map.upsert(p.user_id, hist);
        }
    }
    let n = new_map.len();
    engine.replace_usermap(new_map);
    Ok(n)
}

pub fn rebuild_from_profiles(engine: &Engine, profiles: &[(UserId, &UserProfile)]) {
    let mut new_map = UserKnn::new();
    for (uid, p) in profiles {
        let mut hist: FxHashMap<TrackId, f32> = FxHashMap::default();
        for (&tid, &plays) in p.plays.iter() {
            if plays > 0 {
                hist.insert(tid, (plays as f32).min(5.0) / 5.0);
            }
        }
        if !hist.is_empty() {
            new_map.upsert(*uid, hist);
        }
    }
    engine.replace_usermap(new_map);
}

pub fn spawn_usermap_feeder(engine: Arc<Engine>, interval_secs: u64) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("usermap-feeder".into())
        .spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(interval_secs));
                match rebuild_usermap(&engine) {
                    Ok(n) => tracing::info!(users = n, "usermap rebuilt"),
                    Err(e) => tracing::warn!(err = ?e, "usermap rebuild failed"),
                }
            }
        })
        .expect("spawn usermap feeder")
}