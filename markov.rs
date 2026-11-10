use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;
use crate::ids::TrackId;
use crate::snapshot::MarkovSnapshot;

#[derive(Serialize, Deserialize, Default)]
pub struct MarkovChain {
    trans: FxHashMap<TrackId, SmallVec<[(TrackId, u32); 8]>>,
    rest:  FxHashMap<TrackId, i64>,
}

impl MarkovChain {
    pub fn new() -> Self { Self::default() }

    pub fn snapshot(&self) -> MarkovSnapshot {
        MarkovSnapshot { trans: Arc::new(self.trans.clone()) }
    }

    pub fn clone_via_restore(&self) -> MarkovChain {
        MarkovChain { trans: self.trans.clone(), rest: self.rest.clone() }
    }

    pub fn trans_clone(&self) -> FxHashMap<TrackId, SmallVec<[(TrackId, u32); 8]>> { self.trans.clone() }
    pub fn rest_clone(&self)  -> FxHashMap<TrackId, i64> { self.rest.clone() }

    pub fn restore(
        trans: FxHashMap<TrackId, SmallVec<[(TrackId, u32); 8]>>,
        rest: FxHashMap<TrackId, i64>,
    ) -> Self {
        Self { trans, rest }
    }

    pub fn observe(&mut self, from: TrackId, to: TrackId) {
        let row = self.trans.entry(from).or_default();
        if let Some(slot) = row.iter_mut().find(|(id, _)| *id == to) {
            slot.1 += 1;
        } else {
            row.push((to, 1));
        }
    }

    pub fn rest(&mut self, id: TrackId, until: i64) {
        self.rest.insert(id, until);
    }

    pub fn is_resting(&self, id: TrackId, now: i64) -> bool {
        self.rest.get(&id).map_or(false, |u| now < *u)
    }

    pub fn cleanup(&mut self, now: i64) {
        self.rest.retain(|_, u| now < *u);
    }
}