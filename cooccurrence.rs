use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;
use crate::ids::{AlbumId, ArtistId, TrackId};
use crate::snapshot::CoocSnapshot;

#[derive(Serialize, Deserialize)]
pub struct CoOccurrence {
    pmi_track:    FxHashMap<TrackId, SmallVec<[(TrackId, f32); 16]>>,
    pmi_artist:   FxHashMap<ArtistId, SmallVec<[(ArtistId, f32); 16]>>,
    track_count:  FxHashMap<TrackId, u32>,
    artist_count: FxHashMap<ArtistId, u32>,
    co_track:     FxHashMap<TrackId, SmallVec<[(TrackId, u32); 16]>>,
    co_artist:    FxHashMap<ArtistId, SmallVec<[(ArtistId, u32); 16]>>,
    album_co:     FxHashMap<AlbumId, SmallVec<[(TrackId, f32); 16]>>,
    total_tracks: u32,
    total_artists: u32,
    dirty:        bool,
}

impl CoOccurrence {
    pub fn new() -> Self {
        Self {
            pmi_track: FxHashMap::default(), pmi_artist: FxHashMap::default(),
            track_count: FxHashMap::default(), artist_count: FxHashMap::default(),
            co_track: FxHashMap::default(), co_artist: FxHashMap::default(),
            album_co: FxHashMap::default(), total_tracks: 0, total_artists: 0, dirty: false,
        }
    }

    pub fn snapshot(&self) -> CoocSnapshot {
        CoocSnapshot {
            pmi_track:  Arc::new(self.pmi_track.clone()),
            pmi_artist: Arc::new(self.pmi_artist.clone()),
            album_co:   Arc::new(self.album_co.clone()),
        }
    }

    pub fn observe_sequence(&mut self, seq: &[TrackId], artists: &[ArtistId]) {
        for i in 0..seq.len() {
            *self.track_count.entry(seq[i]).or_insert(0) += 1;
            *self.artist_count.entry(artists[i]).or_insert(0) += 1;
            self.total_tracks += 1;
            self.total_artists += 1;
            for j in (i + 1)..seq.len().min(i + 50) {
                self.bump_track(seq[i], seq[j]);
                self.bump_track(seq[j], seq[i]);
                self.bump_artist(artists[i], artists[j]);
                self.bump_artist(artists[j], artists[i]);
            }
        }
        self.dirty = true;
    }

    pub fn observe_album(&mut self, album: AlbumId, tracks: &[TrackId]) {
        let entry = self.album_co.entry(album).or_default();
        entry.clear();
        for &t in tracks {
            entry.push((t, 1.0 / tracks.len().max(1) as f32));
        }
        self.dirty = true;
    }

    fn bump_track(&mut self, a: TrackId, b: TrackId) {
        let e = self.co_track.entry(a).or_default();
        if let Some(slot) = e.iter_mut().find(|(id, _)| *id == b) { slot.1 += 1; } else { e.push((b, 1)); }
    }
    fn bump_artist(&mut self, a: ArtistId, b: ArtistId) {
        let e = self.co_artist.entry(a).or_default();
        if let Some(slot) = e.iter_mut().find(|(id, _)| *id == b) { slot.1 += 1; } else { e.push((b, 1)); }
    }

    pub fn compact(&mut self, top_k: usize, smoothing: f32) {
        if !self.dirty { return; }
        self.pmi_track.clear();
        self.pmi_artist.clear();

        let total_t = self.total_tracks.max(1) as f32;
        let total_a = self.total_artists.max(1) as f32;

        for (&a, edges) in self.co_track.iter() {
            let pa = (*self.track_count.get(&a).unwrap_or(&0) as f32 + smoothing) / (total_t + smoothing);
            let mut top: SmallVec<[(TrackId, f32); 16]> = SmallVec::new();
            for &(b, cnt) in edges.iter() {
                let pb = (*self.track_count.get(&b).unwrap_or(&0) as f32 + smoothing) / (total_t + smoothing);
                let pab = (cnt as f32 + smoothing) / (total_t + smoothing);
                let score = (pab / (pa * pb)).ln().max(0.0);
                if score > 0.0 { top.push((b, score)); }
            }
            top.sort_unstable_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
            top.truncate(top_k);
            if !top.is_empty() {
                let norm = top.iter().map(|(_, s)| s).fold(0.0f32, f32::max).max(1e-6);
                let v: SmallVec<[(TrackId, f32); 16]> = top.into_iter().map(|(id, s)| (id, s / norm)).collect();
                self.pmi_track.insert(a, v);
            }
        }

        for (&a, edges) in self.co_artist.iter() {
            let pa = (*self.artist_count.get(&a).unwrap_or(&0) as f32 + smoothing) / (total_a + smoothing);
            let mut top: SmallVec<[(ArtistId, f32); 16]> = SmallVec::new();
            for &(b, cnt) in edges.iter() {
                let pb = (*self.artist_count.get(&b).unwrap_or(&0) as f32 + smoothing) / (total_a + smoothing);
                let pab = (cnt as f32 + smoothing) / (total_a + smoothing);
                let score = (pab / (pa * pb)).ln().max(0.0);
                if score > 0.0 { top.push((b, score)); }
            }
            top.sort_unstable_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
            top.truncate(top_k);
            if !top.is_empty() {
                let norm = top.iter().map(|(_, s)| s).fold(0.0f32, f32::max).max(1e-6);
                let v: SmallVec<[(ArtistId, f32); 16]> = top.into_iter().map(|(id, s)| (id, s / norm)).collect();
                self.pmi_artist.insert(a, v);
            }
        }
        self.dirty = false;
    }

    // Геттеры/сеттеры для persistence
    pub fn pmi_track_clone(&self)   -> FxHashMap<TrackId, SmallVec<[(TrackId, f32); 16]>> { self.pmi_track.clone() }
    pub fn pmi_artist_clone(&self)  -> FxHashMap<ArtistId, SmallVec<[(ArtistId, f32); 16]>> { self.pmi_artist.clone() }
    pub fn track_count_clone(&self) -> FxHashMap<TrackId, u32> { self.track_count.clone() }
    pub fn artist_count_clone(&self)-> FxHashMap<ArtistId, u32> { self.artist_count.clone() }
    pub fn co_track_clone(&self)    -> FxHashMap<TrackId, SmallVec<[(TrackId, u32); 16]>> { self.co_track.clone() }
    pub fn co_artist_clone(&self)   -> FxHashMap<ArtistId, SmallVec<[(ArtistId, u32); 16]>> { self.co_artist.clone() }
    pub fn album_co_clone(&self)    -> FxHashMap<AlbumId, SmallVec<[(TrackId, f32); 16]>> { self.album_co.clone() }
    pub fn total_tracks_clone(&self)-> u32 { self.total_tracks }
    pub fn total_artists_clone(&self)-> u32 { self.total_artists }

    pub fn restore(
        pmi_track: FxHashMap<TrackId, SmallVec<[(TrackId, f32); 16]>>,
        pmi_artist: FxHashMap<ArtistId, SmallVec<[(ArtistId, f32); 16]>>,
        track_count: FxHashMap<TrackId, u32>,
        artist_count: FxHashMap<ArtistId, u32>,
        co_track: FxHashMap<TrackId, SmallVec<[(TrackId, u32); 16]>>,
        co_artist: FxHashMap<ArtistId, SmallVec<[(ArtistId, u32); 16]>>,
        album_co: FxHashMap<AlbumId, SmallVec<[(TrackId, f32); 16]>>,
        total_tracks: u32, total_artists: u32,
    ) -> Self {
        Self { pmi_track, pmi_artist, track_count, artist_count, co_track, co_artist, album_co, total_tracks, total_artists, dirty: false }
    }
}