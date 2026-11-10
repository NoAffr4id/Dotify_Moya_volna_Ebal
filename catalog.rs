use std::sync::Arc;
use arc_swap::ArcSwap;
use hnsw_rs::prelude::*;
use rustc_hash::FxHashMap;
use crate::config::HnswConfig;
use crate::error::{Result, WaveError};
use crate::ids::{AlbumId, ArtistId, GenreId, LangId, TrackId};

#[derive(Clone, Debug)]
pub struct Track {
    pub id:             TrackId,
    pub artist_id:      ArtistId,
    pub album_id:       Option<AlbumId>,
    pub genre_id:       GenreId,
    pub language:       Option<LangId>,
    pub release_year:   u16,
    pub duration_sec:   f32,
    pub popularity:     f32,
    pub popularity_pct: f32,
    pub bpm:            Option<f32>,
    pub key_camelot:    Option<u8>,
    pub vocal:          Option<f32>,
    pub energy_curve:   Option<Arc<[f32]>>,
    pub embedding:      Arc<[f32]>,
    pub alt_embedding:  Option<Arc<[f32]>>,
    pub intro_embedding: Option<Arc<[f32]>>,
    pub chunk_embeddings: Option<Arc<[Arc<[f32]>]>>,
}

struct IndexSlot {
    hnsw:   Hnsw<'static, f32, DistCosine>,
    id_map: Vec<TrackId>,
}

pub struct Catalog {
    tracks:       FxHashMap<TrackId, Arc<Track>>,
    by_artist:    FxHashMap<ArtistId, Vec<TrackId>>,
    by_genre:     FxHashMap<GenreId,  Vec<TrackId>>,
    by_album:     FxHashMap<AlbumId,  Vec<TrackId>>,
    by_lang:      FxHashMap<LangId,   Vec<TrackId>>,
    dim:          usize,
    hnsw_cfg:     HnswConfig,
    idx_main:     ArcSwap<Option<IndexSlot>>,
    idx_intro:    ArcSwap<Option<IndexSlot>>,
    idx_alt:      ArcSwap<Option<IndexSlot>>,
    idx_chunk:    ArcSwap<Option<IndexSlot>>,
    dirty:        std::sync::atomic::AtomicBool,
}

impl Catalog {
    pub fn new(dim: usize, hnsw_cfg: HnswConfig) -> Self {
        Self {
            tracks: FxHashMap::default(),
            by_artist: FxHashMap::default(),
            by_genre: FxHashMap::default(),
            by_album: FxHashMap::default(),
            by_lang: FxHashMap::default(),
            dim,
            hnsw_cfg,
            idx_main:  ArcSwap::from_pointee(None),
            idx_intro: ArcSwap::from_pointee(None),
            idx_alt:   ArcSwap::from_pointee(None),
            idx_chunk: ArcSwap::from_pointee(None),
            dirty: std::sync::atomic::AtomicBool::new(true),
        }
    }

    pub fn insert(&mut self, mut t: Track) -> Result<()> {
        if t.embedding.len() != self.dim {
            return Err(WaveError::DimMismatch { expected: self.dim, actual: t.embedding.len() });
        }
        let mut emb: Vec<f32> = t.embedding.to_vec();
        crate::vec::normalize(&mut emb);
        t.embedding = Arc::from(emb);

        if let Some(a) = t.alt_embedding.as_mut() {
            let mut v = a.to_vec();
            crate::vec::normalize(&mut v);
            t.alt_embedding = Some(Arc::from(v));
        }
        if let Some(a) = t.intro_embedding.as_mut() {
            let mut v = a.to_vec();
            crate::vec::normalize(&mut v);
            t.intro_embedding = Some(Arc::from(v));
        }
        if let Some(chunks) = t.chunk_embeddings.as_mut() {
            let mut v: Vec<Arc<[f32]>> = Vec::with_capacity(chunks.len());
            for c in chunks.iter() {
                let mut cc = c.to_vec();
                crate::vec::normalize(&mut cc);
                v.push(Arc::from(cc));
            }
            t.chunk_embeddings = Some(Arc::from(v));
        }

        self.by_artist.entry(t.artist_id).or_default().push(t.id);
        self.by_genre.entry(t.genre_id).or_default().push(t.id);
        if let Some(a) = t.album_id { self.by_album.entry(a).or_default().push(t.id); }
        if let Some(l) = t.language { self.by_lang.entry(l).or_default().push(t.id); }
        self.tracks.insert(t.id, Arc::new(t));
        self.dirty.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    pub fn get(&self, id: TrackId) -> Result<Arc<Track>> {
        self.tracks.get(&id).cloned().ok_or(WaveError::TrackNotFound(id))
    }

    pub fn len(&self) -> usize { self.tracks.len() }
    pub fn is_empty(&self) -> bool { self.tracks.is_empty() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn iter(&self) -> impl Iterator<Item = &Arc<Track>> { self.tracks.values() }

    pub fn tracks_by_artist(&self, a: ArtistId) -> &[TrackId] { self.by_artist.get(&a).map(Vec::as_slice).unwrap_or(&[]) }
    pub fn tracks_by_genre(&self, g: GenreId)   -> &[TrackId] { self.by_genre.get(&g).map(Vec::as_slice).unwrap_or(&[]) }
    pub fn tracks_by_album(&self, a: AlbumId)   -> &[TrackId] { self.by_album.get(&a).map(Vec::as_slice).unwrap_or(&[]) }
    pub fn tracks_by_lang(&self, l: LangId)     -> &[TrackId] { self.by_lang.get(&l).map(Vec::as_slice).unwrap_or(&[]) }

    pub fn needs_rebuild(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn rebuild_indices(&self) -> Result<()> {
        let main  = self.build_index(|t| Some(t.embedding.as_ref()), false);
        let intro = self.build_index(|t| t.intro_embedding.as_deref(), false);
        let alt   = self.build_index(|t| t.alt_embedding.as_deref(), false);
        let chunk = self.build_index(|t| t.chunk_embeddings.as_ref().and_then(|c| c.first().map(|x| x.as_ref())), true);

        self.idx_main.store(Arc::new(main));
        self.idx_intro.store(Arc::new(intro));
        self.idx_alt.store(Arc::new(alt));
        self.idx_chunk.store(Arc::new(chunk));
        self.dirty.store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn build_index<F>(&self, get: F, chunks: bool) -> Option<IndexSlot>
    where
        F: Fn(&Track) -> Option<&[f32]>,
    {
        let mut total = 0usize;
        for t in self.tracks.values() {
            if chunks {
                if let Some(c) = t.chunk_embeddings.as_ref() { total += c.len(); }
            } else if get(t).is_some() {
                total += 1;
            }
        }
        if total == 0 { return None; }

        let cap = self.hnsw_cfg.max_elements.max(total);
        let mut hnsw = Hnsw::new(
            self.hnsw_cfg.m,
            cap,
            self.hnsw_cfg.m * 2,
            self.hnsw_cfg.ef_construction,
            DistCosine,
        );
        hnsw.set_extend_candidates(false);

        let mut id_map: Vec<TrackId> = Vec::with_capacity(total);
        let mut d_id = 0usize;
        for t in self.tracks.values() {
            if chunks {
                if let Some(cs) = t.chunk_embeddings.as_ref() {
                    for c in cs.iter() {
                        hnsw.insert((c.as_ref(), d_id));
                        id_map.push(t.id);
                        d_id += 1;
                    }
                }
            } else if let Some(v) = get(t) {
                hnsw.insert((v, d_id));
                id_map.push(t.id);
                d_id += 1;
            }
        }
        hnsw.set_searching_mode(true);
        Some(IndexSlot { hnsw, id_map })
    }

    pub fn search_content(&self, q: &[f32], k: usize) -> Vec<(TrackId, f32)> {
        Self::search(&self.idx_main.load(), q, k, self.hnsw_cfg.ef_search)
    }
    pub fn search_intro(&self, q: &[f32], k: usize) -> Vec<(TrackId, f32)> {
        Self::search(&self.idx_intro.load(), q, k, self.hnsw_cfg.ef_search)
    }
    pub fn search_alt(&self, q: &[f32], k: usize) -> Vec<(TrackId, f32)> {
        Self::search(&self.idx_alt.load(), q, k, self.hnsw_cfg.ef_search)
    }
    pub fn search_chunk(&self, q: &[f32], k: usize) -> Vec<(TrackId, f32)> {
        Self::search(&self.idx_chunk.load(), q, k, self.hnsw_cfg.ef_search)
    }

    fn search(slot: &Option<Arc<IndexSlot>>, q: &[f32], k: usize, ef: usize) -> Vec<(TrackId, f32)> {
        let Some(idx) = slot.as_ref() else { return Vec::new() };
        idx.hnsw
            .search(q, k, ef)
            .into_iter()
            .map(|n| (idx.id_map[n.d_id], (1.0 - n.distance).clamp(-1.0, 1.0)))
            .collect()
    }
}