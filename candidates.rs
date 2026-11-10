use smallvec::SmallVec;
use rustc_hash::FxHashSet;
use crate::catalog::Catalog;
use crate::coldpool::ColdPool;
use crate::ids::{AlbumId, TrackId, UserId};
use crate::snapshot::{CoocSnapshot, MarkovSnapshot, UserKnnSnapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Content, IntroContent, AltContent, ChunkContent,
    Cooccurrence, Markov, Artist, Genre, Album,
    UserKnn, Explore, Anchor, ColdPool,
}

impl Source {
    pub fn id(self) -> u8 {
        match self {
            Source::Content      => 0, Source::IntroContent => 1,
            Source::AltContent   => 2, Source::ChunkContent => 3,
            Source::Cooccurrence => 4, Source::Markov       => 5,
            Source::Artist       => 6, Source::Genre        => 7,
            Source::Album        => 8, Source::UserKnn      => 9,
            Source::Explore      => 10, Source::Anchor      => 11,
            Source::ColdPool     => 12,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Candidate {
    pub track:        TrackId,
    pub source:       Source,
    pub source_score: f32,
}

pub struct Sources<'a> {
    pub catalog:  &'a Catalog,
    pub cooc:     &'a CoocSnapshot,
    pub markov:   &'a MarkovSnapshot,
    pub usermap:  &'a UserKnnSnapshot,
    pub coldpool: &'a ColdPool,
}

fn sort_trunc(v: &mut SmallVec<[Candidate; 64]>, k: usize) {
    v.sort_unstable_by(|a, b| b.source_score.partial_cmp(&a.source_score).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(k);
}

impl<'a> Sources<'a> {
    pub fn content(&self, seed: &[f32], k: usize) -> SmallVec<[Candidate; 64]> {
        let mut v: SmallVec<[Candidate; 64]> = self.catalog.search_content(seed, k)
            .into_iter().map(|(id, s)| Candidate { track: id, source: Source::Content, source_score: s }).collect();
        sort_trunc(&mut v, k); v
    }

    pub fn intro_content(&self, seed: &[f32], k: usize) -> SmallVec<[Candidate; 64]> {
        let mut v: SmallVec<[Candidate; 64]> = self.catalog.search_intro(seed, k)
            .into_iter().map(|(id, s)| Candidate { track: id, source: Source::IntroContent, source_score: s }).collect();
        sort_trunc(&mut v, k); v
    }

    pub fn alt_content(&self, seed: &[f32], k: usize) -> SmallVec<[Candidate; 64]> {
        let mut v: SmallVec<[Candidate; 64]> = self.catalog.search_alt(seed, k)
            .into_iter().map(|(id, s)| Candidate { track: id, source: Source::AltContent, source_score: s }).collect();
        sort_trunc(&mut v, k); v
    }

    pub fn chunk_content(&self, seed: &[f32], k: usize) -> SmallVec<[Candidate; 64]> {
        let mut v: SmallVec<[Candidate; 64]> = self.catalog.search_chunk(seed, k)
            .into_iter().map(|(id, s)| Candidate { track: id, source: Source::ChunkContent, source_score: s }).collect();
        sort_trunc(&mut v, k); v
    }

    pub fn from_cooc(&self, recent: &[TrackId], k: usize) -> SmallVec<[Candidate; 64]> {
        let mut out: SmallVec<[Candidate; 64]> = SmallVec::new();
        for &seed in recent.iter().take(5) {
            for &(n, w) in self.cooc.pmi_track_neighbors(seed) {
                out.push(Candidate { track: n, source: Source::Cooccurrence, source_score: w });
            }
        }
        sort_trunc(&mut out, k); out
    }

    pub fn from_markov(&self, recent: &[TrackId], k: usize) -> SmallVec<[Candidate; 64]> {
        let Some(&last) = recent.first() else { return SmallVec::new() };
        self.markov.top_next(last, k).iter().map(|&(id, cnt)| Candidate {
            track: id, source: Source::Markov, source_score: (cnt as f32).ln_1p() * 0.1,
        }).collect()
    }

    pub fn from_artist(
        &self,
        artist_affinity: &rustc_hash::FxHashMap<crate::ids::ArtistId, f32>,
        k: usize,
    ) -> SmallVec<[Candidate; 64]> {
        let mut out: SmallVec<[Candidate; 64]> = SmallVec::new();
        let mut artists: Vec<_> = artist_affinity.iter().collect();
        artists.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (a, w) in artists.into_iter().take(10) {
            for &tid in self.catalog.tracks_by_artist(*a).iter().take(k / 10 + 1) {
                out.push(Candidate { track: tid, source: Source::Artist, source_score: *w });
            }
            for &(other, w2) in self.cooc.pmi_artist_neighbors(*a).iter().take(3) {
                for &tid in self.catalog.tracks_by_artist(other).iter().take(2) {
                    out.push(Candidate { track: tid, source: Source::Artist, source_score: *w * w2 });
                }
            }
        }
        sort_trunc(&mut out, k); out
    }

    pub fn from_genre(
        &self,
        genre_affinity: &rustc_hash::FxHashMap<crate::ids::GenreId, f32>,
        k: usize,
    ) -> SmallVec<[Candidate; 64]> {
        let mut out: SmallVec<[Candidate; 64]> = SmallVec::new();
        let mut genres: Vec<_> = genre_affinity.iter().collect();
        genres.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (g, w) in genres.into_iter().take(5) {
            let pool = self.catalog.tracks_by_genre(*g);
            let step = (pool.len() / (k + 1)).max(1);
            for &tid in pool.iter().step_by(step).take(k / 5 + 1) {
                out.push(Candidate { track: tid, source: Source::Genre, source_score: *w * 0.5 });
            }
        }
        sort_trunc(&mut out, k); out
    }

    pub fn from_album(&self, album: AlbumId, k: usize) -> SmallVec<[Candidate; 64]> {
        self.cooc.album_siblings(album).iter().take(k).map(|&(id, w)| Candidate {
            track: id, source: Source::Album, source_score: w,
        }).collect()
    }

    pub fn from_user_knn(
        &self,
        user_id: UserId,
        exclude: impl Fn(TrackId) -> bool,
        k: usize,
    ) -> SmallVec<[Candidate; 64]> {
        crate::usermap::recommend_snapshot(self.usermap, user_id, k, exclude).into_iter().map(|(id, s)| Candidate {
            track: id, source: Source::UserKnn, source_score: s.min(1.0),
        }).collect()
    }

    pub fn explore(&self, k: usize, rng: &mut impl rand::Rng) -> SmallVec<[Candidate; 64]> {
        use rand::seq::IteratorRandom;
        self.catalog.iter().filter(|t| t.popularity > 0.05).choose_multiple(rng, k)
            .into_iter().map(|t| Candidate { track: t.id, source: Source::Explore, source_score: 0.3 }).collect()
    }

    pub fn anchor(
        &self,
        session: &crate::snapshot::SessionSnapshot,
        now: i64,
        k: usize,
    ) -> SmallVec<[Candidate; 64]> {
        let mut out: SmallVec<[Candidate; 64]> = SmallVec::new();
        for e in session.recent.iter().rev().take(5) {
            let s = session.anchor_score(e.track, now);
            if s > 0.0 { out.push(Candidate { track: e.track, source: Source::Anchor, source_score: s }); }
        }
        out.truncate(k); out
    }

    pub fn from_coldpool(
        &self,
        exclude: &FxHashSet<TrackId>,
        k: usize,
        now: i64,
        rng: &mut impl rand::Rng,
    ) -> SmallVec<[Candidate; 64]> {
        self.coldpool.sample(self.catalog, exclude, k, now, rng)
            .into_iter()
            .map(|id| Candidate { track: id, source: Source::ColdPool, source_score: 1.0 })
            .collect()
    }

    pub fn filter_resting(
        &self,
        c: Candidate,
        session: &crate::snapshot::SessionSnapshot,
        now: i64,
    ) -> Option<Candidate> {
        let t = self.catalog.get(c.track).ok()?;
        if session.is_resting(t.id, now) { return None; }
        if session.is_resting_genre(t.genre_id, now) { return None; }
        if session.is_resting_artist(t.artist_id, now) { return None; }
        if let Some(a) = t.album_id { if session.is_resting_album(a, now) { return None; } }
        Some(c)
    }
}