use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use crate::adaptive::ArmState;
use crate::cooccurrence::CoOccurrence;
use crate::ids::{AlbumId, ArtistId, GenreId, LangId, TrackId, UserId};
use crate::markov::MarkovChain;
use crate::profile::{ContextStat, UserProfile};
use crate::usermap::UserKnn;

impl UserProfile {
    pub fn clone_via_restore(&self) -> Self {
        UserProfile {
            user_id: self.user_id,
            long_vec: self.long_vec.clone(),
            artist_affinity: self.artist_affinity.clone(),
            genre_affinity:  self.genre_affinity.clone(),
            lang_affinity:   self.lang_affinity.clone(),
            era_ma: self.era_ma, bpm_ma: self.bpm_ma, vocal_ma: self.vocal_ma,
            energy_ma: self.energy_ma.clone(),
            skip_rate: self.skip_rate.clone(),
            last_played: self.last_played.clone(),
            plays: self.plays.clone(),
            skips: self.skips.clone(),
            co_like: self.co_like.clone(),
            co_skip: self.co_skip.clone(),
            context_stats: self.context_stats.clone(),
            context_genre_stats: self.context_genre_stats.clone(),
            dim: self.dim,
            half_life_hours: self.half_life_hours,
            cold_play_threshold: self.cold_play_threshold,
            total_weight: self.total_weight,
            total_plays: self.total_plays,
        }
    }
}

impl CoOccurrence {
    pub fn clone_via_restore(&self) -> Self {
        CoOccurrence::restore(
            self.pmi_track_clone(), self.pmi_artist_clone(),
            self.track_count_clone(), self.artist_count_clone(),
            self.co_track_clone(), self.co_artist_clone(),
            self.album_co_clone(), self.total_tracks_clone(), self.total_artists_clone(),
        )
    }
}

impl MarkovChain {
    pub fn clone_via_restore(&self) -> MarkovChain {
        MarkovChain::restore(self.trans_clone(), self.rest_clone())
    }
}

impl UserKnn {
    pub fn clone_via_restore(&self) -> Self {
        let mut n = UserKnn::new();
        for (&k, v) in self.users_iter() { n.upsert(k, v.clone()); }
        n
    }
}