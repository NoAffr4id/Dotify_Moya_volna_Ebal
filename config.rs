use serde::{Deserialize, Serialize};
use crate::coldpool::ColdPoolConfig;
use crate::error::{Result, WaveError};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankerWeights {
    pub cosine:            f32,
    pub cosine_short:      f32,
    pub cosine_long:       f32,
    pub intro_cosine:      f32,
    pub alt_cosine:        f32,
    pub chunk_cosine:      f32,
    pub artist_affinity:   f32,
    pub genre_affinity:    f32,
    pub cooccurrence_pmi:  f32,
    pub co_like:           f32,
    pub co_skip:           f32,
    pub user_knn:          f32,
    pub cold_pool:         f32,
    pub markov:            f32,
    pub anchor:            f32,
    pub energy_match:      f32,
    pub bpm_compat:        f32,
    pub vocal_match:       f32,
    pub language_match:    f32,
    pub era_match:         f32,
    pub popularity:        f32,
    pub popularity_percentile: f32,
    pub freshness:         f32,
    pub novelty:           f32,
    pub context_quality:   f32,
    pub bandit:            f32,
    pub adaptive:          f32,
    pub skip_penalty:      f32,
    pub cold_boost:        f32,
    pub anti_pop:          f32,
}

impl Default for RankerWeights {
    fn default() -> Self {
        Self {
            cosine: 0.20, cosine_short: 0.10, cosine_long: 0.10,
            intro_cosine: 0.04, alt_cosine: 0.03, chunk_cosine: 0.03,
            artist_affinity: 0.08, genre_affinity: 0.05,
            cooccurrence_pmi: 0.05, co_like: 0.03, co_skip: 0.03,
            user_knn: 0.03, cold_pool: 0.05,
            markov: 0.05, anchor: 0.04,
            energy_match: 0.03, bpm_compat: 0.02, vocal_match: 0.02,
            language_match: 0.02, era_match: 0.02,
            popularity: 0.02, popularity_percentile: 0.02,
            freshness: 0.03, novelty: 0.04, context_quality: 0.03,
            bandit: 0.03, adaptive: 0.10,
            skip_penalty: 0.35, cold_boost: 0.04, anti_pop: 0.05,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m:              usize,
    pub ef_construction:usize,
    pub ef_search:      usize,
    pub max_elements:   usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 64, max_elements: 1_000_000 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub snapshot_dir:      String,
    pub profile_every_n:   u64,
    pub cooc_every_n:      u64,
    pub markov_every_n:    u64,
    pub policy_every_n:    u64,
    pub usermap_every_n:   u64,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            snapshot_dir:    "./snapshots".to_string(),
            profile_every_n: 500,
            cooc_every_n:    1000,
            markov_every_n:  1000,
            policy_every_n:  500,
            usermap_every_n: 2000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaveConfig {
    pub dim:                    usize,
    pub wave_len:               usize,
    pub pool_per_source:        usize,
    pub session_decay_short:    f32,
    pub session_decay_long:     f32,
    pub short_window:           usize,
    pub long_window:            usize,
    pub diversification_window: usize,
    pub max_same_artist:        usize,
    pub max_same_genre:         usize,
    pub genre_quota_window:     usize,
    pub min_distinct_genres:    usize,
    pub mmr_lambda:             f32,
    pub mmr_position_lambda:    f32,
    pub dpp_enabled:            bool,
    pub dpp_epsilon:            f32,
    pub explore_fraction:       f32,
    pub skip_streak_explore_boost: f32,
    pub skip_streak_threshold:  u32,
    pub drift_threshold:        f32,
    pub profile_half_life_hours: f32,
    pub session_break_hours:    f32,
    pub anchor_half_life_hours: f32,
    pub rest_track_hours:       f32,
    pub rest_genre_hours:       f32,
    pub rest_trigger_streak:    u32,
    pub background_weight:      f32,
    pub pmi_smoothing:          f32,
    pub cold_play_threshold:    u32,
    pub novelty_boost:          f32,
    pub anti_popularity_threshold: f32,
    pub anti_popularity_penalty:   f32,
    pub adaptive_tau_hours:     f32,
    pub adaptive_shrink:        f32,
    pub hnsw:                   HnswConfig,
    pub cold_pool:              ColdPoolConfig,
    pub persistence:            PersistenceConfig,
    pub weights:                RankerWeights,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            dim:                    1024,
            wave_len:               30,
            pool_per_source:        150,
            session_decay_short:    0.65,
            session_decay_long:     0.92,
            short_window:           5,
            long_window:            50,
            diversification_window: 6,
            max_same_artist:        1,
            max_same_genre:         2,
            genre_quota_window:     10,
            min_distinct_genres:    3,
            mmr_lambda:             0.72,
            mmr_position_lambda:    0.15,
            dpp_enabled:            false,
            dpp_epsilon:            0.10,
            explore_fraction:       0.15,
            skip_streak_explore_boost: 0.25,
            skip_streak_threshold:  3,
            drift_threshold:        0.30,
            profile_half_life_hours: 24.0 * 21.0,
            session_break_hours:    0.33,
            anchor_half_life_hours: 0.5,
            rest_track_hours:       2.0,
            rest_genre_hours:       0.5,
            rest_trigger_streak:    3,
            background_weight:      0.30,
            pmi_smoothing:          1.0,
            cold_play_threshold:    3,
            novelty_boost:          1.20,
            anti_popularity_threshold: 0.95,
            anti_popularity_penalty:   0.20,
            adaptive_tau_hours:     24.0 * 14.0,
            adaptive_shrink:        2.0,
            hnsw:                   HnswConfig::default(),
            cold_pool:              ColdPoolConfig::default(),
            persistence:            PersistenceConfig::default(),
            weights:                RankerWeights::default(),
        }
    }
}

impl WaveConfig {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0      { return Err(WaveError::Config("dim must be > 0")); }
        if self.wave_len == 0 { return Err(WaveError::Config("wave_len must be > 0")); }
        if !(0.0..=1.0).contains(&self.session_decay_short) { return Err(WaveError::Config("session_decay_short")); }
        if !(0.0..=1.0).contains(&self.session_decay_long)  { return Err(WaveError::Config("session_decay_long")); }
        if !(0.0..=1.0).contains(&self.mmr_lambda)          { return Err(WaveError::Config("mmr_lambda")); }
        if !(0.0..=1.0).contains(&self.explore_fraction)    { return Err(WaveError::Config("explore_fraction")); }
        Ok(())
    }
}