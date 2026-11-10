use rand::Rng;
use crate::adaptive::{sample_dual, GlobalPolicySnapshot, UserPolicySnapshot};
use crate::bandit::{sample_from_snapshot, Arm, BanditSnapshot};
use crate::candidates::{Candidate, Source};
use crate::catalog::Track;
use crate::config::WaveConfig;
use crate::context::TemporalContext;
use crate::ids::TrackId;
use crate::snapshot::{CoocSnapshot, MarkovSnapshot, ProfileSnapshot, SessionSnapshot};
use crate::vec::{camelot_distance, cosine_normalized};

pub struct RankerCtx<'a, R: Rng> {
    pub profile:       &'a ProfileSnapshot,
    pub session:       &'a SessionSnapshot,
    pub cooc:          &'a CoocSnapshot,
    pub markov:        &'a MarkovSnapshot,
    pub bandit:        &'a BanditSnapshot,
    pub user_policy:   &'a UserPolicySnapshot,
    pub global_policy: &'a GlobalPolicySnapshot,
    pub cfg:           &'a WaveConfig,
    pub now:           i64,
    pub ctx:           TemporalContext,
    pub rng:           &'a mut R,
}

pub fn score<R: Rng>(c: &Candidate, t: &Track, r: &mut RankerCtx<'_, R>) -> f32 {
    let w = &r.cfg.weights;
    let short = &*r.session.short_vec;
    let long  = &*r.session.long_vec;

    let cos_short = cosine_normalized(short, &t.embedding);
    let cos_long  = cosine_normalized(long,  &t.embedding);
    let cos_intro = t.intro_embedding.as_ref().map(|e| cosine_normalized(short, e)).unwrap_or(cos_short);
    let cos_alt   = t.alt_embedding  .as_ref().map(|e| cosine_normalized(short, e)).unwrap_or(cos_short);
    let cos_chunk = t.chunk_embeddings.as_ref()
        .map(|c| c.iter().map(|ch| cosine_normalized(short, ch)).fold(f32::NEG_INFINITY, f32::max))
        .unwrap_or(cos_short);

    let recent: Vec<TrackId> = r.session.recent.iter().rev().take(10).map(|e| e.track).collect();

    let art   = r.profile.artist_affinity.get(&t.artist_id).copied().unwrap_or(0.0);
    let gen   = r.profile.genre_affinity .get(&t.genre_id ).copied().unwrap_or(0.0);
    let pmi   = r.cooc.pmi_score(&recent, t.id);
    let co_lk = recent.first().map(|&s| r.profile.co_score(s, t.id)).unwrap_or(0.0).max(0.0);
    let co_sk = (-recent.first().map(|&s| r.profile.co_score(s, t.id)).unwrap_or(0.0)).max(0.0);
    let mrkv  = r.markov.prob(recent.first().copied().unwrap_or(t.id), t.id);
    let ank   = r.session.anchor_score(t.id, r.now);

    let mut energy = t.energy_curve.as_ref().map(|e| {
        let mut s = 0.0f32;
        let mut n = 0.0f32;
        for (a, b) in e.iter().zip(r.profile.energy_ma.iter()) { s += (a - b).abs(); n += 1.0; }
        1.0 - (s / n.max(1.0)).clamp(0.0, 1.0)
    }).unwrap_or(0.5);
    energy = (energy * r.ctx.activity.energy_bias()).clamp(0.0, 1.5) / 1.5;

    let bpm_sim = t.bpm.map(|b| 1.0 - ((b - r.profile.bpm_ma).abs() / 60.0).clamp(0.0, 1.0)).unwrap_or(0.5);
    let mut voc_sim = t.vocal.map(|v| 1.0 - (v - r.profile.vocal_ma).abs()).unwrap_or(0.5);
    voc_sim = (voc_sim * r.ctx.activity.vocal_bias()).clamp(0.0, 1.5) / 1.5;
    let lang_sim = t.language.map(|l| r.profile.lang_affinity.get(&l).copied().unwrap_or(0.0)).unwrap_or(0.5);
    let era_sim = 1.0 - ((t.release_year as f32 - r.profile.era_ma).abs() / 40.0).clamp(0.0, 1.0);
    let key_sim = t.key_camelot.map(|a| camelot_distance(a, 0)).unwrap_or(0.5);

    let pop   = t.popularity;
    let pop_p = t.popularity_pct;
    let fresh = r.profile.freshness(t.id, r.now);
    let cold  = r.profile.is_cold(t.id);
    let mut novl = if cold { 1.0 } else { 0.0 };
    novl *= r.ctx.activity.novelty_bias();
    let skip  = r.profile.skip_rate.get(&t.id).copied().unwrap_or(0.0);

    let band_src = sample_from_snapshot(r.bandit, Arm::Source(c.source.id()), r.rng);
    let band_gen = sample_from_snapshot(r.bandit, Arm::Genre(t.genre_id.raw() as u32), r.rng);
    let band_gh  = sample_from_snapshot(r.bandit, Arm::GenreHour(t.genre_id.raw() as u32, r.ctx.hour), r.rng);
    let band_gd  = sample_from_snapshot(r.bandit, Arm::GenreDevice(t.genre_id.raw() as u32, r.ctx.device), r.rng);
    let band_art = sample_from_snapshot(r.bandit, Arm::Artist(t.artist_id.raw()), r.rng);
    let band = 0.4 * band_gen + 0.2 * band_gh + 0.1 * band_gd + 0.2 * band_src + 0.1 * band_art;

    let adapt = sample_dual(
        r.user_policy, r.global_policy,
        t.id, t.artist_id, t.genre_id, r.now, r.rng,
    );

    let ctx_q = r.profile.context_quality(&r.ctx, t.genre_id);

    let anti_pop = if t.popularity > r.cfg.anti_popularity_threshold {
        -r.cfg.anti_popularity_penalty * (t.popularity - r.cfg.anti_popularity_threshold)
    } else { 0.0 };

    let novelty_boost = if cold { r.cfg.novelty_boost - 1.0 } else { 0.0 };
    let cold_boost = if t.popularity < 0.1 { 1.0 } else { 0.0 };

    let knn_boost = if matches!(c.source, Source::UserKnn) { c.source_score } else { 0.0 };
    let coldpool_boost = if matches!(c.source, Source::ColdPool) { c.source_score } else { 0.0 };

    w.cosine            * (0.6 * cos_short + 0.4 * cos_long)
        + w.cosine_short    * cos_short
        + w.cosine_long     * cos_long
        + w.intro_cosine    * cos_intro
        + w.alt_cosine      * cos_alt
        + w.chunk_cosine    * cos_chunk
        + w.artist_affinity * art
        + w.genre_affinity  * gen
        + w.cooccurrence_pmi* pmi
        + w.co_like         * co_lk
        - w.co_skip         * co_sk
        + w.user_knn        * knn_boost
        + w.cold_pool       * coldpool_boost
        + w.markov          * mrkv
        + w.anchor          * ank
        + w.energy_match    * energy
        + w.bpm_compat      * bpm_sim
        + w.vocal_match     * voc_sim
        + w.language_match  * lang_sim
        + w.era_match       * era_sim
        + w.popularity      * pop
        + w.popularity_percentile * pop_p
        + w.freshness       * fresh
        + w.novelty         * novl
        + w.context_quality * ctx_q
        + w.bandit          * band
        + w.adaptive        * adapt
        + w.cold_boost      * cold_boost
        + w.anti_pop        * anti_pop
        + novelty_boost
        - w.skip_penalty    * skip
        + if key_sim > 0.9 { 0.01 } else { 0.0 }
}