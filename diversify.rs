use std::collections::HashMap;
use std::sync::Arc;
use crate::catalog::Track;
use crate::config::WaveConfig;
use crate::ids::{ArtistId, GenreId, TrackId};
use crate::vec::cosine_normalized;

pub struct Scored {
    pub track: Arc<Track>,
    pub score: f32,
}

impl Clone for Scored {
    fn clone(&self) -> Self { Self { track: self.track.clone(), score: self.score } }
}

pub fn mmr(mut pool: Vec<Scored>, k: usize, cfg: &WaveConfig) -> Vec<Scored> {
    pool.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<Scored> = Vec::with_capacity(k);
    let mut artist_counts: HashMap<ArtistId, usize> = HashMap::new();
    let mut genre_counts:  HashMap<GenreId,  usize> = HashMap::new();

    while out.len() < k && !pool.is_empty() {
        let mut best: Option<(usize, f32)> = None;
        let pos = out.len() as f32 / k.max(1) as f32;
        let lam = (cfg.mmr_lambda - cfg.mmr_position_lambda * pos).clamp(0.3, 1.0);

        for (i, cand) in pool.iter().enumerate() {
            if *artist_counts.get(&cand.track.artist_id).unwrap_or(&0) >= cfg.max_same_artist { continue; }
            if *genre_counts .get(&cand.track.genre_id ).unwrap_or(&0) >= cfg.max_same_genre  { continue; }

            let max_sim = out.iter()
                .map(|s| cosine_normalized(&s.track.embedding, &cand.track.embedding))
                .fold(0.0f32, f32::max);

            let val = lam * cand.score - (1.0 - lam) * max_sim;
            if best.map_or(true, |(_, b)| val > b) { best = Some((i, val)); }
        }

        let Some((i, _)) = best else { break };
        let chosen = pool.swap_remove(i);

        if out.len() >= cfg.diversification_window {
            let old = &out[out.len() - cfg.diversification_window];
            if let Some(c) = artist_counts.get_mut(&old.track.artist_id) { *c = c.saturating_sub(1); }
            if let Some(c) = genre_counts .get_mut(&old.track.genre_id ) { *c = c.saturating_sub(1); }
        }
        *artist_counts.entry(chosen.track.artist_id).or_insert(0) += 1;
        *genre_counts .entry(chosen.track.genre_id ).or_insert(0) += 1;
        out.push(chosen);
    }
    out
}

pub fn dpp(pool: Vec<Scored>, k: usize, epsilon: f32) -> Vec<Scored> {
    let n = pool.len();
    if n <= k { return pool; }
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut l_inv: Vec<Vec<f32>> = Vec::new();

    let pos_score = |i: usize| pool[i].score.max(0.0);
    let diag = |i: usize| pos_score(i) + epsilon.max(1e-3);

    for _step in 0..k {
        let mut best_s = f32::NEG_INFINITY;
        let mut best_ri = 0;
        let mut best_c: Vec<f32> = Vec::new();
        let mut best_d = 0.0f32;

        for (ri, &i) in remaining.iter().enumerate() {
            let d = diag(i);
            let mut c = Vec::with_capacity(selected.len());
            for &j in selected.iter() {
                let sim = cosine_normalized(&pool[i].track.embedding, &pool[j].track.embedding);
                c.push(sim * (pos_score(i) * pos_score(j)).sqrt());
            }
            let s = if selected.is_empty() { d } else {
                let mut cinv = vec![0.0f32; c.len()];
                for r in 0..c.len() {
                    for cc in 0..c.len() { cinv[r] += l_inv[r][cc] * c[cc]; }
                }
                let ctc: f32 = c.iter().zip(cinv.iter()).map(|(a, b)| a * b).sum();
                d - ctc
            };
            if s > best_s { best_s = s; best_ri = ri; best_c = c; best_d = d; }
        }

        if best_s <= 1e-6 { break; }
        let chosen = remaining.swap_remove(best_ri);
        selected.push(chosen);

        let sz = selected.len();
        let mut new_inv = vec![vec![0.0f32; sz]; sz];
        if sz == 1 {
            new_inv[0][0] = 1.0 / best_d;
        } else {
            let old = sz - 1;
            let s_inv = 1.0 / best_s;
            let mut ainv_c = vec![0.0f32; old];
            for r in 0..old {
                for cc in 0..old { ainv_c[r] += l_inv[r][cc] * best_c[cc]; }
            }
            let mut ct_ainv = vec![0.0f32; old];
            for cc in 0..old {
                for r in 0..old { ct_ainv[cc] += best_c[r] * l_inv[r][cc]; }
            }
            for r in 0..old {
                for c2 in 0..old {
                    new_inv[r][c2] = l_inv[r][c2] + ainv_c[r] * s_inv * ct_ainv[c2];
                }
            }
            for r in 0..old {
                new_inv[r][old] = -ainv_c[r] * s_inv;
                new_inv[old][r] = -ct_ainv[r] * s_inv;
            }
            new_inv[old][old] = s_inv;
        }
        l_inv = new_inv;
    }
    selected.into_iter().map(|i| Scored { track: pool[i].track.clone(), score: pool[i].score }).collect()
}