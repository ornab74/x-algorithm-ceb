use crate::candidate_pipeline::candidate::PostCandidate;
use crate::candidate_pipeline::query::ScoredPostsQuery;
use crate::ceb::{
    bias_field_from_table, bias_field_score, bias_from_position, bias_mix_boost, bias_orb_tree_from_table,
    bias_orb_tree_score, bias_orbit_from_table, bias_orbit_score, llm_color_from_text, rgb_to_hex, BiasTuning,
};
use std::env;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

const BIAS_BOOST: f64 = 0.22;

pub struct CebBiasSelectorScorer;

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for CebBiasSelectorScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        let target = bias_from_position(env_value("CEB_BIAS_X"), env_value("CEB_BIAS_Y"), env_value("CEB_BIAS_SENTIMENT"));
        let orbit = bias_orbit_from_table(env_seed("CEB_BIAS_SEED"));
        let field = bias_field_from_table(env_seed("CEB_BIAS_FIELD_SEED"));
        let tree = bias_orb_tree_from_table(env_seed("CEB_ORB_TREE_SEED"));

        let scored = candidates
            .iter()
            .map(|candidate| {
                let seed = candidate.tweet_id as u64 ^ (candidate.author_id as u64).rotate_left(13);
                let rgb = candidate
                    .ceb_color_merge_hex
                    .as_ref()
                    .and_then(parse_hex_color)
                    .unwrap_or_else(|| llm_color_from_text(&candidate.tweet_text, seed));
                let bias = candidate_bias_from_rgb(rgb, candidate);
                let mix = bias_mix_boost(1.0, bias, target);
                let muted = if target.x <= -0.8 && target.y <= -0.8 && bias.x < 0.0 {
                    0.0
                } else {
                    1.0
                };
                let orbit_score = orbit_score_for_candidate(&orbit, bias);
                let field_score = field_score_for_candidate(&field, bias);
                let tree_score = tree_score_for_candidate(&tree, bias);
                let boosted_score = candidate
                    .score
                    .map(|score| score * (1.0 + BIAS_BOOST * mix * (0.4 + 0.25 * orbit_score + 0.2 * field_score + 0.15 * tree_score)) * muted);

                PostCandidate {
                    score: boosted_score,
                    ceb_bias_x: Some(bias.x),
                    ceb_bias_y: Some(bias.y),
                    ceb_bias_sentiment: Some(bias.sentiment),
                    ceb_bias_score: Some(mix * (0.45 * orbit_score + 0.3 * field_score + 0.25 * tree_score)),
                    ceb_color_merge_hex: Some(rgb_to_hex(rgb)),
                    ..Default::default()
                }
            })
            .collect();

        Ok(scored)
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        if scored.score.is_some() {
            candidate.score = scored.score;
        }
        if scored.ceb_bias_x.is_some() {
            candidate.ceb_bias_x = scored.ceb_bias_x;
        }
        if scored.ceb_bias_y.is_some() {
            candidate.ceb_bias_y = scored.ceb_bias_y;
        }
        if scored.ceb_bias_sentiment.is_some() {
            candidate.ceb_bias_sentiment = scored.ceb_bias_sentiment;
        }
        if scored.ceb_bias_score.is_some() {
            candidate.ceb_bias_score = scored.ceb_bias_score;
        }
        if scored.ceb_color_merge_hex.is_some() {
            candidate.ceb_color_merge_hex = scored.ceb_color_merge_hex;
        }
    }
}

fn env_value(key: &str) -> f64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(-1.0, 1.0)
}

fn env_seed(key: &str) -> u64 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

fn candidate_bias_from_rgb(rgb: (u8, u8, u8), candidate: &PostCandidate) -> BiasTuning {
    let r = rgb.0 as f64 / 255.0;
    let g = rgb.1 as f64 / 255.0;
    let b = rgb.2 as f64 / 255.0;
    let x = (r - b).clamp(-1.0, 1.0);
    let y = (g - 0.5).clamp(-1.0, 1.0);
    let sentiment = candidate
        .ceb_quantum_gain
        .unwrap_or(0.5)
        .mul_add(2.0, -1.0)
        .clamp(-1.0, 1.0);
    bias_from_position(x, y, sentiment)
}

fn orbit_score_for_candidate(orbit: &[crate::ceb::BiasOrbit], bias: BiasTuning) -> f64 {
    if orbit.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for node in orbit.iter().take(24) {
        total += bias_orbit_score(bias, *node);
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { (total / count).clamp(0.0, 1.0) }
}

fn field_score_for_candidate(field: &[crate::ceb::BiasField], bias: BiasTuning) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for node in field.iter().take(24) {
        total += bias_field_score(bias, *node);
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { (total / count).clamp(0.0, 1.0) }
}

fn tree_score_for_candidate(tree: &[crate::ceb::BiasOrbNode], bias: BiasTuning) -> f64 {
    if tree.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for node in tree.iter().take(32) {
        total += bias_orb_tree_score(bias, *node);
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { (total / count).clamp(0.0, 1.0) }
}

fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}
