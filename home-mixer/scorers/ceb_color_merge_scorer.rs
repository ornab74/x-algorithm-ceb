use crate::candidate_pipeline::candidate::PostCandidate;
use crate::candidate_pipeline::query::ScoredPostsQuery;
use crate::ceb::{
    color_merge_boost, idea_overlap_score, idea_tags_from_text, llm_color_from_text, merge_color,
    palette_from_samples, quantum_metrics_from_text, rgb_to_hex, QuantumColorMetrics,
};
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_recsys_proto::user_action_sequence_data_container::Data as ProtoDataContainer;

const COLOR_MERGE_BOOST: f64 = 0.18;
const PALETTE_LIMIT: usize = 6;

pub struct CebColorMergeScorer;

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for CebColorMergeScorer {
    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        let user_profile = build_user_profile(query);

        let scored = candidates
            .iter()
            .map(|candidate| {
                let seed = candidate.tweet_id as u64 ^ (candidate.author_id as u64).rotate_left(9);
                let base_rgb = candidate
                    .ceb_color_hex
                    .as_ref()
                    .and_then(parse_hex_color)
                    .unwrap_or_else(|| llm_color_from_text(&candidate.tweet_text, seed));

                let metrics = candidate_metrics(candidate, &candidate.tweet_text, seed);
                let (merged_rgb, merge_score) = merge_color(base_rgb, &user_profile.palette);
                let merge_boost = color_merge_boost(merge_score, metrics);
                let idea_tags = candidate_idea_tags(candidate, &candidate.tweet_text, seed);
                let idea_overlap = idea_overlap_score(&idea_tags, &user_profile.idea_tags);
                let boosted_score = candidate
                    .score
                    .map(|score| {
                        score * (1.0 + COLOR_MERGE_BOOST * merge_boost * (0.7 + 0.3 * idea_overlap))
                    });

                PostCandidate {
                    score: boosted_score,
                    ceb_color_merge_hex: Some(rgb_to_hex(merged_rgb)),
                    ceb_color_merge_score: Some(merge_score),
                    ceb_entropy: Some(metrics.entropy),
                    ceb_quantum_gain: Some(metrics.quantum_gain),
                    ceb_drift: Some(metrics.drift),
                    ceb_idea_tags: idea_tags,
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
        if scored.ceb_color_merge_hex.is_some() {
            candidate.ceb_color_merge_hex = scored.ceb_color_merge_hex;
        }
        if scored.ceb_color_merge_score.is_some() {
            candidate.ceb_color_merge_score = scored.ceb_color_merge_score;
        }
        if scored.ceb_entropy.is_some() {
            candidate.ceb_entropy = scored.ceb_entropy;
        }
        if scored.ceb_quantum_gain.is_some() {
            candidate.ceb_quantum_gain = scored.ceb_quantum_gain;
        }
        if scored.ceb_drift.is_some() {
            candidate.ceb_drift = scored.ceb_drift;
        }
        if !scored.ceb_idea_tags.is_empty() {
            candidate.ceb_idea_tags = scored.ceb_idea_tags;
        }
    }
}

struct CebUserProfile {
    palette: Vec<(u8, u8, u8)>,
    idea_tags: Vec<String>,
}

fn candidate_metrics(candidate: &PostCandidate, text: &str, seed: u64) -> QuantumColorMetrics {
    if let (Some(entropy), Some(quantum_gain), Some(drift)) = (
        candidate.ceb_entropy,
        candidate.ceb_quantum_gain,
        candidate.ceb_drift,
    ) {
        return QuantumColorMetrics {
            entropy,
            drift,
            quantum_gain,
        };
    }
    quantum_metrics_from_text(text, seed)
}

fn candidate_idea_tags(candidate: &PostCandidate, text: &str, seed: u64) -> Vec<String> {
    if !candidate.ceb_idea_tags.is_empty() {
        return candidate.ceb_idea_tags.clone();
    }
    idea_tags_from_text(text, seed)
}

fn build_user_profile(query: &ScoredPostsQuery) -> CebUserProfile {
    let mut samples: Vec<String> = Vec::new();

    if let Some(sequence) = query.user_action_sequence.as_ref() {
        if let Some(container) = sequence.user_actions_data.as_ref() {
            if let Some(ProtoDataContainer::OrderedAggregatedUserActionsList(list)) =
                container.data.as_ref()
            {
                for action in &list.aggregated_user_actions {
                    samples.push(format!("{:?}", action));
                }
            }
        }
    }

    let palette = if samples.is_empty() {
        Vec::new()
    } else {
        palette_from_samples(&samples, query.user_id as u64, PALETTE_LIMIT)
    };

    let idea_tags = if samples.is_empty() {
        Vec::new()
    } else {
        let mut tags = Vec::new();
        for sample in &samples {
            let seed = query.user_id as u64 ^ (sample.len() as u64).rotate_left(7);
            tags.extend(idea_tags_from_text(sample, seed));
        }
        tags
    };

    CebUserProfile { palette, idea_tags }
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
