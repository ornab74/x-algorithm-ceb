use crate::candidate_pipeline::candidate::PostCandidate;
use crate::candidate_pipeline::query::ScoredPostsQuery;
use crate::ceb::{idea_tags_from_text, llm_color_from_text, quantum_metrics_from_text, rgb_to_hex};
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;

#[derive(Default)]
pub struct CebColorHydrator;

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for CebColorHydrator {
    async fn hydrate(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        let hydrated = candidates
            .iter()
            .map(|candidate| {
                let seed = candidate.tweet_id as u64 ^ (candidate.author_id as u64).rotate_left(9);
                let rgb = llm_color_from_text(&candidate.tweet_text, seed);
                let metrics = quantum_metrics_from_text(&candidate.tweet_text, seed);
                let idea_tags = idea_tags_from_text(&candidate.tweet_text, seed);
                PostCandidate {
                    ceb_color_hex: Some(rgb_to_hex(rgb)),
                    ceb_entropy: Some(metrics.entropy),
                    ceb_quantum_gain: Some(metrics.quantum_gain),
                    ceb_drift: Some(metrics.drift),
                    ceb_idea_tags: idea_tags,
                    ..Default::default()
                }
            })
            .collect();
        Ok(hydrated)
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        if hydrated.ceb_color_hex.is_some() {
            candidate.ceb_color_hex = hydrated.ceb_color_hex;
        }
        if hydrated.ceb_entropy.is_some() {
            candidate.ceb_entropy = hydrated.ceb_entropy;
        }
        if hydrated.ceb_quantum_gain.is_some() {
            candidate.ceb_quantum_gain = hydrated.ceb_quantum_gain;
        }
        if hydrated.ceb_drift.is_some() {
            candidate.ceb_drift = hydrated.ceb_drift;
        }
        if !hydrated.ceb_idea_tags.is_empty() {
            candidate.ceb_idea_tags = hydrated.ceb_idea_tags;
        }
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
