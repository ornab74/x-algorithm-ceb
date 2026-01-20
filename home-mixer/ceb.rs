use std::collections::HashMap;

use crate::ceb_ideas::CEB_IDEAS;
use crate::ceb_bias_fields::CEB_BIAS_FIELDS;
use crate::ceb_orbit_table::CEB_ORBIT_TABLE;
use crate::ceb_orb_tree::CEB_ORB_TREE;

const MAX_RGB_DISTANCE: f64 = 441.6729559300637;
const IDEA_TAG_LIMIT: usize = 4;
const BIAS_RADIUS: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct QuantumColorMetrics {
    pub entropy: f64,
    pub drift: f64,
    pub quantum_gain: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BiasTuning {
    pub x: f64,
    pub y: f64,
    pub sentiment: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BiasOrbit {
    pub x: f64,
    pub y: f64,
    pub frequency: f64,
    pub sentiment: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BiasField {
    pub x: f64,
    pub y: f64,
    pub frequency: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BiasOrbNode {
    pub x: f64,
    pub y: f64,
    pub frequency: f64,
    pub sentiment: f64,
    pub weight: f64,
}

pub fn llm_color_from_text(text: &str, seed: u64) -> (u8, u8, u8) {
    let entropy = shannon_entropy(text);
    let hash = fnv1a_hash(text) ^ seed.rotate_left(17);
    let hue = (hash % 360) as f64;
    let saturation = clamp(0.55 + entropy / 8.0, 0.45, 0.95);
    let lightness = clamp(0.4 + entropy / 18.0, 0.3, 0.7);
    hsl_to_rgb(hue, saturation, lightness)
}

pub fn quantum_metrics_from_text(text: &str, seed: u64) -> QuantumColorMetrics {
    let entropy = shannon_entropy(text);
    let hash = fnv1a_hash(text) ^ seed.rotate_left(29);
    let phase = ((hash % 360) as f64).to_radians();
    let drift = clamp((phase.sin() * 0.6) + (entropy / 10.0), -1.0, 1.0);
    let quantum_gain = clamp((entropy / 6.5) + (phase.cos().abs() * 0.35), 0.0, 1.0);
    QuantumColorMetrics {
        entropy,
        drift,
        quantum_gain,
    }
}

pub fn merge_color(base: (u8, u8, u8), palette: &[(u8, u8, u8)]) -> ((u8, u8, u8), f64) {
    if palette.is_empty() {
        return (base, 0.0);
    }

    let mut closest = palette[0];
    let mut min_dist = color_distance(base, closest);
    for color in palette.iter().skip(1) {
        let dist = color_distance(base, *color);
        if dist < min_dist {
            min_dist = dist;
            closest = *color;
        }
    }

    let similarity = clamp(1.0 - (min_dist / MAX_RGB_DISTANCE), 0.0, 1.0);
    let blend = 0.35 + similarity * 0.4;
    let merged = blend_rgb(base, closest, blend);
    (merged, similarity)
}

pub fn color_merge_boost(merge_score: f64, metrics: QuantumColorMetrics) -> f64 {
    let entropy_gain = clamp(metrics.entropy / 8.0, 0.0, 1.0);
    let drift_gain = clamp(1.0 - metrics.drift.abs(), 0.35, 1.0);
    clamp(merge_score * (0.6 + 0.4 * metrics.quantum_gain) * entropy_gain * drift_gain, 0.0, 1.0)
}

pub fn idea_tags_from_text(text: &str, seed: u64) -> Vec<String> {
    let mut tags = Vec::new();
    if CEB_IDEAS.is_empty() {
        return tags;
    }
    let mut hash = fnv1a_hash(text) ^ seed.rotate_left(11);
    let mut used: HashMap<usize, ()> = HashMap::new();
    for _ in 0..IDEA_TAG_LIMIT {
        hash ^= hash.rotate_left(7);
        let idx = (hash as usize) % CEB_IDEAS.len();
        if used.contains_key(&idx) {
            continue;
        }
        used.insert(idx, ());
        tags.push(CEB_IDEAS[idx].to_string());
    }
    tags
}

pub fn idea_overlap_score(candidate_tags: &[String], user_tags: &[String]) -> f64 {
    if candidate_tags.is_empty() || user_tags.is_empty() {
        return 0.0;
    }
    let mut matches = 0.0;
    for tag in candidate_tags {
        if user_tags.iter().any(|t| t == tag) {
            matches += 1.0;
        }
    }
    clamp(matches / IDEA_TAG_LIMIT as f64, 0.0, 1.0)
}

pub fn bias_from_position(x: f64, y: f64, sentiment: f64) -> BiasTuning {
    BiasTuning {
        x: clamp(x, -1.0, 1.0),
        y: clamp(y, -1.0, 1.0),
        sentiment: clamp(sentiment, -1.0, 1.0),
    }
}

pub fn bias_distance(a: BiasTuning, b: BiasTuning) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let ds = a.sentiment - b.sentiment;
    ((dx * dx) + (dy * dy) + (ds * ds)).sqrt() / (BIAS_RADIUS * 1.7320508075688772)
}

pub fn bias_mix_boost(base: f64, bias: BiasTuning, target: BiasTuning) -> f64 {
    let dist = bias_distance(bias, target);
    let closeness = clamp(1.0 - dist, 0.0, 1.0);
    clamp(base * (0.5 + 0.5 * closeness), 0.0, 1.0)
}

pub fn bias_orbit_from_table(seed: u64) -> Vec<BiasOrbit> {
    let mut out = Vec::with_capacity(CEB_ORBIT_TABLE.len());
    let offset = (seed as usize) % CEB_ORBIT_TABLE.len();
    for i in 0..CEB_ORBIT_TABLE.len() {
        let idx = (i + offset) % CEB_ORBIT_TABLE.len();
        let (x, y, f, s) = CEB_ORBIT_TABLE[idx];
        out.push(BiasOrbit {
            x,
            y,
            frequency: clamp(f, 0.0, 1.0),
            sentiment: clamp(s, -1.0, 1.0),
        });
    }
    out
}

pub fn bias_orbit_score(bias: BiasTuning, orbit: BiasOrbit) -> f64 {
    let target = BiasTuning {
        x: orbit.x,
        y: orbit.y,
        sentiment: orbit.sentiment,
    };
    let dist = bias_distance(bias, target);
    clamp((1.0 - dist) * orbit.frequency, 0.0, 1.0)
}

pub fn bias_field_from_table(seed: u64) -> Vec<BiasField> {
    let mut out = Vec::with_capacity(CEB_BIAS_FIELDS.len());
    let offset = (seed as usize) % CEB_BIAS_FIELDS.len();
    for i in 0..CEB_BIAS_FIELDS.len() {
        let idx = (i + offset) % CEB_BIAS_FIELDS.len();
        let (x, y, f) = CEB_BIAS_FIELDS[idx];
        out.push(BiasField {
            x,
            y,
            frequency: clamp(f, 0.0, 1.0),
        });
    }
    out
}

pub fn bias_field_score(bias: BiasTuning, field: BiasField) -> f64 {
    let dx = bias.x - field.x;
    let dy = bias.y - field.y;
    let dist = ((dx * dx) + (dy * dy)).sqrt();
    clamp((1.0 - dist / 1.4142135623730951) * field.frequency, 0.0, 1.0)
}

pub fn bias_orb_tree_from_table(seed: u64) -> Vec<BiasOrbNode> {
    let mut out = Vec::with_capacity(CEB_ORB_TREE.len());
    let offset = (seed as usize) % CEB_ORB_TREE.len();
    for i in 0..CEB_ORB_TREE.len() {
        let idx = (i + offset) % CEB_ORB_TREE.len();
        let (x, y, f, s, w) = CEB_ORB_TREE[idx];
        out.push(BiasOrbNode {
            x,
            y,
            frequency: clamp(f, 0.0, 1.0),
            sentiment: clamp(s, -1.0, 1.0),
            weight: clamp(w, 0.0, 1.0),
        });
    }
    out
}

pub fn bias_orb_tree_score(bias: BiasTuning, node: BiasOrbNode) -> f64 {
    let target = BiasTuning {
        x: node.x,
        y: node.y,
        sentiment: node.sentiment,
    };
    let dist = bias_distance(bias, target);
    clamp((1.0 - dist) * node.frequency * (0.6 + 0.4 * node.weight), 0.0, 1.0)
}

pub fn rgb_to_hex(rgb: (u8, u8, u8)) -> String {
    format!("#{:02X}{:02X}{:02X}", rgb.0, rgb.1, rgb.2)
}

pub fn palette_from_samples(samples: &[String], seed: u64, limit: usize) -> Vec<(u8, u8, u8)> {
    let mut counts: HashMap<String, (u8, u8, u8, usize)> = HashMap::new();
    for sample in samples {
        let rgb = llm_color_from_text(sample, seed ^ fnv1a_hash(sample));
        let key = rgb_to_hex(rgb);
        let entry = counts.entry(key).or_insert((rgb.0, rgb.1, rgb.2, 0));
        entry.3 += 1;
    }

    let mut palette: Vec<_> = counts
        .values()
        .map(|(r, g, b, count)| ((*r, *g, *b), *count))
        .collect();
    palette.sort_by(|a, b| b.1.cmp(&a.1));
    palette.into_iter().take(limit).map(|(rgb, _)| rgb).collect()
}

fn fnv1a_hash(input: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn shannon_entropy(text: &str) -> f64 {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    for byte in bytes {
        *counts.entry(*byte).or_insert(0) += 1;
    }
    let len = bytes.len() as f64;
    counts
        .values()
        .map(|count| {
            let p = *count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());

    let (r1, g1, b1) = match h_prime {
        h if (0.0..1.0).contains(&h) => (c, x, 0.0),
        h if (1.0..2.0).contains(&h) => (x, c, 0.0),
        h if (2.0..3.0).contains(&h) => (0.0, c, x),
        h if (3.0..4.0).contains(&h) => (0.0, x, c),
        h if (4.0..5.0).contains(&h) => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let m = l - c / 2.0;
    (
        ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn blend_rgb(a: (u8, u8, u8), b: (u8, u8, u8), blend: f64) -> (u8, u8, u8) {
    let blend = clamp(blend, 0.0, 1.0);
    let inv = 1.0 - blend;
    (
        ((a.0 as f64 * inv) + (b.0 as f64 * blend)).round().clamp(0.0, 255.0) as u8,
        ((a.1 as f64 * inv) + (b.1 as f64 * blend)).round().clamp(0.0, 255.0) as u8,
        ((a.2 as f64 * inv) + (b.2 as f64 * blend)).round().clamp(0.0, 255.0) as u8,
    )
}

fn color_distance(a: (u8, u8, u8), b: (u8, u8, u8)) -> f64 {
    let dr = a.0 as f64 - b.0 as f64;
    let dg = a.1 as f64 - b.1 as f64;
    let db = a.2 as f64 - b.2 as f64;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}
