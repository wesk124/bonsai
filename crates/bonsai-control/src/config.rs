//! `bonsai-ctl` configuration file (TOML).
//!
//! Template with every option documented: `config/bonsai-ctl.example.toml`.
//! User guide: `docs/CONTROL_PLANE.md`.
//!
//! Codecs and precisions are referred to by name. Built-in names:
//!
//! | name            | meaning                                                         |
//! |-----------------|-----------------------------------------------------------------|
//! | `bf16`          | uncompressed baseline                                           |
//! | `int8`          | symmetric, 1 scale per (layer, head, token) over head_dim       |
//! | `int4`          | asymmetric g32; K per-channel (KIVI-style), V per-token         |
//! | `int4-pertoken` | asymmetric g32; K and V per-token                               |
//! | `atom-kv4`      | Atom (MLSys'24) KV scheme: asymmetric, 1 group per (head, token) |
//!
//! Every codec name (built-in or `[codecs.*]`) is also a uniform precision.
//! `[precisions.*]` defines per-layer mixed precisions.

use std::{collections::BTreeMap, fmt::Write as _, net::SocketAddr, path::{Path, PathBuf}, time::Duration};

use bonsai_core::{
    BonsaiError, ChunkPlan, ChunkSpec, CodecSpec, ElemType, GroupAxis, KvLayout, LayerOverride, PrecisionMap,
    QuantBits, QuantParams, Result,
};
use serde::Deserialize;

use crate::{
    coordinator::CoordinatorConfig,
    policy::{
        AdaptivePrecisionPolicy, Candidate, CodecPolicy, CodecRates, CostModel, LeastLoaded, PlacementPolicy,
        StaticCodecPolicy,
    },
    sink::{JsonlSink, MemorySink, MetricsSink},
};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CtlConfig {
    pub server: ServerSection,
    pub model: ModelSection,
    #[serde(default)]
    pub admission: AdmissionSection,
    #[serde(default)]
    pub timeouts: TimeoutSection,
    #[serde(default)]
    pub failure: FailureSection,
    #[serde(default)]
    pub placement: PlacementSection,
    pub codec: CodecSection,
    #[serde(default)]
    pub codecs: BTreeMap<String, CodecDef>,
    #[serde(default)]
    pub precisions: BTreeMap<String, PrecisionDef>,
    #[serde(default)]
    pub metrics: MetricsSection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    pub listen: SocketAddr,
    #[serde(default = "d_tick")]
    pub tick_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    /// Workers must register with the same name and layout.
    pub name: String,
    /// `"qwen2.5-1.5b-instruct"`, or omit and give the four fields below.
    pub preset: Option<String>,
    pub num_layers: Option<u16>,
    pub num_kv_heads: Option<u16>,
    pub head_dim: Option<u16>,
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionSection {
    #[serde(default = "d_max_queue")]
    pub max_queue: usize,
    #[serde(default = "d_lease_attempts")]
    pub max_lease_attempts: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeoutSection {
    #[serde(default = "d_heartbeat")]
    pub heartbeat_ms: u64,
    #[serde(default = "d_handoff")]
    pub handoff_ms: u64,
    #[serde(default = "d_decode")]
    pub decode_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureSection {
    #[serde(default = "d_true")]
    pub fallback_to_bf16: bool,
    #[serde(default = "d_retries")]
    pub max_retries: u32,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlacementKind {
    #[default]
    LeastLoaded,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementSection {
    #[serde(default)]
    pub policy: PlacementKind,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodecPolicyKind {
    Static,
    Adaptive,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecSection {
    pub policy: CodecPolicyKind,
    #[serde(rename = "static")]
    pub static_: Option<StaticSection>,
    pub adaptive: Option<AdaptiveSection>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticSection {
    pub precision: String,
    #[serde(default)]
    pub chunk: ChunkDef,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkDef {
    /// 0 = whole sequence.
    #[serde(default)]
    pub tokens: u32,
    /// 0 = all layers.
    #[serde(default)]
    pub layers: u16,
}

impl From<ChunkDef> for ChunkSpec {
    fn from(c: ChunkDef) -> Self {
        ChunkSpec { tokens_per_chunk: c.tokens, layers_per_chunk: c.layers }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveSection {
    pub max_quality_loss: f64,
    #[serde(default = "d_min_gain")]
    pub min_gain: f64,
    #[serde(default = "d_alpha")]
    pub bandwidth_ewma_alpha: f64,
    pub cost_model: CostModelDef,
    pub candidates: Vec<CandidateDef>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateDef {
    pub precision: String,
    #[serde(default)]
    pub chunk: ChunkDef,
    /// `[[max_context_tokens, f1_loss_points], ...]`, ascending. Contexts longer
    /// than the last bucket are treated as unmeasured (candidate not eligible).
    pub quality_loss: Vec<(u32, f64)>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RatesDef {
    pub encode_gbps: f64,
    pub decode_gbps: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostModelDef {
    pub per_write_us: f64,
    pub per_chunk_kernel_us: f64,
    pub link_efficiency: f64,
    pub bf16: RatesDef,
    pub int8: RatesDef,
    pub int4: RatesDef,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisDef {
    HeadDim,
    Token,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantDef {
    pub group_size: u16,
    pub axis: AxisDef,
    pub symmetric: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecDef {
    /// 16 (bf16), 8 or 4.
    pub bits: u8,
    pub key: Option<QuantDef>,
    pub value: Option<QuantDef>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverrideDef {
    /// Half-open `[start, end)`.
    pub layers: [u16; 2],
    pub codec: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionDef {
    pub default: String,
    #[serde(default)]
    pub overrides: Vec<OverrideDef>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsSection {
    /// JSONL file for HandoffReports. Omit to keep reports in memory only.
    pub jsonl: Option<PathBuf>,
}

impl Default for AdmissionSection {
    fn default() -> Self {
        AdmissionSection { max_queue: d_max_queue(), max_lease_attempts: d_lease_attempts() }
    }
}
impl Default for TimeoutSection {
    fn default() -> Self {
        TimeoutSection { heartbeat_ms: d_heartbeat(), handoff_ms: d_handoff(), decode_ms: d_decode() }
    }
}
impl Default for FailureSection {
    fn default() -> Self {
        FailureSection { fallback_to_bf16: true, max_retries: d_retries() }
    }
}

fn d_tick() -> u64 { 50 }
fn d_max_queue() -> usize { 1024 }
fn d_lease_attempts() -> u32 { 2 }
fn d_heartbeat() -> u64 { 2_000 }
fn d_handoff() -> u64 { 10_000 }
fn d_decode() -> u64 { 300_000 }
fn d_true() -> bool { true }
fn d_retries() -> u32 { 1 }
fn d_min_gain() -> f64 { 0.1 }
fn d_alpha() -> f64 { 0.2 }

/// Everything `bonsai-ctl` needs to start, built from a [`CtlConfig`].
pub struct Resolved {
    pub listen: SocketAddr,
    pub tick: Duration,
    pub coordinator: CoordinatorConfig,
    pub placement: Box<dyn PlacementPolicy>,
    pub codec_policy: Box<dyn CodecPolicy>,
}

fn invalid(msg: impl Into<String>) -> BonsaiError {
    BonsaiError::InvalidConfig(msg.into())
}

pub fn builtin_codec(name: &str, layout: &KvLayout) -> Option<CodecSpec> {
    let asym = |g, axis| QuantParams { group_size: g, axis, symmetric: false };
    Some(match name {
        "bf16" => CodecSpec::Bf16,
        "int8" => CodecSpec::int8_default(layout),
        "int4" => CodecSpec::int4_default(),
        "int4-pertoken" => CodecSpec::Quant {
            bits: QuantBits::Int4Packed,
            key: asym(32, GroupAxis::HeadDim),
            value: asym(32, GroupAxis::HeadDim),
        },
        "atom-kv4" => CodecSpec::Quant {
            bits: QuantBits::Int4Packed,
            key: asym(layout.head_dim, GroupAxis::HeadDim),
            value: asym(layout.head_dim, GroupAxis::HeadDim),
        },
        _ => return None,
    })
}

pub const BUILTIN_CODECS: &[&str] = &["bf16", "int8", "int4", "int4-pertoken", "atom-kv4"];

impl CtlConfig {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| invalid(e.to_string()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::from_toml_str(&std::fs::read_to_string(path)?)
    }

    pub fn layout(&self) -> Result<KvLayout> {
        let m = &self.model;
        match m.preset.as_deref() {
            Some("qwen2.5-1.5b-instruct") => Ok(KvLayout::QWEN2_5_1_5B),
            Some(other) => Err(invalid(format!("unknown model preset {other:?}"))),
            None => {
                let need = |v: Option<u16>, f: &str| v.ok_or_else(|| invalid(format!("model.{f} required without preset")));
                let dtype = match m.dtype.as_deref().unwrap_or("bf16") {
                    "bf16" => ElemType::Bf16,
                    "f16" => ElemType::F16,
                    "f32" => ElemType::F32,
                    d => return Err(invalid(format!("unknown model.dtype {d:?}"))),
                };
                Ok(KvLayout {
                    num_layers: need(m.num_layers, "num_layers")?,
                    num_kv_heads: need(m.num_kv_heads, "num_kv_heads")?,
                    head_dim: need(m.head_dim, "head_dim")?,
                    dtype,
                })
            }
        }
    }

    pub fn codec(&self, name: &str) -> Result<CodecSpec> {
        let layout = self.layout()?;
        if let Some(def) = self.codecs.get(name) {
            let q = |d: Option<QuantDef>, which: &str| {
                let d = d.ok_or_else(|| invalid(format!("codecs.{name}.{which} required for {}-bit", def.bits)))?;
                Ok::<_, BonsaiError>(QuantParams {
                    group_size: d.group_size,
                    axis: match d.axis {
                        AxisDef::HeadDim => GroupAxis::HeadDim,
                        AxisDef::Token => GroupAxis::Token,
                    },
                    symmetric: d.symmetric,
                })
            };
            let spec = match def.bits {
                16 => CodecSpec::Bf16,
                8 => CodecSpec::Quant { bits: QuantBits::Int8, key: q(def.key, "key")?, value: q(def.value, "value")? },
                4 => CodecSpec::Quant { bits: QuantBits::Int4Packed, key: q(def.key, "key")?, value: q(def.value, "value")? },
                b => return Err(invalid(format!("codecs.{name}.bits must be 16, 8 or 4 (got {b})"))),
            };
            spec.validate(&layout)?;
            return Ok(spec);
        }
        builtin_codec(name, &layout).ok_or_else(|| invalid(format!("unknown codec {name:?}")))
    }

    pub fn precision(&self, name: &str) -> Result<PrecisionMap> {
        let layout = self.layout()?;
        let p = if let Some(def) = self.precisions.get(name) {
            let mut overrides = Vec::new();
            for o in &def.overrides {
                let [a, b] = o.layers;
                if a >= b || b > layout.num_layers {
                    return Err(invalid(format!(
                        "precisions.{name}: layer range [{a}, {b}) invalid for {} layers",
                        layout.num_layers
                    )));
                }
                overrides.push(LayerOverride { layers: a..b, codec: self.codec(&o.codec)? });
            }
            PrecisionMap { default: self.codec(&def.default)?, overrides }
        } else {
            PrecisionMap::uniform(self.codec(name)?)
        };
        p.validate(&layout)?;
        Ok(p)
    }

    fn check_plan(&self, what: &str, precision: &PrecisionMap, chunk: ChunkSpec) -> Result<()> {
        // A context length that is a multiple of every alignment, so only
        // chunk-size/alignment mistakes are reported.
        let layout = self.layout()?;
        let seq = 4096u32.max(chunk.tokens_per_chunk * 2);
        ChunkPlan::build(layout, precision.clone(), seq, chunk).map(|_| ()).map_err(|e| invalid(format!("{what}: {e}")))
    }

    /// Validate everything and build the coordinator's components.
    pub fn resolve(&self) -> Result<Resolved> {
        let layout = self.layout()?;
        let alpha = self.codec.adaptive.as_ref().map(|a| a.bandwidth_ewma_alpha).unwrap_or(d_alpha());
        let coordinator = CoordinatorConfig {
            layout,
            model: self.model.name.clone(),
            heartbeat_timeout: Duration::from_millis(self.timeouts.heartbeat_ms),
            handoff_timeout: Duration::from_millis(self.timeouts.handoff_ms),
            decode_timeout: Duration::from_millis(self.timeouts.decode_ms),
            max_lease_attempts: self.admission.max_lease_attempts.max(1),
            fallback_to_bf16: self.failure.fallback_to_bf16,
            max_retries: self.failure.max_retries,
            max_queue: self.admission.max_queue,
            bandwidth_ewma_alpha: alpha,
        };
        // Validate every named precision, even unused ones, so typos surface early.
        for name in self.precisions.keys() {
            self.precision(name)?;
        }
        let codec_policy: Box<dyn CodecPolicy> = match self.codec.policy {
            CodecPolicyKind::Static => {
                let s = self.codec.static_.as_ref().ok_or_else(|| invalid("codec.policy = \"static\" needs [codec.static]"))?;
                let precision = self.precision(&s.precision)?;
                self.check_plan("codec.static", &precision, s.chunk.into())?;
                Box::new(StaticCodecPolicy { precision, chunk: s.chunk.into() })
            }
            CodecPolicyKind::Adaptive => {
                let a = self
                    .codec
                    .adaptive
                    .as_ref()
                    .ok_or_else(|| invalid("codec.policy = \"adaptive\" needs [codec.adaptive]"))?;
                if a.candidates.is_empty() {
                    return Err(invalid("codec.adaptive.candidates is empty"));
                }
                let mut candidates = Vec::new();
                for (i, c) in a.candidates.iter().enumerate() {
                    let precision = self.precision(&c.precision)?;
                    self.check_plan(&format!("codec.adaptive.candidates[{i}]"), &precision, c.chunk.into())?;
                    if !c.quality_loss.windows(2).all(|w| w[0].0 < w[1].0) {
                        return Err(invalid(format!("codec.adaptive.candidates[{i}].quality_loss must be ascending by context")));
                    }
                    candidates.push(Candidate { precision, chunk: c.chunk.into(), quality_loss_by_ctx: c.quality_loss.clone() });
                }
                let r = |d: RatesDef| CodecRates { encode_bps: d.encode_gbps * 1e9, decode_bps: d.decode_gbps * 1e9 };
                let m = &a.cost_model;
                Box::new(AdaptivePrecisionPolicy {
                    candidates,
                    model: CostModel {
                        bf16: r(m.bf16),
                        int8: r(m.int8),
                        int4: r(m.int4),
                        per_write_ns: m.per_write_us * 1e3,
                        per_chunk_kernel_ns: m.per_chunk_kernel_us * 1e3,
                        link_efficiency: m.link_efficiency,
                    },
                    max_quality_loss: a.max_quality_loss,
                    min_gain: a.min_gain,
                })
            }
        };
        let placement: Box<dyn PlacementPolicy> = match self.placement.policy {
            PlacementKind::LeastLoaded => Box::new(LeastLoaded),
        };
        Ok(Resolved {
            listen: self.server.listen,
            tick: Duration::from_millis(self.server.tick_ms.max(1)),
            coordinator,
            placement,
            codec_policy,
        })
    }

    /// Opens the metrics sink (creates/truncates the JSONL file).
    pub fn open_sink(&self) -> Result<Box<dyn MetricsSink>> {
        Ok(match &self.metrics.jsonl {
            Some(p) => {
                if let Some(dir) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                    std::fs::create_dir_all(dir)?;
                }
                Box::new(JsonlSink::create(p)?)
            }
            None => Box::new(MemorySink::default()),
        })
    }

    /// Human-readable summary for `bonsai-ctl --check`: every codec/precision
    /// with its size on the wire.
    pub fn summary(&self) -> Result<String> {
        let layout = self.layout()?;
        let raw = layout.raw_bytes_per_token() as f64;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "model {} : {} layers x {} kv heads x {} dim, {:?} -> {} B/token raw\n",
            self.model.name, layout.num_layers, layout.num_kv_heads, layout.head_dim, layout.dtype, raw
        );
        let _ = writeln!(out, "{:<16} {:>10} {:>7} {:>10} {:>10} {:>10}", "precision", "B/token", "ratio", "1K MiB", "8K MiB", "32K MiB");
        let mut names: Vec<String> = BUILTIN_CODECS.iter().map(|s| s.to_string()).collect();
        names.extend(self.codecs.keys().cloned());
        names.extend(self.precisions.keys().cloned());
        for name in names {
            let p = self.precision(&name)?;
            let per_tok = p.encoded_bytes(&layout, 32 * 1024) as f64 / (32.0 * 1024.0);
            let mib = |t: u64| p.encoded_bytes(&layout, t) as f64 / (1 << 20) as f64;
            let _ = writeln!(
                out,
                "{:<16} {:>10.0} {:>6.1}% {:>10.1} {:>10.1} {:>10.1}",
                name,
                per_tok,
                100.0 * per_tok / raw,
                mib(1024),
                mib(8192),
                mib(32768)
            );
        }
        let _ = writeln!(out);
        match self.codec.policy {
            CodecPolicyKind::Static => {
                if let Some(s) = &self.codec.static_ {
                    let _ = writeln!(out, "codec policy: static, precision={} chunk={:?}", s.precision, s.chunk);
                }
            }
            CodecPolicyKind::Adaptive => {
                if let Some(a) = &self.codec.adaptive {
                    let _ = writeln!(
                        out,
                        "codec policy: adaptive, {} candidates, max_quality_loss={}, min_gain={}",
                        a.candidates.len(),
                        a.max_quality_loss,
                        a.min_gain
                    );
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = include_str!("../../../config/bonsai-ctl.example.toml");

    #[test]
    fn example_template_resolves() {
        let cfg = CtlConfig::from_toml_str(TEMPLATE).expect("template parses");
        cfg.resolve().expect("template resolves");
        let s = cfg.summary().unwrap();
        assert!(s.contains("int8-edges"), "{s}");
    }

    #[test]
    fn minimal_config() {
        let cfg = CtlConfig::from_toml_str(
            r#"
            [server]
            listen = "0.0.0.0:7000"
            [model]
            name = "Qwen/Qwen2.5-1.5B-Instruct"
            preset = "qwen2.5-1.5b-instruct"
            [codec]
            policy = "static"
            [codec.static]
            precision = "int8"
            "#,
        )
        .unwrap();
        let r = cfg.resolve().unwrap();
        assert_eq!(r.coordinator.max_queue, 1024);
    }

    #[test]
    fn rejects_typos_and_bad_chunking() {
        let base = r#"
            [server]
            listen = "0.0.0.0:7000"
            [model]
            name = "m"
            preset = "qwen2.5-1.5b-instruct"
        "#;
        // unknown key
        assert!(CtlConfig::from_toml_str(&format!("{base}\n[codec]\npolicy = \"static\"\nprecison = \"x\"")).is_err());
        // unknown precision
        let c = CtlConfig::from_toml_str(&format!("{base}\n[codec]\npolicy = \"static\"\n[codec.static]\nprecision = \"int3\"")).unwrap();
        assert!(c.resolve().is_err());
        // int4 (per-channel K, g32) with a 100-token chunk is misaligned
        let c = CtlConfig::from_toml_str(&format!(
            "{base}\n[codec]\npolicy = \"static\"\n[codec.static]\nprecision = \"int4\"\nchunk = {{ tokens = 100 }}"
        ))
        .unwrap();
        assert!(c.resolve().is_err());
    }

    #[test]
    fn atom_kv4_matches_paper_scheme() {
        let l = KvLayout::QWEN2_5_1_5B;
        let CodecSpec::Quant { bits, key, value } = builtin_codec("atom-kv4", &l).unwrap() else { panic!() };
        assert_eq!(bits, QuantBits::Int4Packed);
        for p in [key, value] {
            assert_eq!((p.group_size, p.axis, p.symmetric), (128, GroupAxis::HeadDim, false));
        }
    }
}
