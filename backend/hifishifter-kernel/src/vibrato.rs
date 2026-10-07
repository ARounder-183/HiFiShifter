//! 颤音预设的持久化模型。
//!
//! 【定位】后端**只做透传存储**：字段语义、取值范围与默认值全部在前端收口
//! （见 `frontend/src/features/vibrato/`）。这与 `UiSettings::custom_scale_presets`
//! 的处理方式一致 —— "支持哪些取值"属于业务知识，写两遍必然漂移。
//!
//! 【为什么是一组宽松字段而不是枚举】波形的"形状"与"包络曲线"在前端是字符串
//! 联合类型，取值集合会随版本演进。这里收成 `String` 并给默认值，用户降级
//! 运行旧版本时未知取值也能原样保留，而不是被反序列化拒绝。
//!
//! 【周期波形】`cycle` 是 `serde_json::Value`：它要么是参数式
//! （`{ kind: "shape", shape, skew }`），要么是采样式（`{ kind: "table", table }`）。
//! 用 `Value` 而不是具体结构体，正是为了让"前端新增波形种类"不必同步改后端。

use serde::{Deserialize, Serialize};

/// 颤音预设。
///
/// 字段名经 `rename_all = "camelCase"` 后与前端 `VibratoPreset` 逐字对应。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VibratoPreset {
    /// `builtin.<name>` 为系统预设（只存在于前端代码，不会被写进用户预设）；
    /// `custom_<rand>` 为用户预设。
    #[serde(default)]
    pub id: String,
    /// 用户可见名。系统预设的名称走前端 i18n，此字段仅作兜底。
    #[serde(default)]
    pub name: String,
    /// 系统预设只读标记。前端以 id 前缀为准，此字段仅作记录。
    #[serde(default)]
    pub builtin: bool,

    /// 归一化周期波形（参数式或采样式，形状由前端定义）。
    #[serde(default)]
    pub cycle: serde_json::Value,

    /// 深度（cents，规范单位）。落到具体参数时由前端按参数族换算。
    #[serde(default)]
    pub depth_cents: f64,

    /// `"hz"` 或 `"cycles"`。
    #[serde(default = "default_vibrato_rate_mode")]
    pub rate_mode: String,
    #[serde(default = "default_vibrato_rate_hz")]
    pub rate_hz: f64,
    #[serde(default = "default_vibrato_cycles")]
    pub cycles: f64,
    #[serde(default = "default_vibrato_one")]
    pub rate_ramp_end: f64,
    #[serde(default)]
    pub start_phase_deg: f64,

    #[serde(default = "default_vibrato_attack_ms")]
    pub attack_ms: f64,
    #[serde(default = "default_vibrato_curve")]
    pub attack_curve: String,
    #[serde(default = "default_vibrato_release_ms")]
    pub release_ms: f64,
    #[serde(default = "default_vibrato_curve")]
    pub release_curve: String,
    /// 渐强：起点 / 终点的深度倍率。
    #[serde(default)]
    pub depth_ramp: serde_json::Value,

    #[serde(default)]
    pub align_cycles: bool,
    #[serde(default)]
    pub irregularity: f64,
    #[serde(default)]
    pub bias_cents: f64,

    /// `"line"` / `"holdStart"` / `"holdEnd"` / `"average"` / `"existing"`。
    #[serde(default = "default_vibrato_baseline")]
    pub baseline: String,
    #[serde(default = "default_vibrato_hundred")]
    pub blend: f64,
}

fn default_vibrato_rate_mode() -> String {
    "hz".to_string()
}
fn default_vibrato_rate_hz() -> f64 {
    5.5
}
fn default_vibrato_cycles() -> f64 {
    6.0
}
fn default_vibrato_one() -> f64 {
    1.0
}
fn default_vibrato_attack_ms() -> f64 {
    90.0
}
fn default_vibrato_release_ms() -> f64 {
    90.0
}
fn default_vibrato_curve() -> String {
    "exp".to_string()
}
fn default_vibrato_baseline() -> String {
    "line".to_string()
}
fn default_vibrato_hundred() -> f64 {
    100.0
}

/// 用户自定义颤音预设的**数量上限不在这里**。
///
/// 上限属于"支持哪些取值"这类业务知识，与字段语义、取值范围、默认值一样由
/// 前端收口（见 `frontend/src/features/vibrato/vibratoPresets.ts` 的
/// `MAX_VIBRATO_PRESETS`）。后端只做透传存储，不重复定义第二份。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_round_trips_through_json() {
        let json = serde_json::json!({
            "id": "custom_abc",
            "name": "My Vibrato",
            "builtin": false,
            "cycle": { "kind": "shape", "shape": "triangle", "skew": 0.4 },
            "depthCents": 42.0,
            "rateMode": "hz",
            "rateHz": 6.5,
            "cycles": 6.0,
            "rateRampEnd": 1.2,
            "startPhaseDeg": 90.0,
            "attackMs": 120.0,
            "attackCurve": "exp",
            "releaseMs": 80.0,
            "releaseCurve": "linear",
            "depthRamp": { "start": 0.4, "end": 1.0 },
            "alignCycles": true,
            "irregularity": 12.0,
            "biasCents": -5.0,
            "baseline": "existing",
            "blend": 80.0
        });
        let preset: VibratoPreset = serde_json::from_value(json).expect("应能反序列化");
        assert_eq!(preset.id, "custom_abc");
        assert_eq!(preset.depth_cents, 42.0);
        assert_eq!(preset.attack_curve, "exp");
        assert_eq!(preset.baseline, "existing");
        assert!(preset.align_cycles);

        // 往返后字段名必须是 camelCase，与前端 TS 类型一致。
        let back = serde_json::to_value(&preset).expect("应能序列化");
        assert_eq!(back["depthCents"], serde_json::json!(42.0));
        assert_eq!(back["rateRampEnd"], serde_json::json!(1.2));
        assert_eq!(back["startPhaseDeg"], serde_json::json!(90.0));
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // 旧配置里没有这些键时，反序列化必须成功并给出可用默认值。
        let preset: VibratoPreset =
            serde_json::from_value(serde_json::json!({})).expect("空对象应能反序列化");
        assert_eq!(preset.rate_mode, "hz");
        assert_eq!(preset.rate_hz, 5.5);
        assert_eq!(preset.attack_curve, "exp");
        assert_eq!(preset.baseline, "line");
        assert_eq!(preset.blend, 100.0);
        assert_eq!(preset.depth_cents, 0.0);
    }

    #[test]
    fn unknown_waveform_shape_is_preserved_verbatim() {
        // 前端将来新增波形种类时，旧后端不应丢弃数据。
        let json = serde_json::json!({
            "id": "custom_x",
            "cycle": { "kind": "table", "table": [0.0, 0.5, 1.0] },
            "baseline": "spiral"
        });
        let preset: VibratoPreset = serde_json::from_value(json).expect("应能反序列化");
        assert_eq!(preset.cycle["kind"], "table");
        assert_eq!(preset.baseline, "spiral");
    }
}
