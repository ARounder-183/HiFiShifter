//! 把 `UiSettings` 下发到**进程级**的消费方。
//!
//! 【为什么单独成模块】这些副作用原先只写在 App 的 `save_ui_settings` 里，插件
//! 那侧一个都没有 —— 于是插件里改「默认拉伸算法」完全不起作用（值存下来了，
//! 没有任何人读它去更新拉伸默认值），推理设备同理。这不是"少写几行"，而是
//! **同一个设置在两个形态里行为不同**：用户在 App 里能感知的开关，进了 REAPER
//! 就是死的。
//!
//! 因此下发逻辑只有这一份，两个宿主都调用它。谁"拥有"这个设置与谁"消费"它无关。
//!
//! 【为什么返回值是"变化项"而不是"是否成功"】下发本身不失败（最坏是记一条日志）。
//! 调用方真正需要知道的是**哪些项真的变了** —— 只有变了才值得让渲染缓存失效、
//! 重排队后台渲染。`get_ui_settings` 是读路径且调用频繁，没有这个去重，每次读设置
//! 都会把三个 ONNX 会话全部拆掉重建（CoreML 上单个模型重编译 0.4~1.2 秒）。

use crate::config::UiSettings;
use crate::time_stretch::UserStretchAlgorithm;
use std::sync::{Mutex, OnceLock};

/// 上一次下发时生效的取值。用来判断"这次是不是真的变了"。
#[derive(Clone, PartialEq)]
struct AppliedKey {
    ort_ep: String,
    ort_device_id: Option<i32>,
    default_stretch_algorithm: UserStretchAlgorithm,
    default_hifigan_mel_stretch: bool,
}

/// 本次下发中真正发生变化的项。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppliedChanges {
    /// 推理设备（EP 或设备序号）变了 —— 会话已被销毁，调用方应让依赖它的缓存失效。
    pub inference_device_changed: bool,
    /// 全局拉伸默认值变了 —— 当前工程若未覆盖该项，渲染结果会变。
    pub stretch_defaults_changed: bool,
}

fn applied_key() -> &'static Mutex<Option<AppliedKey>> {
    static SLOT: OnceLock<Mutex<Option<AppliedKey>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// 比较两次取值，得出真正变化的项（纯函数，便于测试）。
///
/// `None` 表示"此前没有下发过"，因此两项都算变化。
fn changes_between(previous: Option<&AppliedKey>, next: &AppliedKey) -> AppliedChanges {
    match previous {
        Some(previous) => AppliedChanges {
            inference_device_changed: previous.ort_ep != next.ort_ep
                || previous.ort_device_id != next.ort_device_id,
            stretch_defaults_changed: previous.default_stretch_algorithm
                != next.default_stretch_algorithm
                || previous.default_hifigan_mel_stretch != next.default_hifigan_mel_stretch,
        },
        None => AppliedChanges {
            inference_device_changed: true,
            stretch_defaults_changed: true,
        },
    }
}

/// 记下本次取值并返回变化项。与副作用分离，使"去重判定"可被单独测试。
fn record(next: AppliedKey) -> AppliedChanges {
    let mut guard = applied_key().lock().unwrap_or_else(|e| e.into_inner());
    let changes = changes_between(guard.as_ref(), &next);
    *guard = Some(next);
    changes
}

/// 把设置下发给 ONNX 会话、拉伸默认值、导入策略与渲染缓存。
///
/// 幂等：取值未变时不触碰任何会话。首次调用视为"全部变化"（此前没有任何取值生效）。
pub fn apply(settings: &UiSettings) -> AppliedChanges {
    let changes = record(AppliedKey {
        ort_ep: settings.ort_ep.clone(),
        ort_device_id: settings.ort_device_id,
        default_stretch_algorithm: settings.default_stretch_algorithm,
        default_hifigan_mel_stretch: settings.default_hifigan_mel_stretch,
    });

    if changes.inference_device_changed {
        // 三个模型模块各自持有会话；只重建取值真的变了的那一类也要全部走一遍，
        // 因为 EP 是全局选择（`update_ort_ep` 内部按 EP 去重，重复调用是廉价的）。
        crate::nsf_hifigan_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::hnsep_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
        crate::fcpe_onnx::update_ort_ep(&settings.ort_ep, settings.ort_device_id);
    }

    // 以下三项是廉价的纯赋值，不做去重 —— 它们没有"重建会话"那种代价，
    // 而漏掉一次下发的后果（导入策略/拉伸默认值与用户选择不一致）更隐蔽。
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    crate::config::set_loop_new_clips_default(settings.loop_new_clips);
    crate::config::set_sync_edits_across_takes(settings.sync_edits_across_takes);
    crate::config::set_channel_import_policy(&settings.channel_import_policy);
    crate::render_cache::apply_settings(&settings.render_cache);

    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ep: &str, device: Option<i32>) -> AppliedKey {
        AppliedKey {
            ort_ep: ep.to_string(),
            ort_device_id: device,
            default_stretch_algorithm: UserStretchAlgorithm::default(),
            default_hifigan_mel_stretch: true,
        }
    }

    /// 首次下发（此前没有任何取值生效）必须报告两项都变了。
    #[test]
    fn the_first_application_reports_everything_as_changed() {
        let changes = changes_between(None, &key("cpu", None));
        assert_eq!(
            changes,
            AppliedChanges {
                inference_device_changed: true,
                stretch_defaults_changed: true,
            }
        );
    }

    /// 取值完全相同 → 什么都不算变。
    ///
    /// 【为什么这条最重要】`get_ui_settings` 是读路径且被频繁调用；没有这个去重，
    /// 每次读设置都会重建全部 ONNX 会话（CoreML 上单个模型重编译 0.4~1.2 秒）。
    #[test]
    fn identical_values_report_no_change() {
        let previous = key("gpu", Some(0));
        assert_eq!(
            changes_between(Some(&previous), &key("gpu", Some(0))),
            AppliedChanges::default()
        );
    }

    /// 只改设备序号也必须被识别为变化 —— 只比较 EP 会让"换一张显卡"静默失效。
    #[test]
    fn changing_only_the_device_id_counts_as_a_change() {
        let previous = key("gpu", Some(0));
        let changes = changes_between(Some(&previous), &key("gpu", Some(1)));
        assert!(changes.inference_device_changed);
        assert!(!changes.stretch_defaults_changed);
    }

    /// EP 字符串大小写/空白不同视为不同取值：下发侧按原样传给 ORT，
    /// 这里做归一化只会掩盖"用户写了个 ORT 不认识的 EP"。
    #[test]
    fn a_different_ep_string_is_a_change() {
        let previous = key("cpu", None);
        assert!(changes_between(Some(&previous), &key("gpu", None)).inference_device_changed);
    }

    /// 改拉伸默认值只报告拉伸变化，不该顺带把推理设备也判成变了。
    #[test]
    fn changing_the_stretch_default_does_not_touch_the_inference_device() {
        let previous = key("gpu", None);
        let mut next = previous.clone();
        next.default_hifigan_mel_stretch = !next.default_hifigan_mel_stretch;

        let changes = changes_between(Some(&previous), &next);
        assert!(changes.stretch_defaults_changed);
        assert!(!changes.inference_device_changed);
    }

    /// `record` 的去重是跨调用的（进程级记录），不只是同一次比较。
    ///
    /// 这条测试与其它用例共用进程级记录，因此必须串行：用一把测试内互斥锁，
    /// 否则并行执行时另一次 `record` 会改写记录，让断言随机失败。
    #[test]
    fn recording_the_same_values_twice_dedupes_across_calls() {
        static SERIAL: Mutex<()> = Mutex::new(());
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

        let first = record(key("cpu", Some(7)));
        assert!(first.inference_device_changed, "首次记录应视为变化");

        let second = record(key("cpu", Some(7)));
        assert_eq!(
            second,
            AppliedChanges::default(),
            "同一取值被重复记录为变化"
        );

        let third = record(key("cpu", Some(8)));
        assert!(third.inference_device_changed, "设备序号变化未被识别");
    }
}
