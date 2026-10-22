//! Capture 命令：UI 驱动录制与 take 管理的唯一入口。
//!
//! 【为什么这些命令不走 `editor/commands.rs` 的会话分发】那条路面向 `EditorSession`，
//! 而 `EditorSession` 只持有 `Weak<DocumentSession>` —— 它到不了 `ExtensionOwner`，
//! 也就到不了捕获存储。Capture 的全部状态都在 owner 上，所以入口挂在**视图层**
//! （那里已经有 `link.owner()`，`set_transport` 走的是同一条路）。
//!
//! 【为什么返回 `Option`】视图层先问一次"这是不是 Capture 命令"；不是就照旧交给会话。
//! 于是这个模块的接入对既有命令**零影响**。

use crate::render::extension::ExtensionOwner;
use crate::capture::CaptureOptions;
use serde_json::{json, Value};
use std::sync::Arc;

/// 本命令是否属于 Capture。视图层据此决定要不要拦下来。
pub(crate) fn is_capture_command(command: &str) -> bool {
    command.starts_with("capture_")
}

/// 处理一条 Capture 命令。调用方必须先确认 [`is_capture_command`]。
pub(crate) fn handle(
    owner: &Arc<ExtensionOwner>,
    command: &str,
    args: &Value,
) -> Result<Value, String> {
    let capture = owner.capture();
    match command {
        "capture_status" => Ok(status_json(owner)),
        // 整圈块级 trace（诊断包/前端按需拉取）。不做限流：这是用户显式动作。
        "capture_block_trace" => Ok(owner.block_diagnostics_json(256)),
        "capture_start" => {
            // 【为什么必须先挡这一条】宿主已经把音频经 ARA 交给我们时，再从输入总线录一遍
            // 是**另一条**素材来源，两者会互相覆盖。此前它静默地允许了 —— 用户会在 ARA
            // 宿主里录出一份与界面显示无关的东西。
            if owner.is_ara_bound() {
                return Err("capture_not_applicable_ara".into());
            }
            capture.start()?;
            Ok(status_json(owner))
        }
        "capture_stop" => {
            let take = capture.stop()?;
            // 录完立刻重投影并排一次合成：用户按下停止之后期待听到处理结果，而不是等
            // 下一次某个不相关的动作把它带起来。
            owner.request_capture_reload();
            Ok(json!({ "take": take, "status": status_json(owner) }))
        }
        "capture_cancel" => {
            capture.cancel();
            Ok(status_json(owner))
        }
        "capture_clear" => {
            capture.clear();
            // 时间线要跟着变空（否则界面还留着已经删掉的片段）。
            owner.request_capture_reload();
            Ok(status_json(owner))
        }
        "capture_delete_take" => {
            let id = args["takeId"]
                .as_u64()
                .ok_or("capture_delete_take needs a numeric takeId")?;
            capture.delete_take(id)?;
            // 时间线、原始磁带与合成结果都要跟着走：删掉的可能正是活动 take。
            owner.request_capture_reload();
            Ok(status_json(owner))
        }
        "capture_set_options" => {
            let current = capture.options();
            capture.set_options(CaptureOptions {
                max_duration_sec: args["maxDurationSec"]
                    .as_f64()
                    .unwrap_or(current.max_duration_sec),
                segment_sec: args["segmentSec"].as_f64().unwrap_or(current.segment_sec),
                monitor: args["monitor"].as_bool().unwrap_or(current.monitor),
                outside_passthrough: args["outsidePassthrough"]
                    .as_bool()
                    .unwrap_or(current.outside_passthrough),
            });
            Ok(status_json(owner))
        }
        other => Err(format!("unknown capture command: {other}")),
    }
}

/// 状态的线上形状。
///
/// 【为什么 camelCase】与既有 bootstrap / 命令载荷一致；前端只认这一种命名。
///
/// `ara_bound` 让界面能说清"这个宿主不需要 Capture"，而不是让按钮点了没反应 ——
/// 入口现在是**无条件**可见的（见 `captureEntry`：形态判定要等 5 秒宽限期，不能拿它
/// 当可见性判据），所以面板必须自己回答"我在这里有没有用"。
fn status_json(owner: &Arc<ExtensionOwner>) -> Value {
    let status = owner.capture().status();
    let ara_bound = owner.is_ara_bound();
    let recording = status.recording.map(|recording| {
        json!({
            "capturedFrames": recording.captured_frames,
            "sealedSegments": recording.sealed_segments,
            "position": recording.position,
            "peak": [recording.peak[0], recording.peak[1]],
            "gaps": recording.gaps,
            "spares": recording.spares,
        })
    });
    // 【为什么把"录制长到哪了"发出来】录制过程中界面上必须有东西在动。没有这一项，
    // 用户只能看到一个"已录 0.0 秒"的数字，而时间轴毫无变化 —— 这正是"不知道自己
    // 录了什么"的来源。带子由前端画，几何由这里给。
    let capture = status.capture.map(|capture| {
        json!({
            "startFrame": capture.start_frame,
            "endFrame": capture.end_frame,
            "positionFrame": capture.position_frame,
            "sampleRate": capture.sample_rate,
        })
    });
    // 【为什么把"录到了但没有内容"单独报出来】全静音的录制不产生片段，于是时间轴空白、
    // 面板却说"已录制"。把捕获帧数与片段数一起给出来，界面就能说"录到了 12.5 秒，
    // 但输入总线是静音"，而不是让用户面对三个互相矛盾的读数。
    let material = status.material.map(|material| {
        json!({
            "clips": material.clips,
            "capturedFrames": material.captured_frames,
            "coveredFrames": material.covered_frames,
            "spannedFrames": material.spanned_frames,
            "sampleRate": material.sample_rate,
            "silent": material.is_silent(),
        })
    });
    let takes = status
        .takes
        .iter()
        .map(|take| {
            json!({
                "id": take.id,
                "active": take.active,
                "clips": take.clips,
                "startSec": take.span_sec.map(|span| span.0),
                "endSec": take.span_sec.map(|span| span.1),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "state": status.state,
        // 【为什么"待命"是一个独立读数】武装之后、宿主播放之前，素材一定还是零 —— 而那
        // 与"正在录"是两种不同的等待：前者等用户去按播放，后者在积累素材。前端据此
        // 说"等待宿主播放"而不是"已录 0.0 秒"（后者看起来像失败）。
        "standby": status.standby,
        "recording": recording,
        "capture": capture,
        "material": material,
        "hasMaterial": status.has_material,
        "span": status.span.map(|span| json!({
            "startFrame": span.start_frame,
            "endFrame": span.end_frame,
            "sampleRate": span.sample_rate,
        })),
        "options": {
            "maxDurationSec": status.options.max_duration_sec,
            "segmentSec": status.options.segment_sec,
            "monitor": status.options.monitor,
            "outsidePassthrough": status.options.outside_passthrough,
        },
        "takeCount": status.take_count,
        "activeTake": status.active_take,
        "takes": takes,
        // 【为什么这三项是"观察到过"而不是"声明过"】它们描述宿主**实际**做了什么，
        // 而宿主不会声明。前端据此决定显示哪一条引导语：见过位置就能录，没见过位置
        // 就该说"这个宿主没有把播放位置交给插件"，而不是让用户对着按钮猜。
        "sawContext": status.saw_context,
        "sawInput": status.saw_input,
        "sawOffline": status.saw_offline,
        // 录制的**唯一**硬前提是工程位置：音频按 `projectTimeSamples` 对齐，
        // 没有它整条通路不成立。
        "canCapture": status.saw_context,
        // 宿主**此刻**是否在播放：按了录制却没声音进来时，面板据此说清原因。
        "hostPlaying": status.host_playing,
        // 宿主在用 64 位浮点跑本实例 ⇒ 插件按设计静音（平面是 64 位的）。
        // 这是"插上就没声音"的一个**用户可修复**的原因：把宿主精度切回 32-bit。
        "unsupportedPrecision": status.unsupported_precision,
        // 【为什么上限要预警】上限到点会**自己停**，而那是用户设的一个数值 —— 不预警
        // 的话用户只能看到"录着录着就断了"。见 `Capture::limit_warning`。
        "limitWarning": status.limit_warning,
        // 上一次录制**自己**停下的原因（`null` = 没有过、或那一次是用户按的停）。
        // 到上限 / 内存没跟上 / 宿主换了采样率：三者的补救完全不同，所以分开报。
        "stoppedReason": status.stopped_reason,
        // 【为什么把"现在听到的是哪一版"报出来】处理结果、原始录音与宿主原声听起来都
        // "像那么回事"，而它们之间的差别正是"我的编辑生效了没有" —— 用户无法自己判断。
        // 如实报出仲裁结论，他就不必猜（见 `CaptureOutput::as_str`）。
        "output": owner.capture_output().as_str(),
        // 块级诊断（计数 + 捕获侧 lookbehind 结果）。trace 本身走独立命令，
        // 避免 200ms 的状态轮询拖着 256 条记录。
        "blockDiagnostics": owner.block_diagnostics_json(0),
        // 宿主是否**正在**通过 ARA 供音（那时 Capture 不适用，见 `capture_start`）。
        "araBound": ara_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner() -> Arc<ExtensionOwner> {
        let owner = Arc::new(ExtensionOwner::default());
        owner.set_setup_sample_rate(48_000.0);
        owner
    }

    /// 命令名前缀是视图层分流的唯一依据。
    #[test]
    fn only_the_capture_prefix_is_claimed() {
        assert!(is_capture_command("capture_start"));
        assert!(is_capture_command("capture_status"));
        assert!(!is_capture_command("set_transport"));
        assert!(!is_capture_command("get_timeline_state"));
    }

    /// 录制按钮（`canCapture`）必须**在武装之前**就能变真。
    ///
    /// 【为什么这条必须有】此前 `saw_context` 只在武装之后更新，而按钮由它门控 ——
    /// 于是"必须先录一次才能录"：按钮永远是灰的，Capture 在所有宿主上都用不了。
    /// 这条走的是面板真正读的那条路（`capture_status`）。
    #[test]
    fn the_record_button_unlocks_before_anything_is_recorded() {
        let owner = owner();
        assert_eq!(
            handle(&owner, "capture_status", &json!({})).unwrap()["canCapture"],
            false,
            "还没见过宿主上下文时不能声称可以录制"
        );
        let samples: &'static [f32] = Box::leak(vec![0.1_f32; 64].into_boxed_slice());
        // 注意：**没有** `capture_start` —— 观察必须与武装无关。
        owner.capture().observe(&crate::capture::InputBlock {
            start: 0,
            frames: 64,
            sample_rate: 48_000.0,
            planes: [Some(samples), None],
            playing: false,
        });
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(
            status["canCapture"], true,
            "见过宿主上下文之后按钮必须可用，否则用户没有办法开始"
        );
        assert_eq!(status["hostPlaying"], false, "此刻宿主没在播放，如实报出来");
    }

    /// 宿主正在通过 ARA 供音时，Capture 必须**明确拒绝**而不是偷偷再录一份。
    ///
    /// 【为什么这条必须有】`capture_start` 此前不检查这件事：在 ARA 宿主里按下录制会
    /// 从输入总线再采一份**与界面显示无关**的素材，两条来源互相覆盖，用户听到的东西
    /// 与他看到的时间线对不上。入口现在无条件可见（见 `captureEntry`），所以后端必须
    /// 是那道真正的门。
    #[test]
    fn arming_is_refused_while_the_host_provides_audio_through_ara() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        assert!(owner.is_ara_bound(), "这个夹具的宿主是绑定了 ARA 的");
        let error = handle(&owner, "capture_start", &json!({})).unwrap_err();
        assert_eq!(error, "capture_not_applicable_ara");
        assert!(!owner.capture().is_armed(), "拒绝之后不能留下已武装状态");
        // 状态里也要如实报出来，面板才能说清"这个宿主不需要 Capture"。
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["araBound"], true);
    }

    /// 未录制时的状态如实报告"还不能录"，而不是假装可用。
    #[test]
    fn an_untouched_instance_reports_that_it_cannot_capture_yet() {
        let owner = owner();
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["state"], "empty");
        assert_eq!(status["takeCount"], 0);
        assert_eq!(status["activeTake"], Value::Null);
        assert_eq!(
            status["canCapture"], false,
            "没见过 ProcessContext 之前不能声称可以录制"
        );
        assert_eq!(status["recording"], Value::Null);
        assert_eq!(status["araBound"], false, "没绑定 ARA 的实例就是 Capture 的适用对象");
    }

    /// 选项补丁是**逐字段**的：只传一个字段不该把其余字段清掉。
    #[test]
    fn an_options_patch_only_touches_the_fields_it_names() {
        let owner = owner();
        handle(
            &owner,
            "capture_set_options",
            &json!({ "maxDurationSec": 120.0 }),
        )
        .unwrap();
        let options = owner.capture().options();
        assert_eq!(options.max_duration_sec, 120.0);
        assert!(options.monitor, "没提 monitor 就不能改它");
        assert!(options.segment_sec > 0.0);
    }

    /// 未知命令明确报错，而不是静默成功。
    #[test]
    fn an_unknown_capture_command_is_an_error() {
        let owner = owner();
        assert!(handle(&owner, "capture_frobnicate", &json!({})).is_err());
    }

    /// 删除不存在的 take 是错误，而不是静默成功。
    #[test]
    fn deleting_an_unknown_take_is_rejected() {
        let owner = owner();
        assert!(handle(&owner, "capture_delete_take", &json!({ "takeId": 99 })).is_err());
    }

    /// 没声明工程率时不能开始录制，且错误要说明**怎么才能行**。
    #[test]
    fn starting_without_a_declared_rate_explains_what_to_do() {
        let owner = Arc::new(ExtensionOwner::default());
        let error = handle(&owner, "capture_start", &json!({})).unwrap_err();
        assert!(
            error.contains("setupProcessing"),
            "错误必须指向可执行的下一步，而不是只说'失败'：{error}"
        );
    }

    /// 一个长度为 `frames` 的播放中的块。
    fn playing_block(frames: usize) -> crate::capture::InputBlock<'static> {
        let samples: &'static [f32] = Box::leak(vec![0.2_f32; frames].into_boxed_slice());
        crate::capture::InputBlock {
            start: 0,
            frames,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        }
    }

    /// **武装之后、宿主播放之前是"待命"，不是"已录 0.0 秒"。**
    ///
    /// 【为什么这条是用户反馈的回归测试】"点了录制没反应"就是待命被显示成"录制中
    /// 但已录 0.0 秒"。两者对用户是不同的等待：一个要他**去按播放**，一个在积累素材。
    #[test]
    fn arming_before_the_host_plays_reports_standby() {
        let owner = owner();
        handle(&owner, "capture_start", &json!({})).unwrap();
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["state"], "standby", "宿主还没播放 ⇒ 待命");
        assert_eq!(status["standby"], true);
        assert_eq!(status["recording"], Value::Null, "待命期没有任何读数");
        assert_eq!(status["canCapture"], false, "没见过上下文 ⇒ 还不能录");

        // 宿主开始播放 ⇒ 进入录制中。
        owner.capture().on_block(playing_block(64));
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["state"], "recording");
        assert_eq!(status["standby"], false);
        assert!(status["recording"].is_object(), "录制中必须有读数");
    }

    /// 上限预警与"自己停下的原因"都要如实报出来。
    ///
    /// 【为什么两者都要】上限到点会自己停，而那是用户设的一个数值；不预警 + 不归因，
    /// 用户只能看到"录着录着就断了"，且没有任何依据知道该怎么改。
    #[test]
    fn the_limit_warning_and_the_stop_reason_are_reported() {
        let owner = owner();
        handle(
            &owner,
            "capture_set_options",
            &json!({ "maxDurationSec": 20.0 }),
        )
        .unwrap();
        handle(&owner, "capture_start", &json!({})).unwrap();
        // 一秒一块（48000 帧）。上限 20 秒，录到 1 秒时还剩 19 秒 ⇒ 不该预警。
        let second = |index: i64| crate::capture::InputBlock {
            start: index * 48_000,
            frames: 48_000,
            sample_rate: 48_000.0,
            planes: [None, None],
            playing: true,
        };
        owner.capture().on_block(second(0));
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["limitWarning"], false, "还剩 19 秒时不该预警");
        // 录到 11 秒 ⇒ 还剩 9 秒 < 预警线 10 秒 ⇒ 必须预警。
        for index in 1..11_i64 {
            owner.capture().on_block(second(index));
        }
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(status["limitWarning"], true, "还剩不到 10 秒必须预警");
        assert_eq!(status["stoppedReason"], Value::Null, "还没停，没有停止原因");
    }

    /// **到上限后素材不能丢**：`service()` 把它 promote 成一个 take。
    ///
    /// 【为什么这是"到点自动停"的正面验收】捕获自己停下时没有任何用户动作。若没有
    /// 自动 promote，那些段会一直留在捕获里 —— 界面显示"尚未录制"、素材却在内存里，
    /// 而下一次 `capture_start` 会把它们覆盖掉："自动停"于是变成"自动丢"。
    #[test]
    fn a_capture_that_stopped_itself_is_promoted_by_the_service_tick() {
        let owner = owner();
        // 上限 1 秒（选项的下限，`set_options` 会夹到 1.0）@ 48 kHz = 48000 帧；
        // 下面每块 24000 帧（0.5 秒）⇒ 第二块到点。
        handle(
            &owner,
            "capture_set_options",
            &json!({ "maxDurationSec": 1.0 }),
        )
        .unwrap();
        handle(&owner, "capture_start", &json!({})).unwrap();
        for index in 0..4_i64 {
            owner.capture().on_block(crate::capture::InputBlock {
                start: index * 24_000,
                frames: 24_000,
                sample_rate: 48_000.0,
                planes: [None, None],
                playing: true,
            });
        }
        assert!(!owner.capture().is_armed(), "到上限后必须已解除武装");
        // 会话 actor 的心跳：这一轮应当把素材收成一个 take。
        owner.capture().service();
        let status = handle(&owner, "capture_status", &json!({})).unwrap();
        assert_eq!(
            status["takeCount"], 1,
            "自己停下的捕获必须被 promote，否则素材会随下一次录制一起消失"
        );
        assert_eq!(status["stoppedReason"], "limit_reached");
    }
}
