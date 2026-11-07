//! `capture_status` 的线上形状（两种格式共用）。
//!
//! 【为什么这份 JSON 在运行时里】前端的 Capture 面板只认这一种形状：字段少一个，
//! 界面就少一句该说的话；字段含义漂了就变成"按钮说能录、其实没录"。VST3 与 CLAP
//! 各写一份必然漂移，所以形状由运行时给出，格式外壳只补三个**只有它自己知道**的读数
//! （见 [`StatusExtras`]）。

use crate::{CaptureStatus, CaptureStore};
use serde_json::{json, Value};

/// 只有格式外壳才知道的读数。
pub struct StatusExtras<'a> {
    /// 宿主是否**正在**通过 ARA 供音（那时 Capture 不适用）。
    pub ara_bound: bool,
    /// 此刻听到的是哪一版（`CaptureOutput::as_str` 的取值）。
    pub output: &'a str,
    /// 块级诊断（计数 + 捕获侧 lookbehind 结果）。trace 本身走独立命令。
    pub block_diagnostics: Value,
}

/// 把 [`CaptureStatus`] 渲染成前端读的那份 JSON。
pub fn status_json(store: &CaptureStore, extras: StatusExtras<'_>) -> Value {
    let status: CaptureStatus = store.status();
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
        // 【为什么给"区间列表"而不只是并集】并集会把"跳到 30 秒再继续录"画成一条
        // 0..33 秒的连续带子 —— 中间那 30 秒其实什么都没录到。
        let runs: Vec<Value> = capture
            .runs
            .iter()
            .map(|run| json!({ "startFrame": run.start, "endFrame": run.end }))
            .collect();
        json!({
            "startFrame": capture.start_frame,
            "endFrame": capture.end_frame,
            "positionFrame": capture.position_frame,
            "blockFrames": capture.block_frames,
            "sampleRate": capture.sample_rate,
            "runs": runs,
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
    // 【为什么把"清单里有、本地没有"的音频发出来】工程被拷到另一台机器时 sidecar
    // 不会跟过去；界面必须能说清"丢了哪几段、共多久"，而不是只留一条日志。
    let missing = store
        .missing_captures()
        .into_iter()
        .map(|entry| {
            let rate = if entry.sample_rate > 0 {
                entry.sample_rate as f64
            } else {
                1.0
            };
            json!({
                "id": entry.id,
                "frames": entry.frames,
                "seconds": entry.frames as f64 / rate,
                "startSec": entry.start_sec,
                "endSec": entry.end_sec,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "state": status.state,
        "phase": status.phase,
        "missingTakes": missing,
        // 【为什么"待命"是一个独立读数】武装之后、宿主播放之前，素材一定还是零 —— 而那
        // 与"正在录"是两种不同的等待：前者等用户去按播放，后者在积累素材。
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
        // 而宿主不会声明。前端据此决定显示哪一条引导语。
        "sawContext": status.saw_context,
        "sawInput": status.saw_input,
        "sawOffline": status.saw_offline,
        // 录制的**唯一**硬前提是工程位置：音频按工程帧对齐，没有它整条通路不成立。
        "canCapture": status.saw_context,
        // 宿主**此刻**是否在播放：按了录制却没声音进来时，面板据此说清原因。
        "hostPlaying": status.host_playing,
        // 宿主在用 64 位浮点跑本实例 ⇒ 插件按设计静音（平面是 64 位的）。
        "unsupportedPrecision": status.unsupported_precision,
        // 【为什么上限要预警】上限到点会**自己停**，而那是用户设的一个数值。
        "limitWarning": status.limit_warning,
        // 上一次录制**自己**停下的原因（`null` = 没有过、或那一次是用户按的停）。
        "stoppedReason": status.stopped_reason,
        // 【为什么把"现在听到的是哪一版"报出来】处理结果、原始录音与宿主原声听起来都
        // "像那么回事"，而它们之间的差别正是"我的编辑生效了没有" —— 用户无法自己判断。
        "output": extras.output,
        "blockDiagnostics": extras.block_diagnostics,
        // 宿主是否**正在**通过 ARA 供音（那时 Capture 不适用）。
        "araBound": extras.ara_bound,
    })
}
