//! 原GUI命令的插件准入与类型适配；使用共享原参数/历史/波形实现，不接收宿主几何变更。
use super::session::EditorSession;
use crate::sync::MutexExt;
use base64::Engine as _;
use hifishifter_kernel::editor::{history, params, waveform, ParamHost};
use hifishifter_kernel::state::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

pub(super) fn mutates_audio(command: &str) -> bool {
    matches!(
        command,
        "set_param_frames"
            | "restore_param_frames"
            | "set_static_param"
            | "convert_mix_param"
            // 拉伸映射同样改写参数曲线，必须让在途渲染作废（否则拉伸后听到的是
            // 旧曲线渲染出来的音频）。
            | "stretch_track_linked_params"
            // MIDI 导入整段改写音高曲线，与 `set_param_frames` 同一类改动。
            | "import_midi_to_pitch"
            // REAPER 音高互转同样整段改写曲线（两个方向都会写），在途渲染必须作废。
            | "import_host_take_pitch"
            | "export_host_take_pitch"
            | "set_track_state"
            | "move_track"
            | "undo_timeline"
            | "redo_timeline"
            | "undo_parameter_edit"
            | "redo_parameter_edit"
            | "set_history_position"
    )
}

/// 一条命令在哪条线上执行。
///
/// 三条线的划分依据只有一个：**它需要什么**。
/// * 需要 HWND / COM / 宿主回调 ⇒ 宿主 UI 线程（`webview.rs` 的内联 arm）；
/// * 需要 `&EditorSession`（时间线、参数、历史）⇒ actor 队列；
/// * 什么都不需要、但要跑几十秒 ⇒ 慢车道（[`super::slow_lane`]）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lane {
    Actor,
    Slow,
}

/// 慢车道命令表 —— **唯一**的分发依据（`lane` 与 `dispatch_slow` 都从它读）。
///
/// 【加一条新命令时】必须在 [`dispatch_slow`] 里加同名 arm；
/// `every_slow_command_is_dispatchable` 会立刻发现漏加。
const SLOW_COMMANDS: &[&str] = &["run_vocoder_benchmark", "analyze_clip_syllables"];

/// 某条命令该走哪条线。
pub(crate) fn lane(command: &str) -> Lane {
    if SLOW_COMMANDS.contains(&command) {
        Lane::Slow
    } else {
        Lane::Actor
    }
}

/// 该命令是否走慢车道（`webview` 的账本也读这一个判据）。
pub(crate) fn is_slow(command: &str) -> bool {
    lane(command) == Lane::Slow
}

/// 慢车道的实现。
///
/// 【为什么签名里没有 `&EditorSession`】这是"慢任务不得触碰会话状态"的**编译期**保证：
/// 时间线、参数、历史全都只能经由会话访问，签名里没有它，就写不出越界的代码。
/// （`SlowTask.work` 的闭包入口靠纪律，见 `slow_lane` 的模块头；这条按名字分发的路
/// 则把最常用的那条锁死在类型层面。）
/// The slow-lane arm for `command`, if the table has one.
///
/// Kept separate from [`dispatch_slow`] so a test can prove that every table
/// entry has a matching arm **without actually running the arm**.  The
/// benchmark arm builds an ONNX session and runs five real inferences; on a
/// headless CI runner that takes minutes and can hang in the ORT CPU backend.
/// Before the Intel-macOS backend was registered it failed instantly and this
/// test looked cheap, which hid the cost for as long as ORT was broken there.
pub(crate) fn slow_dispatch_target(
    command: &str,
) -> Option<(&'static str, fn() -> Result<Value, String>)> {
    match command {
        "run_vocoder_benchmark" => Some(("run_vocoder_benchmark", || {
            hifishifter_kernel::nsf_hifigan_onnx::run_benchmark().and_then(value)
        })),
        _ => None,
    }
}

pub(crate) fn dispatch_slow(command: &str, _args: &Value) -> Option<Result<Value, String>> {
    slow_dispatch_target(command).map(|(_, run)| run())
}

// 回归诊断测试仅在cfg(test)编译；产品分组历史路径由已加载分支实际执行。
#[cfg(test)]
#[path = "commands_tests.rs"]
mod undo_group_diagnostic;
fn value<T: serde::Serialize>(input: T) -> Result<Value, String> {
    serde_json::to_value(input).map_err(|e| e.to_string())
}
fn args<T: serde::de::DeserializeOwned>(input: Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|e| format!("invalid editor arguments: {e}"))
}

fn syllable_payload(
    clip_id: &str,
    take_id: &str,
    analysis: Option<&hifishifter_kernel::pitch_analysis::TakeSyllableAnalysis>,
    window: (f64, f64),
) -> Value {
    json!({
        "ok": true,
        "clipId": clip_id,
        "takeId": take_id,
        "sourceSpansSec": analysis.map(|a| a.source_spans_sec.iter().map(|s| {
            json!({"startSec": s.start_sec, "endSec": s.end_sec, "midiNote": s.midi_note})
        }).collect::<Vec<_>>()).unwrap_or_default(),
        "sourceNotes": analysis.map(|a| a.source_spans_sec.iter().map(|s| s.midi_note).collect::<Vec<_>>()).unwrap_or_default(),
        "analyzedWindowSec": analysis.map_or(
            [window.0.min(window.1), window.0.max(window.1)],
            |a| a.analyzed_window_sec
        ),
        "algorithm": analysis.map(|a| a.algorithm.clone()).unwrap_or_else(|| {
            hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM.to_string()
        }),
        "algorithmVersion": analysis
            .map(|a| a.algorithm_version)
            .unwrap_or(hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION),
        "analyzed": analysis.is_some(),
    })
}

/// 慢车道上的纯计算入口：输入是 actor 冻结的 owned 快照，输出可序列化结果。
///
/// 【为什么仍然显式接收整个请求类型】只靠 `TimelineState + clipId` 也能算，但那样
/// worker 就得自己再查一次 Clip/Take/源，而查询到的版本未必与分析时一致。把身份
/// 与快照绑在一个不可变值里，计算阶段天然不会“看着新源算旧窗口”。
pub(crate) fn analyze_syllables(
    request: super::session::SyllableAnalysisRequest,
) -> Result<Value, String> {
    let clip = request
        .snapshot
        .clips
        .iter()
        .find(|clip| clip.id == request.clip_id)
        .ok_or("clip_not_found")?;
    let take = clip
        .takes
        .iter()
        .find(|take| take.id == request.take_id)
        .ok_or("take_not_found")?;
    // MIDI 音符本身已有精确边界，不应被音频估计覆盖；但这次仍返回一个明确的
    // “无可分析音频”结果，由 actor 决定保留现有缓存还是回退。
    if take
        .midi_note_data
        .as_ref()
        .is_some_and(|notes| !notes.is_empty())
    {
        hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
        return Ok(json!({
            "clipId": request.clip_id,
            "takeId": request.take_id,
            "sourceSpansSec": [],
            "sourceNotes": [],
            "analyzedWindowSec": [
                request.source_window.0.min(request.source_window.1),
                request.source_window.0.max(request.source_window.1)
            ],
            "algorithm": hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM,
            "algorithmVersion": hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
            "analyzed": false,
        }));
    }
    // 插件这里读到的 `source_path` 一定是本会话物化的授权 PCM（在 actor 冻结快照
    // 前已由 `check_source` 校验）。worker 只做计算，不重新解释路径。
    if take.source_path.is_none() && clip.source_path.is_none() {
        hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
        return Ok(json!({
            "clipId": request.clip_id,
            "takeId": request.take_id,
            "sourceSpansSec": [],
            "sourceNotes": [],
            "analyzedWindowSec": [
                request.source_window.0.min(request.source_window.1),
                request.source_window.0.max(request.source_window.1)
            ],
            "algorithm": hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM,
            "algorithmVersion": hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
            "analyzed": false,
        }));
    }
    let source_path = take
        .source_path
        .as_deref()
        .or(clip.source_path.as_deref())
        .ok_or_else(|| {
            hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
            "source_missing".to_string()
        })?;
    let window_start = request.source_window.0.min(request.source_window.1);
    let window_end = request.source_window.0.max(request.source_window.1);
    hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(Some(
        hifishifter_kernel::pitch_analysis::SyllableAnalysisProgress {
            active: true,
            clip_id: request.clip_id.clone(),
            clip_name: clip.name.clone(),
            progress: 0.02,
            stage: "decoding".into(),
            algorithm: hifishifter_kernel::game_onnx::GAME_ALGORITHM.into(),
            message: None,
        },
    ));
    let game = hifishifter_kernel::game_onnx::resolve_model_dir().map(|dir| {
        hifishifter_kernel::game_onnx::analyze_source_file(
            std::path::Path::new(source_path),
            &dir,
            hifishifter_kernel::game_onnx::GameInferenceConfig::default(),
        )
    });
    let (spans, algorithm, version) = match game {
        Some(Ok(events)) => (
            hifishifter_kernel::game_onnx::full_source_voiced_spans(&events)
                .into_iter()
                .filter(|(start, end, _)| *end > window_start && *start < window_end)
                .map(|span| hifishifter_kernel::pitch_analysis::SyllableSpan {
                    start_sec: span.0,
                    end_sec: span.1,
                    midi_note: Some(span.2),
                    event_id: Some(hifishifter_kernel::pitch_analysis::source_event_id(
                        span.0, span.1,
                    )),
                })
                .collect::<Vec<_>>(),
            hifishifter_kernel::game_onnx::GAME_ALGORITHM,
            hifishifter_kernel::game_onnx::GAME_ALGORITHM_VERSION,
        ),
        Some(Err(error)) => {
            log::warn!("GAME plugin syllable analysis failed: {error}");
            (
                match vad_fallback_spans(&request, clip) {
                    Ok(spans) => spans,
                    Err(error) => {
                        hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
                        return Err(error);
                    }
                },
                hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM,
                hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
            )
        }
        None => (
            match vad_fallback_spans(&request, clip) {
                Ok(spans) => spans,
                Err(error) => {
                    hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
                    return Err(error);
                }
            },
            hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM,
            hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
        ),
    };
    hifishifter_kernel::pitch_analysis::set_syllable_analysis_progress(None);
    Ok(json!({
        "clipId": request.clip_id,
        "takeId": request.take_id,
        "sourceSpansSec": spans.iter().map(|span| json!({
            "startSec": span.start_sec,
            "endSec": span.end_sec,
            "midiNote": span.midi_note,
        })).collect::<Vec<_>>(),
        "sourceNotes": spans.iter().map(|span| span.midi_note).collect::<Vec<_>>(),
        "analyzedWindowSec": [window_start, window_end],
        "algorithm": algorithm,
        "algorithmVersion": version,
        // 空 spans 代表“没检测到发声”，不是分析失败；actor 会保留旧缓存并回退。
        "analyzed": !spans.is_empty(),
    }))
}

fn vad_fallback_spans(
    request: &super::session::SyllableAnalysisRequest,
    clip: &hifishifter_kernel::state::Clip,
) -> Result<Vec<hifishifter_kernel::pitch_analysis::SyllableSpan>, String> {
    let (level, pitch) = hifishifter_kernel::pitch_clip::analyze_clip_for_syllables_isolated(
        &request.snapshot,
        clip,
        &request.root_track_id,
        request.frame_period_ms,
        &|| false,
    )
    .ok_or("source_missing")?;
    let spans = hifishifter_kernel::pitch_analysis::estimate_syllable_spans(
        &level,
        &pitch,
        request.frame_period_ms,
        hifishifter_kernel::pitch_analysis::SyllableConfig::default(),
    );
    let (start, end) = (
        request.source_window.0.min(request.source_window.1),
        request.source_window.0.max(request.source_window.1),
    );
    Ok(spans
        .into_iter()
        .filter(|span| span.end_sec > start && span.start_sec < end)
        .collect())
}
#[cfg(test)]
mod lane_tests {
    use super::*;

    /// **不变量：慢车道表里的每条命令都真的能在慢车道上执行。**
    ///
    /// 【为什么这是回归测试】表是分发的**唯一**真相。若只在表里加一条而忘了
    /// `dispatch_slow` 的 arm，调用方会收到 "command is not dispatchable on the slow
    /// lane" —— 一条只会在用户点下去时才出现的错误。让它在测试里先出现。
    #[test]
    fn every_slow_command_is_dispatchable() {
        for command in SLOW_COMMANDS {
            assert_eq!(lane(command), Lane::Slow, "{command} 必须在慢车道表里");
            // 音节分析是由 actor 冻结快照后派发的计算型慢任务，不经过
            // `dispatch_slow` 的纯函数入口。它仍必须被视为慢命令，否则
            // `enqueue` 会把它排回 actor。
            if *command == "analyze_clip_syllables" {
                assert_eq!(lane(command), Lane::Slow);
                continue;
            }
            assert!(
                slow_dispatch_target(command).is_some_and(|(name, _)| name == *command),
                "{command} 在表里但 dispatch_slow 没有对应 arm"
            );
        }
        assert!(
            !SLOW_COMMANDS.is_empty(),
            "表为空时这条测试没有意义：慢车道至少要有一条命令"
        );
    }

    /// **不变量：基准测试绝不在 actor 上执行。**
    ///
    /// 【为什么这是回归测试】它跑 31–134 秒。此前它在 actor 上同步执行，于是同队列的
    /// 每一次编辑、每一次轮询、以及宿主保存的 flush 全部排队到期 —— 用户看到一串
    /// `editor request N expired without a reply` 和一个不响应的编辑器。
    #[test]
    fn the_benchmark_is_never_on_the_actor_lane() {
        assert_eq!(lane("run_vocoder_benchmark"), Lane::Slow);
        for command in [
            "get_timeline_state",
            "set_param_frames",
            "get_playback_state",
            "export_pitch_to_midi",
        ] {
            assert_eq!(lane(command), Lane::Actor, "{command} 必须留在 actor 上");
        }
    }

    /// **参数块四项命令必须由插件 actor 真正接住。**
    ///
    /// 这条不是“命令名存在”就算过：它验证新机制不会在 VST/ARA 里落到
    /// `Command unavailable`，且查询 / 合并写入不会误改参数曲线。
    #[test]
    fn param_block_commands_are_wired_in_the_plugin() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        let (reply, rx) = std::sync::mpsc::channel();
        let (events, _event_rx) = std::sync::mpsc::sync_channel(256);
        let sink = crate::editor::session::UiSink {
            view_id: "param-block-diagnostic".into(),
            reply,
            events,
            closed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let call = |id, command: &str, args| {
            editor
                .enqueue(crate::editor::session::UiRequest {
                    id,
                    command: command.into(),
                    args,
                    sink: sink.clone(),
                    link: None,
                })
                .unwrap();
            let response: Value = rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
            assert_eq!(response["ok"], true, "{response}");
            response["value"].clone()
        };
        let timeline = call(1, "get_timeline_state", json!({}));
        let track = timeline["tracks"][0]["id"].as_str().unwrap().to_owned();
        {
            let mut timeline = editor.timeline.lock_or_recover();
            timeline.clips[0].takes[0].syllable_analysis =
                Some(hifishifter_kernel::pitch_analysis::TakeSyllableAnalysis {
                    algorithm: hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM.into(),
                    algorithm_version:
                        hifishifter_kernel::pitch_analysis::SYLLABLE_ALGORITHM_VERSION,
                    source_digest: "param-block-test".into(),
                    source_spans_sec: vec![hifishifter_kernel::pitch_analysis::SyllableSpan {
                        start_sec: 0.0,
                        end_sec: 0.001,
                        midi_note: None,
                        event_id: None,
                    }],
                    analyzed_window_sec: [0.0, 0.0],
                });
        }
        let payload = call(
            2,
            "set_param_block_edits",
            json!({
                "rootTrackId": track,
                "expectedEditRevision": 0,
                "edits": [{
                    "id": "manual:test",
                    "geometry": {"kind": "root", "span": {"startSec": 0.0, "endSec": 0.02}},
                    "manual": true,
                    "createdSeq": 1
                }],
                "checkpoint": true
            }),
        );
        assert_eq!(payload["blocks"][0]["id"], "manual:test");
        let read = call(3, "get_param_block_view", json!({"rootTrackId": track}));
        assert_eq!(read["editRevision"], payload["editRevision"]);
        assert_eq!(read["blocks"][0]["kind"], "manual");
        // 通道容量足够容纳本测试产生的 history/state 事件；不读取内容。
    }
}

#[cfg(test)]
mod export_request_tests {
    use super::*;

    /// **目标路径解析：显式路径优先，否则目录 + 文件名并补格式扩展名。**
    ///
    /// 【为什么这条是契约】导出对话框发的是 `projectOutputDir` + `projectFileName`
    /// （见前端 `AdvancedExportRequest`），而另一条调用路径发 `projectOutputPath`。
    /// 两者必须落到同一个文件上，否则"预检说没冲突、导出却写到别处"。
    #[test]
    fn the_export_target_prefers_an_explicit_path_and_otherwise_builds_one() {
        use hifishifter_kernel::encode::OutputFormat;
        let explicit = serde_json::json!({
            "projectOutputPath": "C:/tmp/take.wav",
            "projectOutputDir": "C:/ignored",
            "projectFileName": "ignored",
        });
        assert_eq!(
            export_output_path(&explicit, OutputFormat::Wav).unwrap(),
            std::path::PathBuf::from("C:/tmp/take.wav")
        );
        let composed = serde_json::json!({
            "projectOutputDir": "C:/tmp",
            "projectFileName": "take",
        });
        assert_eq!(
            export_output_path(&composed, OutputFormat::Wav).unwrap(),
            std::path::PathBuf::from("C:/tmp").join("take.wav")
        );
        // 缺目录必须明确拒绝，而不是写到一个猜出来的位置。
        assert!(export_output_path(&serde_json::json!({}), OutputFormat::Wav).is_err());
    }

    /// **编码参数：`encoder` 解不出时退回旧协议的 `bitDepth` → WAV。**
    #[test]
    fn the_export_spec_falls_back_to_the_legacy_bit_depth() {
        let with_encoder = serde_json::json!({
            "encoder": hifishifter_kernel::encode::OutputSpec::wav_32f(),
        });
        assert_eq!(
            export_spec(&with_encoder).unwrap(),
            hifishifter_kernel::encode::OutputSpec::wav_32f()
        );
        let legacy = serde_json::json!({ "bitDepth": 16 });
        let spec = export_spec(&legacy).unwrap();
        assert_eq!(spec.format, hifishifter_kernel::encode::OutputFormat::Wav);
        assert_eq!(
            spec.wav.bit_depth,
            hifishifter_kernel::encode::WavBitDepth::I16
        );
    }

    /// **分轨导出必须明确拒绝，而不是悄悄导出一份与宿主不一致的分轨。**
    #[test]
    fn separated_export_is_refused_with_a_reason() {
        let request = serde_json::json!({
            "mode": "separated",
            "projectOutputDir": "C:/tmp",
            "projectFileName": "take",
        });
        let error = preview_export_audio_plan(&request).unwrap_err();
        assert!(
            error.contains("separated"),
            "拒绝的理由必须说明是分轨不可用：{error}"
        );
    }

    /// 预检只报路径与冲突，不渲染。
    #[test]
    fn the_preview_reports_the_target_and_existing_conflicts() {
        let dir = std::env::temp_dir().join(format!("hfs-export-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let request = serde_json::json!({
            "mode": "project",
            "projectOutputDir": dir.to_string_lossy(),
            "projectFileName": "plan",
        });
        let plan = preview_export_audio_plan(&request).unwrap();
        assert_eq!(plan["ok"], true);
        assert_eq!(plan["mode"], "project");
        let target = plan["targets"][0]["path"].as_str().unwrap();
        assert!(target.ends_with("plan.wav"), "{target}");
        assert!(
            plan["existingPaths"].as_array().unwrap().is_empty(),
            "文件还不存在时不该报冲突"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// 导出取消标志。
///
/// 【为什么是进程级】导出是"整台机器上一次一个"的操作（与独立 App 的
/// `install_export_cancel_flag` 同一口径）：它占大量内存，且用户按一次取消就该停。
static EXPORT_CANCEL: std::sync::Mutex<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>> =
    std::sync::Mutex::new(None);

/// 解析导出请求里的编码参数（与独立 App 共用 `OutputSpec`）。
fn export_spec(
    request: &serde_json::Value,
) -> Result<hifishifter_kernel::encode::OutputSpec, String> {
    // 前端传回的 `encoder` 就是 `OutputSpec` 的 camelCase 形状；解不出就退回
    // 旧协议的 `bitDepth` → WAV。
    if let Some(encoder) = request.get("encoder").filter(|value| !value.is_null()) {
        if let Ok(spec) =
            serde_json::from_value::<hifishifter_kernel::encode::OutputSpec>(encoder.clone())
        {
            return Ok(spec);
        }
    }
    let bit_depth = request["bitDepth"].as_u64().unwrap_or(32) as u32;
    Ok(hifishifter_kernel::encode::OutputSpec::wav_legacy(
        bit_depth,
    ))
}

/// 解析导出目标路径（`projectOutputPath` 优先，否则目录 + 文件名）。
fn export_output_path(
    request: &serde_json::Value,
    format: hifishifter_kernel::encode::OutputFormat,
) -> Result<std::path::PathBuf, String> {
    if let Some(path) = request["projectOutputPath"]
        .as_str()
        .filter(|p| !p.is_empty())
    {
        return Ok(std::path::PathBuf::from(path));
    }
    let dir = request["projectOutputDir"]
        .as_str()
        .filter(|dir| !dir.is_empty())
        .ok_or("projectOutputDir is required")?;
    let name = request["projectFileName"]
        .as_str()
        .filter(|name| !name.is_empty())
        .unwrap_or("export");
    Ok(
        std::path::PathBuf::from(dir).join(hifishifter_kernel::encode::with_format_extension(
            name, format,
        )),
    )
}

/// 导出计划预检：只算路径与冲突，不渲染。
fn preview_export_audio_plan(request: &serde_json::Value) -> Result<Value, String> {
    if request["mode"].as_str().unwrap_or("project") != "project" {
        return Err("separated export is not available in the plug-in".into());
    }
    let spec = export_spec(request)?;
    let path = export_output_path(request, spec.format)?;
    let existing = path.is_file().then(|| path.to_string_lossy().to_string());
    Ok(serde_json::json!({
        "ok": true,
        "mode": "project",
        "targets": [{"trackId": Value::Null, "path": path.to_string_lossy()}],
        "existingPaths": existing.into_iter().collect::<Vec<_>>(),
    }))
}

/// 把一段工程范围渲染并编码到文件。
///
/// 【为什么不支持 separated】分轨导出要按宿主轨道切文件，而"哪条 clip 属于哪条轨道"
/// 在插件里是**投影**（`workspace_timeline_locked`），与宿主的分组 / 文件夹不同口径。
/// 与其导出一份与宿主不一致的分轨，不如明确拒绝。
fn export_audio_to_file(
    session: &EditorSession,
    request: &serde_json::Value,
) -> Result<Value, String> {
    if request["mode"].as_str().unwrap_or("project") != "project" {
        return Err("separated export is not available in the plug-in".into());
    }
    let spec = export_spec(request)?;
    let path = export_output_path(request, spec.format)?;
    if !path.is_absolute() {
        return Err("an absolute export path is required".into());
    }
    if !path.parent().is_some_and(std::path::Path::is_dir) {
        return Err("the export folder does not exist".into());
    }
    let sample_rate = match request["sampleRate"].as_u64() {
        None => 44_100_u32,
        Some(rate) if matches!(rate, 44_100 | 48_000) => rate as u32,
        Some(_) => return Err("unsupported export sample rate".into()),
    };
    let (start_sec, end_sec) = {
        let timeline = session.timeline.lock_or_recover().clone();
        let range = &request["range"];
        match range["kind"].as_str().unwrap_or("all") {
            "custom" => (
                range["startSec"]
                    .as_f64()
                    .ok_or("a custom export range needs startSec")?,
                range["endSec"]
                    .as_f64()
                    .ok_or("a custom export range needs endSec")?,
            ),
            _ => (0., timeline.project_sec),
        }
    };
    // 渲染输入与播放路径**同一份**（PCM + 逐 clip 参数）；
    // 导出与播放因此共用同一条混合数学（`RenderInput::mix_interleaved`）。
    let input = if session.is_capture() {
        // Capture：几何与 PCM 在捕获存储上，用户曲线在会话时间线上 —— owner 是唯一
        // 同时够得着两边的对象（见 `ExtensionOwner::capture_render_input`）。
        session
            .capture_owner()
            .ok_or("the Capture host is gone")?
            // 【导出为什么要求带曲线】导出的产物用户会直接拿去用，而"会话还在加载"
            // 意味着用户画好的曲线读不出来 —— 与其导出一份没有编辑的音频，不如让
            // 用户稍等再按一次（`capture_session_pending` 的说明）。
            .capture_render_input(true)?
            .ok_or("the editor session is still loading; try exporting again in a moment")?
            .3
    } else {
        let document = session.document().ok_or("the ARA document is closed")?;
        let owner = document
            .renderer_owners()
            .into_iter()
            .next()
            .ok_or("no ARA instance is bound to this document yet")?;
        let edits = document.edits.lock_or_recover().clone();
        owner.ara_render_input(&document, &edits, false)?.1
    };
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    *EXPORT_CANCEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(cancel.clone());
    let outcome = input.export_range_to_file(crate::render::input::ExportRequest {
        start_sec,
        end_sec,
        output_path: &path,
        spec: &spec,
        sample_rate,
        cancel,
        progress: None,
    });
    *EXPORT_CANCEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    let summary = outcome?;
    log::info!(
        "[ara] exported {:.3}s to {} ({} byte(s))",
        end_sec - start_sec,
        path.display(),
        summary.bytes_written
    );
    Ok(serde_json::json!({
        "ok": true,
        "mode": "project",
        "path": path.to_string_lossy(),
        "format": spec.format.as_name(),
        "output_dir": path
            .parent()
            .map(|parent| parent.to_string_lossy().to_string())
            .unwrap_or_default(),
        "count": 1,
    }))
}

// ── REAPER take 音高 ⇄ HiFiShifter 参数线（一次性转换）─────────────────────
//
// 【两侧各管各的，**互不叠加**】REAPER 的 take 音高（`D_PITCH` / `PITCHENV`）是
// **源级**播放属性：真机实测（REAPER 7.85）把 +12 半音写进 take 包络后，take FX 里
// 生成的 440 Hz 测试音仍然是 440 Hz —— 音高处理发生在源级、早于 take FX。ARA 插件交给
// 宿主的渲染结果同样落在那层之后（同族证据见
// `docs/2026-11-17-take-envelope-axis-findings.md` 的附节：源级那一层是渲染实测，
// ARA 那一层是推断 + 现场报障），宿主不会对它再做一遍重采样。于是：
//
// - 插件读到、并送去渲染的素材**不含** take 音高 ⇒ `pitch_edit` 必须原样保留；
// - take 包络只在插件被旁路 / 移除时才作用于这段音频 ⇒ 两侧永远不会同时处理。
//
// 所以两个方向都是**复制**（copy）：谁也不清空对方。唯一要动的宿主数值是 item 级
// `D_PITCH` —— 它与包络叠加，而「转换为 REAPER Take 音高包络」这条需求的语义就是
// 让包络成为 REAPER 侧唯一的音高来源，因此导出后把它归零（导入方向本来就会把它的
// 值加进 `pitch_edit`，归零不会丢信息）。

/// 在导入方向运行期间抑制「自动写入 Take 音高」（见 `import_host_take_pitch` 的说明）。
///
/// 【为什么是 Drop 而不是手动复位】那个函数有七处 `?` 早退；漏掉任何一处都会把抑制位
/// 永久留在"开"的位置 —— 表现是"自动写入从此不再发生"，而且没有任何报错。
struct AutoPitchWriteSuppression<'a> {
    session: &'a EditorSession,
}

impl<'a> AutoPitchWriteSuppression<'a> {
    fn begin(session: &'a EditorSession) -> Self {
        session.set_suppress_auto_pitch_write(true);
        Self { session }
    }
}

impl Drop for AutoPitchWriteSuppression<'_> {
    fn drop(&mut self) {
        self.session.set_suppress_auto_pitch_write(false);
    }
}

/// 命令入参里的片段列表。
fn pitch_bridge_clip_ids(input: &Value) -> Result<Vec<String>, String> {
    let ids: Vec<String> = serde_json::from_value(input["clipIds"].clone())
        .map_err(|_| "invalid clip id list".to_string())?;
    if ids.is_empty() || ids.len() > 512 {
        return Err("the pitch bridge needs 1..512 clips".into());
    }
    Ok(ids)
}

/// 这个片段唯一直接绑定的宿主 take（与 host_edit_target 同一口径）。
///
/// 【为什么这里必须先剥掉 namespace 前缀】编辑器时间线里的 id 是**呈现 id**
/// （`hfs-ui-{pid}-{n}-ara-item-{guid}`，见 `EditorSession::rewrite_ids`），而
/// `host_edit_target` 认的是**裸**宿主身份（`ara-item-{guid}`）。此前这条路径
/// 直接把呈现 id 递进去，`strip_prefix("ara-item-")` 恒为 `None`，随后的
/// `clip_ids` 反查也找不到 —— 于是**两个方向都恒失败**，用户看到的是
/// "clip has no unique directly bound host take"（现场日志里每一次音高互转都是它）。
/// `plan_host_edit` 一直有这一句，本函数当初漏了。
fn pitch_bridge_target(
    session: &EditorSession,
    clip_id: &str,
) -> Result<crate::host::HostClipTarget, String> {
    let native_id = clip_id.strip_prefix(&session.namespace).unwrap_or(clip_id);
    let document = session.document().ok_or("host timeline unavailable")?;
    // 【顺序：take FX 先认领】一个 take FX 实例只服务它自己那条 take，它的清单里也只有
    // 那个 item —— 所以"它能不能解析这条 clip"本身就是最精确的判据。反过来（先问轨道
    // FX 实例）会把 clip 交给**同一个 item 的 active take**，而用户的实例可能挂在另一条
    // take 上：那是"转换成功、音高没变"与"把别的 take 改了"的直接成因。
    let owners = document.renderer_owners();
    let (take_fx, others): (Vec<_>, Vec<_>) = owners
        .into_iter()
        .partition(|owner| owner.host_placement().is_take_fx());
    for owner in take_fx.into_iter().chain(others) {
        if let Ok(target) = owner.host_edit_target(native_id) {
            return Ok(target);
        }
    }
    Err("clip has no unique directly bound host take".into())
}

/// 一个片段在桥接时需要的只读事实（全部在**不持时间线锁**时算好）。
struct PitchBridgeClip {
    /// 完整片段状态：导出方向要把它原样交给 `compute_clip_export_pitch_offsets`。
    clip: Clip,
    /// 时间线里的片段 id（含命名空间前缀，与命令入参同一口径）。
    clip_id: String,
    /// 这个片段唯一直接绑定的宿主 take（收集时解析一次，两个方向共用）。
    target: crate::host::HostClipTarget,
    root_track: String,
    /// D_PITCH（半音）：REAPER 的 Item 音高偏移。
    static_semitones: f64,
    /// take 的音高包络点（u 域）。
    points: Vec<hifishifter_kernel::pitch_bridge::EnvelopePoint>,
    /// 这个 take 上**到底有没有**音高（`D_PITCH ≠ 0`，或包络里存在非零值）。
    ///
    /// 【为什么它是一位独立的事实】"没有音高"与"音高是 0"在 REAPER 里是同一件事
    /// （都不变调），但在**导入**方向上是两件完全不同的事：把"没有音高"导进来，
    /// 等于把整条音高线重置成原线 —— 用户画的每一笔都会消失。批量作用域必须能分辨
    /// 这两者，否则"一键导入全部片段"就是一次静默的格式化。
    ///
    /// 【为什么不看 `<PITCHENV>` 块是否存在】`DEFSHAPE` 的默认值不进 `points`
    /// （`read_take_pitch` 只读 `PT` 行），而一个只有默认值的包络在语义上仍然可能
    /// 有音高。这里取**保守**判据：看不到任何非零值就当没有 —— 宁可少导入一条
    /// （用户还能对单条片段走严格模式），也不要误删别人的音高线。
    has_pitch: bool,
}

/// 片段的播放速率（非有限 / 非正 ⇒ 1，与渲染端同一口径）。
///
/// 【为什么 `pub(super)`】会话侧的包络指纹（`auto_pitch_write_fingerprint`）必须与导出
/// 用同一个速率：两处各写一份 `if rate > 0 { rate } else { 1.0 }`，一处改了另一处就会
/// "自动写出来的包络与手动点一次的不一样"。
pub(super) fn pitch_bridge_rate(clip: &Clip) -> f64 {
    if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
        clip.playback_rate as f64
    } else {
        1.0
    }
}

/// 这条片段的音高分析**是否已经落地**（原线按当前布局组装完成）。
///
/// 【为什么用 `pitch_orig_key` 而不是"`pitch_orig` 是不是全 0"】全 0 既可能是
/// "还没分析"，也可能是"分析完了，这段确实没有音高"。前者不能写、后者必须写（写出
/// 空包络 = 让 REAPER 侧也回到"不变调"）。只有键能区分这两者：它描述的是"这一轮
/// 原线是按哪个布局算的"，为空即表示这一轮还没算完。
fn pitch_bridge_analysis_ready(timeline: &TimelineState, root_track: &str) -> bool {
    timeline
        .params_by_root_track
        .get(root_track)
        .is_some_and(|entry| entry.pitch_orig_key.is_some())
}

/// 命令是否走宽容模式（批量作用域）。
///
/// 【为什么默认是严格】单条片段（时间轴右键）映射不到宿主 take 时必须**报错**：
/// 用户点的是那一条，静默跳过等于"点了没反应"。批量作用域相反 —— 500 条里有一条
/// 不是 ARA 片段，不该让另外 499 条一条都转不了。
fn pitch_bridge_tolerant(input: &Value) -> bool {
    input["tolerant"].as_bool().unwrap_or(false)
}

/// 收集片段事实。
///
/// `tolerant` 为真时（批量作用域），映射不到宿主 take 的片段进 `unresolved` 而不是让
/// 整批失败。
fn pitch_bridge_collect(
    session: &EditorSession,
    clip_ids: &[String],
    tolerant: bool,
) -> Result<(Vec<PitchBridgeClip>, Vec<String>), String> {
    let snapshot = session.timeline.lock_or_recover().clone();
    let mut collected = Vec::with_capacity(clip_ids.len());
    let mut unresolved: Vec<String> = Vec::new();
    for clip_id in clip_ids {
        let Some(clip) = snapshot
            .clips
            .iter()
            .find(|clip| clip.id == *clip_id)
            .cloned()
        else {
            if tolerant {
                unresolved.push(clip_id.clone());
                continue;
            }
            return Err("unknown editor clip".into());
        };
        // 一段解析拆成闭包：宽容模式下**任何**一条失败都只影响这一条。
        let resolved = (|| -> Result<PitchBridgeClip, String> {
            let root_track = snapshot
                .resolve_root_track_id(&clip.track_id)
                .ok_or("clip track has no root parameter line")?;
            let target = pitch_bridge_target(session, clip_id)?;
            // 先取出要用的两个数：`target` 随后整份移进结构体，之后再读它的字段
            // 就是 use-after-move。
            let static_semitones = target.geometry.take_pitch;
            let pitch = target.take_pitch_points(&|| true)?;
            let has_pitch = pitch.points.iter().any(|point| point.value.abs() > 1e-6)
                || static_semitones.abs() > 1e-6;
            Ok(PitchBridgeClip {
                clip,
                clip_id: clip_id.clone(),
                target,
                root_track,
                static_semitones,
                points: pitch.points,
                has_pitch,
            })
        })();
        match resolved {
            Ok(entry) => collected.push(entry),
            Err(error) if tolerant => {
                log::debug!("[ara] pitch bridge skipped {clip_id}: {error}");
                unresolved.push(clip_id.clone());
            }
            Err(error) => return Err(error),
        }
    }
    Ok((collected, unresolved))
}

/// 方向 1：REAPER take 音高 → HiFiShifter 参数线。
///
/// 【宿主侧一个字节都不改】`D_PITCH` 与 `PITCHENV` 留在原处：它们只在插件被旁路 /
/// 移除时作用，与插件渲染的音频永远不会同时处理（见本段开头的实测）。此前这条命令
/// 会把宿主侧的包络删掉、`D_PITCH` 归零，用户丢掉的是他自己在 REAPER 里的数据。
///
/// 【批量作用域的两条额外判据】`tolerant` 为真时：
///
/// 1. 映射不到宿主 take 的片段跳过（进 `unresolved`），不让整批失败；
/// 2. **take 上根本没有音高的片段一律不动**。导入一条"没有音高"的片段等于把它的
///    音高线重置成原线；批量跑一遍就是把工程里没碰过音高的片段全部格式化。
///    单条片段仍走严格模式 —— 用户明确点了那一条，"重置成原线"是他的意图。
fn import_host_take_pitch(session: &EditorSession, input: &Value) -> Result<Value, String> {
    let clip_ids = pitch_bridge_clip_ids(input)?;
    let tolerant = pitch_bridge_tolerant(input);
    // 【导入期间不许自动写回】本命令会经 `write_authority_frames` 改参数线，而那条路径
    // 正是「自动写入 Take 音高」的触发点。不抑制的话，导入结束 400 ms 后会来一次自动
    // 导出，而导出的 `D_PITCH = 0` 会把本命令刻意保留在宿主侧的 Item 音高偏移抹掉
    // （见文件开头"两侧各管各的"）。
    let _suppression = AutoPitchWriteSuppression::begin(session);
    let (clips, mut unresolved) = pitch_bridge_collect(session, &clip_ids, tolerant)?;

    let mut converted = 0usize;
    let mut written_frames = 0usize;
    let mut skipped_frames = 0usize;
    let mut windows: Vec<(String, u32, Vec<f32>, Vec<bool>)> = Vec::new();
    {
        let mut timeline = session.timeline.lock_or_recover();
        let frame_period_ms = timeline.frame_period_ms().max(0.1);
        let mut before_by_root: std::collections::BTreeMap<String, Vec<f32>> =
            std::collections::BTreeMap::new();
        for clip in &clips {
            if tolerant && !clip.has_pitch {
                unresolved.push(clip.clip_id.clone());
                continue;
            }
            timeline.ensure_params_for_root(&clip.root_track);
            let Some(entry) = timeline.params_by_root_track.get_mut(&clip.root_track) else {
                return Err("params_missing".into());
            };
            before_by_root
                .entry(clip.root_track.clone())
                .or_insert_with(|| entry.pitch_edit.clone());
            let start_sec = clip.clip.start_sec.max(0.0);
            let length_sec = clip.clip.length_sec.max(0.0);
            let rate = pitch_bridge_rate(&clip.clip);
            // 与导出方向同一个窗口：写回的帧区间必须正好是那个片段自己的那几帧。
            let (start_frame, end_frame) = hifishifter_kernel::frame_grid::clip_frame_window(
                start_sec,
                length_sec,
                frame_period_ms,
            );
            let count = end_frame - start_frame;
            let mut wrote_any = false;
            for index in 0..count {
                let frame = start_frame + index;
                let orig = entry.pitch_orig.get(frame).copied().unwrap_or(0.0);
                if !(orig.is_finite() && orig > 0.0) || frame >= entry.pitch_edit.len() {
                    skipped_frames += 1;
                    continue;
                }
                let offset_sec = index as f64 * frame_period_ms / 1000.0;
                let u = hifishifter_kernel::pitch_bridge::offset_to_take_u(offset_sec, rate);
                let value = clip.static_semitones
                    + hifishifter_kernel::pitch_bridge::sample_envelope(&clip.points, u);
                if !value.is_finite() {
                    skipped_frames += 1;
                    continue;
                }
                entry.pitch_edit[frame] = orig + value as f32;
                written_frames += 1;
                wrote_any = true;
            }
            if wrote_any {
                converted += 1;
            }
        }
        for (root, before) in before_by_root {
            let Some(entry) = timeline.params_by_root_track.get(&root) else {
                continue;
            };
            if let Some((start, values, present)) =
                hifishifter_kernel::editor::midi_import::pitch_write_window(
                    &before,
                    &entry.pitch_edit,
                )
            {
                windows.push((root, start, values, present));
            }
        }
    }
    // 【一次命令 = 一步撤销】旧实现每根根轨道各压一步：多轨选择下一次"导入"在撤销栈
    // 上要按三次，而用户只点了一次。空窗口不压点 —— 那会留下一步"按了什么也没发生"。
    if !windows.is_empty() {
        session.checkpoint_authority_step_as(HistoryOp::HostPitchImport.key());
    }
    // 【权威写在锁外】与 MIDI 导入同一条理由：write_authority_frames 会再取一次
    // 时间线锁，持锁调用即自死锁。撤销点已在上方压过，因此这里传 `false`。
    for (root, start, values, present) in &windows {
        match session.write_authority_frames(root, "pitch", *start, values, Some(present), false) {
            Some(Ok(_)) => {}
            Some(Err(error)) => return Err(error),
            None => return Err("parameter authority disappeared during the pitch import".into()),
        }
    }
    Ok(json!({
        "ok": true,
        "clips": clip_ids.len(),
        "clipsConverted": converted,
        "framesWritten": written_frames,
        "framesSkipped": skipped_frames,
        "unresolved": unresolved,
    }))
}

/// 为一批根轨道补一次 `dyn_orig` 组装（导出侧填充的能量判据要用它）。
///
/// 【必须在取时间线锁**之前**调用】`maybe_schedule_dyn_orig` 内部会
/// `session.timeline.lock()`（见 `dyn_analysis.rs` 的文档），持锁调用即自死锁 ——
/// 与 `editor/params.rs` 的同一条纪律。
///
/// 【为什么插件尤其需要它】独立 App 的引擎在 `handle_update_timeline` 里会为"画过
/// dyn / 动态面板打开"的根轨道组装基线，而插件这条路径此前**完全不组装**。于是插件
/// 里导出时逐帧电平永远是"未知" ⇒ 长间隙一律保守 hold，`dry` 档与 `bridge` 档听
/// 起来完全一样 —— 那会让 A/B 的结论直接失真。
///
/// 【口径是保守的】`dyn_orig` 是整条根轨组**融合后**的电平，不是"本 take 的电平"：
/// 别的 clip 在响 ⇒ 判为可闻 ⇒ hold。宁可少插值，也不要凭空造中间调。
fn schedule_dyn_orig_for_roots<'a>(
    session: &EditorSession,
    roots: impl IntoIterator<Item = &'a str>,
) {
    let mut unique: Vec<&str> = roots.into_iter().collect();
    unique.sort_unstable();
    unique.dedup();
    for root in unique {
        let _ =
            hifishifter_kernel::pitch_analysis::maybe_schedule_dyn_orig(&session.timeline, root);
    }
}

/// 方向 2：HiFiShifter 参数线 → REAPER take 音高包络。
///
/// 【为什么**不**清空 HiFiShifter 的音高线】take 音高是源级属性，ARA 渲染结果在它之后
/// （见本段开头的实测），所以包络根本不会作用到插件渲染出来的音频上 —— 清空只会让
/// 用户"转换完就听不见音高了"（现场报障）。两侧各留一份，谁也不会被叠加两次。
///
/// 【"还没分析"与"音高线是平的"必须分开】`compute_clip_export_pitch_offsets` 返回
/// `None`、或原线还是全 0 时，这条片段的曲线会被当成"平的" ⇒ 空断点 ⇒
/// `rewrite_take_pitch` 删掉整个 `<PITCHENV>`。
///
/// - **单条片段**保持原语义：用户点的是那一条，看得见，而且只有一步撤销。
/// - **批量作用域**里这不行：一次点击会顺手删掉几百条包络，而报告只会说"写了 N 条"。
///   因此批量模式先用 `pitch_orig_key` 判断"这一轮的分析是否落地"，没落地的片段
///   一律跳过并进 `unresolved`（见 `pitch_bridge_analysis_ready`）。
///
/// `auto`（入参）=「自动写入 Take 音高」用的形态：用户没点任何按钮，只是改了一下音高线。
/// 因此它比手动档更保守：没有唯一 take 的片段照样跳过，另外**绝不删掉已经存在的包络**
/// （空点集等于删掉整个 `<PITCHENV>`），也**不接管 take 自带 `D_PITCH` 的片段**
/// （归零会把那个偏移无声抹掉）。两条理由都在函数体内。
pub(super) fn export_host_take_pitch(
    session: &EditorSession,
    input: &Value,
) -> Result<Value, String> {
    let clip_ids = pitch_bridge_clip_ids(input)?;
    // 【自动档（逐次编辑的写回）一定是宽容档】它是"顺带写一下"：一条片段没有唯一 take
    // 不该让整批失败，而**分析还没落地的片段更不该被写成空包络** —— 空包络 = 删掉
    // 用户在 REAPER 里画好的 PITCHENV。
    let auto = input["auto"].as_bool().unwrap_or(false);
    let tolerant = pitch_bridge_tolerant(input) || auto;
    let (clips, mut unresolved) = pitch_bridge_collect(session, &clip_ids, tolerant)?;
    // 【自动档不接管"自带静态音高"的 take】本命令的语义是让包络成为 REAPER 侧唯一的
    // 音高来源，因此它会把 `D_PITCH` 归零。而自动档是用户**没点任何东西**时发生的：
    // 若那个 take 本来带着 +2 半音的 Item 音高偏移（插件这一侧完全看不出来），归零就是
    // 一次无声的数据丢失。跳过它，用户手动转一次即可（那一步是他自己点的，看得见）。
    let clips: Vec<PitchBridgeClip> = if auto {
        clips
            .into_iter()
            .filter(|clip| {
                if clip.static_semitones.abs() <= 1e-6 {
                    return true;
                }
                log::debug!(
                    "[pitch] auto take-pitch write skips {}: the take carries its own D_PITCH={} semitones",
                    clip.clip_id,
                    clip.static_semitones
                );
                unresolved.push(clip.clip_id.clone());
                false
            })
            .collect()
    } else {
        clips
    };
    // 能量通道：必须在下面的 `session.timeline.lock_or_recover()` **之前**
    // （`maybe_schedule_dyn_orig` 内部会再取一次时间线锁）。
    schedule_dyn_orig_for_roots(session, clips.iter().map(|clip| clip.root_track.as_str()));
    let mut converted = 0usize;
    let mut points_written = 0usize;
    let mut plans: Vec<(
        &PitchBridgeClip,
        Vec<hifishifter_kernel::pitch_bridge::EnvelopePoint>,
    )> = Vec::new();
    {
        let timeline = session.timeline.lock_or_recover();
        let frame_period_ms = timeline.frame_period_ms().max(0.1);
        // 【策略只解析一次】放在循环外：循环内每次重算等于每个 clip 读一遍配置、
        // 四次环境变量 —— 而结果对整批 clip 完全相同。
        let fill_policy =
            hifishifter_kernel::pitch_editing::take_pitch_fill_policy(frame_period_ms);
        for clip in &clips {
            // 【批量作用域：分析还没落地的片段一律不写】此时 `pitch_orig` 全是 0，
            // 会被当成"音高线是平的" ⇒ 空包络 ⇒ **删掉用户在 REAPER 里画好的包络**。
            // 单条片段保持原语义：用户点的是那一条，看得见、且只有一步撤销。
            // 【自动档不查"分析是否落地"这个键】那个键是**记账**：布局一变就会被清空，
            // 与"这段到底有没有可写的音高"不是一回事。自动档真正要防的是"把用户在 REAPER
            // 里画好的包络删掉"，那件事由下面那条"空包络守卫"直接回答（用包络本身判断）。
            if tolerant && !auto && !pitch_bridge_analysis_ready(&timeline, &clip.root_track) {
                unresolved.push(clip.clip_id.clone());
                continue;
            }
            // 【单条片段保持原语义】没有分析结果时按"全 0 曲线"处理 ⇒ 写空包络。那一步
            // 是**可见**的（用户点的是那一条，且只有一步撤销）；批量方向由上面的判据
            // 挡住 —— 一次点击不该顺手删掉几百条包络。
            // 【未识别帧必须在这里填掉】渲染端对"这一帧没有 F0"的语义是**不变换**，
            // 而宿主包络是连续控制信号 —— 逐帧写 0 会让移调器在每个辅音处把比率从
            // +12 拉到 0 再拉回 +12。填充只发生在导出边界（见
            // `compute_clip_take_pitch_offsets` 的文档），渲染端一个字节都不改。
            let offsets = hifishifter_kernel::pitch_editing::compute_clip_take_pitch_offsets(
                &timeline,
                &clip.clip,
                &fill_policy,
            )
            .unwrap_or_default();
            let points = hifishifter_kernel::pitch_bridge::build_take_pitch_points(
                &offsets,
                frame_period_ms,
                clip.clip.length_sec.max(0.0),
                pitch_bridge_rate(&clip.clip),
            );
            // 【自动档绝不删掉已经存在的包络】空点集 = 空包络 = 整个 `<PITCHENV>` 被删掉。
            // 手动档这是语义的一部分（用户点名要转，而且只有一步撤销）；自动档不行 ——
            // 它发生在用户**没点任何东西**的时候，而"这一轮算出来是平的"既可能是用户把线
            // 拉平了，也可能是原线还没组装好。分不清就不动。
            if auto && points.is_empty() && !clip.points.is_empty() {
                log::debug!(
                    "[pitch] auto take-pitch write keeps the existing envelope of {} (nothing to write)",
                    clip.clip_id
                );
                unresolved.push(clip.clip_id.clone());
                continue;
            }
            plans.push((clip, points));
        }
    }
    for (clip, points) in plans {
        // 目标在收集阶段已经解析过（并校验过 GUID）；这里**复用**同一份，不再重新查
        // 一次 —— 两次解析之间的时间窗里，片段完全可能被宿主删掉或换掉 take，而
        // 写错对象的表现是"另一个片段的音高变了"。
        let target = &clip.target;
        target.write_take_pitch_points(&points, &|| true)?;
        // 「需重置 Item 音高偏移」：`D_PITCH` 与包络会叠加，而这条命令的语义是让
        // 包络成为 REAPER 侧唯一的音高来源；它的值在导入方向已经折进 `pitch_edit`。
        // 【为 0 时不写】自动档会在每次编辑后跑一遍，而"写一次 0"没有任何效果却要过一次
        // 宿主 API；这一句只是省掉那次空写，语义与无条件写入逐字相同。
        if clip.static_semitones.abs() > 1e-6 {
            target.set_take(c"D_PITCH", 0., &|| true)?;
        }
        target.update(&|| true)?;
        points_written += points.len();
        converted += 1;
    }
    Ok(json!({
        "ok": true,
        "clips": clip_ids.len(),
        "clipsConverted": converted,
        "pointsWritten": points_written,
        "unresolved": unresolved,
    }))
}

/// 逐片段的**源音高参考曲线**（参数编辑器里那条 "Clip 音高参考线"）。
///
/// 【为什么插件也必须回答】独立 App 分析完成后由后端用 `clip_pitch_data` 事件
/// 推送曲线，卫星窗口晚打开时再用 `get_clip_pitch_data` 补一次。插件的钢琴卷帘
/// 默认就打开这条参考线（`showClipPitchOverlay` 默认 true），而这条命令此前落到
/// `Command unavailable` —— 于是每次挂载都刷一条 ERROR，参考线永远空白。
///
/// 【数据来源】插件的原线就是 `params_by_root_track[root].pitch_orig`：**工程时间
/// 轴**上的 MIDI 音高（0 = 无音高）。因此这里只做"按 clip 的时间窗切片"，payload
/// 形状与 App 的 `build_clip_pitch_data_payload` 逐字段一致（`curve_start_sec`
/// = clip 起点，`midi_curve` 每帧一个音高，`frame_period_ms` 与时间线同源）。
///
/// 【为什么没有原线就不返回这个 clip】分析未完成时 `pitch_orig` 是空表，切出来
/// 只有 0；把全 0 曲线交给前端等于让它把"还没算"缓存成"就是没有"，而参考线不会
/// 再取第二次。空表 = 没有 payload，与 App 的 cache-miss 同一处置。
fn clip_pitch_data(session: &EditorSession, input: &Value) -> Result<Value, String> {
    const MAX_CLIPS: usize = 512;
    /// 单条曲线最多回多少帧（5 ms 一帧 ⇒ 10 分钟；再长的片段参考线也没法看）。
    const MAX_FRAMES_PER_CLIP: usize = 120_000;
    /// 一次请求的**总**帧预算：这条命令走 actor 线程，回一份几十 MB 的 JSON 会把
    /// 界面顶住。超出的 clip 直接跳过（不报错）—— 前端在分析完成事件里还会再来。
    const MAX_FRAMES_TOTAL: usize = 600_000;
    let clip_ids: Vec<String> = serde_json::from_value(input["clipIds"].clone())
        .map_err(|_| "invalid clip id list".to_string())?;
    if clip_ids.is_empty() || clip_ids.len() > MAX_CLIPS {
        return Err(format!("the pitch data request needs 1..{MAX_CLIPS} clips"));
    }
    let timeline = session.timeline.lock_or_recover();
    let timeline_period = timeline.frame_period_ms().max(0.1);
    let mut payloads = Vec::new();
    let mut budget = MAX_FRAMES_TOTAL;
    for clip_id in clip_ids {
        let Some(clip) = timeline.clips.iter().find(|clip| clip.id == clip_id) else {
            continue;
        };
        let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
            continue;
        };
        let Some(params) = timeline.params_by_root_track.get(&root) else {
            continue;
        };
        if params.pitch_orig.is_empty() {
            continue;
        }
        let period = if params.frame_period_ms.is_finite() && params.frame_period_ms > 0.0 {
            params.frame_period_ms
        } else {
            timeline_period
        };
        // 与组装侧同一个窗口（`frame_grid`）：参考线必须与导出包络落在同一帧上，
        // 否则用户看到"界面上的线和导出的不一样"却查不出差在哪。
        let (start_frame, end_frame) = hifishifter_kernel::frame_grid::clip_frame_window(
            clip.start_sec,
            clip.length_sec,
            period,
        );
        let count = end_frame - start_frame;
        if count > MAX_FRAMES_PER_CLIP || count > budget {
            continue;
        }
        budget -= count;
        let mut curve = Vec::with_capacity(count);
        let mut voiced = false;
        for index in 0..count {
            let value = params
                .pitch_orig
                .get(start_frame + index)
                .copied()
                .unwrap_or(0.0);
            voiced |= value > 0.0;
            curve.push(value);
        }
        if !voiced {
            // 这一段没有能量（静音 / 还没分析到）⇒ 没有可画的参考线。
            continue;
        }
        payloads.push(json!({
            "clip_id": clip.id,
            "curve_start_sec": clip.start_sec.max(0.0),
            "midi_curve": curve,
            "frame_period_ms": period,
        }));
    }
    Ok(json!({"ok": true, "payloads": payloads}))
}

/// 快速导出**选中片段**：范围取选中片段在工程时间上的并集，其余走同一条路径。
///
/// 【为什么不另写一套渲染】导出与播放共用 `RenderInput::mix_interleaved`
/// （见 `export_audio_to_file` 的说明）；这里只换范围与目标路径，渲染与编码一字不改。
fn quick_export_selected_clips(
    session: &EditorSession,
    request: &serde_json::Value,
) -> Result<Value, String> {
    let clip_ids: Vec<String> = serde_json::from_value(request["clipIds"].clone())
        .map_err(|_| "invalid clip id list".to_string())?;
    let (start_sec, end_sec) = {
        let timeline = session.timeline.lock_or_recover();
        let mut range: Option<(f64, f64)> = None;
        for clip in timeline
            .clips
            .iter()
            .filter(|clip| clip_ids.contains(&clip.id))
        {
            let lo = clip.start_sec;
            let hi = clip.start_sec + clip.length_sec;
            range = Some(match range {
                Some((a, b)) => (a.min(lo), b.max(hi)),
                None => (lo, hi),
            });
        }
        range.ok_or("none of the selected clips is in the timeline")?
    };
    export_audio_to_file(
        session,
        &json!({
            "mode": "project",
            "projectOutputDir": request["outputDir"],
            "projectFileName": request["fileName"],
            "range": {"kind": "custom", "startSec": start_sec, "endSec": end_sec},
            "encoder": request.get("encoder").cloned().unwrap_or(Value::Null),
        }),
    )
}

/// MIDI 输入路径判据：必须是**用户刚在原生对话框里选过**的绝对文件路径。
///
/// 【为什么不复用 `browser_media_path`】后者要么落在已授权目录内，要么必须是媒体
/// 扩展名。MIDI 导入的路径来自插件自己的"打开文件"对话框（`open_midi_dialog`），
/// 用户刚刚亲自选过 —— 与音频导入读真实磁盘路径是同一条口径，不需要目录授权；
/// 而 `.mid` 也不在媒体扩展名表里。仍然要求绝对路径，免得相对路径被当成进程工作
/// 目录下的文件。
fn midi_input_path(input: &str) -> Result<std::path::PathBuf, String> {
    let path = std::path::Path::new(input);
    if !path.is_absolute() {
        return Err("an absolute MIDI path is required".into());
    }
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err("not a file".into());
    }
    Ok(path)
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Frames {
    track_id: String,
    param: String,
    start_frame: u32,
    #[serde(default)]
    frame_count: u32,
    stride: Option<u32>,
    binary: Option<bool>,
    with_sentinel: Option<bool>,
    #[serde(default)]
    values: Vec<f32>,
    /// 与 values 等长的显式编辑位图；缺省 = 全 true（旧前端兼容）。
    present: Option<Vec<bool>>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Static {
    track_id: String,
    param: String,
    value: Option<f64>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Mix {
    track_id: String,
    from: String,
    ranges: Vec<hifishifter_kernel::editor::ConvertRange>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StretchLinked {
    track_id: String,
    mappings: Vec<hifishifter_kernel::state::StretchLinkedRangeSec>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Segment {
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrackPatch {
    track_id: String,
    volume: Option<f32>,
    muted: Option<bool>,
    solo: Option<bool>,
    compose_enabled: Option<bool>,
    pitch_analysis_algo: Option<PitchAnalysisAlgo>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PrivateTrackMove {
    track_id: String,
    target_index: usize,
    parent_track_id: Option<String>,
}

/// 一份时间线载荷的**软**上限：超过它就自动降级为 lite。
///
/// 【为什么要软上限】硬上限（`client::RESPONSE_BUDGET`，8 MiB）在超限时只能如实
/// 回一个错误 —— 界面于是永久停在上一份快照上：看起来能用，其实再也不更新。
/// 软上限让同一次请求**降级**而不是失败：界面继续刷新，代价只是这一帧没有波形
/// 预览与 MIDI 音符。
const PAYLOAD_SOFT_LIMIT_BYTES: usize = 6 * 1024 * 1024;

/// 估算一份**全量**载荷的序列化体积。
///
/// 【为什么估算而不是实测】实测要先把整个 `TimelineStatePayload` 序列化一遍 ——
/// 而那正是最贵的一步，等于最坏情况下付两次代价。这里要回答的只是"是不是已经大到
/// 离谱"，取主导项即可：`waveform_preview`（4096 个 f32 的数组）与
/// `midi_note_data` 是唯一会随 clip **时长**线性增长的字段，其余全是定长标量。
///
/// 【为什么系数大于 1】JSON 里一个 f32 序列化成 8-12 个字符，解析成 JS 数字还要
/// 再占 8 字节；一个 MIDI 音符是一个五字段对象，落在堆上约 100 字节。系数取"解析
/// 后的堆占用"，因为那才是能把渲染进程打死的东西。
fn estimated_full_payload_bytes(
    timeline: &hifishifter_kernel::state::model::TimelineState,
) -> usize {
    // 轨道、工程元数据、tempo map、历史深度的粗略基数。
    let mut bytes = 16 * 1024;
    for clip in &timeline.clips {
        // 每条 clip 的定长字段（id / 路径 / 淡变 / 速率 / 颜色 …）。
        bytes += 1024;
        bytes += clip
            .midi_note_data
            .as_ref()
            .map_or(0, |notes| notes.len() * 96);
        for take in &clip.takes {
            bytes += 256;
            bytes += take
                .midi_note_data
                .as_ref()
                .map_or(0, |notes| notes.len() * 96);
        }
    }
    bytes
}

/// 构造原payload，包括真实本地dirty和历史深度，不伪造工程文件路径。
pub(crate) fn payload(session: &EditorSession, lite: bool) -> Result<Value, String> {
    let (position, _) = session.transport();
    let timeline = session.timeline.lock_or_recover();
    // 【为什么在这里判，而不是在投递处判】只有这里还拿得到 `TimelineState` ——
    // 一旦它变成 `Value`，"降级"就无从做起了。判据是**估算**体积（见上）。
    let estimated = if lite {
        0
    } else {
        estimated_full_payload_bytes(&timeline)
    };
    let degraded = !lite && estimated > PAYLOAD_SOFT_LIMIT_BYTES;
    if degraded {
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "gui_payload_lite",
            format_args!(
                "[gui] timeline payload degraded to lite: estimated {} KiB > {} KiB \
                 (clips={}); waveform previews and MIDI notes are omitted from this frame",
                estimated / 1024,
                PAYLOAD_SOFT_LIMIT_BYTES / 1024,
                timeline.clips.len()
            ),
        );
    }
    let mut payload = if lite || degraded {
        timeline.to_payload_lite()
    } else {
        timeline.to_payload()
    };
    // The editor fetches waveform mipmaps through the binary API. Repeating
    // previews in every timeline reply floods the UI thread and WebView heap.
    for clip in &mut payload.clips {
        clip.waveform_preview = None;
    }
    payload.playhead_sec = position.max(0.);
    let (undo, redo) = history_depths_of(&session.history.lock_or_recover());
    payload.undo_depth = Some(undo);
    payload.redo_depth = Some(redo);
    let project = session.project.lock_or_recover().clone();
    payload.project = Some(hifishifter_kernel::models::ProjectMetaPayload {
        // 形态名字要说实话：Capture 实例与 ARA 文档是两种东西（用户报障时这是
        // 第一个要问的问题）。
        name: if session.is_capture() {
            "HiFiShifter Capture"
        } else {
            "HiFiShifter ARA"
        }
        .into(),
        path: None,
        dirty: session.generation.load(Ordering::Acquire)
            != session.applied.load(Ordering::Acquire),
        recent: vec![],
        notes_markdown: project.notes_markdown,
        base_scale: project.base_scale,
        use_custom_scale: project.use_custom_scale,
        custom_scale: project.custom_scale,
        beats_per_bar: project.beats_per_bar,
        time_signature_denominator: project.time_signature_denominator,
        grid_size: project.grid_size,
        stretch_algorithm_override: project.stretch_algorithm_override,
        hifigan_mel_stretch_override: project.hifigan_mel_stretch_override,
        save_undo_history: project.save_undo_history,
    });
    drop(timeline);
    let mut result = value(payload)?;
    session.decorate_host_fades(&mut result);
    // 宿主音频读数（语言无关分类）：GUI 据此把"看起来能用、其实没内容"的空白 Clip
    // 说明白（见 `render::extension::HostAudioState`）。
    session.decorate_host_audio(&mut result);
    Ok(result)
}
pub(crate) fn history_state(session: &EditorSession) -> Value {
    let history = session.history.lock_or_recover();
    let (undo, redo) = history_depths_of(&history);
    let records: Vec<_> = history
        .records
        .iter()
        .map(|r| json!({"label":r.label,"atMs":r.at_ms}))
        .collect();
    // `backend` 是**判别式**：本地栈的 `label` 是 op key（前端按 `history_op_<key>` 本地化），
    // 宿主栈的 `label` 是宿主自己的可读字符串（前端原样显示）。没有它，前端只能无差别加
    // 前缀，于是宿主字符串被渲染成 `history_op_Move items`（见 `UndoHistoryPanel::labelOf`）。
    json!({"ok":true,"backend":"local","position":history.position,"undoDepth":undo,"redoDepth":redo,
        "records":if records.is_empty() {vec![json!({"label":null,"atMs":history.started_at_ms})]} else {records}})
}
fn track_exists(session: &EditorSession, track: &str) -> Result<(), String> {
    if session
        .timeline
        .lock_or_recover()
        .tracks
        .iter()
        .any(|t| t.id == track)
    {
        Ok(())
    } else {
        Err("unknown host track".into())
    }
}
fn budget(start: u32, count: usize) -> Result<(), String> {
    if count > 1_000_000
        || (start as usize)
            .checked_add(count)
            .is_none_or(|n| n > 1_000_000)
    {
        return Err("parameter frame budget exceeded".into());
    }
    Ok(())
}
fn after_write(session: &EditorSession, result: Value) -> Result<Value, String> {
    // 【为什么在这里留一行】这是**所有**曲线写入命令的统一收尾，而它此前是静默的：
    // 前端只看到一句 `Invoke failed: set_param_frames`，插件侧没有任何一行说明被什么
    // 拒绝。写入是交互里最热的路径（拖动时每帧一次），所以按固定键限流：10 s 一行，
    // 带被压掉的条数。
    if result["ok"] == false {
        let message = result["error"]
            .as_str()
            .unwrap_or("parameter operation rejected")
            .to_owned();
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            "cmd_failed:write",
            format_args!("[cmd] write rejected: {message}"),
        );
        return Err(message);
    }
    // 【为什么不再读 `session.error`】写入命令的成败由 `result["ok"]` 回答。把会话级
    // 错误当成"这次也失败了"会让一次历史失败永久污染之后每一次成功写入：前端收到
    // Err 就把本地已经生效的笔画回滚，用户看到"画完就没了"。会话级错误现在只由
    // `ensure_loaded` 这类**终态**失败设置，并通过 `get_state` 呈现给界面。
    session.emit("history_state", history_state(session));
    session.notify_timeline();
    Ok(result)
}

/// 插件自有音乐上下文（基准音阶 / 自定义音阶 / 音阶变化点）变化后的统一收尾。
///
/// 【为什么两件事必须一起做】
/// 1. **立刻投影到 GUI 时间线**：`payload()` 读的是 `session.timeline`，而它要到下一次
///    `ensure_loaded` 才重建。不投影的话，本命令的**响应**会带着旧的 `project_scale_notes`
///    / `tempo_map`，而前端把响应当权威回声照单应用（`applyTimelineState` 在
///    `tempo_map !== undefined` 时无条件覆盖），用户刚提交的改动会被自己的回声抹掉。
/// 2. **让在途渲染作废**：音阶进渲染缓存键（`scale-signature`），与
///    `refresh_host_geometry` 处理 fade 变化是同一条理由。
///
/// 投影本身是**派生**的（真相在 `settings_store`），所以这里写 `session.timeline`
/// 不违反"派生物会被重建"那条纪律 —— 它会在每次重建后由
/// `project_plugin_musical_context_locked` 重新算出来。
fn after_musical_context_write(session: &EditorSession) {
    let Some(document) = session.document() else {
        return;
    };
    document.project_plugin_musical_context_locked(&mut session.timeline.lock_or_recover());
    document.render_epoch.fetch_add(1, Ordering::AcqRel);
}
fn peaks(
    session: &EditorSession,
    path: &str,
) -> Result<std::sync::Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>, String> {
    session.check_source(path)?;
    if let Some(found) = session.peaks.lock_or_recover().get(path).cloned() {
        return Ok(found);
    }
    let result = std::sync::Arc::new(hifishifter_kernel::hfspeaks_v2::compute_mipmap_peaks(
        std::path::Path::new(path),
    )?);
    let mut cache = session.peaks.lock_or_recover();
    let total: u64 = cache.values().map(|p| p.estimated_byte_size()).sum();
    if total.saturating_add(result.estimated_byte_size()) > 64 * 1024 * 1024 {
        cache.clear();
    }
    cache.insert(path.into(), result.clone());
    Ok(result)
}

/// Read-only startup data has no document/PCM dependency and can bypass the actor.
pub(crate) fn startup_settings(command: &str) -> Option<Value> {
    match command {
        "get_ui_settings" => Some(json!(crate::settings_store::settings())),
        "ui_kv_dump" => Some(json!(crate::settings_store::frontend_prefs())),
        "get_log_settings" => Some(super::log_settings::get_log_settings()),
        _ => None,
    }
}

/// 纯文本剪贴板写入：UI 线程内联，无需宿主 PCM / EditorSession。
/// 返回同形 `{ok,error}`，前端不必按宿主形态分支。
pub(crate) fn startup_clipboard_text(input: &Value) -> Value {
    let Some(text) = input["text"].as_str() else {
        return json!({"ok":false,"error":"clipboard text required"});
    };
    if let Err(reason) = hifishifter_clipboard::system_clipboard::validate_clipboard_text(text) {
        return json!({"ok":false,"error":reason});
    }
    match hifishifter_clipboard::system_clipboard::write_text(text) {
        Ok(()) => json!({"ok":true}),
        Err(error) => json!({"ok":false,"error":error}),
    }
}

/// 仅actor线程调用。宿主PCM不可用/未知命令/平台能力缺失均明确报错。
pub(super) fn dispatch(
    session: &EditorSession,
    command: &str,
    input: Value,
) -> Result<Value, String> {
    if let Some(settings) = startup_settings(command) {
        return Ok(settings);
    }
    if command == "set_log_settings" {
        return Ok(super::log_settings::set_log_settings(&input));
    }
    match command {
        "list_directory"
        | "stat_paths"
        | "get_audio_file_info"
        | "read_audio_preview"
        | "get_media_audio_streams"
        | "collect_folder_media"
        | "search_files_recursive"
        | "create_directory"
        | "rename_path"
        | "delete_paths"
        | "reveal_paths_in_file_manager"
        | "open_path_with_default_app" => return session.browser_command(command, &input),
        // 设置由进程级的 `settings_store` 持有：它属于用户，不属于某一个 ARA 文档，
        // 并且与独立 App 共用同一份配置文件。此前这里是 `session.settings`（每个
        // 文档一份、初始化为出厂默认），保存只改内存、不落盘。
        "get_ui_settings" => return value(crate::settings_store::settings()),
        "save_ui_settings" => {
            return value(crate::settings_store::save_settings_patch(
                &input["settings"],
            )?)
        }
        // 前端偏好（原 localStorage 的 `hifishifter.*` 键）。批量接口：前端启动时
        // 一次取走全部、写入时按去抖批量提交，避免一次设置变更产生多个往返。
        "ui_kv_dump" => return value(crate::settings_store::frontend_prefs()),
        "ui_kv_put" => {
            let patch: std::collections::BTreeMap<String, String> = args(input["patch"].clone())?;
            if patch.len() > 512 {
                return Err("too many preference keys in one write".into());
            }
            return value(crate::settings_store::save_frontend_prefs(patch));
        }
        "ui_kv_delete" => {
            let keys: Vec<String> = args(input["keys"].clone())?;
            if keys.len() > 512 {
                return Err("too many preference keys in one delete".into());
            }
            return value(crate::settings_store::delete_frontend_prefs(&keys));
        }
        "get_about_info" => {
            // 与独立 App **逐字段同形**：前端 `AboutDialog` 只认这一组键。
            // 原先这里返回 `{name, version, host}`，于是插件里版本号显示错、
            // commit 恒空、仓库链接恒走前端兜底值。
            return Ok(hifishifter_kernel::build_info::about_payload(
                crate::plugin_form_label(session.capture_owner().as_deref()),
                crate::VERSION,
            ));
        }
        "plugin_get_apply_state" => return Ok(session.state()),
        // 宿主链接：导出工作区供独立 App 编辑，并回报当前状态。
        //
        // 【为什么导出是异步的】它要写音频（可能上百 MB）。命令分发与 actor 循环是
        // 同一个线程，同步写完会让 UI 在整个导出期间无响应 —— 而"点一下导出，
        // 然后去独立 App 里编辑"正是这个功能唯一的用法。结果由 `host_link_status` 轮询。
        "host_link_export" => {
            let owner = session
                .capture_owner()
                .ok_or("host link is only available in Capture mode")?;
            return crate::render::link::start_export(owner);
        }
        "host_link_status" => return Ok(crate::render::link::status()),
        "host_link_release" => return Ok(crate::render::link::release()),
        // Help 菜单的「打开日志目录」。
        //
        // 【为什么同时回报路径】打开资源管理器是尽力而为（[`reveal_directory`]）；
        // 无论成败都要把目录与文件路径交出去，前端才能在失败时让用户**复制**它，
        // 而不是逼他手抄。
        "open_log_folder" => {
            let dir = crate::diagnostics::log_directory();
            let file = crate::diagnostics::log_file();
            let opened = super::browser_files::reveal_directory(&dir);
            return Ok(json!({
                "ok": true,
                "path": dir.to_string_lossy(),
                "file": file.to_string_lossy(),
                "opened": opened.is_ok(),
            }));
        }
        // 诊断导出与基准测试：导出走 UI 线程的精简实现（见 `webview.rs`，命令名与
        // 独立 App **相同**，前端不必按模式分支）。
        //
        // 【为什么基准测试现在也放行】此前它在这里被拒绝，理由是"初始化 ORT 会话可能
        // 在 GPU/驱动异常的环境下硬崩，而在 DAW 进程里崩会带走用户的整个会话"。那个
        // 顾虑在**当前内核**里已经被系统性缓解（见 `vocoder_ort_session` 的文件头）：
        // 全局单飞锁带租约抢占、烟测有超时且超时即禁用该 EP、GPU 候选创建失败只记
        // 日志并换下一个。而剩下的真正风险是"驱动在 `run()` 里崩"——那是进程级故障，
        // 无论跑不跑基准测试，用户下一次渲染一样会撞上。
        //
        // 反过来，基准测试是用户排查"该选哪个推理设备"的**唯一**手段。用"可能崩"
        // 封掉一个用户主动触发、且自带收敛保护的入口，代价大于收益。
        "run_vocoder_benchmark" => {
            // 不可达：`lane()` 把它路由到慢车道（见 `SLOW_COMMANDS`），由
            // `slow_lane` 在独立线程上执行 `dispatch_slow`。留这条 arm 是为了万一
            // 有人绕过 `enqueue` 直接调 `dispatch` 时给出**可诊断**的失败 ——
            // 而不是让一次 31 秒的推理悄悄占住 actor（那正是本方案要消灭的故障）。
            return Err("run_vocoder_benchmark must run on the slow lane".into());
        }
        "plugin_history_barrier" => {
            session.suppress_history.store(false, Ordering::Release);
            return Ok(json!({"ok":true}));
        }
        "plugin_editor_barrier" => return Ok(json!({"ok":true})),
        "get_timeline_inventory" => {
            // 媒体写入回执只取结构，不为显示新GUID先物化整份宿主PCM或跑分析。
            if let Some(document) = session.document() {
                session.refresh_ui_geometry(&document, true);
            }
            return payload(session, false);
        }
        "select_track" | "select_clip" => {
            // 选择已确认的GUI对象不必等PCM；新粘贴片段先选中，再由后台补音频/波形。
            if let Some(document) = session.document() {
                session.refresh_ui_geometry(&document, true);
            }
            if command == "select_track" {
                let id = input["trackId"].as_str().ok_or("trackId missing")?;
                track_exists(session, id)?;
                session.timeline.lock_or_recover().select_track(id);
            } else {
                let id = input["clipId"].as_str().map(str::to_owned);
                let mut timeline = session.timeline.lock_or_recover();
                if let Some(id) = &id {
                    if !timeline.clips.iter().any(|clip| &clip.id == id) {
                        return Err("unknown host clip".into());
                    }
                }
                timeline.select_clip(id);
            }
            // 只有已授权clip才切换源曲线，未就绪的显示占位绝不读本地原文件。
            let authorized = {
                let timeline = session.timeline.lock_or_recover();
                timeline.selected_clip_id.as_ref().is_none_or(|id| {
                    timeline
                        .clips
                        .iter()
                        .any(|clip| &clip.id == id && clip.source_path.is_some())
                })
            };
            if authorized {
                session.select_source_projection()?;
            }
            session.notify_timeline();
            return payload(session, false);
        }
        "get_playback_state" => {
            // 宿主播放态不依赖曲线载入；Unsupported/Conflict也必须还能观察播放并暂停。
            return Ok(session.playback_state());
        }
        // 共振峰工具窗口：命令名与载荷和独立 App **逐字段同形**，前端不必按模式分支。
        // 音频来自宿主授权的 PCM（插件禁止读本地文件），切片/倒放/下混与 App 共用
        // kernel 的 `formant_cache::analyze_clip_formants_from_interleaved`。
        "analyze_clip_formants" => return value(session.analyze_clip_formants(&input)?),
        // ── 参数块：与独立 App 同形的只读 / 缓存命令 ──
        //
        // 只读查询与合并持久化在第一批闭环；音频重分析在第二批由 `enqueue`
        // 分流到慢车道，worker 只计算、actor 复核后提交。
        "get_clip_syllable_spans" => {
            let clip_id = input["clipId"].as_str().ok_or("clipId missing")?;
            let timeline = session.timeline.lock_or_recover();
            let clip = timeline
                .clips
                .iter()
                .find(|clip| clip.id == clip_id)
                .ok_or("clip_not_found")?;
            let take = clip.active_take();
            let source_path = take
                .source_path
                .as_deref()
                .or(clip.source_path.as_deref())
                .unwrap_or_default();
            let digest = crate::editor::session::syllable_source_digest(
                source_path,
                take,
                timeline.frame_period_ms(),
            );
            let window =
                hifishifter_kernel::param_blocks::projection::ClipProjection::from_clip_take(
                    clip, take,
                )
                .audible_source_window();
            let analysis = take
                .syllable_analysis
                .as_ref()
                .filter(|analysis| analysis.is_current_for_window(&digest, window));
            return Ok(syllable_payload(clip_id, &take.id, analysis, window));
        }
        "analyze_clip_syllables" => {
            // 不可达：`enqueue` 在慢车道分流处改投 `EnqueueSyllableAnalysis`，
            // 由 actor 冻结快照后再派发慢任务。保留这条 arm 是为了有人绕过
            // `enqueue` 直接调 `dispatch` 时得到可诊断的失败，而不是在 actor 上
            // 同步解码（那会让所有编辑与宿主保存一起排队）。
            return Err("analyze_clip_syllables must run on the syllable slow lane".into());
        }
        "get_param_block_view" => {
            return session.param_block_view_payload(
                input["rootTrackId"].as_str().ok_or("rootTrackId missing")?,
            );
        }
        "set_param_block_edits" => {
            let root = input["rootTrackId"].as_str().ok_or("rootTrackId missing")?;
            let expected = input["expectedEditRevision"]
                .as_u64()
                .ok_or("expectedEditRevision missing")?;
            return session.set_param_block_edits(
                root,
                expected,
                args(input["edits"].clone())?,
                input["checkpoint"].as_bool().unwrap_or(true),
                input
                    .get("curvePatches")
                    .map(|value| args(value.clone()))
                    .transpose()?
                    .unwrap_or_default(),
                input["expectedCurveRevision"].as_u64(),
                input["expectedGeometryRevision"].as_u64(),
                input["expectedGeometrySignature"].as_str(),
            );
        }
        "restore_game_param_blocks" => {
            let root = input["rootTrackId"].as_str().ok_or("rootTrackId missing")?;
            /*
             * 前端发送的是 ranges（与独立 App 同一契约）；单 range 字段仅作为
             * 旧调用方的兼容入口保留，不能再成为插件模式的隐藏崩溃点。
             */
            let ranges: Vec<(u32, u32)> = if let Some(items) = input["ranges"].as_array() {
                items
                    .iter()
                    .map(|item| {
                        let start = item["startFrame"]
                            .as_u64()
                            .ok_or("ranges.startFrame missing")?
                            as u32;
                        let count = item["frameCount"]
                            .as_u64()
                            .ok_or("ranges.frameCount missing")?
                            as u32;
                        Ok((start, start.saturating_add(count)))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            } else {
                let start = input["startFrame"]
                    .as_u64()
                    .ok_or("startFrame or ranges missing")? as u32;
                let end = input["endFrameExclusive"]
                    .as_u64()
                    .ok_or("endFrameExclusive or ranges missing")? as u32;
                vec![(start, end)]
            };
            return session.restore_game_param_block_ranges(
                root,
                &ranges,
                input["checkpoint"].as_bool().unwrap_or(true),
            );
        }
        "plugin_refresh" => {
            session.ensure_loaded(input["force"].as_bool().unwrap_or(false))?;
            return payload(session, false);
        }
        // 语言原先只是原样回显（"返回 ok 但什么也没做"）。现在记进与独立 App
        // 共用的前端偏好，用户换回 App 时语言也跟着走。
        "set_ui_locale" => {
            let locale = input["locale"].as_str().ok_or("locale missing")?;
            if locale.len() > 32 {
                return Err("locale too long".into());
            }
            return Ok(json!({"ok":true,"locale":crate::settings_store::set_locale(locale)}));
        }
        "consume_startup_project_path" => return Ok(Value::Null),
        "get_processor_params" => {
            return value(
                hifishifter_kernel::editor::capabilities::get_processor_params(
                    input["algo"].as_str().ok_or("algo missing")?.into(),
                ),
            )
        }
        "transliterate" => {
            let texts: Vec<String> = args(input["texts"].clone())?;
            if texts.len() > 5000 || texts.iter().map(String::len).sum::<usize>() > 1024 * 1024 {
                return Err("text index budget exceeded".into());
            }
            let options: hifishifter_kernel::search::SearchOptions = if input["options"].is_null() {
                Default::default()
            } else {
                args(input["options"].clone())?
            };
            return value(hifishifter_kernel::search::transliterate_batch(
                &texts, &options,
            ));
        }
        "read_system_clipboard_object" => {
            // 【为什么读不到不是错误】这是"看看剪贴板里有没有我们的对象"的探测。
            // 剪贴板被别的进程短暂占用是常态，把它报成 invoke 失败只会让前端把
            // "暂时读不到"显示成故障。读不到就是"没有"。
            return Ok(match hifishifter_clipboard::peek_bytes() {
                Some(bytes) => match String::from_utf8(bytes) {
                    Ok(payload) => json!({"ok":true,"available":true,"payload":payload}),
                    Err(_) => json!({"ok":true,"available":false}),
                },
                None => json!({"ok":true,"available":false}),
            });
        }
        // 纯文本剪贴板：与 UI 线程快速路径共用同一个结构化实现。
        "write_clipboard_text" => return Ok(startup_clipboard_text(&input)),
        "write_system_clipboard_object" => {
            let payload = input["payload"]
                .as_str()
                .ok_or("clipboard payload required")?;
            if payload.len() > 8 * 1024 * 1024 {
                return Err("parameter clipboard budget exceeded".into());
            }
            let decoded: Value = args(serde_json::from_str(payload).map_err(|e| e.to_string())?)?;
            if decoded["kind"] != "param" {
                return Err("timeline clipboard geometry is controlled by host".into());
            }
            hifishifter_clipboard::write_bytes(
                payload.as_bytes(),
                input["textSummary"]
                    .as_str()
                    .unwrap_or("HiFiShifter parameter data copied."),
            )?;
            return Ok(json!({"ok":true}));
        }
        "clipboard_kind" => {
            // 同 `read_system_clipboard_object`：探测读不到即"没有"，不是失败。
            let kind = hifishifter_clipboard::peek_bytes()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|payload| crate::host::clipboard::clipboard_kind(&payload));
            return Ok(json!({"ok":true,"kind":kind}));
        }
        // ── MIDI 导入的前两步 ──
        //
        // 解析文件与解析剪贴板都**不碰时间线**，因此放在 `ensure_loaded` 之前：宿主
        // 尚未就绪时弹窗仍要能列出 MIDI 轨道（否则用户点了"导入 MIDI"只得到一句
        // 与 MIDI 无关的宿主错误）。真正的写入在下面 `import_midi_to_pitch`。
        "get_midi_tracks" => {
            let path = input["midiPath"].as_str().unwrap_or_default().to_owned();
            let guid = input["clipboardGuid"].as_str().map(str::to_owned);
            if guid.as_deref().unwrap_or_default().is_empty() {
                midi_input_path(&path)?;
            }
            return Ok(hifishifter_kernel::editor::midi_import::get_midi_tracks(
                session, path, guid,
            ));
        }
        "read_midi_clipboard_to_memory" => {
            return Ok(
                hifishifter_kernel::editor::midi_import::read_midi_clipboard_to_memory(
                    session,
                    hifishifter_clipboard::read_standard_midi_file(),
                ),
            );
        }
        "emit_ui_event" => {
            let event = input["event"].as_str().ok_or("event missing")?;
            session.emit(event, input["payload"].clone());
            return Ok(Value::Null);
        }
        _ => {}
    }
    if let Err(error) = session.ensure_loaded(false) {
        if matches!(command, "get_timeline_state" | "get_timeline_state_lite") {
            if session.is_loaded() || session.source_files_pending() {
                return payload(session, command == "get_timeline_state_lite");
            }
            if let Some(document) = session.document() {
                if !document.ui_tracks.lock_or_recover().is_empty() {
                    {
                        let mut timeline = session.timeline.lock_or_recover();
                        document.present_host_inventory(&mut timeline, &session.namespace);
                        // 与 `ensure_loaded` 同一条不变量：清单之后补一次事实投影，
                        // 否则倒放窗口在这里会被重新打回镜像。
                        document.project_host_take_facts_locked(&mut timeline);
                    }
                    return payload(session, command == "get_timeline_state_lite");
                }
            }
        }
        let inventory_command = matches!(
            command,
            "get_param_frames"
                | "set_param_frames"
                | "restore_param_frames"
                | "set_static_param"
                | "get_static_param"
                | "set_track_state"
                | "select_clip"
                | "select_track"
                | "undo_parameter_edit"
                | "redo_parameter_edit"
                | "undo_timeline"
                | "redo_timeline"
                | "set_history_position"
        );
        if !session.source_files_pending() || !inventory_command {
            return Err(error);
        }
    }
    match command {
        "get_timeline_state" => payload(session, false),
        "move_track" => {
            let request: PrivateTrackMove = args(input)?;
            session.move_private_track(
                &request.track_id,
                request.target_index,
                request.parent_track_id,
            )?;
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 工程级拉伸覆盖（"拉伸算法 / 声码器 Mel 拉伸"子菜单）。
        //
        // 【为什么插件要支持它，而不是像缓存清理那样禁用】它控制的是**插件自己的**
        // 音频处理（拉伸算法、HiFiGAN 的 mel 拉伸），不是宿主的属性 —— 在插件里
        // 它同样有意义。此前这条命令落到 `Command unavailable`，于是整个子菜单点了
        // 没有任何反应。
        //
        // 【为什么不能只写 ProjectState】真正的效果来自进程级的拉伸配置
        // （`update_project_stretch_overrides`）；只改 `ProjectState` 会让界面上的
        // 勾选要等下一次渲染才可能被读到。
        "set_project_stretch_settings" => {
            let algorithm: Option<hifishifter_kernel::time_stretch::UserStretchAlgorithm> =
                serde_json::from_value(input["stretchAlgorithmOverride"].clone())
                    .map_err(|_| "invalid stretch algorithm override".to_string())?;
            let hifigan: Option<bool> =
                serde_json::from_value(input["hifiganMelStretchOverride"].clone())
                    .map_err(|_| "invalid hifigan mel stretch override".to_string())?;
            {
                let mut project = session.project.lock_or_recover();
                project.stretch_algorithm_override = algorithm;
                project.hifigan_mel_stretch_override = hifigan;
            }
            hifishifter_kernel::time_stretch::update_project_stretch_overrides(algorithm, hifigan);
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 工程基准音阶。
        //
        // 【为什么插件必须支持它】REAPER **没有工程调号概念**（音阶只能是
        // HiFiShifter 自有的设置），而 HiFiShifter 的音高吸附、级数渲染与渲染缓存键
        // 全都锚定它。此前这条命令落到 `Command unavailable`，于是插件里音阶选择器
        // 是灰的 —— 用户拿不到一整套依赖音阶的功能。
        //
        // 【为什么要落盘】`ProjectState` 是 per-ARA-document 且从不写的；只改它
        // 会让用户选的音阶在换工程/重启后归零。真正的家是 `UiSettings`。
        "set_project_base_scale" => {
            let requested = input["baseScale"].as_str().unwrap_or_default();
            let scale = hifishifter_kernel::state::model::normalize_scale_key(requested);
            {
                let mut project = session.project.lock_or_recover();
                project.base_scale = scale.clone();
                project.use_custom_scale = false;
            }
            crate::settings_store::save_settings_patch(&json!({
                "pluginMusicalContext": {
                    "baseScale": scale,
                    "useCustomScale": false,
                }
            }))?;
            after_musical_context_write(session);
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        "set_project_custom_scale" => {
            let custom: hifishifter_kernel::project::CustomScale =
                serde_json::from_value(input["customScale"].clone())
                    .map_err(|_| "invalid custom scale".to_string())?;
            let normalized = custom.normalized();
            {
                let mut project = session.project.lock_or_recover();
                project.custom_scale = Some(normalized.clone());
                project.use_custom_scale = true;
            }
            crate::settings_store::save_settings_patch(&json!({
                "pluginMusicalContext": {
                    "useCustomScale": true,
                    "customScale": normalized,
                }
            }))?;
            after_musical_context_write(session);
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // Tempo Map：插件里**只接受音阶轴**。
        //
        // 【为什么不能整组拒绝】Tempo Map 就是"随时间变化的音阶"的存储
        // （`TempoPointData.scale`），而 ActionBar 的基准音阶选择器在有 Tempo Map 时
        // 写的就是变化点。整组拒绝 = 插件里没有时变音阶 —— 音阶功能被砍掉一半。
        //
        // 【为什么 BPM/拍号要拒】它们是**宿主权威**（`render::transport` 从 VST3
        // 进程上下文读，插件不写）。接受写入就会造出第二个 BPM 真相。
        // 拒绝时给**明确原因**，不是 `Rejected` / `Command unavailable`。
        //
        // 【为什么只落盘、不写 session.timeline】`session.timeline` 是派生物，会被
        // `ensure_loaded` 与 ARA 重新认领整体替换。音阶的真相在设置里，由
        // `workspace_timeline_locked` 每轮重新播种（见
        // `DocumentSession::project_plugin_musical_context_locked`）。
        "set_timeline_tempo_map" => {
            // 前端形态是 `{ tempoMap: [...] | null }`；也接受裸数组/null，
            // 免得换一种调用形态就变成"命令不可用"。
            let raw = match input {
                Value::Object(_) => input.get("tempoMap").cloned().unwrap_or(Value::Null),
                other => other,
            };
            let points: Vec<hifishifter_kernel::models::TempoPointPayload> = match raw {
                Value::Null => Vec::new(),
                value => serde_json::from_value(value)
                    .map_err(|_| "invalid tempo map payload".to_string())?,
            };
            let (host_bpm, host_numerator, host_denominator) = {
                let timeline = session.timeline.lock_or_recover();
                let project = session.project.lock_or_recover();
                (
                    timeline.bpm,
                    project.beats_per_bar,
                    project.time_signature_denominator,
                )
            };
            // 宿主权威字段**以宿主为准**（载荷里带的 BPM/拍号一律不采信）。
            //
            // 【为什么是"覆盖"而不是"拒绝"】界面在插件里把 BPM/拍号显示成只读，
            // 所以载荷里这两个值本来就只是"上一次后端发的读数"。宿主在用户编辑期间
            // 改了速度时，两者会短暂不等 —— 此时拒绝会让**用户真正想改的音阶**一起
            // 丢失（前端会回滚整份 Tempo Map）。覆盖则只丢一个本就不该由插件持有的值。
            // 差异如实记一行日志，不做静默吞掉。
            let coerced = points.iter().any(|point| {
                (point.bpm - host_bpm).abs() > 1e-6
                    || point
                        .numerator
                        .is_some_and(|numerator| numerator != host_numerator)
                    || point
                        .denominator
                        .is_some_and(|denominator| denominator != host_denominator)
            });
            if coerced {
                crate::log_line(&format!(
                    "[ara] tempo map: host-authoritative BPM/time signature override \
                     (payload carried a value differing from host {host_bpm}/{host_numerator}\
                     /{host_denominator}); the plugin stores the scale axis only"
                ));
            }
            let scale_points: Vec<hifishifter_kernel::config::PluginScalePoint> = points
                .iter()
                .filter_map(|point| {
                    let scale = point.scale.as_ref()?;
                    Some(hifishifter_kernel::config::PluginScalePoint {
                        position_sec: point.position_sec,
                        key: scale.key.clone(),
                        name: scale.name.clone(),
                        notes: scale.notes.clone(),
                    })
                })
                .collect();
            // 幂等：内容没变就不写、不触发重载（一次编辑不该产生两次撤销步/两次重渲染）。
            let current = crate::settings_store::settings()
                .plugin_musical_context
                .scale_points;
            if current != scale_points {
                crate::settings_store::save_settings_patch(&json!({
                    "pluginMusicalContext": { "scalePoints": scale_points }
                }))?;
                after_musical_context_write(session);
                after_write(session, json!({"ok":true}))?;
            }
            payload(session, true)
        }
        "get_timeline_state_lite" => payload(session, true),
        "get_project_meta" => Ok(payload(session, true)?["project"].clone()),
        // vslib 是闭源库，插件包**刻意不随附**（见 `hifishifter-plugin/Cargo.toml` 的
        // 说明）。此前这条落到 `Command unavailable`，于是每次启动都有一次失败
        // invoke，前端只能把 vslib 显示成"未知"。如实回答"没编进来 + 为什么"，
        // 算法列表才能按能力过滤掉它。
        "get_vslib_status" => Ok(json!({
            "compiled": false,
            "available": false,
            "version": serde_json::Value::Null,
            "error": "vslib is not bundled with the plug-in",
        })),
        // 推理设备（ORT Execution Provider）只读状态。插件与独立 App **共用同一份设置、
        // 同一个内核**：`ui_settings_apply::apply` 已把 `ort_ep` / `ort_device_key` 下发给
        // 会话，所以这几条读数在插件里同样成立。
        //
        // 【为什么必须补】此前它们落到 `Command unavailable`，于是 `MenuBar` 把整个
        // 推理设备子菜单禁用（`disabled={isPluginMode()}`）—— 用户能在独立 App 里选
        // GPU，在插件里却只能看着 CPU。三个读数都**不初始化 ORT**（`is_available` 是
        // 只读快照，GPU/DXGI 枚举是纯查询），不会像 `run_vocoder_benchmark` 那样在
        // DAW 进程里触发重量级会话重建。
        "get_onnx_status" => {
            let compiled = hifishifter_kernel::nsf_hifigan_onnx::compiled();
            Ok(json!({
                "compiled": compiled,
                "available": compiled && hifishifter_kernel::nsf_hifigan_onnx::is_available(),
                "error": hifishifter_kernel::nsf_hifigan_onnx::model_load_error(),
                "epChoice": hifishifter_kernel::nsf_hifigan_onnx::ep_choice(),
            }))
        }
        "get_gpu_devices" => value(hifishifter_kernel::gpu_info::enumerate_gpus()),
        "get_dml_adapters" => value(hifishifter_kernel::dml_adapters::enumerate_dml_adapters()),
        // 记事本正文：独立 App 把它写进工程文件；插件没有工程文件，写进插件自己的
        // 数据目录（见 `editor/notebook.rs`）。此前这条落到 `Command unavailable`，
        // 而前端把失败吞掉 —— 症状是"在插件里写的笔记重载即消失"。
        //
        // 【为什么不登记为撤销步】插件的撤销栈只记时间轴与参数；把记事本塞进去会
        // 让"撤销"在两套权威之间跳。用户要撤销的是编辑动作，不是打字。
        "set_project_notes" => {
            let markdown = input["notesMarkdown"].as_str().unwrap_or_default();
            super::notebook::store().set_notes(markdown)?;
            session.project.lock_or_recover().notes_markdown = markdown.to_owned();
            payload(session, true)
        }
        "seal_project_notes_history" => Ok(super::notebook::seal_notes_history()),
        "notebook_put_asset" => {
            let result = super::notebook::store().put_asset(
                input["assetId"].as_str().unwrap_or_default(),
                input["kind"].as_str().unwrap_or("image"),
                input["ext"].as_str().unwrap_or("bin"),
                input["mime"].as_str(),
                input["dataBase64"].as_str().unwrap_or_default(),
                input.get("meta").cloned().filter(|v| !v.is_null()),
            )?;
            Ok(result)
        }
        "notebook_read_asset" => {
            Ok(super::notebook::store().read_asset(input["assetId"].as_str().unwrap_or_default()))
        }
        "notebook_list_assets" => Ok(super::notebook::store().list_assets()),
        "notebook_remove_asset" => Ok(
            super::notebook::store().remove_asset(input["assetId"].as_str().unwrap_or_default())
        ),
        "notebook_prune_assets" => Ok(super::notebook::store().prune_assets()),
        // 纯文件读取：不碰工程状态，也不需要宿主。
        "notebook_read_file_base64" => Ok(super::notebook::read_file_base64(
            input["path"].as_str().unwrap_or_default(),
            input["maxBytes"].as_u64(),
        )),
        "notebook_read_clipboard_image" => Ok(super::notebook::read_clipboard_image()),
        // 记事本「暂存剪贴板载荷」：把插件自己写进系统剪贴板的那份 JSON 原样交出去。
        //
        // 【为什么不是独立 App 的 `ProjectFragment`】插件写进剪贴板的从来不是 App 的
        // 工程片段：那是"宿主 item state + 参数种子"，粘贴时要按宿主 item 重建
        // （见 `host_clipboard`）。把 App 的 `.hsf` 片段格式搬过来，只会产出一份谁也
        // 粘不回去的字节。因此这里读的就是 `host_clipboard` 写出去的那一份，恢复时
        // 原样写回，粘贴链路一行都不用改。
        //
        // 【为什么 `encoding` 仍是 `fragment`】记事本的围栏格式只有两个编码取值
        // （`fragment` / `param`，见前端 `hifiClipBlock.ts` 的解析），而这里的字节
        // 都是**自包含的片段载荷**，与"fragment"的含义一致；为插件新增第三个取值是
        // 一次跨产品的格式变更，不值得为一句说明付出。
        "notebook_read_clipboard_payload" => {
            let Some(bytes) = hifishifter_clipboard::read_bytes()? else {
                return Ok(json!({"ok":true,"available":false}));
            };
            if bytes.is_empty() {
                return Ok(json!({"ok":true,"available":false}));
            }
            let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
                return Ok(json!({"ok":true,"available":false}));
            };
            let Some(kind) = crate::host::clipboard::clipboard_kind(&value) else {
                return Ok(json!({"ok":true,"available":false}));
            };
            let (summary, ext, encoding) = if kind == "param" {
                // 参数线载荷：v2 是 `segments`，v1 是单段 `values`，两种都要能画。
                let segments: Vec<&Value> = match value["segments"].as_array() {
                    Some(list) => list.iter().collect(),
                    None if value.get("values").is_some() => vec![&value],
                    None => Vec::new(),
                };
                let mut total = 0usize;
                let mut collected: Vec<f64> = Vec::new();
                for segment in segments {
                    if let Some(values) = segment["values"].as_array() {
                        total += values.len();
                        collected.extend(values.iter().map(|v| v.as_f64().unwrap_or(0.)));
                    }
                }
                // 与 App 同口径降采样到 ≤200 点：暂存块的迷你曲线不需要原始精度，
                // 而上万帧整条塞进 IPC 只是白搬几百 KB。
                let step = (collected.len() / 200).max(1);
                let sparkline: Vec<f64> = collected.iter().step_by(step).copied().collect();
                (
                    json!({"clipKind":"param","clipCount":0,"trackCount":0,
                        "sourceProject":Value::Null,"durationSec":0.0,"preview":[],
                        "param":{"param":value["param"].as_str().unwrap_or(""),
                            "framePeriodMs":value["framePeriodMs"].as_f64().unwrap_or(0.),
                            "frameCount":total,"sparkline":sparkline}}),
                    "hsp",
                    "param",
                )
            } else {
                let empty = Vec::new();
                let items = value["items"].as_array().unwrap_or(&empty);
                let tracks = value["tracks"].as_array().unwrap_or(&empty);
                let origin = value["origin_sec"].as_f64().unwrap_or(0.);
                let mut end = origin;
                let mut rows = Vec::new();
                for item in items {
                    let before = &item["before"];
                    let start = before["start_sec"].as_f64().unwrap_or(origin);
                    let length = before["length_sec"].as_f64().unwrap_or(0.);
                    end = end.max(start + length);
                    let slot = item["track"].as_u64().unwrap_or(0) as usize;
                    rows.push(json!({
                        "trackId":slot.to_string(),
                        "trackName":tracks.get(slot).and_then(|t|t["name"].as_str()).unwrap_or(""),
                        "name":before["name"].as_str().unwrap_or(""),
                        "startSec":start,"lengthSec":length,
                    }));
                }
                (
                    json!({"clipKind":"clips","clipCount":items.len(),"trackCount":tracks.len(),
                        "sourceProject":Value::Null,"durationSec":(end-origin).max(0.0),
                        "preview":rows}),
                    "hsf",
                    "fragment",
                )
            };
            Ok(
                json!({"ok":true,"available":true,"kind":kind,"encoding":encoding,
                "ext":ext,"mime":"application/json","byteLen":bytes.len(),
                "base64":base64::engine::general_purpose::STANDARD.encode(&bytes),
                "summary":summary}),
            )
        }
        // 把暂存的载荷原样写回系统剪贴板。
        //
        // 【为什么必须校验格式】写回去的字节随后会被粘贴链路解码；随便什么字节都
        // 接受，等于让记事本把一个"看起来像暂存块"的东西变成一条必然失败的粘贴。
        "notebook_write_clipboard_payload" => {
            let encoded = input["payloadBase64"]
                .as_str()
                .ok_or("clipboard payload required")?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|_| "clipboard payload is not valid base64".to_string())?;
            if bytes.len() > 8 * 1024 * 1024 {
                return Err("clipboard payload budget exceeded".into());
            }
            let value: Value = serde_json::from_slice(&bytes)
                .map_err(|_| "clipboard payload is not native HiFiShifter data".to_string())?;
            if crate::host::clipboard::clipboard_kind(&value).is_none() {
                return Err("clipboard payload is not native HiFiShifter data".into());
            }
            hifishifter_clipboard::write_bytes(
                &bytes,
                input["textSummary"]
                    .as_str()
                    .unwrap_or("HiFiShifter data restored."),
            )?;
            Ok(json!({"ok":true}))
        }
        "get_runtime_info" => {
            // 【为什么不再序列化整条时间线】此前这里调 `payload(session, true)` ——
            // 一次"运行时状态"查询要锁住 timeline/history/project 并序列化全部片段，
            // 于是状态栏的一次失败与内容加载强耦合（大工程上这个查询还会明显卡顿）。
            // 时间线本来就有自己的通路（前端启动时的 `fetch_timeline` 与
            // `plugin_refresh`），独立 App 的同一命令也**从不**带 timeline
            // （`RuntimeInfoPayload.timeline = None`）。这里与 App 对齐，只回报摘要。
            let (_, playing) = session.transport();
            let clip_count = session.timeline.lock_or_recover().clips.len();
            Ok(json!({
                "ok": true,
                "device": "Host / ARA",
                "model_loaded": hifishifter_kernel::world_vocoder::is_available(),
                "audio_loaded": clip_count > 0,
                "has_synthesized": session.applied.load(Ordering::Acquire) > 0,
                "is_playing": playing,
                "playback_target": if playing { Some("synthesized") } else { None },
                "gpu_backend": "",
                "clip_count": clip_count,
            }))
        }
        "get_history_state" => Ok(history_state(session)),
        "begin_undo_group" => {
            // 组内写入用 `checkpoint:false`，因此"组开始前"的权威必须在这里先挂上；
            // 否则 undo 找不到可恢复的快照，整组曲线会留在原位（右键重置后"没反应"）。
            session.checkpoint_authority_step();
            let timeline = session.timeline.lock_or_recover();
            history::checkpoint(
                &mut session.history.lock_or_recover(),
                &timeline,
                input["label"].as_str().unwrap_or("batch").into(),
                || None,
            );
            session.suppress_history.store(true, Ordering::Release);
            drop(timeline);
            session.emit("history_state", history_state(session));
            payload(session, false)
        }
        "end_undo_group" => {
            session.suppress_history.store(false, Ordering::Release);
            Ok(json!({"ok":true}))
        }
        "set_transport" => {
            if input["bpm"].is_number() {
                return Err("tempo is controlled by the host".into());
            }
            if let Some(position) = input["playheadSec"].as_f64() {
                if position.is_finite() {
                    session.timeline.lock_or_recover().playhead_sec = position.max(0.);
                }
            }
            payload(session, true)
        }
        // 【为什么必须支持它】ActionBar 的网格下拉与"吸附/网格设置"对话框都走这条
        // 命令；插件此前对它回 `Command unavailable`，于是那些控件**看起来能用、
        // 点了什么也不会发生**。
        //
        // 【为什么拍号在这里被忽略】拍号是宿主权威（`render::transport` 从 VST3
        // 进程上下文读，见 `time_signature`），插件不写它。但前端在改网格时会把
        // **当前**拍号一并放进同一个补丁，所以这里不能因为看到拍号就报错 ——
        // 只接受网格那部分。真正要改拍号时 UI 是灰的（`ActionBar` 里按插件模式禁用）。
        "set_project_timeline_settings" => {
            let grid = input["gridSize"]
                .as_str()
                .map(hifishifter_kernel::config::TimelineSnapSettings::normalize_grid_size);
            if let Some(grid) = grid {
                session.project.lock_or_recover().grid_size = grid.clone();
                // 网格是 HiFiShifter 自有的设置（宿主没有对应概念），因此要**落盘**：
                // 只写进 per-document 的 `ProjectState` 会在换工程/重启后归零。
                crate::settings_store::save_settings_patch(&json!({ "gridSize": grid }))?;
            }
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 长分析的轮询兜底。事件通道（`pitch_orig_analysis_progress`）在正常情况下
        // 就够了，但事件可能在窗口尚未挂载时发出 —— 那时前端只能靠轮询这条把
        // "正在分析第 3/8 个片段"补回来。此前这里恒回 `null`，于是**首次**分析
        // 期间界面完全没有进度，看起来像卡死。
        "get_pitch_analysis_progress" => Ok(
            match hifishifter_kernel::pitch_clip::get_clip_pitch_batch_progress() {
                Some(batch) => json!({
                    "rootTrackId": "",
                    "progress": batch.progress,
                    "currentClipName": batch.current_clip_name,
                    "completedClips": batch.completed_clips,
                    "totalClips": batch.total_clips,
                }),
                None => Value::Null,
            },
        ),
        "get_syllable_analysis_progress" => Ok(
            hifishifter_kernel::pitch_analysis::syllable_analysis_progress()
                .map(|progress| serde_json::to_value(progress).unwrap_or(Value::Null))
                .unwrap_or(Value::Null),
        ),
        // 把编辑后的音高曲线导出成 MIDI。
        //
        // 【为什么插件里能做】整条链路是纯变换（读音高数据 → 写文件），不需要任何
        // 宿主写接口 —— 与"在 REAPER 里建 MIDI item"是两件事。目标路径来自插件自己
        // 的"另存为"对话框（见 `webview.rs` 的 `pick_midi_output_path`）。
        //
        // 【为什么先克隆时间线】导出可能触发 FCPE 音高分析（长素材可达数分钟）。
        // 持有时间线锁跑完会冻结全部命令与 UI 轮询，因此快照必须在**锁外**用。
        "export_pitch_to_midi" => {
            let request: hifishifter_kernel::midi_export::MidiExportRequest = args(input)?;
            if request.tracks.len() > 4096 {
                return Err("MIDI export track budget exceeded".into());
            }
            let path = std::path::Path::new(&request.output_path);
            if !path.is_absolute()
                || !path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("mid") || ext.eq_ignore_ascii_case("midi")
                    })
            {
                return Err("an absolute .mid output path is required".into());
            }
            if !path.parent().is_some_and(std::path::Path::is_dir) {
                return Err("the output folder does not exist".into());
            }
            let timeline = session.timeline.lock_or_recover().clone();
            Ok(hifishifter_kernel::midi_export::export_pitch_to_midi(
                &timeline, request,
            ))
        }
        // ── MIDI 导入 ──
        //
        // 【为什么插件里能做】`import_midi_to_pitch` 写的是 `params_by_root_track` 里的
        // 音高曲线 —— 那是插件自己的权威（与参数编辑器同一份数据），不碰宿主几何，也
        // 不要求任何宿主写接口。共享编排在内核里，与独立 App 逐字同一条路径。
        //
        // 【为什么只做"导入到曲线"】`import_midi_as_clip` / `replace_midi_clip_data`
        // 建的是**本地片段**，而插件的时间线是宿主清单的投影
        // （`workspace_timeline_locked` 只保留已分配 region 的 clip）：造出来的片段
        // 在下一次宿主同步时就会消失。那不是"没实现"，是在这里做不到 —— 前端因此
        // 只对"导入到音高曲线"开闸。
        "import_midi_to_pitch" => {
            let request: hifishifter_kernel::editor::midi_import::ImportMidiToPitchRequest =
                args(input)?;
            if request
                .clipboard_guid
                .as_deref()
                .unwrap_or_default()
                .is_empty()
            {
                midi_input_path(&request.midi_path)?;
            }
            let result =
                hifishifter_kernel::editor::midi_import::import_midi_to_pitch(session, request);
            if result["ok"] == false {
                // 失败原因原样交给前端：`MidiTrackSelectDialog` 按这些键给出具体文案
                // （`pitch_requires_compose` / `no_pitch_line_selected` / …）。包成
                // `Err` 会把它压成一句通用的"导入失败"，与独立 App 的表现分叉。
                return Ok(result);
            }
            after_write(session, result)
        }
        "get_track_summary" => {
            let timeline = session.timeline.lock_or_recover();
            Ok(
                json!({"ok":true,"track_id":input["trackId"].as_str().map(str::to_owned).or_else(||timeline.selected_track_id.clone()),
                "waveform_preview":[],"pitch_range":{"min":-24,"max":24}}),
            )
        }
        "get_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.frame_count as usize)?;
            value(params::get_param_frames(
                session,
                a.track_id,
                a.param,
                a.start_frame,
                a.frame_count,
                a.stride,
                a.binary,
                a.with_sentinel,
            ))
        }
        "set_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.values.len())?;
            after_write(
                session,
                params::set_param_frames_with_present(
                    session,
                    a.track_id,
                    a.param,
                    a.start_frame,
                    a.values,
                    a.present,
                    a.checkpoint,
                ),
            )
        }
        "restore_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.frame_count as usize)?;
            after_write(
                session,
                params::restore_param_frames(
                    session,
                    a.track_id,
                    a.param,
                    a.start_frame,
                    a.frame_count,
                    a.checkpoint,
                ),
            )
        }
        "get_static_param" => {
            let a: Static = args(input)?;
            track_exists(session, &a.track_id)?;
            value(params::get_static_param(session, a.track_id, a.param))
        }
        "set_static_param" => {
            let a: Static = args(input)?;
            track_exists(session, &a.track_id)?;
            let number = a
                .value
                .filter(|n| n.is_finite())
                .ok_or("finite static parameter required")?;
            after_write(
                session,
                params::set_static_param(session, a.track_id, a.param, number, a.checkpoint),
            )
        }
        "convert_mix_param" => {
            let a: Mix = args(input)?;
            track_exists(session, &a.track_id)?;
            for range in &a.ranges {
                budget(range.start_frame, range.frame_count as usize)?;
            }
            after_write(
                session,
                params::convert_mix_param(session, a.track_id, a.from, a.ranges),
            )
        }
        // 「拉伸时锁定参数线」的时域映射。
        //
        // 【为什么必须有】它是**无条件**被调用的：前端在 clip 拉伸手势收尾时总会
        // 发它（见 `params.ts` 的调用点）。此前这条落到 `Command unavailable`，
        // 于是拉伸后参数线与片段长度对不上 —— 而失败只留一行日志，用户看到的是
        // "拉伸坏了"。它只改 HFS 自己的参数曲线、不碰宿主几何，所以插件里完全可用。
        "stretch_track_linked_params" => {
            let a: StretchLinked = args(input)?;
            track_exists(session, &a.track_id)?;
            let _ = a.checkpoint;
            if a.mappings.len() > 4096 {
                return Err("stretch mapping budget exceeded".into());
            }
            // ARA 模式下几何变化由内核的几何事务（`GeometryOp::Move/Stretch`）完成；这里再改
            // 一次本地数组会把已经投影过的曲线二次映射（偏移/截断/默认值）。
            // 保留命令是为了独立 App 与旧前端兼容，但在插件内必须是显式 no-op。
            log::info!(
                "[ara] stretch_track_linked_params skipped in ARA mode; the atlas projects \
                 source edits from the host geometry"
            );
            Ok(serde_json::json!({"ok": true, "skipped": "ara-atlas"}))
        }
        "set_track_state" => {
            let object = input.as_object().ok_or("track arguments required")?;
            for (key, value) in object {
                if !matches!(
                    key.as_str(),
                    "trackId"
                        | "muted"
                        | "solo"
                        | "volume"
                        | "composeEnabled"
                        | "pitchAnalysisAlgo"
                ) && !value.is_null()
                {
                    return Err(format!("track property controlled by host: {key}"));
                }
            }
            let patch: TrackPatch = args(input.clone())?;
            // 工程反序列化为前向兼容保留Unknown；实时GUI命令不能把它当可执行算法。
            if patch.pitch_analysis_algo.as_ref().is_some_and(|algo| {
                matches!(
                    algo,
                    PitchAnalysisAlgo::Unknown | PitchAnalysisAlgo::VocalShifterVslib
                )
            }) {
                return Err("algorithm unavailable in plugin mode".into());
            }
            let id = patch.track_id.as_str();
            track_exists(session, id)?;
            if patch.volume.is_some_and(|n| {
                !n.is_finite() || !(0.0..=hifishifter_kernel::gain::GAIN_MAX).contains(&n)
            }) {
                return Err("track volume out of range".into());
            }
            let mut timeline = session.timeline.lock_or_recover();
            let mut candidate = timeline.clone();
            let track = candidate.tracks.iter_mut().find(|t| t.id == id).unwrap();
            if let Some(volume) = patch.volume {
                track.volume = volume;
            }
            if let Some(value) = patch.muted {
                track.muted = value;
            }
            if let Some(value) = patch.solo {
                track.solo = value;
            }
            if let Some(value) = patch.compose_enabled {
                track.compose_enabled = value;
            }
            if let Some(value) = patch.pitch_analysis_algo {
                track.pitch_analysis_algo = value;
            }
            session.checkpoint_timeline(&timeline, HistoryOp::EditTrack);
            *timeline = candidate;
            session.mark_dirty();
            session.publish_timeline(timeline.clone());
            drop(timeline);
            after_write(session, json!({"ok":true}))?;
            payload(session, false)
        }
        "undo_timeline"
        | "redo_timeline"
        | "undo_parameter_edit"
        | "redo_parameter_edit"
        | "set_history_position" => {
            let mut timeline = session.timeline.lock_or_recover();
            let mut recorded = session.history.lock_or_recover();
            let (target, intent) = match command {
                "undo_timeline" | "undo_parameter_edit" => {
                    (recorded.position.saturating_sub(1), HistoryJumpIntent::Undo)
                }
                "redo_timeline" | "redo_parameter_edit" => {
                    (recorded.position.saturating_add(1), HistoryJumpIntent::Redo)
                }
                _ => (
                    input["position"]
                        .as_u64()
                        .ok_or("history position missing")? as usize,
                    HistoryJumpIntent::Jump,
                ),
            };
            if let Some((next, _, selection)) =
                history::jump(&mut recorded, &timeline, target, intent, None)
            {
                let layout_only = hifishifter_kernel::param_blocks::timeline_equal_except_blocks(
                    &timeline, &next,
                );
                if matches!(command, "undo_parameter_edit" | "redo_parameter_edit") {
                    // 参数快照只恢复控制数据，绝不把旧clip位置/源窗口写回或覆盖当前宿主几何。
                    timeline.params_by_root_track = next.params_by_root_track;
                    for track in &mut timeline.tracks {
                        if let Some(old) = next.tracks.iter().find(|old| old.id == track.id) {
                            track.volume = old.volume;
                            track.muted = old.muted;
                            track.solo = old.solo;
                            track.compose_enabled = old.compose_enabled;
                            track.pitch_analysis_algo = old.pitch_analysis_algo.clone();
                        }
                    }
                } else {
                    *timeline = next;
                }
                // 参数权威是独立历史层：显示数组不再作为快照回放。
                drop(recorded);
                let timeline_for_publish = timeline.clone();
                drop(timeline);
                let restored_curve = session.restore_authority_step(target);
                // 撤销 / 重做把整份参数权威换掉了：自动写入要按「我们写过的片段」重扫一遍
                // （窗口未知 —— 见 `note_pitch_edit_of_every_written_clip`）。
                if restored_curve == Some(true) {
                    session.note_pitch_edit_of_every_written_clip();
                }
                if !layout_only || restored_curve == Some(true) {
                    session.mark_dirty();
                    session.publish_timeline(timeline_for_publish);
                } else {
                    session.mark_metadata_dirty();
                }
                session.emit("history_state", history_state(session));
                session.notify_timeline();
                let mut payload = payload(session, false)?;
                if let Some(selection) = selection {
                    payload["param_selection_restore"] = json!(selection);
                }
                if let Some(error) = session.error.lock_or_recover().clone() {
                    return Err(error);
                }
                Ok(payload)
            } else {
                drop(recorded);
                drop(timeline);
                payload(session, false)
            }
        }
        "get_waveform_mipmap_binary" => {
            let path = input["sourcePath"].as_str().ok_or("sourcePath missing")?;
            let level = input["level"].as_u64().unwrap_or(2).min(2) as usize;
            Ok(Value::String(
                base64::engine::general_purpose::STANDARD
                    .encode(peaks(session, path)?.to_binary_level(level)),
            ))
        }
        "preload_waveform_mipmap" => {
            peaks(
                session,
                input["sourcePath"].as_str().ok_or("sourcePath missing")?,
            )?;
            Ok(json!({"ok":true}))
        }
        "batch_get_waveform_mipmap" => {
            let paths: Vec<String> = args(input["sourcePaths"].clone())?;
            let levels: Option<Vec<usize>> = args(input["levels"].clone())?;
            if paths.len() > 64 {
                return Err("too many waveform sources".into());
            }
            let mut result = serde_json::Map::new();
            for path in paths {
                let data = peaks(session, &path)?;
                let encoded: Vec<_> = (0..3)
                    .map(|level| {
                        if levels
                            .as_ref()
                            .is_none_or(|l| l.is_empty() || l.contains(&level))
                        {
                            base64::engine::general_purpose::STANDARD
                                .encode(data.to_binary_level(level))
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                result.insert(path, json!(encoded));
            }
            Ok(Value::Object(result))
        }
        "get_root_mix_waveform_peaks_segment" | "get_track_mix_waveform_peaks_segment" => {
            let a: Segment = args(input)?;
            track_exists(session, &a.track_id)?;
            if !a.start_sec.is_finite() || !a.duration_sec.is_finite() || a.duration_sec <= 0. {
                return Err("invalid waveform range".into());
            }
            if a.duration_sec * 48000.0 * 2.0 * 4.0 > 64.0 * 1024.0 * 1024.0 {
                return Err("waveform mix buffer budget exceeded".into());
            }
            if command == "get_root_mix_waveform_peaks_segment" {
                value(waveform::get_root_mix_waveform_peaks_segment(
                    session,
                    a.track_id,
                    a.start_sec,
                    a.duration_sec,
                    a.columns,
                ))
            } else {
                value(waveform::get_track_mix_waveform_peaks_segment(
                    session,
                    a.track_id,
                    a.start_sec,
                    a.duration_sec,
                    a.columns,
                ))
            }
        }
        // ── 导出音频到文件 ──
        //
        // 【为什么插件里能做】ARA 对音频读取**没有角色限制**
        // （`ARAInterface.h:1784-1789`），授权 PCM 就在渲染输入里
        // （`RenderInput::sources`），编码器在内核里。这条路径不碰宿主几何、
        // 也不要求任何宿主写接口，因此 ARA-only 宿主上同样成立。
        //
        // 【为什么这是"功能不全"里最大的一项】独立 App 一直能导出处理结果，
        // ARA 版此前**完全没有**这条路径 —— 用户在 Cubase 里做完处理却拿不出文件。
        "export_audio_advanced" => export_audio_to_file(session, &args(input)?),
        // 冲突预检：只解析目标路径，不渲染。前端据此弹出"是否覆盖"。
        "preview_export_audio_plan" => preview_export_audio_plan(&args(input)?),
        "cancel_export_audio" => {
            let active = EXPORT_CANCEL
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
                .map(|flag| {
                    flag.store(true, std::sync::atomic::Ordering::Release);
                })
                .is_some();
            Ok(serde_json::json!({"ok": true, "active": active}))
        }
        // 导出对话框的默认值。插件没有"工程文件"，因此输出目录取
        // 用户已授权目录里的第一个（没有则取用户主目录）。
        "get_export_audio_defaults" => {
            let dir = session
                .browser_roots
                .lock_or_recover()
                .first()
                .map(|path| path.to_string_lossy().to_string())
                .or_else(|| {
                    std::env::var_os("USERPROFILE")
                        .or_else(|| std::env::var_os("HOME"))
                        .map(|home| std::path::PathBuf::from(home).to_string_lossy().to_string())
                })
                .unwrap_or_default();
            Ok(serde_json::json!({
                "ok": true,
                "projectName": "export",
                "documentsDir": dir,
                "projectOutputDir": dir,
                "projectFileName": "export",
                "separatedOutputDir": dir,
                "separatedFileName": "export",
                "sampleRate": 44100,
                "bitDepth": 32,
                "format": "wav",
                "encoder": hifishifter_kernel::encode::OutputSpec::wav_32f(),
            }))
        }
        // ── B 组：不需要任何宿主 API 的命令 ──
        //
        // 判据（见方案 §10）：**只读 PCM + 只写插件自己的状态或用户选定的文件**。
        // 这些命令在 ARA-only 宿主上与独立 App 里完全同义 —— 此前它们落到
        // `Command unavailable`，于是前端对应的整块控件不可用。
        "get_clip_linked_params" => {
            // `extract_*` 取 `&mut self`（为借用便利），但它不改时间线：克隆一份调用。
            let mut timeline = session.timeline.lock_or_recover().clone();
            match timeline.extract_clip_linked_params(input["clipId"].as_str().unwrap_or_default())
            {
                Some(linked) => Ok(json!({"ok": true, "linkedParams": linked})),
                None => Ok(json!({"ok": false, "error": "clip_not_found"})),
            }
        }
        "apply_clip_linked_params" => {
            let clip_id = input["clipId"].as_str().unwrap_or_default().to_owned();
            let linked: hifishifter_kernel::state::LinkedParamCurvesPayload =
                args(input.get("linkedParams").cloned().unwrap_or(Value::Null))?;
            let mut timeline = session.timeline.lock_or_recover();
            session.checkpoint_timeline(&timeline, HistoryOp::ApplyLinkedParams);
            let mut candidate = timeline.clone();
            candidate.apply_linked_params_to_clip(&clip_id, &linked);
            *timeline = candidate;
            session.mark_dirty();
            session.publish_timeline(timeline.clone());
            drop(timeline);
            after_write(session, json!({"ok": true}))?;
            payload(session, false)
        }
        // 音高参考：两者都**只改参数**，不碰宿主几何 —— 因此插件里同样成立。
        "convert_clips_to_pitch_reference" | "update_pitch_reference" => {
            let clip_ids: Vec<String> = serde_json::from_value(input["clipIds"].clone())
                .map_err(|_| "invalid clip id list".to_string())?;
            let mut timeline = session.timeline.lock_or_recover();
            session.checkpoint_timeline(&timeline, HistoryOp::PitchReference);
            let mut candidate = timeline.clone();
            if command == "convert_clips_to_pitch_reference" {
                candidate.convert_clips_to_pitch_reference(&clip_ids);
            } else {
                candidate.update_pitch_reference_from_track_params(&clip_ids);
            }
            *timeline = candidate;
            session.mark_dirty();
            session.publish_timeline(timeline.clone());
            drop(timeline);
            after_write(session, json!({"ok": true}))?;
            payload(session, false)
        }
        // 缓存清理：插件**没有**跨会话的渲染 / 波形缓存（它们随进程结束而消失），
        // 因此这里是**显式成功**的 no-op，而不是"命令不存在" —— 前端据此保留同一个
        // 菜单项与同一条成功路径，用户不会看到一条指向不存在的功能的失败。
        "clear_render_cache" | "clear_waveform_cache" => {
            Ok(json!({"ok": true, "cleared": 0, "detail": "the plugin keeps no persistent cache"}))
        }
        // 快速导出选中片段：与 `export_audio_advanced` 共用渲染与编码路径，
        // 只是把范围取成选中片段的并集、目标路径由 `outputDir` + `fileName` 组成。
        "quick_export_selected_clips" => quick_export_selected_clips(session, &args(input)?),
        // 参数编辑器的 "Clip 音高参考线"：独立 App 在卫星窗口懒取时用同一条命令。
        "get_clip_pitch_data" => clip_pitch_data(session, &input),
        // REAPER take 音高 ⇄ 参数线（一次性迁移，两侧都清零对侧）。
        "import_host_take_pitch" => import_host_take_pitch(session, &input),
        "export_host_take_pitch" => export_host_take_pitch(session, &input),
        // 合成结果另存：插件的渲染输出**就是**合成结果，因此与工程导出同一条路径。
        "save_synthesized" => {
            let path = input["outputPath"].as_str().unwrap_or_default().to_owned();
            export_audio_to_file(
                session,
                &json!({"mode": "project", "projectOutputPath": path}),
            )
        }
        _ => Err(format!("Command unavailable in plug-in mode: {command}")),
    }
}

#[cfg(test)]
mod pitch_bridge_tests {
    use super::*;

    /// 音高互转必须能从**呈现 id** 找到宿主 take。
    ///
    /// 【为什么这是回归】编辑器时间线的 id 带 `hfs-ui-{pid}-{n}-` 前缀
    /// （`EditorSession::rewrite_ids`），而 `host_edit_target` 认的是**裸**宿主身份
    /// （`ara-item-{guid}`）。漏掉剥前缀时`pitch_bridge_target` 恒失败 —— 两个方向
    /// 都是"点了没反应 + clip has no unique directly bound host take"（现场日志里每
    /// 一次音高互转都是这一句），而菜单、能力位、命令分发看起来全都正常。
    #[test]
    fn the_pitch_bridge_accepts_a_presented_clip_id() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let native = clip
            .id
            .strip_prefix(&editor.namespace)
            .expect("夹具 clip 必须是呈现 id（带 namespace）")
            .to_owned();
        // 前提：裸 id 本来就解析得到 —— 否则这条用例证明不了"前缀"这一层。
        assert!(
            pitch_bridge_target(&editor, &native).is_ok(),
            "裸 id 必须能解析到宿主 take"
        );
        assert!(
            pitch_bridge_target(&editor, &clip.id).is_ok(),
            "呈现 id 必须先剥掉 namespace 再交给 host_edit_target"
        );
        document.close();
    }

    /// 【导出不得清空 HiFiShifter 的音高线】take 音高是源级属性，ARA 渲染结果在它之后
    /// —— 包络不会作用到插件渲染出来的音频上，清空只会让用户"转换完就听不见音高"。
    /// 同时钉住「需重置 Item 音高偏移」那一半：`D_PITCH` 必须归零。
    #[test]
    fn exporting_the_take_envelope_keeps_the_hifishifter_line() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 2.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let (clip_id, root) = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            // 窗口拉长到 8 帧（5 ms 一帧）：旧实现的 `restore_authority_frames` 会正好
            // 覆盖这 8 帧，所以这条用例对"顺手清空"是敏感的。
            let clip = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == clip_id)
                .unwrap();
            clip.start_sec = 0.0;
            clip.length_sec = 0.04;
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            (clip_id, root)
        };

        // 【必须走权威】`params_by_root_track` 只是派生缓存，直接改会被下一次合成抹掉
        // （见 `midi_import` 的同一条注释）；这里写入的编辑才是"用户的音高线"。
        editor
            .write_authority_frames(&root, "pitch", 0, &[2.0_f32; 8], None, true)
            .expect("插件宿主必须应答权威写口")
            .expect("权威写入必须成功");
        let read_line = || {
            editor
                .read_authority_frames(&root, "pitch", 0, 8, None, Some(true), None)
                .expect("插件宿主必须应答权威读口")
                .expect("权威读取必须成功")
        };
        let before = read_line();
        assert!(
            before.edit.iter().all(|value| (*value - 2.0).abs() < 1e-6),
            "夹具必须真的写入一段音高线：{:?}",
            before.edit
        );

        export_host_take_pitch(&editor, &json!({ "clipIds": [clip_id.clone()] })).unwrap();

        assert_eq!(
            host.item_value("D_PITCH"),
            Some(0.),
            "Item 音高偏移必须归零（包络成为 REAPER 侧唯一来源）"
        );
        let after = read_line();
        assert_eq!(
            after.edit, before.edit,
            "导出 take 音高包络**不得**改动 HiFiShifter 的音高线"
        );
        assert_eq!(after.edited, before.edited);
        document.close();
    }

    /// 【导入不得删掉宿主的数据】`PITCHENV` 与 `D_PITCH` 只在插件旁路时才作用于那段
    /// 音频，与插件渲染互不叠加 —— 导入是复制，不是把 REAPER 侧的东西搬走。
    #[test]
    fn importing_the_take_pitch_leaves_the_host_envelope_alone() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        let chunk = "<ITEM\nPOSITION 0\nLENGTH 0.00009070294784580499\nSEL 1\n\
                     IGUID {11111111-1111-1111-1111-111111111111}\n\
                     SOFFS 0\nPLAYRATE 1 1 0\nGUID {22222222-2222-2222-2222-222222222222}\n\
                     <SOURCE WAVE\nFILE \"ara://source\"\n>\n\
                     <PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\n\
                     DEFSHAPE 0 12 -1\nPT 0.00000000000000000 4.00000000000000000 0\n>\n>\n";
        host.enable_item_state(chunk);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 3.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let (clip_id, root) = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            (clip_id, root)
        };
        // 【预热：必须先跑掉"第一次解析宿主身份"】那一步会重建一次 `ui_tracks`，并把
        // 派生参数缓存按基线重装一遍 —— 注入的值会被它抹掉。不预热，这条用例就会随
        // 线程时序红绿不定（注入 → 重建 → 命令读到全 0）。
        pitch_bridge_target(&editor, &clip_id).expect("夹具的宿主身份必须可解析");
        {
            let mut timeline = editor.timeline.lock_or_recover();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.frame_period_ms = 5.0;
            params.pitch_orig = vec![60.0; 8];
            params.pitch_edit = vec![60.0; 8];
        }

        let result =
            import_host_take_pitch(&editor, &json!({ "clipIds": [clip_id.clone()] })).unwrap();
        assert!(
            result["framesWritten"].as_u64().unwrap() > 0,
            "导入必须真的写帧：{result}"
        );
        assert_eq!(
            host.item_state_chunk.borrow().as_str(),
            chunk,
            "导入不得改写宿主 take 的状态（包络原样保留）"
        );
        assert_eq!(
            host.item_value("D_PITCH"),
            Some(3.),
            "导入不得把宿主 Item 音高偏移归零"
        );
        document.close();
    }

    /// 【填充是可逆的】导回来的包络在"没有原线"的帧上必须被**跳过**，而不是写回填充值。
    ///
    /// 导出侧会把辅音处的偏移填成邻域值（让宿主的连续控制信号不跳变）。如果导入不跳过
    /// 这些帧，用户"导出 → 导入"来回一次就会在辅音上得到一条被移调的音高线 —— 而那是
    /// 他从没画过的东西，且**没有任何界面能让他看见**。
    ///
    /// 两条导入路径都靠 `pitch_orig > 0` 挡住：本命令（`commands.rs` 的逐帧判据）与
    /// pending 回退（`schedule.rs` 的 `pending_pitch_offset` 应用）。这条用例钉住前者。
    #[test]
    fn importing_a_filled_envelope_leaves_unvoiced_frames_alone() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        // 恒为 +4 的包络：正是"填充之后"的样子（辅音处不再掉回 0）。
        let chunk = "<ITEM\nPOSITION 0\nLENGTH 0.00009070294784580499\nSEL 1\n\
                     IGUID {11111111-1111-1111-1111-111111111111}\n\
                     SOFFS 0\nPLAYRATE 1 1 0\nGUID {22222222-2222-2222-2222-222222222222}\n\
                     <SOURCE WAVE\nFILE \"ara://source\"\n>\n\
                     <PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\n\
                     DEFSHAPE 0 12 -1\nPT 0.00000000000000000 4.00000000000000000 0\n>\n>\n";
        host.enable_item_state(chunk);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 0.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let (clip_id, root) = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            (clip_id, root)
        };
        // 预热：必须先跑掉"第一次解析宿主身份"（它会按宿主重建一次时间线，把此前写的
        // 片段几何与派生参数一起冲掉）。
        pitch_bridge_target(&editor, &clip_id).expect("夹具的宿主身份必须可解析");
        {
            let mut timeline = editor.timeline.lock_or_recover();
            // 【几何必须在预热之后写】否则被上面那次重建还原成宿主的 D_LENGTH
            // （4 样本 ⇒ 1 帧），这条用例就退化成"只导一帧"。
            if let Some(clip) = timeline.clips.iter_mut().find(|clip| clip.id == clip_id) {
                clip.start_sec = 0.0;
                clip.length_sec = 0.04; // 8 帧（5 ms 一帧）
            }
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.frame_period_ms = 5.0;
            // 帧 2..5 没有 F0（辅音）；其余有声。编辑线 = 原线（用户还没编辑过）。
            params.pitch_orig = vec![60.0, 60.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0];
            params.pitch_edit = params.pitch_orig.clone();
        }

        let result =
            import_host_take_pitch(&editor, &json!({ "clipIds": [clip_id.clone()] })).unwrap();
        assert_eq!(
            result["framesSkipped"].as_u64(),
            Some(3),
            "没有 F0 的帧必须被跳过，实际 {result}"
        );
        assert_eq!(
            result["framesWritten"].as_u64(),
            Some(5),
            "有声帧必须照常写入，实际 {result}"
        );
        document.close();
    }

    /// 批量作用域：一个 id 在时间线上根本不存在，不该让整批失败；严格模式必须报错。
    ///
    /// 【为什么两种模式都要钉】宽容模式是"全部片段"能用的前提（工程在命令执行途中被
    /// 宿主改动、片段被删掉都会走到这里），而严格模式是单条片段的正确语义 —— 用户点的
    /// 是那一条，静默跳过等于"点了没反应"。同一段代码两种行为，只有成对断言才守得住。
    #[test]
    fn a_batch_skips_an_unknown_clip_id_but_a_single_clip_still_fails() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 2.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip_id = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.frame_period_ms = 5.0;
            params.pitch_orig = vec![60.0; 8];
            params.pitch_edit = vec![60.0; 8];
            clip_id
        };

        // 【顺序不能反】夹具把 `pitch_orig` 直接写进派生缓存，而命令会在宿主线程上刷新
        // 一次文档（缓存按权威重建）。因此**需要读到那段音高**的那次调用必须排在最前面；
        // 严格模式那次在收集阶段就因为未知 id 失败了，根本不读参数线。
        let result = import_host_take_pitch(
            &editor,
            &json!({ "clipIds": [clip_id.clone(), "ghost-clip"], "tolerant": true }),
        )
        .expect("宽容模式不该因为一条片段失败");
        // 【只断言"跳过了谁"，不断言"写了几帧"】夹具把 `pitch_orig` 直接写进派生缓存，
        // 而命令会在宿主线程上刷新一次文档（缓存按权威重建）—— 写帧数因此取决于一次
        // 线程竞态。跳过名单不依赖参数线，是这条用例真正要钉的东西。
        assert_eq!(result["clips"].as_u64(), Some(2), "{result}");
        assert_eq!(result["unresolved"][0], "ghost-clip", "{result}");
        assert_eq!(
            result["unresolved"].as_array().map(|items| items.len()),
            Some(1),
            "{result}"
        );

        let strict = import_host_take_pitch(
            &editor,
            &json!({ "clipIds": [clip_id.clone(), "ghost-clip"] }),
        );
        assert!(
            strict.is_err(),
            "严格模式必须报错而不是静默跳过：{strict:?}"
        );
        document.close();
    }

    /// 片段在时间线上、但没有任何宿主 take 绑定它（take 被宿主删掉、或它来自另一个
    /// 实例）：批量作用域必须把它报进 `unresolved`，而不是让整批失败。
    ///
    /// 【为什么这条与上一条分开】上一条走的是"id 不在时间线上"的分支（收集阶段就
    /// 查不到片段）；这一条走的是"片段在、宿主身份解析失败"的分支 —— 两条路各有
    /// 自己的失败点，合成一条用例只会让断言落在先命中的那一条上。
    #[test]
    fn a_clip_without_a_host_take_is_reported_but_does_not_fail_the_batch() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 2.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip_id = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            // 第二条片段：时间线上存在，但没有任何宿主 take 绑定它。
            let mut ghost = timeline.clips[0].clone();
            ghost.id = "ghost-clip".into();
            timeline.clips.push(ghost);
            clip_id
        };

        let result = import_host_take_pitch(
            &editor,
            &json!({ "clipIds": [clip_id.clone(), "ghost-clip"], "tolerant": true }),
        )
        .expect("一条解析不到宿主 take 的片段不该让整批失败");
        assert_eq!(result["clips"].as_u64(), Some(2), "{result}");
        assert_eq!(result["unresolved"][0], "ghost-clip", "{result}");
        assert_eq!(
            result["unresolved"].as_array().map(|items| items.len()),
            Some(1),
            "{result}"
        );
        document.close();
    }

    /// 批量导入**不得**把"take 上没有音高"当成"音高是 0"。
    ///
    /// 【为什么这是最危险的一条】导入的语义是 `pitch_edit = pitch_orig + take 音高`。
    /// take 上没有音高时这个式子退化成"把音高线重置成原线" —— 对单条片段那是用户
    /// 明确点下的意图，对"全部片段"就是把工程里所有没碰过音高的编辑一次抹掉。
    #[test]
    fn a_batch_import_leaves_lines_alone_when_the_take_has_no_pitch() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        // `single_take_chunk` 既没有 `<PITCHENV>`，`D_PITCH` 也是 0 —— 正是绝大多数
        // 真实片段的形状。
        host.enable_item_state(&host.single_take_chunk());
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 0.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip_id = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.frame_period_ms = 5.0;
            params.pitch_orig = vec![60.0; 8];
            // 用户画过的两个半音：批量导入一旦把它重置成 60，这条断言就会失败。
            params.pitch_edit = vec![62.0; 8];
            clip_id
        };

        let tolerant = import_host_take_pitch(
            &editor,
            &json!({ "clipIds": [clip_id.clone()], "tolerant": true }),
        )
        .unwrap();
        assert_eq!(tolerant["framesWritten"].as_u64(), Some(0), "{tolerant}");
        assert_eq!(tolerant["clipsConverted"].as_u64(), Some(0), "{tolerant}");
        // 【这一条才是判据】没有 guard 时这条片段会被正常处理（帧因为没有原线而被跳过），
        // 但它**不会**出现在 `unresolved` 里。名单里有它，说明"take 上没有音高"被识别
        // 出来并主动放过了。
        assert_eq!(tolerant["unresolved"][0], clip_id, "{tolerant}");
        document.close();
    }

    /// 批量导出时"分析还没落地"的片段**不得**被当成全 0 曲线。
    ///
    /// 【为什么这是数据丢失】全 0 曲线经 `build_take_pitch_points` 得到空断点，而空
    /// 断点 = 删除整个 `<PITCHENV>`。批量跑一遍，工程里所有还没分析完的片段都会失去
    /// 用户在 REAPER 里画好的包络 —— 而报告只会说"写了 N 条"。
    ///
    /// 【为什么判据是 `pitch_orig_key` 而不是"原线全 0"】全 0 也可能是"分析完了，
    /// 这段确实没有音高"；那种情况下写出空包络是**正确**的（让 REAPER 侧回到不变调）。
    #[test]
    fn a_batch_export_leaves_the_host_envelope_alone_until_analysis_lands() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        let chunk = "<ITEM\nPOSITION 0\nLENGTH 0.00009070294784580499\nSEL 1\n\
                     IGUID {11111111-1111-1111-1111-111111111111}\n\
                     SOFFS 0\nPLAYRATE 1 1 0\nGUID {22222222-2222-2222-2222-222222222222}\n\
                     <SOURCE WAVE\nFILE \"ara://source\"\n>\n\
                     <PITCHENV\nACT 1 -1\nVIS 1 1 1\nLANEHEIGHT 0 0\nARM 0\n\
                     DEFSHAPE 0 12 -1\nPT 0.00000000000000000 4.00000000000000000 0\n>\n>\n";
        host.enable_item_state(chunk);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_PITCH", 3.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip_id = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            let track_id = timeline.clips[0].track_id.clone();
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            // 只建参数线、**不填** `pitch_orig`：模拟"分析还没跑完"。
            timeline.ensure_params_for_root(&root);
            clip_id
        };

        let tolerant = export_host_take_pitch(
            &editor,
            &json!({ "clipIds": [clip_id.clone()], "tolerant": true }),
        )
        .unwrap();
        assert_eq!(tolerant["clipsConverted"].as_u64(), Some(0), "{tolerant}");
        assert_eq!(tolerant["unresolved"][0], clip_id, "{tolerant}");
        assert_eq!(
            host.item_state_chunk.borrow().as_str(),
            chunk,
            "批量模式同样不得改动宿主 take"
        );
        assert_eq!(
            host.item_value("D_PITCH"),
            Some(3.),
            "没写成时连 D_PITCH 都不能归零"
        );
        document.close();
    }
}

#[cfg(test)]
mod clip_pitch_tests {
    use super::*;

    /// 参考线命令必须按 clip 的**时间窗**切出原线，并且没分析到就不返回 payload。
    ///
    /// 【为什么钉死这两条】插件里这条命令此前根本不存在（`Command unavailable`），
    /// 前端每次挂载都刷一条 ERROR 且参考线永远空白。补上之后最危险的错误形态是
    /// "把全 0 当曲线发回去" —— 那会让前端把"还没算"缓存成"就是没有"，
    /// 而参考线不会取第二次（见 `useClipPitchHydration` 的 requested 集合）。
    #[test]
    fn the_pitch_reference_curve_is_sliced_from_the_root_line() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let (clip_id, root, start_sec, length_sec, period) = {
            let mut timeline = editor.timeline.lock_or_recover();
            let clip_id = timeline.clips[0].id.clone();
            // 夹具的 clip 只有 4 个采样点；这里给它一个可分辨的时间窗（50 ms 起、100 ms 长）。
            let track_id = {
                let clip = timeline
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == clip_id)
                    .unwrap();
                clip.start_sec = 0.05;
                clip.length_sec = 0.1;
                clip.track_id.clone()
            };
            let root = timeline.resolve_root_track_id(&track_id).unwrap();
            timeline.ensure_params_for_root(&root);
            let params = timeline.params_by_root_track.get_mut(&root).unwrap();
            params.frame_period_ms = 5.0;
            // 时间线是 5 ms 一帧：在 clip 窗口上填一段 57/59 交替的音高。
            let period = params.frame_period_ms;
            let start_frame = 10usize; // 0.05 s ÷ 5 ms
            let count = 20usize; // 0.1 s ÷ 5 ms
            params.pitch_orig = vec![0.0; start_frame + count + 4];
            for index in 0..count {
                params.pitch_orig[start_frame + index] = if index % 2 == 0 { 57.0 } else { 59.0 };
            }
            (clip_id, root, 0.05_f64, 0.1_f64, period)
        };

        let payload = clip_pitch_data(&editor, &json!({"clipIds": [clip_id.clone()]})).unwrap();
        let payloads = payload["payloads"].as_array().unwrap();
        assert_eq!(
            payloads.len(),
            1,
            "有原线的 clip 必须回一条 payload：{payload}"
        );
        assert_eq!(payloads[0]["clip_id"], json!(clip_id));
        assert!((payloads[0]["curve_start_sec"].as_f64().unwrap() - start_sec).abs() < 1e-9);
        assert!((payloads[0]["frame_period_ms"].as_f64().unwrap() - period).abs() < 1e-9);
        let curve = payloads[0]["midi_curve"].as_array().unwrap();
        let expected = (((length_sec.max(0.0) * 1000.0) / period).ceil()).max(1.0) as usize;
        assert_eq!(curve.len(), expected);
        assert_eq!(curve[0], json!(57.0));
        assert_eq!(curve[1], json!(59.0));

        // 分析未完成（原线为空）⇒ 不返回 payload，而不是一条全 0 曲线。
        {
            let mut timeline = editor.timeline.lock_or_recover();
            timeline
                .params_by_root_track
                .get_mut(&root)
                .unwrap()
                .pitch_orig
                .clear();
        }
        let empty = clip_pitch_data(&editor, &json!({"clipIds": [clip_id]})).unwrap();
        assert_eq!(empty["payloads"].as_array().unwrap().len(), 0);
    }
}
