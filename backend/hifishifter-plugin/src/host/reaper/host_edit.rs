//! 原GUI几何命令的纯规划与真实宿主写入；不修改actor私有timeline冒充REAPER成功。
use super::{checked, HostClipTarget};
use crate::editor::session::EditorSession;
use crate::host::geometry::host_value_compatible;
use crate::render::extension::ExtensionOwner;
use crate::sync::MutexExt;
use hifishifter_kernel::state::{Clip, ClipStatePatch, TimelineState};
use serde_json::Value;
use std::ffi::{c_void, CStr};
use std::sync::Arc;

/// 宿主 `I_GROUPID` 的上界（0 保留给"未编组"）。
///
/// 【为什么是一份共享常量】写入侧（本文件）与生成侧（`editor/webview.rs` 的
/// `host_group_id_for`）必须落在同一区间；两处各写一个字面量时，只改一处就会让
/// 编组在生成侧合法、在写入侧被整条拒绝。
pub(crate) const HOST_GROUP_ID_MAX: i32 = 2_000_000_000;

pub(crate) struct ClipEdit {
    pub native_id: String,
    pub source_track: String,
    pub destination_track: String,
    pub before: Clip,
    pub after: Clip,
    pub patch: ClipStatePatch,
    /// 逐 take 的方向写回（**非 active take**）。
    ///
    /// 【为什么 active take 不在这里】active take 就是绑定 take：`patch.reversed`
    /// 走既有的几何/镜像/完成门路径（`host_take_direction` + D_STARTOFFS 镜像写），
    /// 回执也按 clip 级字段核对。只有**兄弟 take** 需要按 GUID 现场定位 —— 那不是
    /// "多写一个字段"，是另一条目标寻址。
    pub take_direction: Option<TakeDirectionEdit>,
    /// 逐 take 的声道模式写回（**非 active take**；`I_CHANMODE` 是 take 属性）。
    pub take_channel_mode: Option<TakeChannelModeEdit>,
}

/// 一次"把某条 take 的方向切到 reversed"的逐 take 写回。
#[derive(Debug, Clone)]
pub(crate) struct TakeDirectionEdit {
    /// 内核 take id（带 GUI namespace 前缀）；用于回执核对。
    pub take_id: String,
    /// 宿主 take GUID（去前缀）；用于 chunk 改写定位与指针认领。
    pub take_guid: String,
    /// 目标方向。
    pub reversed: bool,
}

/// 一次"把某条 take 的声道模式设为 mode"的逐 take 写回。
#[derive(Debug, Clone)]
pub(crate) struct TakeChannelModeEdit {
    /// 内核 take id（带 GUI namespace 前缀）；用于回执核对。
    pub take_id: String,
    /// 宿主 take GUID（去前缀）。
    pub take_guid: String,
    /// 0..=4，对齐 REAPER CHANMODE（与 clip 级 `channelMode` 同一值域）。
    pub channel_mode: i32,
}
pub(crate) struct HostEditPlan {
    pub edits: Vec<ClipEdit>,
}

/// 宿主setter完成后只等待本次确切clip的几何回流，不把旧的可读timeline当成功。
#[derive(Debug)]
pub(crate) struct HostEditReceipt {
    clips: Vec<ClipReceipt>,
}

/// 一条 clip 的回执期望：顶层字段 + （逐 take 写回时）**兄弟 take** 的字段。
///
/// 【第三个分量为什么存在】回执现在要回答**两个不同的问题**：
///   1. "宿主回声最终与 `after` 一致吗" —— 诊断（`divergences`）；
///   2. "这次补丁真正写过哪些字段" —— 决定哪些字段可以按 `after` 覆盖载荷（`overlay`）。
///
/// 两者不能用同一份清单：`fields` 里始终包含 `start_sec` / `clip_gain` 等"结构性"
/// 字段（哪怕这次补丁一位都没动），拿它们做覆盖会把**没写过的宿主真值**改掉。
#[derive(Debug)]
struct ClipReceipt {
    id: String,
    /// `(载荷键, 应用后的权威值, 本次补丁是否真的写过它)`
    fields: Vec<(&'static str, Value, bool)>,
    /// 逐 take 期望：键为内核 take id（与载荷 `takes[].id` 同一空间）。
    takes: Vec<(String, Vec<(&'static str, Value)>)>,
}
impl HostEditPlan {
    /// 冻结本次确切clip及预期宿主几何，单独的fade宽度也必须进入完成门。
    ///
    /// 【为什么形状 / 曲率也必须进门】它们此前不在门里，于是"只改曲率"的补丁
    /// `matches()` 恒真：响应立刻带着**改动前**的 dir/shape 回来，前端按权威回声整份
    /// 应用，用户看到的曲线会**弹回**，直到 UI 定时器刷新几何才纠正 —— 一次可见的抖动。
    /// 长度一直在门里，所以只有形状 / 曲率拖拽会抖。
    ///
    /// 【为什么期望值随轴版本而变】连续轴（≥7.81）上形状预设写的是预设自己的
    /// `(curvature, S)`，补丁里同带的 HFS `dir` 会被忽略（见写回块）；旧轴上写的才是
    /// `C_FADE*SHAPE` 与 `D_FADE*DIR` 本身。
    ///
    /// 【为什么 `fade_*_s` 现在也进门】此前不进门的理由是"GUI 载荷没有这个字段
    /// （S 只在 `host_fades` 读数里）"，拿它当判据会让门永远不成立 —— 那个理由在
    /// 载荷补上**顶层** `fade_*_s`（`decorate_host_fades_locked`）之后已经失效。
    /// 不进门的后果与当年 dir/shape 的抖动同源：只改 S 的补丁 `matches()` 恒真，
    /// 回显带着改动前的 S 回来，曲线弹回。
    pub(crate) fn receipt(&self, namespace: &str, fade_axes_v2: Option<bool>) -> HostEditReceipt {
        HostEditReceipt {
            clips: self
                .edits
                .iter()
                .map(|edit| {
                    let after = &edit.after;
                    // "这一位被补丁写到了吗"——只有写过的位才允许按 `after` 覆盖载荷。
                    let moved = edit.source_track != edit.destination_track;
                    let mut fields = vec![
                        (
                            "start_sec",
                            serde_json::json!(after.start_sec),
                            edit.patch.start_sec.is_some(),
                        ),
                        (
                            "length_sec",
                            serde_json::json!(after.length_sec),
                            edit.patch.length_sec.is_some(),
                        ),
                        (
                            "source_start_sec",
                            serde_json::json!(after.source_start_sec),
                            edit.patch.source_start_sec.is_some(),
                        ),
                        (
                            "playback_rate",
                            serde_json::json!(after.playback_rate),
                            edit.patch.playback_rate.is_some(),
                        ),
                        (
                            "track_id",
                            serde_json::json!(format!("{namespace}{}", edit.destination_track)),
                            moved,
                        ),
                        // 【为什么两层增益都要进门，且都取 `after`】REAPER 里 item 增益
                        // （`D_VOL`）与 take 增益是两个独立旋钮，链路里也分两路写
                        // （`write-item:` 与 `write-take:`）。此前门里只有补丁里的
                        // `gain`（take 层），item 层的写入**完全没有回流判据**：只改
                        // item 增益的补丁 `matches()` 恒真，前端按权威回声整份应用，
                        // 旋钮会弹回旧值，直到下一次几何刷新才纠正 —— 与 dir/shape
                        // 当年那次抖动同源。
                        //
                        // 取 `after` 而不是补丁值：`after` 是应用后的权威快照（含
                        // `clamp_gain` 的结果），补丁值只是请求；用请求当判据会把
                        // "宿主把 16 夹到 15.8" 这类事判成成功。
                        (
                            "clip_gain",
                            serde_json::json!(after.clip_gain),
                            edit.patch.item_gain.is_some(),
                        ),
                        (
                            "gain",
                            serde_json::json!(after.gain),
                            edit.patch.gain.is_some(),
                        ),
                        // 【为什么方向与循环源也必须进门】倒放 / 循环都是**只改这一位**的
                        // 补丁：整段媒体上的位置、长度、源窗口一位都不动，于是没有这两项时
                        // `matches()` 恒真 —— 响应立刻带着**改动前**的 `reversed=false`
                        // 回来，前端按权威回声整份应用，用户看到的波形是"倒放 → 正放 →
                        // 倒放"（真机日志 18:14:45 那次倒放，2 s 后才被几何刷新纠正）。
                        // 与当年 dir/shape 弹回同一类缺陷：**凡是被补丁改到、又出现在载荷
                        // 里的字段，都必须进完成门**。
                        (
                            "reversed",
                            serde_json::json!(after.reversed),
                            edit.patch.reversed.is_some(),
                        ),
                        (
                            "loop_enabled",
                            serde_json::json!(after.loop_enabled),
                            edit.patch.loop_enabled.is_some(),
                        ),
                    ];
                    for (value, key) in [
                        (edit.patch.fade_in_sec, "fade_in_sec"),
                        (edit.patch.fade_out_sec, "fade_out_sec"),
                        (edit.patch.auto_fade_in_sec, "auto_fade_in_sec"),
                        (edit.patch.auto_fade_out_sec, "auto_fade_out_sec"),
                        (edit.patch.snap_offset_sec, "snap_offset_sec"),
                    ] {
                        if let Some(value) = value {
                            fields.push((key, serde_json::json!(value), true));
                        }
                    }
                    for (shape, dir, s, shape_key, dir_key, s_key) in [
                        (
                            edit.patch.fade_in_shape,
                            edit.patch.fade_in_dir,
                            edit.patch.fade_in_s,
                            "fade_in_shape",
                            "fade_in_dir",
                            "fade_in_s",
                        ),
                        (
                            edit.patch.fade_out_shape,
                            edit.patch.fade_out_dir,
                            edit.patch.fade_out_s,
                            "fade_out_shape",
                            "fade_out_dir",
                            "fade_out_s",
                        ),
                    ] {
                        if let Some(shape) = shape {
                            match fade_axes_v2 {
                                // 连续轴：形状没有可写号，写的是预设的 curvature；
                                // 载荷的 `fade_*_dir` 正是宿主回读的 curvature。
                                Some(true) => {
                                    if let Some((curvature, preset_s)) =
                                        hifishifter_kernel::fade_axes::host_fade_preset_axes(shape)
                                    {
                                        fields.push((dir_key, serde_json::json!(curvature), true));
                                        // 形状预设自带 S；补丁同带 S 时以补丁为准
                                        // （与写回块同一口径）。
                                        fields.push((
                                            s_key,
                                            serde_json::json!(s.unwrap_or(preset_s)),
                                            true,
                                        ));
                                    }
                                }
                                // 旧轴（或版本读不出来）：形状号本身可写，且 dir 原样写回。
                                _ => {
                                    fields.push((shape_key, serde_json::json!(shape), true));
                                    if let Some(dir) = dir {
                                        fields.push((dir_key, serde_json::json!(dir), true));
                                    }
                                }
                            }
                            continue;
                        }
                        if let Some(dir) = dir {
                            fields.push((dir_key, serde_json::json!(dir), true));
                        }
                        // 【为什么 S 也要进门】此前只比 curvature，理由是"GUI 载荷没有
                        // 这个字段"。现在载荷已带**顶层** `fade_*_s`
                        // （`decorate_host_fades_locked`），不比它会让"只改 S"的补丁
                        // `matches()` 恒真、回显带着改动前的 S 回来 ⇒ 曲线弹回。
                        if let Some(s) = s {
                            fields.push((s_key, serde_json::json!(s), true));
                        }
                    }
                    if let Some(group) = edit.patch.host_group_id {
                        fields.push((
                            "group_id",
                            if group == 0 {
                                serde_json::Value::Null
                            } else {
                                serde_json::json!(format!("host-group-{group}"))
                            },
                            true,
                        ));
                    }
                    // 【逐 take 写回也必须进门】凡是被补丁改到、又出现在载荷里的字段，
                    // 都要等真实回声——兄弟 take 的方向 / 声道模式落在 `takes[]` 里，
                    // 单独收集（见 `matches` 的 take 匹配）。
                    let mut takes: Vec<(String, Vec<(&'static str, Value)>)> = Vec::new();
                    if let Some(direction) = &edit.take_direction {
                        takes.push((
                            direction.take_id.clone(),
                            vec![("reversed", serde_json::json!(direction.reversed))],
                        ));
                    }
                    if let Some(mode) = &edit.take_channel_mode {
                        takes.push((
                            mode.take_id.clone(),
                            vec![("channel_mode", serde_json::json!(mode.channel_mode))],
                        ));
                    }
                    ClipReceipt {
                        id: edit.before.id.clone(),
                        fields,
                        takes,
                    }
                })
                .collect(),
        }
    }
}
impl HostEditReceipt {
    /// 逐字段比较"应用后的权威快照"（`after`）与载荷回声，返回**不一致的明细**。
    ///
    /// 【为什么不再把它当成功判据】回执此前的唯一用途是"写入是否成功"，而它比的是
    /// **载荷**（另一条异步通路的产物：编辑器时间线 → 图集投影 → 序列化）。用异步
    /// 通路的进度判定一次同步写入的结果，必然出两种错：回声慢 ⇒ 误报失败；字段错
    /// （例如本轮修掉的 `clip_gain` 从不被投影）⇒ **永远**失败。现在这条判据只用于
    /// **诊断**：非空表示"宿主说的和我们写下的不一样"，值得记一笔，但写入本身已经
    /// 由 `execute_managed` 的返回码与结构读回核对确认过了（那是宿主的直接答复）。
    ///
    /// 【为什么必须与写回前置检查共用同一个容差】写回前置检查用的是
    /// `host_value_compatible`（相对容差 `1e-6 + 1e-9·max`，见 `host::geometry`），
    /// 而这里此前用的是**纯绝对** `1e-6`。两侧不对称会造出一个窗口：写回被接受
    /// （差值在相对容差内），完成门却判定"宿主还没跟上"。`playback_rate` 尤其容易
    /// 撞上：kernel 里是 f32，写回时与 REAPER 的 f64 比，往返本身就有 ~1e-7 相对误差；
    /// 长工程上的 `start_sec` / `length_sec` 也在绝对阈值的边缘。
    pub(crate) fn divergences(&self, payload: &Value) -> Vec<String> {
        let Some(clips) = payload["clips"].as_array() else {
            return vec!["payload has no clips array".to_string()];
        };
        let mut out = Vec::new();
        for ClipReceipt { id, fields, takes } in &self.clips {
            let Some(clip) = clips.iter().find(|clip| clip["id"] == *id) else {
                // 载荷里找不到这条 clip 是最强的一种分歧：整条缺席。
                out.push(format!("clip {id} missing from the payload"));
                continue;
            };
            for (key, expected, _) in fields {
                if !value_compatible(&clip[*key], expected) {
                    out.push(format!(
                        "clip {id} {key}={} expected {expected}",
                        clip[*key]
                    ));
                }
            }
            for (take_id, take_fields) in takes {
                let Some(take) = clip["takes"].as_array().and_then(|takes| {
                    takes
                        .iter()
                        .find(|take| take["id"].as_str() == Some(take_id.as_str()))
                }) else {
                    out.push(format!("clip {id} take {take_id} missing from the payload"));
                    continue;
                };
                for (key, expected) in take_fields {
                    if !value_compatible(&take[*key], expected) {
                        out.push(format!(
                            "clip {id} take {take_id} {key}={} expected {expected}",
                            take[*key]
                        ));
                    }
                }
            }
        }
        out
    }

    /// 把"本次补丁真正写过的字段"按 `after` 覆盖进载荷。
    ///
    /// 【I-E2】一次成功写入的权威是**我们写下的值**，不是宿主模型的回声。载荷是前端
    /// 拿到之后**整份应用**的那份数据：只要它还停在旧值，用户就会看到"改完回弹"
    /// —— 哪怕写入成功、哪怕我们随后返回 `ok`。
    ///
    /// 【为什么只覆盖"写过的位"】`fields` 始终带着 `start_sec` / `clip_gain` 这类
    /// 结构性字段（补丁一位都没动它们时也在），无条件覆盖会把**没写过**的宿主真值改掉。
    /// 只覆盖补丁触及的位，等价于"把 after 叠加到载荷上"，其余一律继续信宿主。
    pub(crate) fn overlay(&self, payload: &mut Value) {
        let Some(clips) = payload["clips"].as_array_mut() else {
            return;
        };
        for ClipReceipt { id, fields, takes } in &self.clips {
            let Some(clip) = clips.iter_mut().find(|clip| clip["id"] == *id) else {
                continue;
            };
            for (key, expected, touched) in fields {
                if *touched {
                    clip[*key] = expected.clone();
                }
            }
            for (take_id, take_fields) in takes {
                let Some(take) = clip["takes"].as_array_mut().and_then(|takes| {
                    takes
                        .iter_mut()
                        .find(|take| take["id"].as_str() == Some(take_id.as_str()))
                }) else {
                    continue;
                };
                for (key, expected) in take_fields {
                    take[*key] = expected.clone();
                }
            }
        }
    }

    /// 载荷回声是否与 `after` 逐字段相容（`divergences` 为空）。
    ///
    /// 只保留给"另一次异步投影是否追上了"这类**诊断性**判断使用；它不再是写入成功的
    /// 判据（见 `divergences` 的说明）。生产路径已经没有调用方，因此只在测试里编出来 ——
    /// 让"回执仍然认得出陈旧回声"这条不变量继续被钉住，同时不给库里留一个随时会被
    /// 误用回成功判据的死方法。
    #[cfg(test)]
    pub(crate) fn matches(&self, payload: &Value) -> bool {
        self.divergences(payload).is_empty()
    }
}

/// 载荷值与期望值是否相容：数字走共享容差，其余走逐字相等。
fn value_compatible(actual: &Value, expected: &Value) -> bool {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(actual), Some(expected)) => {
            crate::host::geometry::host_value_compatible(actual, expected)
        }
        _ => actual == expected,
    }
}

/// 只拦截已实现的宿主命令；未知命令仍走原actor的明确错误路径。
pub(crate) fn is_clip_edit(command: &str) -> bool {
    matches!(
        command,
        "move_clip" | "move_clips" | "set_clip_state" | "set_clips_state_bulk"
    )
}

/// 请求里是否包含淡变轴字段。
fn requests_fade_axes(patch: &ClipStatePatch) -> bool {
    patch.fade_in_shape.is_some()
        || patch.fade_out_shape.is_some()
        || patch.fade_in_dir.is_some()
        || patch.fade_out_dir.is_some()
        || patch.fade_in_s.is_some()
        || patch.fade_out_s.is_some()
}

/// 淡变轴写入的合法性 —— **纯函数**，因为"哪些字段落在哪套轴上"是领域知识，
/// 不该埋在一个 200 行的循环里（那里既难读也难测）。
///
/// 官方头文件把两套轴标成互补区间（`sdk/reaper_plugin_functions.h`）：
///   `C_FADE*SHAPE` / `D_FADE*DIR`        —— v7.80 and earlier
///   `D_FADE*DIR_NEW` / `D_FADE*DIR2_NEW` —— v7.81 and later
///
/// 因此：
/// - 版本读不出来（`None`）→ 拒绝，不猜；
/// - 旧轴宿主 + S 参数 → 拒绝（S 只存在于新轴，做不到的事不静默丢弃）；
/// - 新轴宿主 + 形状预设 → **接受**。这里此前是拒绝的，理由是"预设 → (curvature, S)
///   的映射尚未校准"；该映射现已实测落地（[`hifishifter_kernel::fade_axes`]），
///   所以预设按钮在新轴宿主上也能用。
///   表外的形状号（越界或小数变体）仍然拒绝 —— 宿主会把它静默变成别的形状
///   （实测写 `5.1` 读回 `SHAPE=7`）。
fn validate_fade_axes(axes_new: Option<bool>, patch: &ClipStatePatch) -> Result<(), String> {
    if !requests_fade_axes(patch) {
        return Ok(());
    }
    match axes_new {
        None => Err("host fade axis version unavailable; cannot write the fade shape".into()),
        Some(false) if patch.fade_in_s.is_some() || patch.fade_out_s.is_some() => {
            Err("fade S parameter requires REAPER 7.81 or later".into())
        }
        Some(true) => {
            for shape in [patch.fade_in_shape, patch.fade_out_shape]
                .into_iter()
                .flatten()
            {
                if hifishifter_kernel::fade_axes::host_fade_preset_axes(shape).is_none() {
                    return Err(format!(
                        "unknown fade shape {shape}: REAPER 7.81+ has no preset at that index"
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// 未知非空字段必须失败，不能用户改shape却被当成成功的长度修改。
fn patch(input: &Value) -> Result<ClipStatePatch, String> {
    let object = input.as_object().ok_or("clip patch must be an object")?;
    for (key, value) in object {
        if value.is_null() {
            continue;
        }
        if !matches!(
            key.as_str(),
            "clipId"
                | "checkpoint"
                | "name"
                | "startSec"
                | "lengthSec"
                | "sourceStartSec"
                | "sourceEndSec"
                | "playbackRate"
                | "clipPlaybackRate"
                // 两层增益：`itemGain` 写宿主 item 的 D_VOL，`gain` 写 take 的 D_VOL。
                | "gain"
                | "itemGain"
                | "muted"
                | "snapOffsetSec"
                | "fadeInSec"
                | "fadeOutSec"
                | "autoFadeInSec"
                | "autoFadeOutSec"
                | "fadeInShape"
                | "fadeOutShape"
                | "fadeInDir"
                | "fadeOutDir"
                | "fadeInS"
                | "fadeOutS"
                | "reversed"
                | "loopEnabled"
                | "channelMode"
                | "hostGroupId"
                // 逐 take 写回（前端 `set_clip_take_reversed` / `set_clip_take_channel_mode`
                // 的重写形态）。值为 `{takeId, reversed|channelMode}` 对象。
                | "hostTakeReversed"
                | "hostTakeChannelMode"
                // 插件自有、宿主表达不到的属性。宿主没有 clip 颜色 / 共振峰变形的概念，
                // 所以它们**不写宿主**，而是落到本地权威 `EditState::clip_local`
                // （与 `fades` 同一手法）。此前它们在白名单外 ⇒ 任何带这两项的
                // `set_clip_state` 都被整条拒绝，用户看到"音频块编辑被拒绝"。
                | "color"
                | "formantMorph"
        ) {
            return Err(format!("host clip property is not yet writable: {key}"));
        }
        if key == "hostTakeReversed" {
            let take = value
                .get("takeId")
                .and_then(|id| id.as_str())
                .ok_or("host take direction requires a takeId")?;
            if take.is_empty() || take.len() > 256 || take.contains('\0') {
                return Err("invalid host take id".into());
            }
            if value
                .get("reversed")
                .and_then(|reversed| reversed.as_bool())
                .is_none()
            {
                return Err("host take direction requires a boolean reversed".into());
            }
            continue;
        }
        if key == "hostTakeChannelMode" {
            let take = value
                .get("takeId")
                .and_then(|id| id.as_str())
                .ok_or("host take channel mode requires a takeId")?;
            if take.is_empty() || take.len() > 256 || take.contains('\0') {
                return Err("invalid host take id".into());
            }
            let mode = value
                .get("channelMode")
                .and_then(|mode| mode.as_f64())
                .filter(|mode| (0.0..=4.0).contains(mode) && mode.fract() == 0.0)
                .ok_or("host take channel mode outside 0..=4")?;
            let _ = mode;
            continue;
        }
        if key == "name" {
            if value
                .as_str()
                .is_none_or(|name| name.len() > 8192 || name.contains('\0'))
            {
                return Err("invalid take name".into());
            }
            continue;
        }
        if key == "color" {
            if value
                .as_str()
                .is_none_or(|color| color.len() > 64 || color.contains('\0'))
            {
                return Err("invalid clip color".into());
            }
            continue;
        }
        if key == "formantMorph" {
            if !value.is_object() {
                return Err("invalid clip formant morph".into());
            }
            continue;
        }
        if !matches!(
            key.as_str(),
            "clipId" | "checkpoint" | "muted" | "reversed" | "loopEnabled"
        ) {
            let number = value
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("finite numeric clip edit required")?;
            if matches!(
                key.as_str(),
                "fadeInDir" | "fadeOutDir" | "fadeInS" | "fadeOutS"
            ) {
                // 官方头文件把两个新轴都标为 -1..1（curvature / S parameter）。
                if !(-1.0..=1.0).contains(&number) {
                    return Err("fade curvature/S outside -1..1".into());
                }
            } else if key != "hostGroupId" && !(0. ..=1_000_000.).contains(&number) {
                // 【为什么 hostGroupId 不走通用上限】通用上限服务的是"秒/倍率/增益"
                // 这些几何量；编组 id 的值域是 `0..=HOST_GROUP_ID_MAX`（2e9，
                // `host_group_id_for` 的哈希取模就落在整个区间上）。让它吃几何
                // 上限会把 ~99.95% 的编组在解析阶段整条拒绝 —— 用户看到的就是
                // "编组点了没反应"，而下面那条专属检查（同值域、还要求整数）
                // 永远不可达。两者各管各的值域。
                return Err("clip edit outside supported finite range".into());
            }
            if matches!(key.as_str(), "fadeInShape" | "fadeOutShape") && number >= 7. {
                return Err("unknown fade shape".into());
            }
            if matches!(
                key.as_str(),
                "lengthSec" | "playbackRate" | "clipPlaybackRate"
            ) && number <= 0.
            {
                return Err("positive clip length/rate required".into());
            }
            if key == "channelMode" && (number.fract() != 0. || number > 4.) {
                return Err("invalid host channel mode".into());
            }
            if key == "gain" && number > f64::from(hifishifter_kernel::gain::GAIN_MAX) {
                return Err(format!(
                    "clip volume outside the supported range 0..{}",
                    hifishifter_kernel::gain::GAIN_MAX
                ));
            }
            if key == "hostGroupId"
                && (!(0.0..=HOST_GROUP_ID_MAX as f64).contains(&number) || number.fract() != 0.)
            {
                return Err("invalid host item group id".into());
            }
        }
    }
    let patch: ClipStatePatch = serde_json::from_value(input.clone()).map_err(|e| e.to_string())?;
    // 【倒放现在可写了】旧实现在这里整条拒绝，理由是"本模块只写数值/字符串字段，
    // 建不了 `SOURCE SECTION`"。那条理由是对的 —— 只是结论过期：方向改由
    // `source_direction::rewrite_take_direction` 做**结构**写回（见 `execute_managed`）。
    // 是否真的能写由 `HostClipTarget::can_reverse_clips()` 在预检里判定，
    // 前端另有同一来源的能力位决定菜单项出不出来。
    Ok(patch)
}

impl EditorSession {
    /// UI只短暂读取当前clip身份/几何；源参数大数组不复制，不在这里materialize或分析。
    pub(crate) fn plan_host_edit(
        &self,
        command: &str,
        input: &Value,
    ) -> Result<HostEditPlan, String> {
        let requests = match command {
            "move_clip" | "set_clip_state" => vec![input.clone()],
            "move_clips" => input["moves"]
                .as_array()
                .ok_or("move list missing")?
                .clone(),
            "set_clips_state_bulk" => input["updates"]
                .as_array()
                .ok_or("patch list missing")?
                .clone(),
            _ => return Err("unknown host clip command".into()),
        };
        if requests.is_empty() || requests.len() > 512 {
            return Err("host edit batch must contain 1..512 clips".into());
        }
        // 【为什么用 checked 取锁】本函数把编辑器时间线**写回宿主**。若这把锁此前被
        // 某线程的 panic 毒化过，`lock_or_recover` 会交回一份**可能半写**的 clip 状态；
        // 拿它去改 DAW 会把一次局部 panic 升级成工程级损坏（S1）。因此这里在恢复即
        // 拒绝本次写回，让用户走"刷新/重开"这条明确的恢复路径，而不是默默写坏。
        // 【为什么不再是 mut】参数线不再在这里搬（见下面那段注释），这份时间线只被读。
        let (timeline, recovered) = self.timeline.lock_or_recover_checked();
        if recovered {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Error,
                "editor.timeline-poisoned",
                format_args!(
                    "[ara] editor timeline was poisoned by a panic; refusing to write host edits"
                ),
            );
            return Err("editor state was reset after an internal error; refresh required".into());
        }
        // ── 移动类命令：参数线由**图集投影**负责，不在这里搬 ─────────────────
        //
        // 【为什么删掉了 `TimelineState::move_clips` 那一步】它搬的是 `EditorSession.timeline`
        // 里的参数数组，而这份时间线每一帧都会被快照整份重建、`params_by_root_track`
        // 又被图集投影整份替换（`EditState::apply`）—— 搬完的下一帧就被盖回去。于是
        // 那一行既没有用户可见效果，又让"锁定参数线"看起来时灵时不灵。
        //
        // 现在曲线位置**只有一个**决定者：`ParamAuthority` 的工程时间 run 按**当前**
        // 设置选锚点（见 `anchor_is_locked`）。它由几何的写回方调用（ARA 的 `end_editing`
        // 与原生扩展的几何刷新），而本函数正是"几何要变了"的那个信号源。
        //
        // 【为什么仍要先把 before 克隆出来】下面的预检要拿"编辑前"的位置去比宿主的
        // 当前几何。读晚了就会拿新值比旧值，把一次正常编辑误判成"宿主在我们编辑期间
        // 变过"。
        let mut before_by_id: std::collections::HashMap<String, hifishifter_kernel::state::Clip> =
            std::collections::HashMap::new();
        for request in &requests {
            let Some(id) = request["clipId"].as_str() else {
                continue;
            };
            if let Some(clip) = timeline.clips.iter().find(|clip| clip.id == id) {
                before_by_id.insert(id.to_owned(), clip.clone());
            }
        }

        let mut seen = std::collections::HashSet::new();
        let mut edits = Vec::new();
        for request in requests {
            let id = request["clipId"].as_str().ok_or("clipId missing")?;
            if !seen.insert(id.to_owned()) {
                return Err("duplicate clip in host edit batch".into());
            }
            let before = before_by_id.get(id).cloned().ok_or("unknown editor clip")?;
            let destination = request["trackId"].as_str().unwrap_or(&before.track_id);
            if !timeline.tracks.iter().any(|track| track.id == destination) {
                return Err("unknown target host track".into());
            }
            let mut changes = request.clone();
            changes
                .as_object_mut()
                .ok_or("clip request missing")?
                .remove("trackId");
            // 旧前端仍可能带这一维（偏好快照）；它不属于 clip 状态，先摘掉再解析。
            // 曲线的位置由图集投影按**当前**设置决定，与这个字段无关。
            changes.as_object_mut().unwrap().remove("moveLinkedParams");
            let mut patch = patch(&changes)?;
            // ── 逐 take 写回的规划 ─────────────────────────────────────────────
            //
            // 前端把 `set_clip_take_reversed` / `set_clip_take_channel_mode` 重写成
            // `set_clip_state` 的 `hostTakeReversed` / `hostTakeChannelMode`
            // （`{takeId, ...}` 对象）。kernel `ClipStatePatch` 不认识它们（serde
            // 丢弃未知字段），因此从原始请求里取。
            //
            // 【active take 折进 clip 级补丁】绑定 take 就是 active take：折进
            // `patch.reversed` / `patch.channel_mode` 之后，既有路径（几何换算、
            // 镜像 SOFFS、回执门）原样生效。兄弟 take 才进 side 字段，由写循环
            // 按 GUID 现场定位。
            let mut take_direction: Option<TakeDirectionEdit> = None;
            let mut take_channel_mode: Option<TakeChannelModeEdit> = None;
            if let Some(direction) = changes.get("hostTakeReversed").filter(|v| !v.is_null()) {
                let take_id = direction["takeId"].as_str().unwrap_or_default().to_owned();
                let reversed = direction["reversed"].as_bool().unwrap_or(false);
                // 只验存在性：兄弟 take 的指针在写循环里按 GUID 现场认领。
                if !before.takes.iter().any(|take| take.id == take_id) {
                    return Err("unknown host take".into());
                }
                let take_guid = take_id
                    .strip_prefix(&self.namespace)
                    .ok_or("take belongs to another editor session")?
                    .to_owned();
                if before.active_take_id.as_deref() == Some(take_id.as_str()) {
                    patch.reversed = Some(reversed);
                } else {
                    take_direction = Some(TakeDirectionEdit {
                        take_id,
                        take_guid,
                        reversed,
                    });
                }
            }
            if let Some(mode) = changes.get("hostTakeChannelMode").filter(|v| !v.is_null()) {
                let take_id = mode["takeId"].as_str().unwrap_or_default().to_owned();
                let channel_mode = mode["channelMode"].as_f64().unwrap_or(f64::NAN);
                if !(0.0..=4.0).contains(&channel_mode) || channel_mode.fract() != 0.0 {
                    return Err("invalid host take channel mode".into());
                }
                if !before.takes.iter().any(|take| take.id == take_id) {
                    return Err("unknown host take".into());
                }
                let take_guid = take_id
                    .strip_prefix(&self.namespace)
                    .ok_or("take belongs to another editor session")?
                    .to_owned();
                if before.active_take_id.as_deref() == Some(take_id.as_str()) {
                    patch.channel_mode = Some(channel_mode as i32);
                } else {
                    take_channel_mode = Some(TakeChannelModeEdit {
                        take_id,
                        take_guid,
                        channel_mode: channel_mode as i32,
                    });
                }
            }
            // 整体宽度/交叉点patch会带未改的shape；不能把它误判为自定义渐变请求。
            if patch.fade_in_shape == Some(before.fade_in_shape) {
                patch.fade_in_shape = None;
            }
            if patch.fade_out_shape == Some(before.fade_out_shape) {
                patch.fade_out_shape = None;
            }
            if patch.fade_in_dir == Some(before.fade_in_dir) {
                patch.fade_in_dir = None;
            }
            if patch.fade_out_dir == Some(before.fade_out_dir) {
                patch.fade_out_dir = None;
            }
            // S 同理：连续轴宿主上每次提交都会带上当前 S（前端发整份 clip 状态），
            // 不在这里去重就会让**每一次**几何编辑都触发一次淡变轴写回。
            if patch.fade_in_s == Some(before.fade_in_s) {
                patch.fade_in_s = None;
            }
            if patch.fade_out_s == Some(before.fade_out_s) {
                patch.fade_out_s = None;
            }
            // 复用原Clip级乘数/Take有效倍率和裁切语义，而不是把clipPlaybackRate当take速率。
            let mut view = TimelineState {
                clips: vec![before.clone()],
                ..Default::default()
            };
            view.patch_clip_state(id, patch.clone());
            let mut after = view.clips.remove(0);
            if let Some(end) = patch.source_end_sec {
                if patch.length_sec.is_none() {
                    // 【Loop 的窗口不是区间】`source_start_sec / source_end_sec` 在
                    // 循环源上是**回绕相位锚点**（前端 `slipWindow` 对两端取模回绕，
                    // 回绕后 end < start 完全合法；内核引擎同样按锚点 rem_euclid
                    // 消费）。`(end − start) / rate` 的区间算式对它们必然得出
                    // 负数 —— 此前这里直接拒绝，症状正是"loop clip 的一次 slip
                    // 提交报 source trim window must have positive length"。
                    // 长度只能来自补丁本身（trim 会带 lengthSec）或编辑前的值
                    // （slip 不改长度），两种来源都已经落在 `after` 上。
                    if !after.loop_enabled {
                        after.length_sec =
                            (end - after.source_start_sec) / (after.playback_rate as f64);
                        if after.length_sec <= 0. {
                            return Err("source trim window must have positive length".into());
                        }
                    }
                }
            }
            let native_id = id
                .strip_prefix(&self.namespace)
                .ok_or("clip belongs to another editor session")?
                .to_owned();
            let destination_track = destination
                .strip_prefix(&self.namespace)
                .ok_or("target track belongs to another editor session")?
                .to_owned();
            let source_track = before
                .track_id
                .strip_prefix(&self.namespace)
                .ok_or("source track belongs to another editor session")?
                .to_owned();
            edits.push(ClipEdit {
                native_id,
                source_track,
                destination_track,
                before,
                after,
                patch,
                take_direction,
                take_channel_mode,
            });
        }
        Ok(HostEditPlan { edits })
    }
}

/// 宿主 `D_STARTOFFS` 的值：倒放 take 存的是**镜像**起点（`D − 正向终点`），
/// 正放（以及倒放 + 循环源，走内核回绕分支）就是正向起点本身。
///
/// 【为什么需要它】`project_host_take_facts_locked` 把宿主镜像坐标复原成正向窗口
/// 交给内核（见 `hifishifter_kernel::take_geometry`），这里是它的**逆运算** ——
/// 写回时必须再翻回镜像域，否则倒放 take 会被整体平移 `媒体长 − 正向终点`。
/// 媒体总长取不到时返回 `None`：宁可不写，也不写一个错的起点。
fn host_start_offset(clip: &hifishifter_kernel::state::Clip) -> Option<f64> {
    if !clip.reversed {
        return Some(clip.source_start_sec);
    }
    // 【为什么倒放**不分** loop / 非 loop】读取侧对两者都给"正向窗口"，倒放时窗口的
    // **终点**就是锚点（loop 时是"锚点 → 可用边界"的右端，非 loop 时是"终点 − 消耗量"
    // 的右端）。镜像关系 `soffs = 轴 − 正向终点` 对两者是同一条式子 —— 旧实现给 loop
    // 单开一支、直接把 `source_start_sec` 写回去，而读取侧那个窗口的终点被钳到
    // `min_bound`，于是循环 + 倒放的 take 每写一次就漂一个 `soffs`。
    let axis = hifishifter_kernel::take_geometry::take_axis_sec(
        None,
        hifishifter_kernel::state::clip_source_media_duration_sec(clip),
        None,
    );
    // 与读取侧同一个逆运算（`forward_source_window` 的逆），避免两侧各写一份。
    let (start, end) = hifishifter_kernel::state::clip_playback_window_sec(clip);
    let window = hifishifter_kernel::take_geometry::ForwardWindow::new(start, end)?;
    hifishifter_kernel::take_geometry::host_start_offset(window, axis).map(|offset| offset.value())
}

/// 宿主几何换算出的**正向**窗口起点 —— 与 `project_host_take_facts_locked` 走同一份
/// 解算（`hifishifter_kernel::take_geometry`）。
///
/// 【为什么需要它】写回前置检查此前直接拿 `target.geometry.source_start_sec`
/// （宿主**镜像**域）与 `edit.before.source_start_sec`（编辑器时间线，倒放时是**正向**
/// 域）比较。倒放 clip 上这是两个不同坐标系的值：检查要么恒失败（用户看到
/// "host source window changed before GUI commit"，一次编辑被拒），要么碰巧通过。
/// 统一换算到正向域之后，判据对两个方向都成立。
pub(super) fn host_forward_window_start(
    geometry: &crate::host::geometry::HostClipGeometry,
    media_duration_sec: Option<f64>,
) -> Option<f64> {
    hifishifter_kernel::take_geometry::TakeGeometry {
        start_offset_sec: geometry.source_start_sec,
        item_length_sec: geometry.duration_sec,
        play_rate_signed: geometry.play_rate_signed,
        media_duration_sec,
        section: geometry.section,
        reversed: Some(geometry.reversed),
        loop_enabled: geometry.loop_source,
    }
    .forward_source_window()
    .map(|window| window.start())
}

/// 由 clip 与媒体总长算出**写回 take 方向**的参数（SECTION 与 SOFFS）。
///
/// 【编码只有一种（真机取证）】REAPER 7.85 的 `Item properties: Toggle take
/// reverse`（41051）写出的就是：
///
/// ```text
/// SOFFS = D − 正向窗口终点
/// <SOURCE SECTION LENGTH <正向跨度> MODE 3 STARTPOS 0 OVERLAP 0.01> … </SOURCE SECTION>
/// ```
///
/// 其中 `MODE 3` = "整份媒体 + 反向"（位语义见 [`super::source_direction`]）。
/// 于是**倒放**不论是否 Loop，参数都是这一组：
///
/// * `LENGTH` = 正向窗口跨度（Loop 时窗口本来就是 `[0, 锚点]`，跨度 = 锚点）；
/// * `SOFFS` = `D − 正向窗口终点`（宿主镜像域）。
///
/// 读取侧（`take_geometry`）对 `MODE 3` 拿到的段是"整份媒体"，轴 = D，
/// 锚点 = D − SOFFS = 正向终点 —— 与这里逐位互逆。
///
/// **正放**：拆掉 SECTION、`SOFFS = 正向窗口起点`（绝对坐标）。这正是宿主
/// 自己 toggle 回去时写的东西。
///
/// 媒体总长取不到、或正向窗口落到媒体之外时返回 `Err`：**宁可不写**，
/// 也不写一个必然读不回来的结构。
fn host_take_direction(
    clip: &hifishifter_kernel::state::Clip,
    media_duration_sec: Option<f64>,
) -> Result<super::source_direction::DirectionEdit, String> {
    if !clip.reversed {
        return Ok(super::source_direction::DirectionEdit::forward(
            clip.source_start_sec,
        ));
    }
    let duration = media_duration_sec
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or("cannot write a reversed take without its media duration")?;
    if clip.loop_enabled {
        // Loop：回绕域是整份媒体，锚点取 `min(source_end, D) mod D` —— 与读取侧
        // `(min_bound.min(anchor), anchor)` 同一条式子（窗口起点在读取侧就是 0）。
        let anchor = clip.source_end_sec.min(duration).rem_euclid(duration);
        return Ok(super::source_direction::DirectionEdit::reversed(
            anchor.max(1e-9),
            (duration - anchor).rem_euclid(duration),
        ));
    }
    let (start, end) = hifishifter_kernel::state::clip_playback_window_sec(clip);
    let window = hifishifter_kernel::take_geometry::ForwardWindow::new(start, end)
        .ok_or("reversed take has no valid forward window")?;
    if window.end() > duration + 1e-9 || window.start() < -1e-9 {
        return Err("reversed take window falls outside the media".into());
    }
    Ok(super::source_direction::DirectionEdit::reversed(
        window.span().max(1e-9),
        (duration - window.end()).max(0.0),
    ))
}

/// 逐 take（**兄弟 take**）的方向写回：解析 take 指针、算编码、走 chunk/动作两条路。
///
/// 【DirectionEdit 为什么在这里算】兄弟 take 的正向窗口与媒体时长只有宿主线程上
/// 有（SECTION 读数 / PCM 源时长），规划阶段拿不到。从宿主读数（SOFFS/LENGTH/
/// PLAYRATE/SECTION）经 `take_geometry` 解算正向窗口，再按目标方向折叠成编码 ——
/// 镜像式子与 [`host_take_direction`] 同一条（`soffs = D − 正向终点`）。
fn write_sibling_take_direction(
    target: &HostClipTarget,
    edit: &TakeDirectionEdit,
    authorized: &impl Fn() -> bool,
) -> Result<DirectionWrite, String> {
    if !target.can_reverse_clips() {
        return Err("the host cannot write the take direction".into());
    }
    let take = target.take_pointer_by_guid(&edit.take_guid, authorized)?;
    let expected = edit.reversed;
    if !target.hosts_this_instance() {
        let original = target.item_state_chunk(authorized)?;
        let rewritten = {
            let direction = sibling_take_direction_edit(target, take, expected, authorized)?;
            super::source_direction::rewrite_take_direction(&original, &edit.take_guid, &direction)?
        };
        match target.set_item_state_chunk(&rewritten, authorized) {
            Ok(()) => {
                if target.take_reversed_now(take, authorized)? != expected {
                    return Err("the host did not apply the requested take direction".into());
                }
                return Ok(DirectionWrite::Chunk { original });
            }
            Err(error) if error == super::write::ITEM_STATE_REFUSED => {
                crate::log_line(&format!(
                    "[ara] the host refused the item state ({error}); \
                     writing the sibling take direction through its own action"
                ));
            }
            Err(error) => return Err(error),
        }
    }
    // 宿主动作路径：`run_item_action` 先把 I_CURTAKE 切到目标 take、跑完切回来
    // （音高包络创建走的就是这条既有纪律）。方向由宿主自己的 toggle 计算 ——
    // 它写出的结构与我们的编码逐字节一致（真机对照过，见 `source_direction`）。
    let before = target.take_reversed_now(take, authorized)?;
    target.apply_take_direction_on(take, expected, authorized)?;
    if target.take_reversed_now(take, authorized)? != expected {
        return Err("the host did not apply the requested take direction".into());
    }
    Ok(DirectionWrite::Action { before, take })
}

/// 把一条**兄弟 take**（非绑定 take）的声道模式写到宿主上，返回 (take 指针, 旧值)。
///
/// `I_CHANMODE` 是 take 属性：`GetSetMediaItemTakeInfo_Value` 对 item 上**任意**
/// take 指针都成立，指针由 [`HostClipTarget::take_pointer_by_guid`] 按 GUID 现场认领。
/// 旧值供整批失败时回滚。
fn set_take_channel_mode_for_take(
    target: &HostClipTarget,
    edit: &TakeChannelModeEdit,
    authorized: &impl Fn() -> bool,
) -> Result<(usize, i32), String> {
    let take = target.take_pointer_by_guid(&edit.take_guid, authorized)?;
    let geometry = target
        .host_geometry_api()
        .ok_or("REAPER geometry API unavailable")?;
    let read = |name: &CStr| -> Result<f64, String> {
        let value = checked(authorized, || unsafe {
            (geometry.take_value)(take as *mut c_void, name.as_ptr())
        })?;
        if !value.is_finite() {
            return Err("the host reported a nonfinite take channel mode".into());
        }
        Ok(value)
    };
    let old = read(c"I_CHANMODE")?;
    if old.fract() != 0.0 || !(0.0..=4.0).contains(&old) {
        return Err("the host reported an invalid take channel mode".into());
    }
    if old as i32 != edit.channel_mode {
        target.set_take_pointer_value(take, c"I_CHANMODE", edit.channel_mode as f64, authorized)?;
        // 写→读→比：宿主拒绝或换算过值都要如实报错，"点了没反应"不伪装成成功。
        let now = read(c"I_CHANMODE")?;
        if now as i32 != edit.channel_mode {
            return Err("the host did not apply the requested take channel mode".into());
        }
    }
    Ok((take, old as i32))
}

/// 由兄弟 take 的宿主读数算出方向写回的编码（SECTION 与 SOFFS）。
fn sibling_take_direction_edit(
    target: &HostClipTarget,
    take: usize,
    desired_reversed: bool,
    authorized: &impl Fn() -> bool,
) -> Result<super::source_direction::DirectionEdit, String> {
    let geometry = target
        .host_geometry_api()
        .ok_or("REAPER geometry API unavailable")?;
    let facts = target.host_take_direction_facts(take as *mut c_void, authorized);
    // 【读不到方向位 = 正放，不是错误】`PCM_Source_GetSectionInfo` 对没有 SECTION
    // 包装的裸源返回 false —— 那是"从未倒放过"的正常形态，与 `reversed_now`
    // （读不到按正放）是同一口径。只有**写倒放**时媒体时长缺失才报错。
    let (current_reversed, section) =
        facts.map_or((false, None), |facts| (facts.reversed, facts.section));
    let read = |name: &CStr| -> Result<f64, String> {
        let value = checked(authorized, || unsafe {
            (geometry.take_value)(take as *mut c_void, name.as_ptr())
        })?;
        if !value.is_finite() {
            return Err(format!(
                "nonfinite sibling take field {}",
                name.to_string_lossy()
            ));
        }
        Ok(value)
    };
    let offs = read(c"D_STARTOFFS")?;
    let length = read(c"D_LENGTH")?;
    let rate = read(c"D_PLAYRATE")?;
    let media = target
        .host_take_source_length(take as *mut c_void, authorized)
        .filter(|value| *value > 0.0);
    let take_geometry = hifishifter_kernel::take_geometry::TakeGeometry {
        start_offset_sec: offs,
        item_length_sec: length,
        play_rate_signed: rate,
        media_duration_sec: media,
        section,
        reversed: Some(current_reversed),
        loop_enabled: false,
    };
    let forward = take_geometry
        .forward_source_window()
        .ok_or("the sibling take window is not resolvable")?;
    if !desired_reversed {
        // 回到正放：SOFFS = 正向窗口起点，SECTION 拆掉（`rewrite_take_direction`
        // 对本来就不是 SECTION 的源是幂等的）。
        return Ok(super::source_direction::DirectionEdit::forward(
            forward.start(),
        ));
    }
    let duration = media.ok_or("cannot write a reversed take without its media duration")?;
    if forward.end() > duration + 1e-9 || forward.start() < -1e-9 {
        return Err("reversed take window falls outside the media".into());
    }
    Ok(super::source_direction::DirectionEdit::reversed(
        forward.span().max(1e-9),
        (duration - forward.end()).max(0.0),
    ))
}

/// 先冻结并验证整批对象，再开始Undo和setter；任一预检失败都不写入任何宿主item。
// 保留无undo上下文的一次性执行入口，当前调用方统一走execute_managed。
#[cfg(test)]
pub(crate) fn execute(
    owner: &Arc<ExtensionOwner>,
    plan: HostEditPlan,
    authorized: impl Fn() -> bool,
) -> Result<(), String> {
    execute_managed(owner, plan, authorized, false)
}
pub(crate) fn execute_managed(
    owner: &Arc<ExtensionOwner>,
    plan: HostEditPlan,
    authorized: impl Fn() -> bool,
    managed: bool,
) -> Result<(), String> {
    let document = owner.editor_document()?;
    let mut targets: Vec<(ClipEdit, HostClipTarget, Option<HostClipTarget>)> = Vec::new();
    let mut styles = std::collections::BTreeMap::new();
    // 每个 edit 对应的方向结构写回计划（None = 本次不改方向）。
    let mut reverse_plans: Vec<Option<super::source_direction::DirectionEdit>> = Vec::new();
    // 插件自有、宿主表达不到的逐片段属性（颜色 / 共振峰变形），键为宿主稳定 item 身份。
    let mut locals = std::collections::BTreeMap::new();
    for edit in plan.edits {
        if !authorized() {
            return Err(format!(
                "host {}",
                crate::editor::errors::LEASE_REVOKED.message
            ));
        }
        let target = owner.host_edit_target(&edit.native_id)?;
        if !target.geometry.markers.is_empty()
            && (edit.patch.length_sec.is_some()
                || edit.patch.source_start_sec.is_some()
                || edit.patch.source_end_sec.is_some()
                || edit.patch.playback_rate.is_some()
                || edit.patch.clip_playback_rate.is_some())
        {
            return Err("nonlinear host stretch-marker editing is not supported".into());
        }
        // 与绑定同一判据（`host::geometry::host_value_compatible`）。此前这里用 1e-6
        // 绝对阈值，而绑定用 1e-7 —— `before` 是**派生物**（经 `normalize_takes` 的 f32
        // 往返），与宿主原值在两者之间的正常舍入会被判成"宿主变了"，用户看到
        // "host clip changed before GUI commit"，一次编辑被拒。
        if !host_value_compatible(target.geometry.start_sec, edit.before.start_sec)
            || !host_value_compatible(target.geometry.duration_sec, edit.before.length_sec)
        {
            // 【为什么把数值写进消息】这句话是用户与开发者**唯一**能拿到的现场：
            // 状态条只显示"音频块编辑被拒绝"，日志里也只有这一行。没有数字时，
            // "差一个轨道头""差一个镜像域""真的差了一次拖动"在外部完全不可区分。
            // 保留原前缀是因为前端按子串把它归为可重试（见 clipEditFailure）。
            return Err(format!(
                "host clip changed before GUI commit; refresh required (host start={:.9} length={:.9}, editor start={:.9} length={:.9})",
                target.geometry.start_sec,
                target.geometry.duration_sec,
                edit.before.start_sec,
                edit.before.length_sec,
            ));
        }
        if (edit.patch.source_start_sec.is_some()
            || edit.patch.source_end_sec.is_some()
            || edit.patch.length_sec.is_some()
            || edit.patch.playback_rate.is_some()
            || edit.patch.clip_playback_rate.is_some())
            && (!host_value_compatible(
                // 两侧统一到**正向**域再比：宿主几何是镜像/段相对域，而 `before` 来自
                // 编辑器时间线（倒放时已被复原成正向）。媒体总长取 `before` 自带的
                // ARA 源时长 —— 它是"编辑前"的权威读数，且与 clip 一起冻结。
                host_forward_window_start(
                    &target.geometry,
                    hifishifter_kernel::state::clip_source_media_duration_sec(&edit.before),
                )
                // 【拿不到窗口 ⇒ 拒绝】方向读不到 / 倒放但没有镜像轴时，"宿主起点"没有
                // 可比较的值。放行等于用两个不同坐标系的值做等值判断 —— 那正是
                // "倒放 clip 上每一次编辑都被判成宿主变了"与"碰巧通过"两种症状的来源。
                .ok_or(
                    "the host take window is not resolvable (direction or mirror axis                      missing); refresh the plugin window before editing a reversed take",
                )?,
                edit.before.source_start_sec,
            ) || !host_value_compatible(
                target.geometry.playback_rate,
                edit.before.playback_rate as f64,
            ))
        {
            return Err(format!(
                "host source window/rate changed before GUI commit; refresh required (host start={:.9} rate={:.9}, editor start={:.9} rate={:.9})",
                host_forward_window_start(
                    &target.geometry,
                    hifishifter_kernel::state::clip_source_media_duration_sec(&edit.before),
                )
                .unwrap_or(target.geometry.source_start_sec),
                target.geometry.playback_rate,
                edit.before.source_start_sec,
                edit.before.playback_rate as f64,
            ));
        }
        let destination = if edit.source_track == edit.destination_track {
            None
        } else {
            let clip = {
                let timeline = document.timeline.lock_or_recover();
                timeline
                    .as_ref()
                    .ok_or("host timeline unavailable")?
                    .clips
                    .iter()
                    .find(|clip| clip.track_id == edit.destination_track)
                    .map(|clip| clip.id.clone())
                    .ok_or("target track has no directly bound ARA item")?
            };
            Some(owner.host_edit_target(&clip)?)
        };
        if targets
            .first()
            .is_some_and(|(_, first, _)| !first.same_project(&target))
        {
            return Err("host batch spans multiple projects".into());
        }
        if destination
            .as_ref()
            .is_some_and(|dest| !target.same_project(dest))
        {
            return Err("target track belongs to another project".into());
        }
        // ── 淡变形状 / 曲率 / S 参数 ────────────────────────────────────────
        //
        // 【落到哪个宿主字段取决于宿主版本】官方头文件把两套轴标成互补区间
        // （`sdk/reaper_plugin_functions.h`）：
        //   `C_FADE*SHAPE` / `D_FADE*DIR`      —— v7.80 and earlier
        //   `D_FADE*DIR_NEW` / `D_FADE*DIR2_NEW` —— v7.81 and later
        // 所以这里**必须**按 `fade_axes_v2` 分流，而不是按模式或猜。
        if requests_fade_axes(&edit.patch) {
            // 委托（宿主把边界包络交给 HFS 渲染）时才存 HFS 自有样式；
            // 否则直接写宿主自己的轴。此前这里**无条件**要求委托 ——
            // 而 REAPER 从不委托，于是整条形状编辑路径是死的。
            let delegated = {
                let key = {
                    let ids = document.clip_ids.lock_or_recover();
                    ids.iter()
                        .find(|(_, id)| id.as_str() == edit.native_id)
                        .map(|(key, _)| *key)
                        .ok_or("fade region identity missing")?
                };
                document
                    .regions
                    .lock_or_recover()
                    .get(&key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
            };
            if delegated {
                // 【为什么委托时不校验轴语义】委托时形状由 **HFS 自己**渲染
                // （`FadeStyle` 进插件状态），宿主轴根本不参与 —— 拿宿主版本去
                // 拦它会把一条本来能用的路径堵死。但 S 参数在 `FadeStyle` 里没有
                // 位置，所以仍然拒绝，而不是静默丢弃。
                if edit.patch.fade_in_s.is_some() || edit.patch.fade_out_s.is_some() {
                    return Err("fade S parameter is not part of the HiFiShifter fade style".into());
                }
                let mut style = document
                    .edits
                    .lock_or_recover()
                    .fades
                    .get(&target.geometry.item_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(value) = edit.patch.fade_in_shape {
                    style.in_shape = value;
                }
                if let Some(value) = edit.patch.fade_out_shape {
                    style.out_shape = value;
                }
                if let Some(value) = edit.patch.fade_in_dir {
                    style.in_dir = value;
                }
                if let Some(value) = edit.patch.fade_out_dir {
                    style.out_dir = value;
                }
                style.validate()?;
                styles.insert(target.geometry.item_id.clone(), style);
            } else {
                // 直接写宿主轴：这里才需要轴语义校验（版本未知/新轴不接受预设）。
                validate_fade_axes(target.geometry.fade_axes_v2, &edit.patch)?;
            }
        }
        // 插件自有、宿主表达不到的逐片段属性：只登记，不写宿主。
        // 键用 `edit.native_id`（形如 `ara-item-{guid}`）—— 与 `EditState::apply` 里
        // `clip.id` 的口径一致，冷重建后仍能对上（`fades` 用裸 GUID 是因为它走几何查表，
        // 两套键口径不同，不要互相套用）。
        if edit.patch.color.is_some() || edit.patch.formant_morph.is_some() {
            let entry = locals
                .entry(edit.native_id.clone())
                .or_insert_with(crate::state_channel::ClipLocalState::default);
            if let Some(color) = &edit.patch.color {
                entry.color = Some(color.clone());
            }
            if let Some(morph) = &edit.patch.formant_morph {
                entry.formant_morph = Some(morph.clone());
            }
        }
        // ── 方向（倒放）结构写回：预检 ──────────────────────────────────────
        // 【为什么这里只做纯计算】真正的 chunk 读写放在写循环**末尾**：它必须读
        // "位置/长度/速率都写完"之后的那份原文，否则会把刚写的数值覆盖回去。
        let reverse_plan = if edit.before.reversed != edit.after.reversed {
            if !target.can_reverse_clips() {
                return Err("the host cannot write the take direction".into());
            }
            if !target.geometry.markers.is_empty() {
                return Err("nonlinear host stretch-marker editing is not supported".into());
            }
            // 【镜像轴取两侧读数的较小者】读取侧 `take_axis_sec` 对"段末端"与"ARA
            // 源时长"取 min；写回若用较大的那个，读回来的锚点会差一个 `D − ara`
            // （容器头级别的几毫秒 ⇒ 内容轻微错位）。两侧同一口径即可逐位互逆。
            let media = match (
                hifishifter_kernel::state::clip_source_media_duration_sec(&edit.after),
                target.geometry.source_length_sec,
            ) {
                (Some(ara), Some(host)) => Some(ara.min(host)),
                (ara, host) => ara.or(host),
            };
            Some(host_take_direction(&edit.after, media)?)
        } else {
            None
        };
        // ── 逐 take 写回：预检 ──────────────────────────────────────────────
        // 【为什么只验能力位、不解析指针】兄弟 take 的指针与几何要到写循环里才
        // 解析（写前最新的宿主读数才可信）；这里先把"这个宿主做不做得到"判掉，
        // 让注定失败的批次在碰第一个 setter 之前就整条拒绝。
        if edit.take_direction.is_some() && !target.can_reverse_clips() {
            return Err("the host cannot write the take direction".into());
        }
        if edit.take_channel_mode.is_some() && !target.host_can_write_take_fields() {
            return Err("the host cannot write take fields".into());
        }
        reverse_plans.push(reverse_plan);
        targets.push((edit, target, destination));
    }
    let undo = if managed {
        None
    } else {
        Some(targets[0].1.begin_undo(&authorized)?)
    };
    // 已开始写入的片段。**在写入之前**登记：写入中途失败时，最后一个片段可能只写了
    // 一半（例如位置已改、长度还没改），它同样必须被回滚。
    let mut applied: Vec<(&ClipEdit, &HostClipTarget)> = Vec::new();
    // 已落地的**结构**写回：回滚按同一条路撤（chunk 写回原文 / 动作再 toggle 一次）。
    let mut applied_structural: Vec<(HostClipTarget, DirectionWrite)> = Vec::new();
    // 已落地的**兄弟 take 声道模式**写回：回滚按写入前的旧值写回
    // （take 指针、I_CHANMODE 旧值）。与 `applied` 分开——它们写的是另一条 take。
    let mut applied_take_modes: Vec<(HostClipTarget, usize, i32)> = Vec::new();
    let result: Result<(), String> = (|| {
        for (index, (edit, target, destination)) in targets.iter().enumerate() {
            applied.push((edit, target));
            let before = &edit.before;
            let after = &edit.after;
            // 写速率时明确保调；普通移动/裁切不擅自改变用户B_PPITCH。
            if (before.playback_rate - after.playback_rate).abs() > 1e-6 {
                target.set_take(c"B_PPITCH", 1., &authorized)?;
                // 【为什么保留符号】负速率是 REAPER 表达倒放的**方式之一**（另一个是
                // `SECTION` 的 `MODE` 反向位）。此前这里恒写正值 —— 用户只要改一次速率，倒放就被
                // 静默改成正放。符号取自宿主的原始读数，不由 clip 推断。
                let sign = if target.geometry.play_rate_signed < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                target.set_take(
                    c"D_PLAYRATE",
                    sign * after.playback_rate as f64,
                    &authorized,
                )?;
            }
            // 【倒放必须写**镜像**起点】宿主 `D_STARTOFFS` 在倒放 take 上是
            // `D − 正向终点`（F-4 实测），而 clip 持有的是镜像复原之后的**正向**窗口。
            //
            // 【本次改方向的 clip 在这里不写】SOFFS 与 SECTION 是同一份计算的两个分量
            // （段内偏移 vs 绝对坐标），分开写会互相覆盖 —— 交给下面的结构写回一次写完。
            if reverse_plans[index].is_none() {
                let after_offs = host_start_offset(after)
                    .ok_or("cannot write a reversed take without its media duration")?;
                if host_start_offset(before).is_none_or(|b| (b - after_offs).abs() > 1e-9) {
                    target.set_take(c"D_STARTOFFS", after_offs, &authorized)?;
                }
            }
            if (before.start_sec - after.start_sec).abs() > 1e-9 {
                target.set_item(c"D_POSITION", after.start_sec, &authorized)?;
            }
            if (before.length_sec - after.length_sec).abs() > 1e-9 {
                target.set_item(c"D_LENGTH", after.length_sec, &authorized)?;
            }
            if let Some(value) = edit.patch.muted {
                target.set_item(c"B_MUTE", if value { 1. } else { 0. }, &authorized)?;
            }
            // Item 增益 → 宿主 item 的 D_VOL（用户在时间轴旋钮上拧的那一层）。
            if let Some(value) = edit.patch.item_gain {
                target.set_item(c"D_VOL", value as f64, &authorized)?;
            }
            // Take 增益 → 宿主 take 的 D_VOL（take 属性面板那一层）。
            //
            // 【为什么必须分开写】两者在 REAPER 里是两个字段、两个旋钮。此前只有
            // 一个 `gain` 且写向 item，take 侧永远写不到；反过来（早期）又把一个
            // 值写进 take，用户在时间轴拧的旋钮就跑到 take 上去了。
            if let Some(value) = edit.patch.gain {
                target.set_take(c"D_VOL", value as f64, &authorized)?;
            }
            if let Some(enabled) = edit.patch.loop_enabled {
                // 双向：`B_LOOPSRC` 是 item 属性，读回来的就是它（见 `HostClipGeometry`）。
                target.set_item(c"B_LOOPSRC", if enabled { 1. } else { 0. }, &authorized)?;
            }
            if let Some(value) = edit.patch.channel_mode {
                if !(0..=4).contains(&value) {
                    return Err("invalid host channel mode".into());
                }
                target.set_take(c"I_CHANMODE", value as f64, &authorized)?;
            }
            if let Some(value) = edit.patch.host_group_id {
                if !(0..=HOST_GROUP_ID_MAX).contains(&value) {
                    return Err("invalid host item group id".into());
                }
                target.set_item(c"I_GROUPID", value as f64, &authorized)?;
            }
            if let Some(value) = edit
                .patch
                .name
                .as_deref()
                .filter(|value| *value != before.name)
            {
                target.set_take_name(value, &authorized)?;
            }
            if let Some(value) = edit.patch.snap_offset_sec {
                target.set_item(c"D_SNAPOFFSET", value, &authorized)?;
            }
            for (value, name) in [
                (edit.patch.fade_in_sec, c"D_FADEINLEN"),
                (edit.patch.fade_out_sec, c"D_FADEOUTLEN"),
                (edit.patch.auto_fade_in_sec, c"D_FADEINLEN_AUTO"),
                (edit.patch.auto_fade_out_sec, c"D_FADEOUTLEN_AUTO"),
            ] {
                if let Some(value) = value {
                    target.set_item(name, value, &authorized)?;
                }
            }
            if let Some(destination) = destination {
                target.move_to(destination, &authorized)?;
            }
            // 【为什么分两种情形】委托时（宿主把边界包络交给 HFS 渲染）宿主只留默认
            // 样式，HFS 的 shape/dir 进插件状态；**不委托时直接写宿主自己的轴** ——
            // 那才是让用户"能调淡变类型和曲率"的那条路。
            if styles.contains_key(&target.geometry.item_id) {
                // REAPER只留默认样式；自定义shape/dir进入HFS状态，而不是宿主c/S轴。
                match target.geometry.fade_axes_v2 {
                    Some(true) => {
                        for (name, value) in [
                            (c"D_FADEINDIR_NEW", target.geometry.fade_in_dir_new),
                            (c"D_FADEOUTDIR_NEW", target.geometry.fade_out_dir_new),
                            (c"D_FADEINDIR2_NEW", target.geometry.fade_in_dir2_new),
                            (c"D_FADEOUTDIR2_NEW", target.geometry.fade_out_dir2_new),
                        ] {
                            if value != 0. {
                                target.set_item(name, 0., &authorized)?;
                            }
                        }
                    }
                    Some(false) => {
                        for (name, value) in [
                            (c"C_FADEINSHAPE", target.geometry.fade_in_shape),
                            (c"C_FADEOUTSHAPE", target.geometry.fade_out_shape),
                            (c"D_FADEINDIR", target.geometry.fade_in_dir),
                            (c"D_FADEOUTDIR", target.geometry.fade_out_dir),
                        ] {
                            if value != 0. {
                                target.set_item(name, 0., &authorized)?;
                            }
                        }
                    }
                    None => {
                        return Err(
                            "host fade version unavailable; cannot normalize default style".into(),
                        )
                    }
                }
            } else if let Some(axes_new) = target.geometry.fade_axes_v2 {
                // 直接写宿主轴。`fade_axes_v2` 已在上面的校验里排除 `None`，
                // 这里再取一次是为了拿到值（校验阶段只做了拒绝）。
                if axes_new {
                    // 【形状预设要写成**一对**分量】≥7.81 没有"形状号"这个可写状态，
                    // 形状由 `(curvature, S)` 决定；七个预设各自对应一组确定的
                    // 坐标（实测表见 `hifishifter_kernel::fade_axes`）。所以写形状
                    // 就是同时写两个分量。
                    //
                    // 【同一次补丁里形状压过 dir】界面切换预设时会**同时**送
                    // `fadeInShape` 与该形状的默认 `fadeInDir`（HFS 自己的曲率轴，
                    // `DEFAULT_FADE_DIR_BY_SHAPE`）。但两套轴不是同一套参数化 ——
                    // 实测旧 `D_FADEINDIR` 与新 `D_FADEINDIR_NEW` 互相读回都是被重
                    // 映射过的值。若让 HFS 的 dir
                    // 覆盖预设的 curvature，用户点了"轻微凸"会得到线性
                    // （预设 1 = `c 0.5`，而 HFS 给该形状的默认 dir 是 0）。
                    // 曲率滑杆走的是**只带 dir、不带 shape** 的补丁，不受此规则影响。
                    for (shape, dir, s, dir_key, s_key) in [
                        (
                            edit.patch.fade_in_shape,
                            edit.patch.fade_in_dir,
                            edit.patch.fade_in_s,
                            c"D_FADEINDIR_NEW",
                            c"D_FADEINDIR2_NEW",
                        ),
                        (
                            edit.patch.fade_out_shape,
                            edit.patch.fade_out_dir,
                            edit.patch.fade_out_s,
                            c"D_FADEOUTDIR_NEW",
                            c"D_FADEOUTDIR2_NEW",
                        ),
                    ] {
                        if let Some(shape) = shape {
                            let (curvature, preset_s) =
                                hifishifter_kernel::fade_axes::host_fade_preset_axes(shape)
                                    .ok_or_else(|| format!("unknown fade shape {shape}"))?;
                            target.set_item(dir_key, curvature, &authorized)?;
                            // 【为什么补丁自带的 S 优先】此前这里直接写预设的 S 并
                            // `continue`，把同一补丁里的 `fade_*_s` **静默丢掉**。补丁
                            // 显式给了 S 就是用户/调用方更具体的要求，按它写；没给才用
                            // 预设自带的 S。收据（`receipt`）按同一口径比对。
                            target.set_item(s_key, s.unwrap_or(preset_s), &authorized)?;
                            continue;
                        }
                        if let Some(value) = dir {
                            target.set_item(dir_key, value, &authorized)?;
                        }
                        if let Some(value) = s {
                            target.set_item(s_key, value, &authorized)?;
                        }
                    }
                } else {
                    for (value, name) in [
                        (edit.patch.fade_in_shape, c"C_FADEINSHAPE"),
                        (edit.patch.fade_out_shape, c"C_FADEOUTSHAPE"),
                        (edit.patch.fade_in_dir, c"D_FADEINDIR"),
                        (edit.patch.fade_out_dir, c"D_FADEOUTDIR"),
                    ] {
                        if let Some(value) = value {
                            target.set_item(name, value, &authorized)?;
                        }
                    }
                }
            }
            if let Some(direction) = &reverse_plans[index] {
                // 结构写回**最后**做：它读的是"位置/长度/速率都写完"之后的那份原文。
                let write = write_take_direction(target, direction, &authorized)?;
                applied_structural.push((target.clone(), write));
            }
            if let Some(edit) = &edit.take_direction {
                // 兄弟 take 的方向写回：同一"最后做"的纪律（chunk 读到的是本批次
                // 数值已生效的原文）。回滚走同一条 applied_structural 撤链。
                let write = write_sibling_take_direction(target, edit, &authorized)?;
                applied_structural.push((target.clone(), write));
            }
            if let Some(edit) = &edit.take_channel_mode {
                // 兄弟 take 的声道模式：字段 setter + 写→读→比。旧值进
                // applied_take_modes，整批失败时按原值写回。
                let (take, old) = set_take_channel_mode_for_take(target, edit, &authorized)?;
                applied_take_modes.push((target.clone(), take, old));
            }
            target.update(&authorized)?;
        }
        Ok(())
    })();
    if result.is_ok() && !styles.is_empty() {
        {
            let _transaction = document.transaction.lock_or_recover();
            let mut edits = document.edits.lock_or_recover();
            edits.fades.extend(styles);
            edits.revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
        }
        for owner in document.renderer_owners() {
            owner.prepare();
        }
    }
    // 本地属性与几何写回同一提交点：宿主写回失败时一并回滚（`locals` 被丢弃）。
    // 这里**不** bump revision —— 几何已经 bump 过；只带本地属性的批次不写宿主，
    // 但仍要重新投影，所以放在同一事务里 merge 后由下面统一 `prepare()`。
    if result.is_ok() && !locals.is_empty() {
        let changed = {
            let _transaction = document.transaction.lock_or_recover();
            let mut edits = document.edits.lock_or_recover();
            let mut changed = false;
            for (item, local) in locals {
                let entry = edits.clip_local.entry(item).or_default();
                if *entry != local {
                    *entry = local;
                    changed = true;
                }
            }
            if changed {
                edits.revision = edits
                    .revision
                    .checked_add(1)
                    .ok_or("edit revision exhausted")?;
            }
            changed
        };
        if changed {
            for owner in document.renderer_owners() {
                owner.prepare();
            }
        }
    }
    let result = match result {
        Ok(()) => Ok(()),
        Err(error) => {
            // 【为什么必须回滚】写循环是逐片段提交的：第 N 个失败时前 N−1 个已经落到
            // 宿主上。若不回滚，用户看到的是"部分片段移动了 + 一条拒绝"，而那句
            // "被拒绝"会被理解成"什么都没发生" —— 两者的差别正是用户报障里最刺眼的
            // 部分（编辑明明生效了却报错）。回滚让"拒绝"与"没改"重新成为同一件事。
            //
            // 【先撤结构、再撤数值】结构写回的是**整份 chunk**，把它写回原文就等于连
            // 该 clip 的数值编辑一起撤销；随后只对"没被结构覆盖"的 clip 做数值回滚，
            // 否则会把刚复原的原文再按镜像域改写一遍。
            let mut rollback_errors: Vec<String> = Vec::new();
            for (target, write) in applied_structural.iter().rev() {
                let rollback = match write {
                    DirectionWrite::Chunk { original } => {
                        target.set_item_state_chunk(original, &authorized)
                    }
                    DirectionWrite::Action { before, take } => {
                        // 【take 在哪条上就撤哪条】绑定 take 用原入口；兄弟 take 的
                        // 方向动作要按同一条 take 的指针再 toggle 回去 —— 撤到
                        // active take 上等于把用户没动过的 take 翻转一遍。
                        if *take == target.bound_take() {
                            target.apply_take_direction(*before, &authorized)
                        } else {
                            target.apply_take_direction_on(*take, *before, &authorized)
                        }
                    }
                };
                if let Err(rollback) = rollback {
                    rollback_errors.push(rollback);
                }
            }
            // 兄弟 take 的声道模式按旧值写回。它写的是另一条 take，不受上面的
            // chunk 回滚影响（chunk 回滚整份原文反而已经把它一起撤了 —— 因此
            // 只对**没有** chunk 写回的 item 撤）。
            for (target, take, old) in applied_take_modes.iter().rev() {
                let covered_by_chunk = applied_structural.iter().any(|(done, write)| {
                    matches!(write, DirectionWrite::Chunk { .. }) && done.same_item(target)
                });
                if covered_by_chunk {
                    continue;
                }
                if let Err(rollback) =
                    target.set_take_pointer_value(*take, c"I_CHANMODE", *old as f64, &authorized)
                {
                    rollback_errors.push(rollback);
                }
            }
            let numeric: Vec<(&ClipEdit, &HostClipTarget)> = applied
                .iter()
                .filter(|(_, target)| {
                    // chunk 写回的是**整份** item，已经连数值一起撤销；动作那条只换了方向，
                    // 位置/长度/SOFFS 仍要按几何逐字段回滚。
                    !applied_structural.iter().any(|(done, write)| {
                        matches!(write, DirectionWrite::Chunk { .. }) && done.same_item(target)
                    })
                })
                .copied()
                .collect();
            if let Err(rollback) = restore_applied_geometry(&numeric, &authorized) {
                rollback_errors.push(rollback);
            }
            if rollback_errors.is_empty() {
                Err(format!(
                    "host clip edit rolled back; nothing was changed ({error})"
                ))
            } else {
                Err(format!(
                    "host clip edit failed and could not be rolled back; use host Undo ({error}; rollback: {})",
                    rollback_errors.join("; ")
                ))
            }
        }
    };
    drop(undo);
    // 【写回成功 ≠ 完成门能收口】数值/结构写回只落在**宿主**上；完成门比对的是
    // **编辑器时间线**，而它的 take 级事实（reversed / loop / channel_mode）只有一条
    // 更新通路：原生清单 `ui_tracks` 重发布 → `ui_geometry_revision` 推进 →
    // `refresh_ui_geometry` 投影。那条通路被三道节流按住：
    //   * 清单早退"令牌未变 + 距上次采集 < 1 s"（方向位**不**推进宿主的
    //     `GetProjectStateChangeCount`，令牌不变是常态）；
    //   * 定时器 100 ms 枚举节流（忙时静默跳过）；
    //   * `host_query_transaction` 在后台事务持有文档时直接放弃本轮。
    // 三道叠加的真机后果（23 次现场失败）：完成门把预算（24 次退避 ≈ 10.6 s）耗尽，
    // 而 REAPER 的 ARA 文档再同步在 +11.2 s 才到 —— 用户看到
    // "clip was edited in the host but its ARA geometry has not caught up"，
    // 乐观更新被回滚，之后由几何刷新再"追认"一遍。剪贴板写回路径
    // （`execute_paste`/`execute_delete`）早已用 `refresh_host_transport()`
    // 同步拉回新几何；几何写回恰恰是最需要它的（它本身就是几何编辑），漏掉它是缺口。
    //
    // 【为什么清 `ui_inventory_sampled_at` 而不是只调刷新】`refresh_host_transport`
    // 里的清单刷新会被"刚采过样"早退拦下 —— 而写回刚把宿主改了，那份"刚采的样"
    // 恰恰是写前快照。清掉采样时刻，下一轮枚举必然执行；方向位变化会走
    // `facts_changed` → 立即发布 → `ui_geometry_revision` 推进 → 完成门在下一次
    // `get_timeline_state` 就能对上。
    if result.is_ok() {
        if let Ok(document) = owner.editor_document() {
            // 【为什么收敛到这一个入口】写回成功只落在**宿主**上；完成门比对的却是
            // **编辑器时间线**。清单采样时刻、几何缓存、呈现代次三者必须一起失效 ——
            // 少任何一个，`HostEditReceipt` 都会读到写前那份几何，把一次成功的写回
            // 报成"宿主没跟上"（用户看到的正是"淡化/自动交叉淡化没生效"）。
            ExtensionOwner::note_host_write_completed(&document);
        }
        owner.refresh_host_transport();
    }
    result.map_err(|error| format!("host clip edit failed (Undo block retained): {error}"))
}

/// 一次方向写回实际走了哪条路。
///
/// 【为什么要区分】两条路的回滚方式不同：chunk 写回的是一份**整 item** 的原文（连本批次
/// 刚写的数值一起撤销），而宿主动作只换了方向，数值仍要按几何逐字段回滚。混着撤会把
/// 已经复原的字段再按镜像域改写一遍。
enum DirectionWrite {
    /// item chunk 整份写回；`original` 就是回滚内容。
    Chunk { original: String },
    /// 宿主自己的动作；回滚时按 `before` 再 toggle 一次。`take` 记录动作作用在
    /// **哪一条** take 上（绑定 take 或兄弟 take）—— 回滚必须撤同一条，撤错了
    /// 就是把用户没动过的 take 翻转一遍。
    Action { before: bool, take: usize },
}

/// 把一条 take 的播放方向写到宿主上。
///
/// 【两条路】
/// 1. **item chunk 整份改写**（[`super::source_direction::rewrite_take_direction`]）——
///    逐字节保留除目标 take 的 `SOURCE`/`SOFFS` 之外的一切，是多 take item 上唯一能
///    "只改这一条 take"的写法；
/// 2. **宿主自己的 `Item properties: Toggle take reverse`** —— 宿主**不允许**扩展重建
///    带在线 ARA FX 的 item（真机取证见 [`super::take_env`]），而 take FX 放置下那个
///    item 恰好就是挂着本插件的那个。这条路写出的结构与第 1 条逐字节一致，因此两者在
///    模型看来是同一种状态。
///
/// 【选择顺序】先看"这个 item 是不是本实例自己挂着的"（放置位置缓存，不花宿主调用）：
/// 是就**直接**走动作，不做那次注定被拒的几百 KB 读写。不是的话先走 chunk（精确到 take），
/// 万一被拒（同工程里可能还有别的 ARA 实例）再退到动作。
///
/// 【写完必须核对】`SECTION MODE` 的取值属于宿主，仓库里没有第二份证据能替代这次实测。
/// 不符就返回错误 —— 调用方整批回滚，"点了没反应"不会伪装成成功。
fn write_take_direction(
    target: &HostClipTarget,
    direction: &super::source_direction::DirectionEdit,
    authorized: &impl Fn() -> bool,
) -> Result<DirectionWrite, String> {
    let expected = direction.section.is_some();
    if !target.hosts_this_instance() {
        let original = target.item_state_chunk(authorized)?;
        let rewritten = super::source_direction::rewrite_take_direction(
            &original,
            &target.geometry.take_id,
            direction,
        )?;
        match target.set_item_state_chunk(&rewritten, authorized) {
            Ok(()) => {
                if target.reversed_now(authorized)? != expected {
                    return Err("the host did not apply the requested take direction".into());
                }
                return Ok(DirectionWrite::Chunk { original });
            }
            Err(error) if error == super::write::ITEM_STATE_REFUSED => {
                crate::log_line(&format!(
                    "[ara] the host refused the item state ({error}); \
                     writing the take direction through its own action"
                ));
            }
            Err(error) => return Err(error),
        }
    }
    let before = target.reversed_now(authorized)?;
    target.apply_take_direction(expected, authorized)?;
    // 【SOFFS 由模型写死】宿主的 toggle 会自己镜像 `D_STARTOFFS`，但镜像基准是**它**手上
    // 的旧值（本批次还没写这一条）。写死成 `DirectionEdit` 算出来的那个，往返才不依赖
    // 宿主此刻的状态 —— 真机实测这个 setter 在倒放段上照样可写。
    target.set_take(c"D_STARTOFFS", direction.s_offs, authorized)?;
    if target.reversed_now(authorized)? != expected {
        return Err("the host did not apply the requested take direction".into());
    }
    Ok(DirectionWrite::Action {
        before,
        take: target.bound_take(),
    })
}

/// 把已写入的片段**按编辑前的几何**写回去。
///
/// 范围刻意限定为几何（位置 / 长度 / 源起点）：批量移动只改这些，回滚它们才能让
/// "拒绝 = 没改"成立。音量与淡变**不回滚** —— 在 item / take 两层增益模型落地前，
/// 它们的"原值"是含糊的（patch.gain 目前写 item 的 D_VOL，而 before.gain 是 take 级），
/// 用错的来源写回去比不写更糟。这类失败仍会如实报告，并保留 Undo 块。
fn restore_applied_geometry(
    applied: &[(&ClipEdit, &HostClipTarget)],
    authorized: &impl Fn() -> bool,
) -> Result<(), String> {
    let mut first_error: Option<String> = None;
    for (edit, target) in applied.iter().rev() {
        let before = &edit.before;
        let restore = (|| -> Result<(), String> {
            if let Some(offset) = host_start_offset(before) {
                target.set_take(c"D_STARTOFFS", offset, authorized)?;
            }
            target.set_item(c"D_POSITION", before.start_sec, authorized)?;
            target.set_item(c"D_LENGTH", before.length_sec, authorized)?;
            target.update(authorized)
        })();
        if let Err(error) = restore {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    /// clip音量写真实item D_VOL，支持零；不能写take音量/极性或靠本地gain冒充成功。
    #[test]
    fn host_clip_gain_writes_item_volume_and_preserves_take_polarity() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
        host.set_take_volume(-0.25);
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
        for gain in [0.5, 0.] {
            let plan = editor
                .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"itemGain":gain}))
                .unwrap();
            let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
            host.reset();
            execute(&owner, plan, || document.is_alive()).unwrap();
            let native = clip.id.strip_prefix(&editor.namespace).unwrap();
            let g = owner.host_edit_target(native).unwrap().geometry;
            assert_eq!(g.item_gain, gain);
            assert_eq!(g.take_gain, -0.25);
            assert!(host.calls().contains(&"write-item:D_VOL".into()));
            assert!(!host.calls().iter().any(|call| call == "write-take:D_VOL"));
            // 载荷按 `TimelineClip` 的真实形状写：它同时带 `clip_gain`（item 层）与
            // `gain`（active take 的投影）。本次只改 item 层，因此 `gain` 保持夹具原值。
            let payload = json!({"clips":[{"id":clip.id,"start_sec":clip.start_sec,"length_sec":clip.length_sec,
                "source_start_sec":clip.source_start_sec,"playback_rate":clip.playback_rate,"track_id":clip.track_id,
                "clip_gain":gain,"gain":clip.gain,
                // 真实载荷**总是**带这两项（`decorate_host_*` 从几何与 take 读），
                // 完成门也按这两项判方向/循环源 —— 夹具缺了它们就不再是"回声"。
                "reversed":clip.reversed,"loop_enabled":clip.loop_enabled}]});
            assert!(receipt.matches(&payload), "{receipt:?} vs {payload}");
            let mut stale = payload;
            stale["clips"][0]["clip_gain"] = json!(1.);
            assert!(!receipt.matches(&stale), "item 增益的陈旧回显必须被门拦下");
        }
        // 负增益没有意义（反相是宿主的 take 极性，不是增益），必须拒绝。
        assert!(editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"gain":-1.}))
            .is_err());
        // **+24 dB（线性 16）现在必须被接受** —— 这正是值域扩大的目的：REAPER 的
        // item 增益上限是 +24.08 dB，此前上界是 4.0（+12 dB），于是任何超过 +12 dB
        // 的 clip 在写回宿主时都会被整条拒绝。
        //
        // 【为什么直接调 `patch` 而不是 `plan_host_edit`】这里要钉的是**值域判据**，
        // 而 `plan_host_edit` 会顺带走一遍宿主几何查询（那是另一件事，且已由上面
        // 那个循环覆盖）。只验判据能让这条断言在没有任何宿主夹具的情况下成立。
        assert!(patch(&json!({"clipId":clip.id,"gain":16.})).is_ok());
        // 超出宿主自己的上限才拒绝。
        assert!(patch(&json!({"clipId":clip.id,"gain":17.})).is_err());
        document.close();
    }
    /// **回归：陈旧回声必须在交给前端之前被 `after` 修正，且不得波及没写过的位。**
    ///
    /// 【现场】ARA 里调 item 增益：REAPER 的 item 收到了，HiFiShifter 的 Clip 却回弹。
    /// 这条缺陷有两个成因，缺一都修不干净：
    ///   1. 载荷里前端读的 `clip_gain` 从不被宿主投影写出来（见
    ///      `workspace::the_item_gain_the_ui_reads_is_projected_into_the_payload`）；
    ///   2. 完成门把"载荷回声"当成**写入成功**的判据，于是回声一旧就退避重试、最后
    ///      回一个终态错误，前端按 `rejectWithValue` 回滚乐观更新。
    ///
    /// 这里钉住的是第 2 条的补救方式：`overlay` 只按 `after` 覆盖**补丁写过的那几位**，
    /// 于是前端整份应用到的载荷与写入一致；没写过的位（例如宿主吸附调整过的
    /// `start_sec`）继续信宿主，不会被我们的快照碾掉。
    #[test]
    fn a_stale_echo_is_repaired_before_the_ui_applies_it() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"itemGain":1.7}))
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        // 回声：`clip_gain` 还停在旧值；`start_sec` 与模型不同（宿主把它吸附到了网格点）。
        let host_snapped_start = clip.start_sec + 0.125;
        let mut payload = json!({"clips":[{
            "id":clip.id,
            "start_sec":host_snapped_start,
            "length_sec":clip.length_sec,
            "source_start_sec":clip.source_start_sec,
            "playback_rate":clip.playback_rate,
            "track_id":clip.track_id,
            "clip_gain":1.0,
            "gain":clip.gain,
            "reversed":clip.reversed,
            "loop_enabled":clip.loop_enabled,
        }]});
        let diverged = receipt.divergences(&payload);
        assert!(
            diverged.iter().any(|line| line.contains("clip_gain")),
            "陈旧回显必须被**看见**（诊断），实际：{diverged:?}"
        );
        receipt.overlay(&mut payload);
        // `clip_gain` 在内核里是 f32，1.7 落到 JSON 是 1.700000047…；用容差比。
        let written = payload["clips"][0]["clip_gain"].as_f64().unwrap();
        assert!(
            (written - 1.7).abs() < 1e-6,
            "前端读的就是这一栏：载荷必须带着我们写下去的值（实际 {written}）"
        );
        assert_eq!(
            payload["clips"][0]["start_sec"],
            json!(host_snapped_start),
            "本次补丁没写过 start_sec，宿主真值不得被快照覆盖"
        );
        // 覆盖之后，"本次写过的位"不得再有分歧；剩下的分歧只允许是**没写过**的宿主管辖
        // 字段（这里是被宿主吸附过的 `start_sec`）—— 那正是"回声只用于诊断"的意思。
        let remaining = receipt.divergences(&payload);
        assert!(
            !remaining.iter().any(|line| line.contains("clip_gain")),
            "写过的位覆盖之后不得再有分歧：{remaining:?}"
        );
        assert!(
            remaining.iter().all(|line| line.contains("start_sec")),
            "覆盖只碰本次写过的位：{remaining:?}"
        );
        document.close();
    }

    /// S1：编辑器时间线被 panic 毒化后，**不得**拿半写状态去改宿主 —— 规划必须拒绝。
    #[test]
    fn a_poisoned_editor_timeline_refuses_host_edits() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip_id = editor.timeline.lock_or_recover().clips[0].id.clone();
        // 在另一线程持锁 panic ⇒ 毒化时间线锁（模拟"某线程在写时间线时崩了"）。
        let poisoned = Arc::clone(&editor);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.timeline.lock_or_recover();
            panic!("poison the editor timeline");
        })
        .join();
        assert!(editor.timeline.is_poisoned(), "前提：时间线锁被毒化");
        let error =
            match editor.plan_host_edit("set_clip_state", &json!({"clipId":clip_id,"gain":0.5})) {
                Ok(_) => panic!("毒化后必须拒绝写回宿主"),
                Err(error) => error,
            };
        assert!(error.contains("refresh required"), "{error}");
    }
    /// 规划复用原Clip×Take倍率，且未开始宿主写之前整批拒绝非法/未知字段。
    #[test]
    fn host_edit_planning_preserves_effective_rate_and_rejects_unknown_batch_fields() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"lengthSec":1.,"clipPlaybackRate":2.}),
            )
            .unwrap();
        assert_eq!(plan.edits[0].after.playback_rate, 2.);
        assert_eq!(plan.edits[0].after.length_sec, 1.);
        // 【倒放不再是"规划期拒绝"】它以前在这里被拒，理由是"本模块只写数值字段，
        // 建不了 `SOURCE SECTION`"。方向现在走**结构**写回（`source_direction`），
        // 能不能写由执行期的宿主能力（`can_reverse_clips`）判定，因此规划阶段接受它。
        assert!(editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"reversed":false})
            )
            .is_ok());
        // 循环源**可以**规划：`B_LOOPSRC` 是纯 item 属性，内核语义与它一致。
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"loopEnabled":true})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"loopEnabled":false})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostGroupId":1234})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"name":"renamed active take"})
            )
            .is_ok());
        assert!(editor.plan_host_edit("set_clips_state_bulk",&json!({"updates":[{"clipId":clip.id,"startSec":1.},{"clipId":"other","lengthSec":-1.}]})).is_err());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":99.})
            )
            .is_err());
        assert_eq!(
            editor.timeline.lock_or_recover().clips[0].start_sec,
            clip.start_sec,
            "纯规划不得修改actor私有几何"
        );
        editor.close();
    }
    /// **Loop clip 的 slip 提交不得按区间算式反推长度**。
    ///
    /// 【为什么这是回归】循环源的 `source_start_sec / source_end_sec` 是**回绕相位
    /// 锚点**（前端 `slipWindow` 对两端取模回绕，回绕后 end < start 完全合法）。
    /// 此前规划器用 `(end − start) / rate` 反推长度，对回绕窗口必然得出负数 ⇒
    /// 一次 slip 提交整批被拒："source trim window must have positive length"。
    /// 修复后 Loop 分支不动长度（长度只能来自补丁或编辑前的值）。
    #[test]
    fn a_loop_slip_with_a_wrapped_window_keeps_its_length() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        // 回绕窗口：媒体长 2.0，slip 把窗口推过文件头，end(0.4) < start(1.6)。
        // 前端 `computeSlipWindow` 的取模输出就长这样。
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({
                    "clipId": clip.id,
                    "loopEnabled": true,
                    "sourceStartSec": 1.6,
                    "sourceEndSec": 0.4
                }),
            )
            .expect("loop 的回绕窗口必须被接受");
        let edit = &plan.edits[0];
        assert!(edit.after.loop_enabled, "补丁打开循环源");
        assert!(
            (edit.after.length_sec - clip.length_sec).abs() < 1e-9,
            "slip 不改长度：before={:?} after={:?}",
            clip.length_sec,
            edit.after.length_sec
        );
        assert!(
            (edit.after.source_start_sec - 1.6).abs() < 1e-9
                && (edit.after.source_end_sec - 0.4).abs() < 1e-9,
            "回绕锚点必须原样入库：{:?}",
            (edit.after.source_start_sec, edit.after.source_end_sec)
        );
        editor.close();
    }

    /// 使用生产clip路由、官方typed getter/setter与Undo块；没有把私有timeline赋值当同步。
    #[test]
    fn host_edit_writes_directly_bound_item_under_one_project_undo_block() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan=editor.plan_host_edit("set_clip_state",&json!({"clipId":clip.id,"startSec":3.,"lengthSec":8./44100.,"clipPlaybackRate":0.5})).unwrap();
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert_eq!(
            calls
                .iter()
                .filter(|name| name.as_str() == "undo-begin")
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|name| name.as_str() == "undo-end")
                .count(),
            1
        );
        assert!(calls.contains(&"write-item:D_POSITION".into()));
        assert!(calls.contains(&"write-take:D_PLAYRATE".into()));
        let geometry = owner.host_geometry().unwrap_err();
        assert!(
            geometry.to_string().contains("incompatible"),
            "未模拟ARA回流，不能伪造已同步模型"
        );
        assert_eq!(editor.timeline.lock_or_recover().clips[0].start_sec, 0.);
        document.close();
    }
    /// 旧快照仍ok也不能结束移动；长度/倍率/淡变宽度均须准确回流，且缺失clip不可成功。
    #[test]
    fn host_edit_receipt_rejects_readable_old_geometry_and_waits_for_widths() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"startSec":3.,"fadeInSec":0.2}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let mut payload = crate::editor::commands::payload(&editor, false).unwrap();
        assert!(!receipt.matches(&payload));
        payload["clips"][0]["start_sec"] = json!(3.);
        assert!(!receipt.matches(&payload));
        payload["clips"][0]["fade_in_sec"] = json!(0.2);
        assert!(receipt.matches(&payload), "{receipt:?} vs {payload}");
        payload["clips"][0]["id"] = json!("other");
        assert!(!receipt.matches(&payload));
        editor.close();
    }
    /// **不变量：完成门与写回前置检查共用同一个容差。**
    ///
    /// 【为什么这是回归测试】写回前置检查用 `host_value_compatible`（相对容差
    /// `1e-6 + 1e-9·max`），完成门此前用纯绝对 `1e-6`。两侧不对称会造出"写回被接受、
    /// 完成门却判宿主还没跟上"的窗口 ⇒ `geometry_waiting` 常真 ⇒ 前端拿到终态错误
    /// "clip was edited in REAPER but its ARA geometry has not caught up"（用户表现为
    /// "音频块编辑被拒绝"）。`playback_rate` 在 kernel 里是 f32，与 REAPER 的 f64 往返
    /// 本身就有 ~1e-7 相对误差，长工程上的 `start_sec` 也在绝对阈值边缘。
    #[test]
    fn host_edit_receipt_uses_the_same_tolerance_as_the_writeback_precheck() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"startSec":3.}))
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let mut payload = crate::editor::commands::payload(&editor, false).unwrap();
        payload["clips"][0]["start_sec"] = json!(3.);
        assert!(receipt.matches(&payload));

        // f32 往返级的相对误差必须仍判为"宿主已跟上"。
        payload["clips"][0]["start_sec"] = json!(3. * (1.0 + 1e-7));
        assert!(
            receipt.matches(&payload),
            "f32 往返误差不得把完成门判成失败，否则每次编辑都会报\"音频块编辑被拒绝\""
        );

        // 真实移动（1e-4 秒 ≈ 44.1k 上约 4 个采样点）仍必须被拒。
        payload["clips"][0]["start_sec"] = json!(3.0001);
        assert!(!receipt.matches(&payload), "真实几何变化不得被容差吞掉");
        editor.close();
    }
    /// **不变量：宿主表达不到的逐片段属性不再被整条拒绝，且落到本地权威、不写宿主。**
    ///
    /// 【为什么这是回归测试】`color` / `formantMorph` 此前不在 `patch()` 的白名单里，
    /// 任何带它们的 `set_clip_state` 都被整条拒成
    /// "host clip property is not yet writable: color" ⇒ 前端显示"音频块编辑被拒绝"。
    /// 而它们是**插件自有**属性（宿主没有 clip 颜色 / 共振峰变形的概念），正确处置是
    /// 接受并落到 `EditState::clip_local`（与 `fades` 同一手法），由共同投影入口重新
    /// 播种；`formant_morph` 还参与渲染，必须能跨重建存活。
    #[test]
    fn host_edit_accepts_plugin_local_clip_properties_without_writing_the_host() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();

        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"color":"#123456","formantMorph":{"enabled":true}}),
            )
            .expect("插件自有属性不得被拒");
        assert_eq!(plan.edits[0].patch.color.as_deref(), Some("#123456"));
        assert!(
            plan.edits[0]
                .patch
                .formant_morph
                .as_ref()
                .is_some_and(|morph| morph.enabled),
            "formantMorph 必须能反序列化进 patch"
        );
        // 它们不是宿主字段：写回计划里不得出现对应的宿主属性名（`patch()` 只接受
        // 已实现的可写键，宿主侧由 `execute_managed` 的写回列表决定）。
        assert!(
            !plan.edits[0]
                .patch
                .color
                .as_deref()
                .is_some_and(|color| color.contains("D_")),
            "颜色不得被当成宿主属性"
        );
        editor.close();
    }
    /// **回归：自动淡化的写回必须能在**一轮**内被完成门确认。**
    ///
    /// 【为什么这条是用户报障的可执行形式】自动交叉淡化走的是 `D_FADE*LEN_AUTO`
    /// 两个 **item 级** 读数，而它们**不推进** `GetProjectStateChangeCount`。
    /// 写回之后若不清几何缓存，`refresh_host_geometry` 会命中"令牌未变"的早退，
    /// 完成门读回的仍是**写前**那份几何 —— 于是它一直等到重试预算耗尽，报出
    /// "clip was edited in the host but its ARA geometry has not caught up"。
    /// 用户看到的就是"自动交叉淡化没有执行"。
    ///
    /// 这里同时钉住两件事：① 两个 AUTO 字段确实被写到宿主且能读回原值；
    /// ② 完成门按**同一份载荷**立刻匹配（不需要等下一轮几何刷新）。
    #[test]
    fn host_edit_auto_fade_round_trips_and_the_receipt_matches_at_once() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({
                    "clipId": clip.id,
                    "autoFadeInSec": 0.2,
                    "autoFadeOutSec": 0.25,
                }),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        // ① 宿主上真的写下了这两个值。
        assert_eq!(host.item_value("D_FADEINLEN_AUTO"), Some(0.2));
        assert_eq!(host.item_value("D_FADEOUTLEN_AUTO"), Some(0.25));
        // ② 完成门用同一份载荷**立刻**匹配：不需要等下一轮几何刷新。
        let payload = json!({"clips":[{
            "id": clip.id,
            "start_sec": clip.start_sec,
            "length_sec": clip.length_sec,
            "source_start_sec": clip.source_start_sec,
            "playback_rate": clip.playback_rate,
            "track_id": clip.track_id,
            "clip_gain": clip.clip_gain,
            "gain": clip.gain,
            "reversed": clip.reversed,
            "loop_enabled": clip.loop_enabled,
            "auto_fade_in_sec": 0.2,
            "auto_fade_out_sec": 0.25,
        }]});
        assert!(
            receipt.matches(&payload),
            "自动淡化的完成门必须一轮内匹配：{receipt:?} vs {payload}"
        );
        // 对照：旧值不得通过（否则这条门对 auto fade 形同虚设）。
        let mut stale = payload;
        stale["clips"][0]["auto_fade_in_sec"] = json!(0.0);
        assert!(!receipt.matches(&stale), "自动淡化的陈旧回显必须被门拦下");
        document.close();
    }

    /// 宽度patch冗余带着原shape不要求委托；实际改shape仍保持明确的责任门。
    #[test]
    fn host_edit_fade_width_writes_reaper_without_delegation_or_redundant_shape() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan=editor.plan_host_edit("set_clip_state",&json!({"clipId":clip.id,"fadeInSec":0.00002,"fadeOutSec":0.00003,
            "autoFadeInSec":0.,"autoFadeOutSec":0.,"fadeInShape":clip.fade_in_shape,"fadeOutShape":clip.fade_out_shape,
            "fadeInDir":clip.fade_in_dir,"fadeOutDir":clip.fade_out_dir})).unwrap();
        assert!(plan.edits[0].patch.fade_in_shape.is_none());
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        for field in [
            "D_FADEINLEN",
            "D_FADEOUTLEN",
            "D_FADEINLEN_AUTO",
            "D_FADEOUTLEN_AUTO",
        ] {
            assert!(calls.contains(&format!("write-item:{field}")));
        }
        assert!(!calls
            .iter()
            .any(|call| call.starts_with("write-item:D_FADEINDIR")
                || call.starts_with("write-item:C_FADE")));
        // 曲率与 S 落到新轴；形状预设现在也能写，落成 (curvature, S) 一对分量。
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInDir":0.5,"fadeInS":-0.25}),
            )
            .unwrap();
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert!(calls.contains(&"write-item:D_FADEINDIR_NEW".to_string()));
        assert!(calls.contains(&"write-item:D_FADEINDIR2_NEW".to_string()));
        assert_eq!(host.item_value("D_FADEINDIR_NEW"), Some(0.5));
        assert_eq!(host.item_value("D_FADEINDIR2_NEW"), Some(-0.25));
        // 旧轴不得被顺手改掉 —— 它不是权威值。
        assert!(!calls.contains(&"write-item:D_FADEINDIR".to_string()));
        assert!(!calls.contains(&"write-item:C_FADEINSHAPE".to_string()));

        // 形状预设：写 `fadeInShape` 就是同时写两个新轴分量，值取自实测表
        // （预设 3「陡峭凸」= `(curvature 1, S 0)`）。同一次补丁里带的 `fadeInDir`
        // 不参与 —— 界面切预设时会顺手送 HFS 自己的默认 dir，那是**另一套参数化**，
        // 让它覆盖 curvature 会让"陡峭凸"变成"陡峭凹"（预设 4 = `(-1, 0)`）。
        host.set_value("D_FADEINDIR_NEW", -0.2);
        host.set_value("D_FADEINDIR2_NEW", -0.3);
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":3.,"fadeInDir":-1.}),
            )
            .unwrap();
        assert_eq!(plan.edits[0].patch.fade_in_shape, Some(3.));
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert!(calls.contains(&"write-item:D_FADEINDIR_NEW".to_string()));
        assert!(calls.contains(&"write-item:D_FADEINDIR2_NEW".to_string()));
        assert_eq!(host.item_value("D_FADEINDIR_NEW"), Some(1.));
        assert_eq!(host.item_value("D_FADEINDIR2_NEW"), Some(0.));
        assert!(!calls.contains(&"write-item:D_FADEINDIR".to_string()));
        assert!(!calls.contains(&"write-item:C_FADEINSHAPE".to_string()));

        // 形状 + S 同发：S 不能被形状分支**静默吞掉**（此前那里 `continue` 掉了
        // 同补丁的 `fadeInS`）。预设 5「轻微 S」自带 S=0.5；补丁显式给 0.9 时按 0.9 写。
        host.set_value("D_FADEINDIR_NEW", -0.2);
        host.set_value("D_FADEINDIR2_NEW", -0.3);
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":5.,"fadeInS":0.9}),
            )
            .unwrap();
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        assert_eq!(
            host.item_value("D_FADEINDIR2_NEW"),
            Some(0.9),
            "补丁显式给的 S 优先于形状预设自带的 S，不能被吞掉"
        );
        assert_eq!(host.item_value("D_FADEINDIR_NEW"), Some(0.));
        document.close();
    }

    /// 倒放写回必须翻回宿主**镜像**域：`D_STARTOFFS = D − 正向终点`。
    ///
    /// 【为什么这是回归测试】`workspace::reversed_forward_window` 把宿主镜像坐标复原成
    /// 正向窗口交给内核；写回时若不翻回去，宿主会按"起点"重新解释那个正向值 —— 倒放
    /// take 被整体平移 `D − start − span`。
    #[test]
    fn reversed_write_back_uses_the_mirrored_host_offset() {
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id":"c","track_id":"t","name":"n","start_sec":0,"length_sec":1,
            "duration_sec":2.0,
            "reversed":true,
            "takes":[{"id":"k","source_start_sec":0.25,"source_end_sec":1.25,"reversed":true,
                      "duration_sec":2.0}]
        }))
        .unwrap();
        clip.normalize_takes();
        // 正向窗 [0.25, 1.25]、媒体总长 2.0 ⇒ 宿主镜像起点 2.0 − 1.25 = 0.75。
        assert_eq!(host_start_offset(&clip), Some(0.75));
        // 正放：就是正向起点本身。
        clip.reversed = false;
        assert_eq!(host_start_offset(&clip), Some(0.25));
        // 【倒放 + 循环源也走镜像】**旧行为在这里返回 `source_start_sec`（0.25）**，
        // 而读取侧给的是 `(min_bound.min(anchor), anchor)` —— 窗口**终点**是锚点。
        // 两边用不同的量当"镜像起点"，于是循环 + 倒放的 take 每写一次就漂一个 `soffs`
        // （用户报的"倒放 take 的偏移量算错了"在循环源上更明显）。镜像关系
        // `soffs = 轴 − 正向终点` 对 loop / 非 loop 是同一条式子，这里不再分叉。
        clip.reversed = true;
        clip.loop_enabled = true;
        assert_eq!(host_start_offset(&clip), Some(0.75));
        // 媒体总长取不到 ⇒ 返回 None（宁可不写，也不写一个错的起点）。
        clip.loop_enabled = false;
        clip.duration_sec = None;
        assert_eq!(host_start_offset(&clip), None);
    }

    /// 方向写回的三种编码必须与读取侧**互逆**。
    ///
    /// 【为什么钉死这一条】倒放没有 setter：写下去的是 `SOURCE SECTION` 加一个 SOFFS，
    /// 而读回来的是 `take_geometry` 的解算。两边的编码只要差一个坐标域
    /// （绝对 vs 段内、正向 vs 镜像），用户看到的就是"片段整体平移了半个媒体"——
    /// 而波形、时长、淡变全都正常，极难归因。
    #[test]
    fn host_take_direction_matches_the_read_side_encoding() {
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(json!({
            "id":"c","track_id":"t","name":"n","start_sec":0,"length_sec":1,
            "duration_sec":4.0,
            "takes":[{"id":"k","source_start_sec":0.5,"source_end_sec":1.5,"duration_sec":4.0}]
        }))
        .unwrap();
        clip.normalize_takes();

        // 正放：拆段，SOFFS = 正向窗口起点（绝对坐标）。
        let forward = host_take_direction(&clip, Some(4.0)).unwrap();
        assert!(forward.section.is_none(), "正放不该带 SECTION");
        assert_eq!(forward.s_offs, 0.5);

        // 倒放：SECTION = `MODE 3` + `LENGTH = 跨度` + `STARTPOS 0`，SOFFS 是镜像起点
        // `D − 正向终点` = 4.0 − 1.5 = 2.5（宿主自己的编码，见函数文档）。
        clip.reversed = true;
        let reversed = host_take_direction(&clip, Some(4.0)).unwrap();
        let section = reversed.section.expect("倒放必须建段");
        assert_eq!(
            (section.start_sec, section.length_sec),
            (0.0, 1.0),
            "STARTPOS 恒 0、LENGTH 是窗口跨度"
        );
        assert_eq!(
            section.mode,
            super::super::source_direction::SECTION_MODE_REVERSED
        );
        assert_eq!(reversed.s_offs, 2.5, "SOFFS 必须是镜像起点 D − 正向终点");

        // 倒放 + Loop：同一套编码（宿主的 `MODE 3` 不区分 loop），跨度是
        // `[0, 锚点)` —— 与读取侧的 loop 分支同一条式子。
        clip.loop_enabled = true;
        let looped = host_take_direction(&clip, Some(4.0)).unwrap();
        let section = looped.section.expect("倒放必须建段");
        assert_eq!((section.start_sec, section.length_sec), (0.0, 1.5));
        assert_eq!(looped.s_offs, 2.5);

        // 媒体总长未知 ⇒ 拒绝（宁可不写，也不写一个读不回来的结构）。
        assert!(host_take_direction(&clip, None).is_err());
    }

    /// 倒放必须走**结构**写回：换 SOURCE 子树、写 SOFFS，并在写后自校验方向位。
    #[test]
    fn reverse_is_written_back_as_a_source_section() {
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
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .expect("倒放补丁必须被接受");
        // 镜像起点由**写回后的**几何决定（补丁翻转方向时内核会重算消费窗口），
        // 轴与生产代码同一条取值顺序（clip 自带的 ARA 源时长优先，其次宿主读数）。
        let (_, window_end) =
            hifishifter_kernel::state::clip_playback_window_sec(&plan.edits[0].after);
        let axis = hifishifter_kernel::state::clip_source_media_duration_sec(&plan.edits[0].after)
            .unwrap_or(2.0);
        let expected_offs = axis - window_end;
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("倒放写回必须成功");

        let calls = host.calls();
        assert!(
            calls.contains(&"set-item-state".to_string()),
            "倒放必须写成结构，而不是某个数值字段：{calls:?}"
        );
        let chunk = host.item_state_chunk.borrow().clone();
        assert!(chunk.contains("<SOURCE SECTION"), "{chunk}");
        assert!(
            chunk.contains("MODE 3"),
            "倒放必须写成宿主自己的 MODE 3（整份媒体 + 反向）：{chunk}"
        );
        assert!(chunk.contains("STARTPOS 0.00000000000000000"), "{chunk}");
        // 镜像起点：夹具媒体总长 2.0 s，`SOFFS = D − 正向窗口终点`。
        assert!(
            chunk.contains(&format!("SOFFS {expected_offs:.17}")),
            "写回的必须是镜像域起点 {expected_offs}：{chunk}"
        );
        assert!(
            chunk.contains("ara://source"),
            "源本身必须原样保留：{chunk}"
        );
        assert!(host.take_reversed.get(), "写后方向位必须已经变成倒放");
        document.close();
    }

    /// **方向必须进完成门**：倒放/循环都是"只改这一位"的补丁。
    ///
    /// 【为什么这条是本轮的核心回归】完成门此前只比 start/length/source_start/rate/
    /// gain/淡变轴。整段媒体上的倒放一位几何都不动，于是 `matches()` 对**改动前**的
    /// 载荷恒真 —— 前端把权威回声（`reversed=false`）整份应用，波形先倒放、再回退
    /// 正放、最后被 20 ms 的几何刷新纠正成倒放。用户看到的就是那三次翻转。
    #[test]
    fn a_direction_only_patch_is_not_confirmed_by_the_stale_payload() {
        let (_model, owner, _id) = crate::editor::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let payload = |after: &Clip, reversed: bool, loop_enabled: bool| {
            json!({"clips":[{
                "id": after.id,
                "start_sec": after.start_sec,
                "length_sec": after.length_sec,
                "source_start_sec": after.source_start_sec,
                "playback_rate": after.playback_rate,
                "track_id": after.track_id,
                "clip_gain": after.clip_gain,
                "gain": after.gain,
                "reversed": reversed,
                "loop_enabled": loop_enabled,
            }]})
        };
        // 倒放：整段媒体上几何一位不动，唯一变化的就是 `reversed`。
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let after = plan.edits[0].after.clone();
        assert!(
            !receipt.matches(&payload(&after, false, after.loop_enabled)),
            "方向还是改动前的回声，完成门必须拦下（否则前端会弹回正放）"
        );
        assert!(
            receipt.matches(&payload(&after, true, after.loop_enabled)),
            "方向已经翻过来的回声才算完成"
        );
        // 循环源同理：它是 `B_LOOPSRC` 那一位，同样只在补丁里出现。
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"loopEnabled":true}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let after = plan.edits[0].after.clone();
        assert!(
            !receipt.matches(&payload(&after, after.reversed, false)),
            "循环源还是旧值时不算完成"
        );
        assert!(receipt.matches(&payload(&after, after.reversed, true)));
    }

    /// take FX 下 item chunk 一律写不进去（真机行为）时，倒放改走宿主自己的动作。
    ///
    /// 【为什么这条是本次修复的核心】`SetItemStateChunk` 对"挂着在线 ARA FX 的 item"连
    /// 原样写回都拒绝，而 take FX 放置下要改方向的那个 item 正是挂着本插件的那个 —— 于是
    /// 用户点倒放，收到的是 `host clip edit failed ... REAPER rejected the item state`，
    /// 而工程里什么都没变。宿主的 `Item properties: Toggle take reverse` 写出的结构与
    /// 我们自己的编码一致（真机对照过），所以借它动作既不改语义、也不需要重建 item。
    #[test]
    fn a_refused_item_chunk_sends_the_reverse_through_the_host_action() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        // 动作路径要把"只有目标 item 被选中"这件事做对，需要能枚举 item。
        host.inventory_enabled.set(true);
        host.take_env_enabled.set(true);
        host.item_state_write_refused.set(true);
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
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .expect("倒放补丁必须被接受");
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("take FX 下倒放必须走宿主动作");

        let calls = host.calls();
        assert!(host.take_reversed.get(), "方向位必须真的翻过来");
        assert!(
            calls.iter().any(|call| call.starts_with("run-command-ex:")),
            "必须借宿主自己的动作切方向：{calls:?}"
        );
        let chunk = host.item_state_chunk.borrow().clone();
        assert!(
            !chunk.contains("<SOURCE SECTION"),
            "chunk 从未被真正改写（宿主拒绝了它）：{chunk}"
        );
        document.close();
    }

    /// **不变量：几何写回成功后，编辑器时间线必须立刻投影到新的宿主事实。**
    ///
    /// 【为什么这是回归】`execute_managed` 的完成门比对的是 `get_timeline_state`
    /// 载荷，而它的 take 级事实（`reversed` / `loop_enabled` / `channel_mode`）
    /// 只能从原生清单 `ui_tracks` 投影进编辑器时间线。那条通路被"令牌未变 + 刚采过
    /// 样"早退按住：方向位**不**推进宿主的 `GetProjectStateChangeCount`（take 级
    /// 属性），令牌不变是常态。真机后果：写回已成功，完成门把 24 次退避预算（≈10.6 s）
    /// 耗尽后报 "ARA geometry has not caught up"，乐观更新被前端回滚 —— 而工程里
    /// 方向其实已经翻了。修复：写回成功后清掉采样时刻并同步 `refresh_host_transport`
    /// —— 同步重采发布清单后，一轮重投内完成门就能对上。
    #[test]
    fn a_successful_direction_write_projects_host_facts_into_the_editor_timeline_immediately() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.inventory_enabled.set(true);
        host.take_env_enabled.set(true);
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
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .expect("倒放补丁必须被接受");
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("倒放写回必须成功");
        assert!(host.take_reversed.get(), "方向位必须已经翻过来");

        // 回复泵的重投入口就是 `get_timeline_state` → `ensure_loaded`。真实链路里
        // `refresh_host_transport` 已在写回线程同步重采并发布清单（写回先清掉了采样
        // 时刻，"令牌未变 + 刚采过样"的早退拦不住这一轮）；这一轮重投再把新事实投影
        // 进时间线。完成门的方向判据到此必然成立 —— 不必再等 ARA 文档收口。
        editor.ensure_loaded(false).expect("写回后的重投必须成功");
        let payload = crate::editor::commands::payload(&editor, false).unwrap();
        // 不做整张回执比对：夹具的本地 clip 与宿主清单在增益等非方向字段上本就
        // 不同源，那是夹具口径，不是本回归的对象。完成门在此等待的**方向判据**
        // 单独钉死：方向位必须在一轮重投内到达载荷。
        let clip = &payload["clips"][0];
        assert_eq!(
            clip["reversed"],
            serde_json::json!(true),
            "一轮重投内方向位必须到达载荷：{clip}"
        );
        // 方向唯一补丁上 `reversed` 就是完成门的**唯一**阻塞判据：其余字段
        // （start/length/rate/gain/loop）一位都没动。它到达 ⇒ 门可收口。
        // 窗口坐标不在此断言：夹具的 clip 没有 ARA 源时长（`duration_sec` 为空），
        // 镜像复原按设计退回"保留宿主镜像坐标"——那是另一条已独立钉住的通路。
        document.close();
    }

    /// 兄弟 take 的逐 take 方向写回同样必须当场投影（回声走 `takes[].reversed`）。
    ///
    /// 【为什么单独一条】兄弟 take 的完成门期望在 `takes[]` 里，而清单投影写的是
    /// clip 级字段与 active take；这条测试钉住"逐 take 回声也能在写回线程内成立"，
    /// 与上一条（clip 级 `reversed`）共同覆盖完成门的两个消费面。
    #[test]
    fn a_sibling_take_direction_write_also_projects_immediately() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        let chunk = "<ITEM
POSITION 0
LENGTH 0.00009070294784580499
SEL 1
         IGUID {11111111-1111-1111-1111-111111111111}
         SOFFS 0
PLAYRATE 1 1 0
GUID {22222222-2222-2222-2222-222222222222}
         <SOURCE WAVE
FILE \"ara://source\"
>
         TAKE
         SOFFS 0.5
PLAYRATE 1 1 0
GUID {55555555-5555-5555-5555-000000000001}
         <SOURCE WAVE
FILE \"ara://source\"
>
>
"
        .to_string();
        host.enable_item_state(&chunk);
        host.enable_takes(1);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 0.5);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let sibling = clip
            .takes
            .iter()
            .find(|take| take.id.ends_with("{55555555-5555-5555-5555-000000000001}"))
            .expect("清单必须枚举到兄弟 take")
            .id
            .clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostTakeReversed":{"takeId":sibling,"reversed":true}}),
            )
            .expect("逐 take 倒放补丁必须被接受");
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("兄弟 take 写回必须成功");

        editor.ensure_loaded(false).expect("写回后的重投必须成功");
        let payload = crate::editor::commands::payload(&editor, false).unwrap();
        // 同上：不做整张 `receipt.matches` —— 夹具所有 take 共享同一个源指针，
        // chunk 改写后 `PCM_Source_GetSectionInfo` 对 active take 也报倒放（真机每条
        // take 有独立源，不存在这个伪影），clip 级字段随之偏移。完成门等待的
        // **take 级回声**单独钉死：兄弟 take 的方向必须在一轮重投内到达载荷。
        let echoed = payload["clips"][0]["takes"]
            .as_array()
            .expect("载荷必须带 takes")
            .iter()
            .find(|take| take["id"] == serde_json::json!(sibling))
            .expect("载荷必须仍枚举到兄弟 take")["reversed"]
            .as_bool();
        assert_eq!(
            echoed,
            Some(true),
            "兄弟 take 的方向回声必须在一轮重投内到达载荷"
        );
        document.close();
    }

    /// **方向位缺席（无 SECTION 包装的 take）不是错误**。
    ///
    /// 【为什么这是回归】`PCM_Source_GetSectionInfo` 对从未倒放过、也没有显式
    /// SECTION 的 take 返回 false —— 那是"正放"的正常读数。此前
    /// `reversed_now` 把它当"宿主不肯报告"，直接返回错误：take FX 放置下
    /// 动作路径的第一步（读 `before`）就整批失败，真机日志
    /// "the host did not report the take direction" 正是它（而轨道 FX 走 chunk
    /// 路径没有这次预读，所以"轨道正常、take FX 被拒"）。修复后"读不到 = 正放"，
    /// 与几何读取侧同一纪律；动作切换之后方向位出现，写→读→比照常成立。
    #[test]
    fn a_take_fx_reverse_succeeds_when_the_host_never_reported_a_direction() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.inventory_enabled.set(true);
        host.take_env_enabled.set(true);
        host.item_state_write_refused.set(true);
        // 【前提】`section_reports` 保持 false：模拟"从未倒放过、无 SECTION 包装"
        // 的 take —— `PCM_Source_GetSectionInfo` 返回 false。此前这一格让整条
        // 动作路径第一步就报 "the host did not report the take direction"。
        host.section_reports.set(false);
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
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .expect("倒放补丁必须被接受");
        host.reset();
        execute(&owner, plan, || document.is_alive())
            .expect("方向位缺席时倒放必须仍走宿主动作成功");

        let calls = host.calls();
        assert!(host.take_reversed.get(), "方向位必须真的翻过来");
        assert!(
            calls.iter().any(|call| call.starts_with("run-command-ex:")),
            "必须借宿主自己的动作切方向：{calls:?}"
        );
        // 动作之后宿主写出了 SECTION 包装，方向位应可读回为倒放。
        assert!(
            host.section_reports.get() || host.take_reversed.get(),
            "toggle 动作之后宿主侧必须处于可读的倒放形态"
        );
        document.close();
    }

    /// **兄弟 take 的方向写回**：chunk 按 take GUID 精确改写那一条，其余 take 逐字节保留。
    ///
    /// 【为什么要这条测试】插件 take 菜单的逐 take 倒放走的是"非 active take"的
    /// 新路径：目标不是绑定 take，而是同一 item 上的兄弟 take。chunk 改写是
    /// 多 take item 上唯一能"只动这一条"的写法 —— 改错 take 是工程损坏。
    #[test]
    fn a_sibling_take_direction_write_touches_only_that_take() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        let chunk = "<ITEM
POSITION 0
LENGTH 0.00009070294784580499
SEL 1
         IGUID {11111111-1111-1111-1111-111111111111}
         SOFFS 0
PLAYRATE 1 1 0
GUID {22222222-2222-2222-2222-222222222222}
         <SOURCE WAVE
FILE \"ara://source\"
>
         TAKE
         SOFFS 0.5
PLAYRATE 1 1 0
GUID {55555555-5555-5555-5555-000000000001}
         <SOURCE WAVE
FILE \"ara://source\"
>
>
"
        .to_string();
        host.enable_item_state(&chunk);
        host.enable_takes(1);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 0.5);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let sibling = clip
            .takes
            .iter()
            .find(|take| take.id.ends_with("{55555555-5555-5555-5555-000000000001}"))
            .expect("清单必须枚举到兄弟 take")
            .id
            .clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostTakeReversed":{"takeId":sibling,"reversed":true}}),
            )
            .expect("逐 take 倒放补丁必须被接受");
        // 兄弟 take 不得折进 clip 级补丁：折进去会把绑定 take 的窗口一并翻转。
        assert!(
            plan.edits[0].patch.reversed.is_none(),
            "兄弟 take 的方向必须走逐 take 路径"
        );
        assert!(plan.edits[0].take_direction.is_some());
        // 回执门必须按这一条 take 的回声收口。
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let stale = json!({"clips":[{"id":clip.id,"takes":[{"id":sibling,"reversed":false}]}]});
        assert!(!receipt.matches(&stale), "回声还是改动前时完成门必须拦下");
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("兄弟 take 的倒放写回必须成功");
        let written = host.item_state_chunk.borrow().clone();
        let lines: Vec<&str> = written.lines().collect();
        // 主 take 原样：源还是裸 WAVE（没有 SECTION），SOFFS 保持 0。
        let main = super::super::take_chunk::section_with_guid(
            &lines,
            "{22222222-2222-2222-2222-222222222222}",
        )
        .expect("主 take 必须还在 chunk 里");
        let main_source =
            super::super::take_chunk::block(&lines, &main, "SOURCE").expect("主 take 必须有源");
        assert!(
            lines[main_source.start].contains("SOURCE WAVE"),
            "主 take 不得被改写：{}",
            lines[main_source.start]
        );
        // 兄弟 take：源被 SECTION（MODE 3）包住，SOFFS 折成镜像域。
        let sibling = super::super::take_chunk::section_with_guid(
            &lines,
            "{55555555-5555-5555-5555-000000000001}",
        )
        .expect("兄弟 take 必须还在 chunk 里");
        let sibling_source = super::super::take_chunk::block(&lines, &sibling, "SOURCE")
            .expect("兄弟 take 必须有源");
        assert!(
            lines[sibling_source.start].contains("SOURCE SECTION"),
            "兄弟 take 的源必须被 SECTION 包住：{}",
            lines[sibling_source.start]
        );
        let written_lines = written.lines().collect::<Vec<_>>();
        let mode_line = written_lines[sibling_source.start + 2];
        assert_eq!(
            mode_line.trim(),
            "MODE 3",
            "必须是宿主自己的 MODE 3 编码：{written}"
        );
        document.close();
    }

    /// **兄弟 take 的声道模式**：`I_CHANMODE` 是 take 属性，写必须落在目标 take 的
    /// 指针上，写后读回核对；回执门按 `takes[].channel_mode` 收口。
    #[test]
    fn a_sibling_take_channel_mode_write_lands_on_that_take() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        host.enable_takes(1);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 0.5);
        host.set_value("D_PLAYRATE", 1.);
        // 写口只更新已存在的键（见夹具 set_take 的注释）：先注册旧值 0。
        host.set_value("I_CHANMODE", 0.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let sibling = clip
            .takes
            .iter()
            .find(|take| take.id.ends_with("{55555555-5555-5555-5555-000000000001}"))
            .expect("清单必须枚举到兄弟 take")
            .id
            .clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostTakeChannelMode":{"takeId":sibling,"channelMode":2}}),
            )
            .expect("逐 take 声道模式补丁必须被接受");
        assert!(plan.edits[0].patch.channel_mode.is_none());
        assert!(plan.edits[0].take_channel_mode.is_some());
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("兄弟 take 的声道模式写回必须成功");
        assert!(
            host.calls().contains(&"write-take:I_CHANMODE".to_string()),
            "必须按 take 指针写 I_CHANMODE：{:?}",
            host.calls()
        );
        assert_eq!(host.item_value("I_CHANMODE"), Some(2.0));
        document.close();
    }

    /// **active take 的逐 take 补丁折进 clip 级路径**：绑定 take 走既有的几何/镜像/
    /// 完成门（`patch.reversed`），不进逐 take side 字段。
    #[test]
    fn an_active_take_direction_patch_folds_into_the_clip_level_path() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_item_state(&host.single_take_chunk());
        // 清单必须启用：take id 的命名空间前缀来自清单投影（ARA-only 形态保留原始
        // take id，见 session.rs 的 namespace_takes 注释），而本路径只服务原生宿主。
        host.enable_takes(0);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 0.5);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_host_adapter(host.context());
        }
        owner.refresh_host_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock_or_recover().clips[0].clone();
        let active_take = clip
            .active_take_id
            .clone()
            .expect("夹具 clip 必须有 active take");
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostTakeReversed":{"takeId":active_take,"reversed":true}}),
            )
            .expect("active take 的逐 take 补丁必须被接受");
        assert_eq!(plan.edits[0].patch.reversed, Some(true));
        assert!(plan.edits[0].take_direction.is_none());
        document.close();
    }

    /// **编组 → 宿主 `I_GROUPID`**：补丁解析、setter、完成门三段同测。
    ///
    /// 【为什么要端到端】编组是"多片段各写一位"的批次：补丁键（`hostGroupId`）、
    /// 写口白名单（`I_GROUPID`）、完成门（`group_id` 回声）三处各一段代码 —— 任何
    /// 一处拒绝都以"编组点了没反应"收场（真机曾经整批拒绝过越界 id）。完成门
    /// 按**字符串**回声比对（"host-group-N" 来自清单对 `I_GROUPID` 的读回），
    /// 回声没带上编组 id 之前不算完成 —— 否则前端会拿旧状态覆盖乐观更新。
    #[test]
    fn a_group_patch_writes_host_group_ids_and_waits_for_the_echo() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
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
        let group = 123_456_789_i32;
        let plan = editor
            .plan_host_edit(
                "set_clips_state_bulk",
                &json!({"updates":[{"clipId":clip.id,"hostGroupId":group}],"checkpoint":true}),
            )
            .expect("编组补丁必须被接受");
        let before = plan.edits[0].before.clone();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let echo = |group_id: serde_json::Value| {
            json!({"clips":[{
                "id": before.id,
                "start_sec": before.start_sec,
                "length_sec": before.length_sec,
                "source_start_sec": before.source_start_sec,
                "playback_rate": before.playback_rate,
                "track_id": before.track_id,
                "clip_gain": before.clip_gain,
                "gain": before.gain,
                "reversed": before.reversed,
                "loop_enabled": before.loop_enabled,
                "group_id": group_id,
            }]})
        };
        assert!(
            !receipt.matches(&echo(json!(""))),
            "回声没有编组 id 时完成门必须拦下"
        );
        assert!(
            receipt.matches(&echo(json!(format!("host-group-{group}")))),
            "回声带上编组 id 才算完成"
        );
        host.reset();
        execute(&owner, plan, || document.is_alive()).expect("编组写回必须成功");
        assert_eq!(
            host.item_value("I_GROUPID"),
            Some(group as f64),
            "编组必须按 I_GROUPID 写到宿主 item 上"
        );
        // 解组：期望回声是 group_id 缺席（Null），任何残留编组 id 都不算完成。
        let plan = editor
            .plan_host_edit(
                "set_clips_state_bulk",
                &json!({"updates":[{"clipId":clip.id,"hostGroupId":0}],"checkpoint":true}),
            )
            .expect("解组补丁必须被接受");
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        assert!(receipt.matches(&echo(serde_json::Value::Null)));
        assert!(!receipt.matches(&echo(json!("host-group-1"))));
        document.close();
    }

    /// 形状 / 曲率必须进完成门：否则"只改曲率"的补丁 `matches()` 恒真，响应会带着
    /// **改动前**的 dir 回来，前端按权威回声整份应用 → 用户拖完曲率看到曲线弹回。
    ///
    /// 【为什么这条只在插件里有意义】独立 App 直接改自己的模型，没有"宿主回流"这一步。
    #[test]
    fn fade_axis_edits_are_part_of_the_completion_receipt() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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

        // 回执的字段表现在是三元的（键 / 期望值 / 本次补丁是否写过它）；
        // 这条用例只关心"哪些字段进了门"，把第三位剥掉即可。
        let fields_of = |receipt: &super::HostEditReceipt| -> Vec<(&'static str, Value)> {
            receipt.clips[0]
                .fields
                .iter()
                .map(|(name, value, _)| (*name, value.clone()))
                .collect()
        };
        let key_value = |fields: &[(&'static str, Value)], key: &str| -> Option<Value> {
            fields
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.clone())
        };

        // ① 纯曲率拖拽：补丁里**没有**任何长度字段。
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"fadeInDir":0.5}))
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        let fields = fields_of(&receipt);
        assert_eq!(
            key_value(&fields, "fade_in_dir").and_then(|value| value.as_f64()),
            Some(0.5),
            "曲率必须在完成门里，否则响应会带着改动前的值回来、曲线弹回"
        );
        assert!(
            key_value(&fields, "fade_in_sec").is_none(),
            "长度没改就不该等长度：门只等本次真正请求的字段"
        );

        // ② 形状预设（连续轴）：落盘的是预设自己的 curvature，不是同补丁带的 dir。
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":3.,"fadeInDir":-1.}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, Some(true));
        let fields = fields_of(&receipt);
        // 预设 3「陡峭凸」= (curvature 1, S 0)；若照抄 dir(-1) 会变成预设 4。
        assert_eq!(
            key_value(&fields, "fade_in_dir").and_then(|value| value.as_f64()),
            Some(1.),
            "连续轴上形状的期望值取自实测表，不是补丁里同带的 HFS dir"
        );
        assert!(
            key_value(&fields, "fade_in_shape").is_none(),
            "连续轴上形状号是派生读数（可能是 -1），不能当判据"
        );

        // ③ 旧轴：形状号本身可写，照常进门。
        let receipt = plan.receipt(&editor.namespace, Some(false));
        let fields = fields_of(&receipt);
        assert_eq!(
            key_value(&fields, "fade_in_shape").and_then(|value| value.as_f64()),
            Some(3.)
        );

        // ④ 只改 S：S 必须进完成门，且门**能真正闭合**（载荷带顶层 `fade_in_s`）。
        //    此前不比 S 的后果与当年 dir 的抖动同源：只改 S 的补丁 `matches()` 恒真，
        //    回显带着改动前的 S 回来，曲线弹回。
        let plan = editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"fadeInS":0.5}))
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, Some(true));
        let fields = fields_of(&receipt);
        assert_eq!(
            key_value(&fields, "fade_in_s").and_then(|value| value.as_f64()),
            Some(0.5),
            "S 必须进完成门"
        );
        assert!(
            !receipt.matches(&crate::editor::commands::payload(&editor, false).unwrap()),
            "写回前宿主读数还是旧 S ⇒ 门不成立"
        );
        execute(&owner, plan, || document.is_alive()).unwrap();
        owner.refresh_host_transport();
        let payload = crate::editor::commands::payload(&editor, false).unwrap();
        assert!(
            receipt.matches(&payload),
            "写回后宿主读数必须让 S 的门闭合（载荷顶层带 `fade_in_s`）"
        );
        document.close();
    }

    /// 吸附偏移经真实typed setter和UI元数据回流，不能写完后永远等默认零值。
    #[test]
    fn host_edit_snap_offset_receipt_uses_current_host_metadata() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"snapOffsetSec":2./44100.}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace, owner.host_fade_axes());
        assert!(!receipt.matches(&crate::editor::commands::payload(&editor, false).unwrap()));
        execute(&owner, plan, || document.is_alive()).unwrap();
        owner.refresh_host_transport();
        assert_eq!(
            owner.host_geometry_metadata().unwrap()[0]
                .geometry
                .snap_offset_sec,
            2. / 44100.
        );
        let payload = crate::editor::commands::payload(&editor, false).unwrap();
        assert!(receipt.matches(&payload));
        // 官方只说明秒域，不因用户已有负偏移使整个host几何失效；GUI自行按原约定绘制。
        host.set_value("D_SNAPOFFSET", -1. / 44100.);
        owner.refresh_host_transport();
        assert_eq!(
            owner.host_geometry_metadata().unwrap()[0]
                .geometry
                .snap_offset_sec,
            -1. / 44100.
        );
        document.close();
    }
    /// 起点/时长未变但宿主已经改源窗口时，裁切预检必须在Undo/写API之前拒绝。
    #[test]
    fn host_edit_source_trim_rejects_stale_source_offset_before_any_write() {
        let (model, owner, _id) = crate::editor::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::MockHost::new();
        host.enable_writer();
        host.clear_markers();
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
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"sourceStartSec":1./44100.}),
            )
            .unwrap();
        host.set_value("D_STARTOFFS", 2. / 44100.);
        host.reset();
        assert!(execute(&owner, plan, || document.is_alive())
            .unwrap_err()
            .contains("source window/rate changed"));
        assert!(!host
            .calls()
            .iter()
            .any(|call| call == "undo-begin" || call.starts_with("write-")));
        document.close();
    }
}

#[cfg(test)]
mod fade_axis_validation_tests {
    use super::*;

    fn shape(shape: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_shape: Some(shape),
            ..Default::default()
        }
    }
    fn curvature(dir: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_dir: Some(dir),
            ..Default::default()
        }
    }
    fn s_param(value: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_s: Some(value),
            ..Default::default()
        }
    }

    /// 不含淡变轴字段的补丁永远合法 —— 长度/位置编辑不该被轴语义拦住。
    #[test]
    fn patches_without_fade_axes_are_always_accepted() {
        let patch = ClipStatePatch {
            length_sec: Some(1.),
            ..Default::default()
        };
        for axes in [None, Some(true), Some(false)] {
            assert!(validate_fade_axes(axes, &patch).is_ok(), "{axes:?}");
        }
    }

    /// 版本读不出来时一律拒绝：不猜哪套轴是权威的。
    #[test]
    fn an_unknown_axis_set_rejects_every_fade_axis_write() {
        for patch in [shape(3.), curvature(0.5), s_param(0.2)] {
            assert!(validate_fade_axes(None, &patch).is_err());
        }
    }

    /// 旧轴宿主（≤7.80）：形状与曲率可以写，S 参数不行（它只存在于新轴）。
    #[test]
    fn legacy_hosts_take_the_shape_and_curvature_but_not_the_s_parameter() {
        assert!(validate_fade_axes(Some(false), &shape(3.)).is_ok());
        assert!(validate_fade_axes(Some(false), &curvature(0.5)).is_ok());
        let error = validate_fade_axes(Some(false), &s_param(0.2)).expect_err("S 必须被拒绝");
        assert!(error.contains("7.81"), "{error}");
    }

    /// 新轴宿主（≥7.81）：曲率、S、以及**形状预设**都可以写。
    ///
    /// 形状此前被拒绝，理由是"预设 → (curvature, S) 的映射尚未校准"。该映射现已实测
    /// 落地（`hifishifter_kernel::fade_axes`），所以预设按钮在新轴宿主上也能用。
    #[test]
    fn continuous_axis_hosts_take_curvature_s_and_shape_presets() {
        assert!(validate_fade_axes(Some(true), &curvature(0.5)).is_ok());
        assert!(validate_fade_axes(Some(true), &s_param(-0.25)).is_ok());
        for preset in 0..7 {
            assert!(
                validate_fade_axes(Some(true), &shape(preset as f64)).is_ok(),
                "preset {preset}"
            );
        }
    }

    /// 表外的形状号必须拒绝。宿主对越界值不会报错，而是静默变成别的形状 ——
    /// 实测写 `5.1` 读回 `SHAPE=7`。
    /// 宁可报错，也不要用户点了 A 得到 B。
    #[test]
    fn continuous_axis_hosts_reject_shapes_outside_the_measured_table() {
        for value in [-1.0, 7.0, 1.1, 5.1, 6.5] {
            let error =
                validate_fade_axes(Some(true), &shape(value)).expect_err("表外形状必须被拒绝");
            assert!(error.contains("unknown fade shape"), "{error}");
        }
    }

    /// 旧轴宿主上形状仍是整数预设号，不走那张表 —— 旧轴宿主本来就有形状号。
    #[test]
    fn legacy_hosts_keep_accepting_plain_shape_indices() {
        assert!(validate_fade_axes(Some(false), &shape(6.)).is_ok());
    }

    /// 任一淡变轴字段都会触发校验（不能只检查 `fade_in_*`）。
    #[test]
    fn every_fade_axis_field_triggers_validation() {
        for patch in [
            ClipStatePatch {
                fade_out_shape: Some(4.),
                ..Default::default()
            },
            ClipStatePatch {
                fade_out_dir: Some(0.1),
                ..Default::default()
            },
            ClipStatePatch {
                fade_out_s: Some(0.1),
                ..Default::default()
            },
        ] {
            assert!(validate_fade_axes(None, &patch).is_err(), "{patch:?}");
        }
    }
}
