//! 外部时间轴的准入校验：ARA 与桥接**共用同一套规则**，按策略整份拒绝或逐条降级。
//!
//! 【为什么要有这一层】外部时间轴**不可信**：它来自宿主、第三方扩展，可能版本漂移、
//! 字段语义不同、局部损坏。把校验收敛到一个具名函数，两侧就不会各自长出一套判据
//! （判据分叉正是过去"一个 96 kHz 素材让插件打不开"那类故障的成因）。
//!
//! 【为什么有两种策略】规则相同，处置不同：
//! - [`Admission::Strict`]：任何一条不合格都**整份拒绝**。这是独立 App 的 ARA 导入
//!   沿用至今的纪律，逐位保持（判据与 [`super::host_pcm::materialize`] 一致）。
//! - [`Admission::Degrade`]：**逐条降级**。桥接的时间轴来自第三方代码，一条坏 clip
//!   让整份失败只会让用户看到"连接失败"而不知道是哪一条；跳过并报出原因，用户才能
//!   据以行动。这与 `render/input.rs` 的既有原则（"一条坏 clip 不该停掉整条时间线"）
//!   是同一条纪律。
//!
//! 【不在这里处理什么】**倒放**（`Clip.reversed` / `ClipTake.reversed`）由
//! [`super::host_pcm::materialize`] 的 `ReversePolicy` 决定：整份拒绝 / 逐 clip 隔离 /
//! 照常物化三种策略的**唯一实现**在那一侧，在这里再判一次只会分叉。

use super::host_pcm::PcmView;
use crate::state::TimelineState;
use std::collections::HashSet;

/// 准入策略。规则是同一套，只有处置不同。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// 任何一条不合格都整份拒绝（独立 App 的 ARA 导入既有纪律）。
    Strict,
    /// 逐条降级：不合格的素材/clip 跳过，未知轨道引用挂到兜底轨道，并报出原因。
    Degrade,
}

/// 一条被拒或被改写的条目。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    pub id: String,
    pub reason: String,
}

/// 准入结果；[`Admission::Strict`] 下必定为空（有问题就已经整份拒绝了）。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SanitizeReport {
    /// 被拒的素材（`persistent_id`）。
    pub dropped_sources: Vec<Rejection>,
    /// 被跳过的 clip（`Clip.id`）。
    pub dropped_clips: Vec<Rejection>,
    /// 轨道引用不存在、被挂到兜底轨道的 clip（`Clip.id`）。
    pub remapped_clips: Vec<Rejection>,
}

impl SanitizeReport {
    /// 没有发生任何降级。
    pub fn is_clean(&self) -> bool {
        self.dropped_sources.is_empty()
            && self.dropped_clips.is_empty()
            && self.remapped_clips.is_empty()
    }

    /// 一行人类可读摘要；用于日志与"跳过了什么"的 UI 提示。
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (label, items) in [
            ("sources", &self.dropped_sources),
            ("clips", &self.dropped_clips),
            ("remapped", &self.remapped_clips),
        ] {
            if items.is_empty() {
                continue;
            }
            let detail = items
                .iter()
                .map(|item| format!("{} ({})", item.id, item.reason))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("{} {}: {detail}", items.len(), label));
        }
        parts.join("; ")
    }
}

/// 外部时间轴的准入校验；就地改写 `timeline` 与 `sources`，返回降级报告。
///
/// 【为什么就地改写而不是返回新值】调用方随后要把同一份 `timeline` 交给
/// [`super::host_pcm::materialize`]，而 `sources` 里的 `PcmView` 借用着调用方持有的
/// PCM 平面 —— 复制一份既贵又借不动。
///
/// `byte_limit` 是 PCM 的累计字节上限（与 `materialize` 的同一判据）：内嵌形态受
/// 64 MiB 帧上限约束；文件引用形态不受帧约束，调用方按需给更大的上限。
pub fn sanitize_host_timeline(
    timeline: &mut TimelineState,
    sources: &mut Vec<PcmView<'_>>,
    byte_limit: usize,
    admission: Admission,
) -> Result<SanitizeReport, String> {
    // 与 `materialize` 同一条顶层判据：这两个量不合格意味着时间轴本身不可信
    // （不是"某一条坏了"），因此两种策略下都是整份拒绝。
    if !timeline.bpm.is_finite()
        || timeline.bpm <= 0.0
        || !timeline.project_sec.is_finite()
        || timeline.project_sec < 0.0
    {
        return Err("invalid host timeline".into());
    }
    let mut report = SanitizeReport::default();
    // ── 素材：率 / 声道 / 有限性 / 唯一性 / 尺寸 ──
    let mut accepted = Vec::with_capacity(sources.len());
    let mut seen: HashSet<String> = HashSet::new();
    let mut bytes = 0_usize;
    for pcm in std::mem::take(sources) {
        let frames = pcm.planes.first().map(Vec::len).unwrap_or(0);
        // 尺寸溢出是算术上的不可能，与策略无关：任何一侧都不该接受它。
        let size = frames
            .checked_mul(pcm.planes.len())
            .and_then(|n| n.checked_mul(4))
            .ok_or("host PCM size overflow")?;
        let total = bytes.checked_add(size).ok_or("host PCM size overflow")?;
        let reason = if pcm.persistent_id.is_empty() {
            Some("empty persistent_id")
        } else if !(8000..=384_000).contains(&pcm.sample_rate) {
            Some("sample rate outside 8000..=384000")
        } else if !(1..=2).contains(&pcm.planes.len()) {
            Some("channel count is not 1 or 2")
        } else if frames == 0 {
            Some("no frames")
        } else if pcm
            .planes
            .iter()
            .any(|plane| plane.len() != frames || plane.iter().any(|n| !n.is_finite()))
        {
            Some("planes are ragged or contain NaN/Inf")
        } else if seen.contains(pcm.persistent_id) {
            Some("duplicate persistent_id")
        } else if total > byte_limit {
            Some("exceeds the PCM byte budget")
        } else {
            None
        };
        match reason {
            None => {
                seen.insert(pcm.persistent_id.to_owned());
                bytes = total;
                accepted.push(pcm);
            }
            Some(reason) => match admission {
                Admission::Strict => return Err("invalid or oversized host PCM".into()),
                Admission::Degrade => report.dropped_sources.push(Rejection {
                    id: pcm.persistent_id.to_owned(),
                    reason: reason.into(),
                }),
            },
        }
    }
    *sources = accepted;

    // ── clip：几何 / 源引用 / 轨道引用 ──
    let known: HashSet<&str> = sources.iter().map(|pcm| pcm.persistent_id).collect();
    let fallback_track = timeline.tracks.first().map(|track| track.id.clone());
    if admission == Admission::Degrade && fallback_track.is_none() {
        // 兜底轨道都不存在时"把 clip 挂上去"是不可能的；没有轨道的工程本身也不可用
        // （契约要求 `tracks` 非空）。
        return Err("host timeline has no tracks".into());
    }
    let mut clips = Vec::with_capacity(timeline.clips.len());
    for mut clip in std::mem::take(&mut timeline.clips) {
        // 与 `materialize` 同一步：先让 Take 与扁平投影一致，再判几何与引用，
        // 否则会出现"改了 Take 没改投影"造成的判据抖动。
        clip.normalize_takes();
        let geometry_ok = clip.start_sec.is_finite()
            && clip.length_sec.is_finite()
            && clip.length_sec > 0.0
            && clip.takes.iter().all(|take| {
                take.source_start_sec.is_finite()
                    && take.source_end_sec.is_finite()
                    && take.playback_rate.is_finite()
                    && take.playback_rate > 0.0
            });
        let references_ok = std::iter::once(&clip.source_path)
            .chain(clip.takes.iter().map(|take| &take.source_path))
            .all(|id| id.as_deref().is_some_and(|id| known.contains(id)));
        let track_ok = timeline
            .tracks
            .iter()
            .any(|track| track.id == clip.track_id);
        if !geometry_ok || !references_ok {
            match admission {
                Admission::Strict => {
                    return Err(if geometry_ok {
                        "missing host PCM".into()
                    } else {
                        "invalid or unsupported host clip geometry".into()
                    })
                }
                Admission::Degrade => {
                    report.dropped_clips.push(Rejection {
                        id: clip.id.clone(),
                        reason: if geometry_ok {
                            "references a source that was not provided"
                        } else {
                            "invalid clip geometry"
                        }
                        .into(),
                    });
                    continue;
                }
            }
        }
        if !track_ok {
            match admission {
                // 既有行为：悬空的 `track_id` 保持原样（ARA 的几何由宿主拥有，
                // 这里改写它会改变本地对宿主事实的表示）。
                Admission::Strict => {}
                Admission::Degrade => {
                    clip.track_id = fallback_track.clone().expect("checked above");
                    report.remapped_clips.push(Rejection {
                        id: clip.id.clone(),
                        reason: "track_id is not in the timeline; attached to the first track"
                            .into(),
                    });
                }
            }
        }
        clips.push(clip);
    }
    timeline.clips = clips;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TimelineState;

    const INLINE_LIMIT: usize = 64 * 1024 * 1024;

    fn timeline() -> TimelineState {
        serde_json::from_value(serde_json::json!({
            "tracks": [{"id": "track", "name": "T", "order": 0}],
            "bpm": 120.0,
            "project_sec": 8.0,
            "clips": [
                {"id": "good", "name": "good", "track_id": "track", "start_sec": 0.0,
                 "length_sec": 1.0,
                 "takes": [{"id": "t1", "source_path": "host-a",
                            "source_end_sec": 1.0}]},
                {"id": "bad-geometry", "name": "bad", "track_id": "track", "start_sec": 1.0,
                 "length_sec": 0.0,
                 "takes": [{"id": "t2", "source_path": "host-a",
                            "source_end_sec": 1.0}]},
                {"id": "dangling-track", "name": "moved", "track_id": "gone", "start_sec": 2.0,
                 "length_sec": 1.0,
                 "takes": [{"id": "t3", "source_path": "host-a",
                            "source_end_sec": 1.0}]},
                {"id": "missing-source", "name": "gone", "track_id": "track", "start_sec": 3.0,
                 "length_sec": 1.0,
                 "takes": [{"id": "t4", "source_path": "host-missing",
                            "source_end_sec": 1.0}]}
            ]
        }))
        .unwrap()
    }

    fn planes() -> Vec<Vec<f32>> {
        vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.3, 0.2, 0.1]]
    }

    /// 严格策略：任何一条不合格都整份拒绝，且与 `materialize` 的判据一致。
    #[test]
    fn strict_admission_rejects_the_whole_timeline_on_any_bad_item() {
        let storage = planes();
        let mut sources = vec![PcmView {
            persistent_id: "host-a",
            sample_rate: 44100,
            planes: &storage,
        }];
        let error = sanitize_host_timeline(
            &mut timeline(),
            &mut sources,
            INLINE_LIMIT,
            Admission::Strict,
        )
        .expect_err("一条坏 clip 必须让严格策略整份拒绝");
        assert_eq!(error, "invalid or unsupported host clip geometry");

        // 越界采样率同理。
        let mut sources = vec![PcmView {
            persistent_id: "host-a",
            sample_rate: 20_000_000,
            planes: &storage,
        }];
        assert!(sanitize_host_timeline(
            &mut timeline(),
            &mut sources,
            INLINE_LIMIT,
            Admission::Strict
        )
        .is_err());
    }

    /// 降级策略：坏 clip 跳过、悬空轨道改写、其余照常，并逐条报出原因。
    #[test]
    fn degrade_admission_skips_bad_items_and_reports_each_reason() {
        let storage = planes();
        let mut sources = vec![
            PcmView {
                persistent_id: "host-a",
                sample_rate: 44100,
                planes: &storage,
            },
            // 越界采样率 → 该素材被拒，其引用者随之被跳过。
            PcmView {
                persistent_id: "host-b",
                sample_rate: 1_000,
                planes: &storage,
            },
        ];
        let mut input = timeline();
        let mut extra = input.clips[0].clone();
        extra.id = "on-bad-source".into();
        extra.takes[0].source_path = Some("host-b".into());
        input.clips.push(extra);

        let report =
            sanitize_host_timeline(&mut input, &mut sources, INLINE_LIMIT, Admission::Degrade)
                .unwrap();

        assert_eq!(sources.len(), 1, "越界素材必须被拒");
        assert_eq!(report.dropped_sources.len(), 1);
        assert_eq!(report.dropped_sources[0].id, "host-b");
        assert!(report.dropped_sources[0].reason.contains("sample rate"));

        let kept: Vec<&str> = input.clips.iter().map(|clip| clip.id.as_str()).collect();
        assert_eq!(kept, vec!["good", "dangling-track"]);
        let dropped: Vec<&str> = report
            .dropped_clips
            .iter()
            .map(|item| item.id.as_str())
            .collect();
        assert_eq!(
            dropped,
            vec!["bad-geometry", "missing-source", "on-bad-source"]
        );
        assert_eq!(report.remapped_clips.len(), 1);
        assert_eq!(report.remapped_clips[0].id, "dangling-track");
        assert_eq!(input.clips[1].track_id, "track", "兜底轨道是第一条轨道");
        assert!(!report.is_clean());
        assert!(report.summary().contains("host-b"));
    }

    /// 两种策略都不接受损坏的顶层时间轴（这不是"某一条坏了"）。
    #[test]
    fn both_admissions_reject_a_corrupt_top_level_timeline() {
        for admission in [Admission::Strict, Admission::Degrade] {
            for mutate in [0, 1, 2, 3] {
                let mut input = timeline();
                match mutate {
                    0 => input.bpm = f64::NAN,
                    1 => input.bpm = 0.0,
                    2 => input.project_sec = f64::NEG_INFINITY,
                    _ => input.project_sec = -1.0,
                }
                let storage = planes();
                let mut sources = vec![PcmView {
                    persistent_id: "host-a",
                    sample_rate: 44100,
                    planes: &storage,
                }];
                assert!(
                    sanitize_host_timeline(&mut input, &mut sources, INLINE_LIMIT, admission)
                        .is_err(),
                    "损坏的 bpm/project_sec 必须整份拒绝"
                );
            }
        }
    }

    /// 降级策略下"没有轨道"是不可恢复的：兜底轨道都不存在，clip 无处可挂。
    #[test]
    fn degrade_admission_needs_at_least_one_track() {
        let mut input = timeline();
        input.tracks.clear();
        let storage = planes();
        let mut sources = vec![PcmView {
            persistent_id: "host-a",
            sample_rate: 44100,
            planes: &storage,
        }];
        assert!(
            sanitize_host_timeline(&mut input, &mut sources, INLINE_LIMIT, Admission::Degrade)
                .is_err()
        );
    }

    /// 累计字节预算：超限的素材被拒，其后更小的仍可接受（不是"一超全废"）。
    #[test]
    fn degrade_admission_applies_the_byte_budget_per_source() {
        let big = vec![vec![0.0_f32; 1024]];
        let small = vec![vec![0.0_f32; 4]];
        let mut sources = vec![
            PcmView {
                persistent_id: "host-a",
                sample_rate: 44100,
                planes: &big,
            },
            PcmView {
                persistent_id: "host-b",
                sample_rate: 44100,
                planes: &small,
            },
        ];
        let mut input = timeline();
        // 只留 good（引用 host-a），其余 clip 会因引用缺失被跳过 —— 不影响本用例。
        let report =
            sanitize_host_timeline(&mut input, &mut sources, 2048, Admission::Degrade).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].persistent_id, "host-b");
        assert_eq!(report.dropped_sources.len(), 1);
        assert!(report.dropped_sources[0].reason.contains("byte budget"));
    }
}

// ─── 宿主几何归属 ─────────────────────────────────────────────────────────────

/// 几何差异列表的上限。
///
/// 【为什么有上限】拒绝原因是给人看的。一条规则动辄产生上百项差异时，
/// 列全反而没有信息量；前八项足以定位用户动了什么。
pub const GEOMETRY_DIFF_LIMIT: usize = 8;

/// 宿主拥有、本地不该改的字段的**正向**投影。
///
/// 【为什么是正向名单，而不是「整份状态减掉几个例外」】旧判据把整个 `TimelineState`
/// 序列化后手工挖掉十几项，等于用「整份状态相等」表达「这几个字段不许变」。
/// 但 `TimelineState` 是独立 App 的**活状态**：后台子系统（假立体声折叠扫描、
/// 波形/音高分析、渲染缓存回填）会在用户**没有做任何编辑**时写它。只要写到的字段
/// 不在例外名单里，提交就被拒 —— 而报错文案固定指控「几何/增益/名字」，与实际原因无关。
///
/// 正向名单把判据变成「我知道宿主拥有哪些字段，只比这些」。
///
/// 【为什么 `bpm` / `project_sec` 不在名单里】二者都由宿主单方面决定：插件每次导出都
/// 从宿主时钟重读 `bpm`，并重算 `project_sec`。本地改动既提交不上去，
/// 也不代表「未保存的宿主编辑」，放进判据只会制造假失败。
///
/// 【为什么放在内核里】两侧都要用它：**消费者**导出时留下基线，**生产者**在回写时
/// 比对。两处各写一份必然漂移，而漂移的症状是「门禁在两边给出不同答案」——
/// 最难排查的一类。判据只有一份实现。
pub fn host_geometry_projection(timeline: &TimelineState) -> serde_json::Value {
    let mut normalized = timeline.clone();
    // 【为什么要按 take 归一化，而不是按扁平投影】拖 clip 的扁平字段是**投影**，
    // `takes` 才是权威。宿主侧的一次"访问切换"会重新派生时间轴，扁平字段的填充
    // 时机随之改变 —— 实测同一份宿主状态下 `source_end_sec` 会从 `0.0` 变成
    // `9.07e-5`，于是判据把"什么都没发生"报成"你改了宿主字段"。
    //
    // 反方向（扁平 → take）就是旧实现的写法，它把投影当成了真值，正是这类抖动的来源。
    // 按 take 归一化之后，判据只依赖权威表示，与扁平字段恰好在哪一次被填充无关。
    for clip in &mut normalized.clips {
        clip.normalize_takes();
    }
    let clips: Vec<_> = normalized
        .clips
        .iter()
        .map(|clip| {
            serde_json::json!({
                "id": clip.id,
                "track_id": clip.track_id,
                "name": clip.name,
                "start_sec": clip.start_sec,
                "length_sec": clip.length_sec,
                "source_start_sec": clip.source_start_sec,
                "source_end_sec": clip.source_end_sec,
                "playback_rate": clip.playback_rate,
                "clip_playback_rate": clip.clip_playback_rate,
                "gain": clip.gain,
                "muted": clip.muted,
                "snap_offset_sec": clip.snap_offset_sec,
                "fade_in_sec": clip.fade_in_sec,
                "fade_out_sec": clip.fade_out_sec,
                "fade_in_shape": clip.fade_in_shape,
                "fade_out_shape": clip.fade_out_shape,
                "fade_in_dir": clip.fade_in_dir,
                "fade_out_dir": clip.fade_out_dir,
                "auto_fade_in_sec": clip.auto_fade_in_sec,
                "auto_fade_out_sec": clip.auto_fade_out_sec,
            })
        })
        .collect();
    let tracks: Vec<_> = normalized
        .tracks
        .iter()
        .map(|track| {
            serde_json::json!({
                "id": track.id,
                "name": track.name,
                "order": track.order,
                "parent_id": track.parent_id,
            })
        })
        .collect();
    serde_json::json!({ "clips": clips, "tracks": tracks })
}

/// 逐路径收集两棵几何投影的差异，最多 [`GEOMETRY_DIFF_LIMIT`] 条。
///
/// 拒绝时用户唯一能据以行动的信息就是这份路径列表：说「几何变了」没有用，
/// 说 `clips[2].start_sec: 0.0 -> 0.25` 才有用。
pub fn geometry_differences(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
) -> Vec<String> {
    let mut out = Vec::new();
    collect_json_differences(baseline, current, "", GEOMETRY_DIFF_LIMIT, &mut out);
    out
}

fn collect_json_differences(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
    path: &str,
    limit: usize,
    out: &mut Vec<String>,
) {
    if out.len() >= limit {
        return;
    }
    match (baseline, current) {
        (serde_json::Value::Object(before), serde_json::Value::Object(after)) => {
            for (key, value) in before {
                if out.len() >= limit {
                    return;
                }
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                match after.get(key) {
                    Some(other) => collect_json_differences(value, other, &child, limit, out),
                    None => out.push(format!("{child}: removed")),
                }
            }
            for key in after.keys() {
                if out.len() >= limit {
                    return;
                }
                if !before.contains_key(key) {
                    let child = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    out.push(format!("{child}: added"));
                }
            }
        }
        (serde_json::Value::Array(before), serde_json::Value::Array(after)) => {
            if before.len() != after.len() {
                out.push(format!(
                    "{path}: length {} -> {}",
                    before.len(),
                    after.len()
                ));
                return;
            }
            for (index, (x, y)) in before.iter().zip(after.iter()).enumerate() {
                if out.len() >= limit {
                    return;
                }
                collect_json_differences(x, y, &format!("{path}[{index}]"), limit, out);
            }
        }
        _ => {
            if !json_values_agree(baseline, current) {
                out.push(format!(
                    "{path}: {} -> {}",
                    compact_json_value(baseline),
                    compact_json_value(current)
                ));
            }
        }
    }
}

/// 两个标量是否**在判据意义上**相等。
///
/// 【为什么浮点要带容差】这个判据回答的是「宿主的时间轴变了吗」，不是「比特相同吗」。
/// 而同一份 `TimelineState` 经过不同的计算路径会给出最后一位不同的浮点值：
/// 实测同一份导出，`length_sec` 在两次投影之间差 1 ULP（`9.0702947845805e-5` 对
/// `9.070294784580499e-5`）。把这种噪声报成"你改了宿主字段"，正是旧实现里
/// 用户无法据以行动的那一类假失败。
///
/// 【容差为什么是 1e-9】它只在**远小于**任何真实编辑的尺度上放行：最细的编辑也是
/// 一个采样点（48 kHz 下 20 µs = 2e-5 s），比容差大四个数量级；
/// 而 ULP 噪声在 1e-4 这个量级上只有 1e-20 s，比容差小十一个数量级。
/// 两边各有巨大的余量，所以它既不会掩饰真实改动，也不会再制造假失败。
fn json_values_agree(baseline: &serde_json::Value, current: &serde_json::Value) -> bool {
    if let (Some(a), Some(b)) = (baseline.as_f64(), current.as_f64()) {
        return (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
    }
    baseline == current
}

fn compact_json_value(value: &serde_json::Value) -> String {
    let text = value.to_string();
    if text.chars().count() <= 64 {
        return text;
    }
    let mut truncated: String = text.chars().take(61).collect();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::state::TimelineState;

    fn timeline_with_one_clip() -> TimelineState {
        serde_json::from_value(serde_json::json!({
            "bpm": 120.0,
            "project_sec": 8.0,
            "tracks": [{"id": "t1", "name": "Track", "order": 0}],
            "clips": [{
                "id": "c1", "track_id": "t1", "name": "Clip",
                "start_sec": 0.0, "length_sec": 2.0,
                "source_start_sec": 0.0, "source_end_sec": 2.0,
                "playback_rate": 1.0,
            }],
        }))
        .unwrap()
    }

    /// 同一份时间线的两次投影必须逐位相同 —— 否则每次提交都会误报漂移。
    #[test]
    fn the_projection_is_deterministic() {
        let timeline = timeline_with_one_clip();
        assert_eq!(
            host_geometry_projection(&timeline),
            host_geometry_projection(&timeline)
        );
    }

    /// **参数改动不是几何漂移。** 曲线与轨道混音由用户拥有，不属于正向名单。
    #[test]
    fn parameter_edits_are_not_geometry_drift() {
        let before = timeline_with_one_clip();
        let mut after = before.clone();
        after.tracks[0].volume = 0.5;
        after.tracks[0].muted = true;
        after.params_by_root_track.insert(
            "t1".into(),
            serde_json::from_value(serde_json::json!({
                "frame_period_ms": 10.0,
                "pitch_edit": [1.0, 2.0],
            }))
            .unwrap(),
        );
        assert_eq!(
            host_geometry_projection(&before),
            host_geometry_projection(&after)
        );
        assert!(geometry_differences(
            &host_geometry_projection(&before),
            &host_geometry_projection(&after)
        )
        .is_empty());
    }

    /// **几何改动必须被指名字段地报出来。**
    #[test]
    fn geometry_drift_names_the_field() {
        let before = timeline_with_one_clip();
        let mut after = before.clone();
        after.clips[0].start_sec = 0.25;
        after.clips[0].fade_in_sec = 0.1;
        let differences = geometry_differences(
            &host_geometry_projection(&before),
            &host_geometry_projection(&after),
        );
        assert!(
            differences
                .iter()
                .any(|line| line.contains("clips[0].start_sec")),
            "{differences:?}"
        );
        assert!(
            differences
                .iter()
                .any(|line| line.contains("clips[0].fade_in_sec")),
            "{differences:?}"
        );
    }

    /// 删除与新增都是差异，且长度变化被单独指出。
    #[test]
    fn clip_add_and_remove_are_both_drift() {
        let before = timeline_with_one_clip();
        let mut after = before.clone();
        after.clips.clear();
        assert!(
            geometry_differences(
                &host_geometry_projection(&before),
                &host_geometry_projection(&after)
            )
            .iter()
            .any(|line| line.contains("length 1 -> 0")),
            "整条 clips 数组的变化必须被指出"
        );
    }

    /// **最后一位的浮点噪声不是漂移**，而真实的编辑仍然是漂移。
    ///
    /// 【为什么这条必须钉住】同一份时间轴经过不同的计算路径会给出 1 ULP 的差
    /// （JSON 往返、f32/f64、take 归一化顺序都会）。把噪声报成"你改了宿主字段"，
    /// 用户拿到的是一句无法据以行动的指控 —— 这正是旧实现最大的可用性问题。
    #[test]
    fn last_bit_float_noise_is_not_drift_but_a_real_edit_is() {
        let mut before = timeline_with_one_clip();
        before.clips[0].length_sec = 9.0702947845805e-5;
        let mut noisy = before.clone();
        noisy.clips[0].length_sec = 9.070294784580499e-5;
        assert!(
            geometry_differences(
                &host_geometry_projection(&before),
                &host_geometry_projection(&noisy)
            )
            .is_empty(),
            "1 ULP 的差不是漂移"
        );

        // 一个采样点（48 kHz 下 20 µs）是真实编辑，必须被报出来。
        let mut edited = before.clone();
        edited.clips[0].length_sec += 2.0e-5;
        let differences = geometry_differences(
            &host_geometry_projection(&before),
            &host_geometry_projection(&edited),
        );
        assert!(
            differences.iter().any(|line| line.contains("length_sec")),
            "{differences:?}"
        );
    }

    /// 差异列表必须有上限 —— 拒绝原因是给人看的。
    #[test]
    fn differences_are_bounded() {
        let before = timeline_with_one_clip();
        let mut after = before.clone();
        let mut clips = Vec::new();
        for index in 0..64 {
            let mut clip = after.clips[0].clone();
            clip.id = format!("c{index}");
            clip.start_sec = index as f64;
            clips.push(clip);
        }
        after.clips = clips;
        let differences = geometry_differences(
            &host_geometry_projection(&before),
            &host_geometry_projection(&after),
        );
        assert!(differences.len() <= GEOMETRY_DIFF_LIMIT, "{differences:?}");
    }
}
