//! 宿主 item 优先的 Clip 装配 —— **唯一**的一份 clip 集合。
//!
//! ## 这个模块存在的理由
//!
//! 插件里同一个 Clip 曾经同时由两处产生：
//!
//! - `ara/mapping.rs` 按 **ARA playback region** 造一条（每节循环一条）；
//! - `editor/workspace.rs::present_host_inventory` 又按 **宿主 item** 造一条。
//!
//! 两者之间靠"四项几何全等且唯一"关联。而 REAPER 对一个**循环延伸**的 item 会按
//! 循环节给出多条 region —— 每节一条，item 自身的 `D_LENGTH` 却是整段。于是这些
//! region 与 item **永远不可能全等**：它们既拿不到 item 身份，也不会被任何归并逻辑
//! 收走，结果就是"延伸出一个循环，界面上多出 N 条片段，而且一条都拖不动"。
//!
//! 修法不是把归并写得更聪明 —— 那只会把漏洞推到下一个不一致的维度（长度、速率、
//! 源窗、方向……）。修法是**让 clip 只有一个来源**：
//!
//! > 宿主暴露 item 时，**一个 item 恰好一条 clip**；
//! > region 不再产生 clip，它只回答"这段时间播放素材的哪一段"。
//!
//! ## 为什么是纯函数、为什么在内核
//!
//! 输入是两张表（item 表 + region 表），输出是一份 clip 列表。它不碰宿主、不碰锁、
//! 不碰 ARA —— 因此可以（也必须）放在 `hifishifter-kernel` 里被单测覆盖。
//!
//! ## 与"今日行为"的关系
//!
//! `items` 为空时本模块的行为与旧的 `ara_document_to_timeline` **逐字相同**
//! （ARA-only 宿主：Cubase / Studio One 等不暴露 item 的后端）。所有既有映射测试
//! 都跑在这条回退路径上，因此"零回归"是可断言的，而不是口头承诺。

use crate::state::{Clip, TimelineState};
use crate::take_geometry::TakeGeometry;
use serde_json::json;
use std::collections::BTreeMap;

/// "一条 region 落在一条 item 之内"的容差（秒）。
///
/// 与 `host::geometry::host_value_compatible` 的绝对项同量级：两者都回答
/// "这两个浮点读数是不是同一个事实"，用不同的阈值会让"绑定接受、装配拒绝"
/// 这类不对称重新出现。
pub const CONTAINMENT_EPS_SEC: f64 = 1e-6;

/// 一条 ARA playback region。**输入**，不是 clip。
///
/// 字段是**已经折算到内核域**的读数（源窗口已按工程速率校正、起点已按
/// `projectRate / sourceRate` 折算）—— 折算逻辑留在插件侧的映射层，本模块只做装配。
#[derive(Debug, Clone, PartialEq)]
pub struct RegionRow {
    /// ARA 在同一文档内保证稳定的 region key；`0` = 宿主没给。
    pub key: u64,
    /// 这条 region 若**单独成 clip**（ARA-only 宿主）时使用的 id。
    ///
    /// 【为什么由调用方给定而不是这里派生】身份来自 region key 与宿主的
    /// model-ref（见插件侧 `MappingIdentities`），而"key 缺失时退回文档下标"
    /// 这条规则只有映射层知道。装配器再派生一次，就等于把同一份身份写成两个来源 ——
    /// 那正是这一轮要删掉的东西。
    pub clip_id: String,
    /// 归属的模型轨道 id（由映射层的 region-sequence 归属给出）。
    pub track_id: String,
    /// 播放时间轴上的起点（秒）。
    pub start_sec: f64,
    /// 播放时间轴上的时长（秒）。
    pub duration_sec: f64,
    /// 已折算到源域的消费窗口起点（秒）。
    pub source_start_sec: f64,
    /// 已折算到源域的消费窗口终点（秒）。
    pub source_end_sec: f64,
    /// 宿主声明的拉伸比值 × 速率校正。
    pub playback_rate: f64,
    /// 片段显示名（region 名 → 源名 → id）。
    pub name: String,
    /// ARA 音频源 persistentID —— 内核渲染层用它找 PCM（**不是**文件路径）。
    pub source_id: String,
    /// ARA 音频修改（≈ take）persistentID。
    ///
    /// 【为什么装配需要它】`TrackBindings`（跨会话的轨道签名）由
    /// `(modification, source)` 组成，而它是"这条轨还是不是原来那条轨"的唯一凭据。
    /// 少了它，装配产物就再也拼不出那张表。
    pub modification_id: String,
    pub source_sample_rate: f64,
    /// ARA `AudioSource.durationSeconds`；`0` = 未知。
    pub source_duration_sec: f64,
    pub source_channels: i32,
}

/// 一条 take 的宿主读数 —— 与 [TakeGeometry] **逐字段同形**，
/// 这样装配器可以直接把解算交给内核既有实现，而不是再写一遍镜像数学。
#[derive(Debug, Clone, PartialEq)]
pub struct TakeRow {
    /// 宿主 take GUID（跨会话稳定）。
    pub id: String,
    pub name: String,
    pub active: bool,
    pub geometry: TakeGeometry,
    pub channel_mode: i32,
}

/// 一个宿主 item —— **输入**，一行。
#[derive(Debug, Clone, PartialEq)]
pub struct ItemRow {
    /// 宿主 item GUID（REAPER 保证唯一且跨会话稳定）。
    pub guid: String,
    /// 宿主轨道的模型 id（清单层给出的线索，见 [assemble] 的轨道归属说明）。
    pub track_id: String,
    pub name: String,
    /// `D_POSITION`
    pub start_sec: f64,
    /// `D_LENGTH`（循环延伸之后是**整段**长度）
    pub length_sec: f64,
    pub takes: Vec<TakeRow>,
    /// active take 在 `takes` 里的下标；越界时取 0。
    pub active_take: usize,
    /// `B_LOOPSRC`
    pub loop_source: bool,
    pub muted: bool,
    /// `D_VOL`（item 级增益）
    pub gain: f64,
}

/// 一条轨道（装配器只负责把它原样写进时间线）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackRow {
    pub id: String,
    pub name: String,
    pub order: i32,
    pub color: String,
}

/// 装配结果：唯一的一份 clip 集合，外加两张**诊断/授权**用的边。
#[derive(Debug, Clone, Default)]
pub struct Assembly {
    pub clips: Vec<Clip>,
    /// clip id → 落在它里面的 region 行下标（按 region 起点升序）。
    /// 循环 item 时长度 = 循环节数；**渲染不读它**（循环由 `loop_enabled` 表达）。
    pub spans: BTreeMap<String, Vec<usize>>,
    /// 没被任何 item 吃掉的 region 行下标。**它们不产生 clip**，只进读数。
    pub unbound: Vec<usize>,
    /// 周期未知的循环 clip id（见 §B2）：`loop_period_sec` 无法从宿主读数确定。
    pub unknown_loop_period: Vec<String>,
}

impl Assembly {
    /// 把装配结果写成一份时间线。
    pub fn timeline(&self, tracks: &[TrackRow], bpm: f64) -> TimelineState {
        let project_sec = self
            .clips
            .iter()
            .map(|clip| clip.start_sec + clip.length_sec.max(0.0))
            .fold(0.0_f64, f64::max);
        let mut timeline = TimelineState::default();
        timeline.tracks = tracks
            .iter()
            .filter_map(|track| {
                serde_json::from_value::<crate::state::Track>(json!({
                    "id": track.id,
                    "name": track.name,
                    "order": track.order,
                    "color": track.color,
                }))
                .ok()
            })
            .collect();
        timeline.clips = self.clips.clone();
        timeline.bpm = bpm;
        timeline.project_sec = project_sec;
        timeline
    }

    /// 一个 clip 吃掉了几条 region（读数用；`1` 是普通 item 的常态）。
    pub fn regions_per_clip(&self) -> BTreeMap<&str, usize> {
        self.spans
            .iter()
            .map(|(id, rows)| (id.as_str(), rows.len()))
            .collect()
    }
}

/// 一条 region 是否属于一条 item。
///
/// 判据只有**包含**一条：
///
/// ```text
/// row.start >= item.start − EPS
/// row.start + row.duration <= item.start + item.length + EPS
/// ```
///
/// 【为什么不是"几何全等"】循环延伸的 item 给出的是 N 条**每节一条**的 region，
/// 它们与整段 item 永远不全等 —— 旧判据在这里恒为假，这正是用户报障的成因。
///
/// 【为什么不是"相交"】相交会让一条跨越两条 item 的 region 同时挂两边；包含是
/// REAPER 的实际形状（region 永远在 item 之内），而越界的 region 应当被**计数**
/// 而不是被猜。
///
/// 【为什么没有"同源"判据】ARA 的 `source.persistentID` 与宿主
/// `GetMediaSourceFileName` 是两个互不相通的域（一个是宿主分配的不透明句柄，
/// 一个是文件路径），仓库里没有任何一处能把它们对上。硬造一个"文件名 == 源名"
/// 的启发式，正是这一轮要删掉的那类猜测：它会在"两个 item 用同一个文件"的常见
/// 布局上给出**错的**结论，而包含判据在那里的答案本来就是对的。
pub fn region_belongs_to(row: &RegionRow, item: &ItemRow) -> bool {
    if !(row.start_sec.is_finite() && row.duration_sec.is_finite()) {
        return false;
    }
    if !(item.start_sec.is_finite() && item.length_sec.is_finite()) {
        return false;
    }
    let row_start = row.start_sec;
    let row_end = row.start_sec + row.duration_sec.max(0.0);
    let item_start = item.start_sec;
    let item_end = item.start_sec + item.length_sec.max(0.0);
    row_start >= item_start - CONTAINMENT_EPS_SEC && row_end <= item_end + CONTAINMENT_EPS_SEC
}

/// 为每条 region 选一个**唯一**的宿主 item。
///
/// 候选里取**跨度最小**的那个；跨度相同取**起点最小**；再相同 ⇒ 判为歧义、不挂。
///
/// 【为什么"最小跨"能消歧】宿主把一条 item 拆成多段 region 时，region 的跨度严格
/// 小于 item 的跨度；两条 item 时间重叠时，"把这条 region 抱得最紧的那一条"就是
/// 它真正所属的那一条。两者都相同时（两个几何完全重合的 item）任何判据都只能是
/// 猜 —— 猜错的代价是写回到**别的** item 上，所以这里如实返回 `None`。
pub fn attach(items: &[ItemRow], rows: &[RegionRow]) -> Vec<Option<usize>> {
    rows.iter()
        .map(|row| {
            let mut best: Option<(f64, f64, usize)> = None;
            let mut ambiguous = false;
            for (index, item) in items.iter().enumerate() {
                if !region_belongs_to(row, item) {
                    continue;
                }
                let span = item.length_sec.max(0.0);
                let start = item.start_sec;
                match best {
                    None => {
                        best = Some((span, start, index));
                        ambiguous = false;
                    }
                    Some((best_span, best_start, _)) => {
                        // 浮点相等按"宿主读数相容"判，而不是按位相等：同一个
                        // 几何在两处被独立解算时，尾差必然存在。
                        let same_span = (span - best_span).abs() <= CONTAINMENT_EPS_SEC;
                        let same_start = (start - best_start).abs() <= CONTAINMENT_EPS_SEC;
                        if same_span && same_start {
                            ambiguous = true;
                        } else if span < best_span - CONTAINMENT_EPS_SEC
                            || (same_span && start < best_start)
                        {
                            best = Some((span, start, index));
                            ambiguous = false;
                        }
                    }
                }
            }
            if ambiguous {
                None
            } else {
                best.map(|(_, _, index)| index)
            }
        })
        .collect()
}

/// 装配：**唯一**的 clip 集合。
///
/// 规则（顺序即优先级）：
///
/// 1. `items` 为空 ⇒ 每条 region 一条 clip（ARA-only 宿主，零回归）。
/// 2. 否则 ⇒ 每个 item 一条 clip，id = `ara-item-{guid}`。
/// 3. 每条 region 挂到唯一一个 item 上（见 [attach]）；挂不上的进 `unbound`。
/// 4. clip 的**工程几何**只来自 item；clip 的**源窗口**只来自 take 几何。
///
/// 【轨道归属】item 有 region 时取**第一条 region 的轨道**，否则取 `item.track_id`。
/// 这条规则不是随手写的：插件里"轨道"的模型身份由 ARA region sequence 派生
/// （`ara-sequence-{key}`），参数曲线按它索引；宿主轨道的 GUID 只是**呈现**层的
/// 命名。让 item 的 clip 落到它自己 region 的轨道上，参数曲线才不会在"宿主清单
/// 迟到一轮"时被整批换键。
pub fn assemble(items: &[ItemRow], rows: &[RegionRow]) -> Assembly {
    let attached = attach(items, rows);
    let mut assembly = Assembly::default();
    if items.is_empty() {
        for (index, row) in rows.iter().enumerate() {
            match clip_for_region(row) {
                Ok(clip) => {
                    // 【回退路径也要记 `spans`】它不只是"一个 clip 吃掉几条 region"的
                    // 读数：`clip_ids`（region key → clip id）就是由它反算的。少了这一
                    // 行，ARA-only 宿主上 `clip_ids` 会恒为空 —— 而
                    // `workspace_timeline_locked` 按它过滤，于是**一条 clip 都发布不出去**。
                    assembly.spans.insert(clip.id.clone(), vec![index]);
                    assembly.clips.push(clip);
                }
                Err(error) => log::warn!("[ara] region {} cannot be assembled: {error}", row.key),
            }
        }
        return assembly;
    }
    // 每条 item 的归属 region（保持 region 顺序；下面按起点排序一次）。
    let mut owned: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    for (index, owner) in attached.iter().enumerate() {
        match owner {
            Some(item) => owned[*item].push(index),
            None => assembly.unbound.push(index),
        }
    }
    for (item_index, item) in items.iter().enumerate() {
        let mut mine = owned[item_index].clone();
        mine.sort_by(|a, b| {
            rows[*a]
                .start_sec
                .partial_cmp(&rows[*b].start_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // 轨道归属：第一条 region 的轨道优先（见函数文档）。
        let track_id = mine
            .first()
            .map(|index| rows[*index].track_id.clone())
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| item.track_id.clone());
        // 媒体身份：取第一条 region 指向的 ARA 源。**它是文档级事实** —— 与
        // "这一轮有没有被授权 PCM"无关（见 §4.5），因此重开工程后仍然成立。
        let media = mine.first().map(|index| &rows[*index]);
        match clip_for_item(item, &track_id, media) {
            Ok(clip) => {
                if clip.loop_enabled && clip.loop_period_sec.is_none() {
                    assembly.unknown_loop_period.push(clip.id.clone());
                }
                if !mine.is_empty() {
                    assembly.spans.insert(clip.id.clone(), mine);
                }
                assembly.clips.push(clip);
            }
            Err(error) => log::warn!("[ara] host item {} cannot be assembled: {error}", item.guid),
        }
    }
    assembly
}

/// 一条 region 的 clip（ARA-only 宿主的回退路径）。
///
/// 【为什么不直接用结构体字面量】`Clip` 有五十多个字段且全部依赖 serde 默认值；
/// 手写字面量意味着每一处默认值都有**两个**来源（serde 与这里），而它们迟早会分叉。
/// 走同一条反序列化路径，本函数产出的 clip 与"产品加载工程时"的 clip 逐字段同构。
pub fn clip_for_region(row: &RegionRow) -> Result<Clip, String> {
    let clip_id = row.clip_id.clone();
    let take_id = format!("{clip_id}-take-1");
    let name = if row.name.trim().is_empty() {
        clip_id.clone()
    } else {
        row.name.clone()
    };
    build(json!({
        "id": clip_id,
        "track_id": row.track_id,
        "name": name,
        "start_sec": row.start_sec,
        "length_sec": row.duration_sec.max(0.0),
        "active_take_id": take_id,
        "clip_playback_rate": 1.0,
        "takes": [{
            "id": take_id,
            "name": name,
            "source_path": row.source_id,
            "source_start_sec": row.source_start_sec,
            "source_end_sec": row.source_end_sec,
            "playback_rate": row.playback_rate,
            // 【必须是整型】`ClipTake::source_sample_rate` 是 `Option<u32>`；
            // 传浮点会让整条 clip 反序列化失败（而失败是静默的 —— 用户看到的
            // 只是"片段不见了"）。
            "source_sample_rate": row.source_sample_rate.max(0.0) as u32,
            "duration_sec": row.source_duration_sec,
            "source_channels": row.source_channels.max(0) as u16,
        }],
    }))
}

/// 一个宿主 item 的 clip。
///
/// `media` 是**已挂到本 item 上**的第一条 region（没有则 `None`）：它只提供
/// **媒体身份**（ARA 源 persistentID），不提供任何几何 —— 几何必须来自 item 与 take，
/// 否则"循环节的长度"会再一次盖住"整段的长度"（这正是原来的缺陷）。
pub fn clip_for_item(
    item: &ItemRow,
    track_id: &str,
    media: Option<&RegionRow>,
) -> Result<Clip, String> {
    let active_index = if item.active_take < item.takes.len() {
        item.active_take
    } else {
        0
    };
    let clip_id = format!("ara-item-{}", item.guid);
    let name = if !item.name.trim().is_empty() {
        item.name.clone()
    } else {
        item.takes
            .get(active_index)
            .map(|take| take.name.clone())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| clip_id.clone())
    };
    let window_of = |take: &TakeRow| -> (f64, f64) {
        take.geometry
            .forward_source_window()
            .map(|window| (window.start(), window.end()))
            // 【拿不到窗口时保留宿主坐标】丢掉它等于"用户在宿主里的片段从插件里消失"，
            // 那比"偏移待定"严重得多。方向读不到（`reversed: None`）同理：界面仍然
            // 看得见这一段（可以选中、可以画线），而渲染与投影**不会**把它当成正放
            // （见 `project_host_take_facts_locked` 的 `ara.reversed-window`）。
            .unwrap_or_else(|| {
                let end = take.geometry.start_offset_sec + take.geometry.consumed_sec();
                (take.geometry.start_offset_sec, end)
            })
    };
    let loop_period_sec = if item.loop_source {
        item.takes
            .get(active_index)
            .and_then(|take| take.geometry.loop_period_sec())
    } else {
        None
    };
    let takes: Vec<serde_json::Value> = item
        .takes
        .iter()
        .map(|take| {
            let (start, end) = window_of(take);
            json!({
                "id": take.id,
                "name": take.name,
                "source_path": media.map(|row| row.source_id.clone()),
                "source_start_sec": start,
                "source_end_sec": end,
                "playback_rate": take.geometry.play_rate_magnitude(),
                "reversed": take.geometry.is_reversed(),
                "loop_enabled": item.loop_source,
                "channel_mode": take.channel_mode,
                "source_sample_rate": media.map(|row| row.source_sample_rate.max(0.0) as u32),
                "duration_sec": media.map(|row| row.source_duration_sec),
                "source_channels": media.map(|row| row.source_channels.max(0) as u16),
            })
        })
        .collect();
    let active_take_id = item.takes.get(active_index).map(|take| take.id.clone());
    build(json!({
        "id": clip_id,
        "track_id": track_id,
        "name": name,
        "start_sec": item.start_sec,
        "length_sec": item.length_sec.max(0.0),
        "active_take_id": active_take_id,
        "clip_playback_rate": 1.0,
        "clip_gain": item.gain,
        "muted": item.muted,
        "loop_period_sec": loop_period_sec,
        "takes": takes,
    }))
}

fn build(value: serde_json::Value) -> Result<Clip, String> {
    let mut clip: Clip = serde_json::from_value(value).map_err(|error| error.to_string())?;
    // 把 active take 物化到扁平投影 —— 与产品加载工程时走同一步。
    clip.normalize_takes();
    Ok(clip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take_geometry::SourceSection;

    fn geometry(
        soffs: f64,
        item_length: f64,
        media: Option<f64>,
        section: Option<SourceSection>,
        reversed: bool,
        looped: bool,
    ) -> TakeGeometry {
        TakeGeometry {
            start_offset_sec: soffs,
            item_length_sec: item_length,
            play_rate_signed: 1.0,
            media_duration_sec: media,
            section,
            reversed: Some(reversed),
            loop_enabled: looped,
        }
    }

    fn take(id: &str, geometry: TakeGeometry) -> TakeRow {
        TakeRow {
            id: id.to_string(),
            name: id.to_string(),
            active: true,
            geometry,
            channel_mode: 0,
        }
    }

    fn item(guid: &str, start: f64, length: f64, takes: Vec<TakeRow>) -> ItemRow {
        ItemRow {
            guid: guid.to_string(),
            track_id: format!("host-track-{guid}"),
            name: guid.to_string(),
            start_sec: start,
            length_sec: length,
            takes,
            active_take: 0,
            loop_source: false,
            muted: false,
            gain: 1.0,
        }
    }

    fn row(key: u64, start: f64, duration: f64) -> RegionRow {
        RegionRow {
            key,
            clip_id: format!("ara-region-{key}"),
            track_id: "track-a".to_string(),
            start_sec: start,
            duration_sec: duration,
            source_start_sec: 0.0,
            source_end_sec: 3.0,
            playback_rate: 1.0,
            name: format!("r{key}"),
            source_id: format!("source-{key}"),
            modification_id: format!("modification-{key}"),
            source_sample_rate: 44_100.0,
            source_duration_sec: 3.0,
            source_channels: 2,
        }
    }

    fn track(id: &str) -> TrackRow {
        TrackRow {
            id: id.to_string(),
            name: id.to_string(),
            order: 0,
            color: String::new(),
        }
    }

    /// K1：没有宿主清单（ARA-only 后端）时，每条 region 一条 clip，id 与旧映射逐字相同。
    ///
    /// 【为什么这是本轮最要紧的一条回归】Cubase / Studio One 不暴露 item 枚举，
    /// 它们走的就是这条回退路径。回退路径一变，所有既有映射测试都会失去意义。
    #[test]
    fn k1_no_items_yields_one_clip_per_region_with_the_legacy_ids() {
        let rows = vec![row(11, 0.0, 1.0), row(22, 1.0, 2.0), row(33, 3.0, 0.5)];
        let assembly = assemble(&[], &rows);
        assert_eq!(assembly.clips.len(), 3);
        assert_eq!(
            assembly
                .clips
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>(),
            vec!["ara-region-11", "ara-region-22", "ara-region-33"]
        );
        // 回退路径同样记 `spans`（1 条 region → 1 条 clip）：`clip_ids` 就是由它
        // 反算的，缺了它 ARA-only 宿主上一条 clip 都发布不出去。
        assert_eq!(assembly.spans.len(), 3);
        assert_eq!(assembly.spans["ara-region-22"], vec![1]);
        assert!(assembly.unbound.is_empty());
        // 源窗口与倍率来自 region 行本身。
        assert_eq!(assembly.clips[1].start_sec, 1.0);
        assert_eq!(assembly.clips[1].length_sec, 2.0);
        assert_eq!(assembly.clips[1].source_path.as_deref(), Some("source-22"));
        assert_eq!(assembly.clips[1].source_start_sec, 0.0);
        assert_eq!(assembly.clips[1].source_end_sec, 3.0);
    }

    /// K2：**clip 条数 == item 条数**。
    #[test]
    fn k2_clip_count_equals_item_count() {
        let items: Vec<ItemRow> = (0..12)
            .map(|index| {
                item(
                    &format!("guid-{index}"),
                    index as f64,
                    1.0,
                    vec![take("t", geometry(0.0, 1.0, Some(3.0), None, false, false))],
                )
            })
            .collect();
        let rows: Vec<RegionRow> = (0..12)
            .map(|index| row(index as u64 + 1, index as f64, 1.0))
            .collect();
        let assembly = assemble(&items, &rows);
        assert_eq!(assembly.clips.len(), 12);
        assert!(assembly.unbound.is_empty());
        for (index, clip) in assembly.clips.iter().enumerate() {
            assert_eq!(clip.id, format!("ara-item-guid-{index}"));
        }
    }

    /// K3：**用户报障 1**。一个 3 s 素材被循环延伸到 12 s，宿主给出 4 条每节一条的
    /// region —— 装配结果必须是**一条** 12 s 的 clip，而不是 1 + 4 条。
    #[test]
    fn k3_a_loop_extended_item_presents_exactly_one_clip() {
        let mut host = item(
            "loop",
            0.0,
            12.0,
            vec![take("t", geometry(0.0, 12.0, Some(3.0), None, false, true))],
        );
        host.loop_source = true;
        let rows: Vec<RegionRow> = (0..4)
            .map(|index| row(index + 1, index as f64 * 3.0, 3.0))
            .collect();
        let assembly = assemble(&[host], &rows);
        assert_eq!(assembly.clips.len(), 1, "一个 item 只允许一条 clip");
        let clip = &assembly.clips[0];
        assert_eq!(clip.id, "ara-item-loop");
        assert_eq!(clip.length_sec, 12.0, "长度必须是整段，不是一节");
        assert!(clip.loop_enabled);
        assert_eq!(clip.loop_period_sec, Some(3.0), "周期是媒体总长");
        assert_eq!(assembly.spans["ara-item-loop"], vec![0, 1, 2, 3]);
        assert!(assembly.unbound.is_empty(), "4 条 region 全部有归属");
    }

    /// K4：包含判据。落在 item 之内 ⇒ 挂上；跨出 item 边界 ⇒ 不挂。
    #[test]
    fn k4_containment_decides_and_crossing_the_boundary_does_not_attach() {
        let host = item(
            "a",
            2.0,
            4.0,
            vec![take(
                "t",
                geometry(0.0, 4.0, Some(10.0), None, false, false),
            )],
        );
        let inside = row(1, 3.0, 2.0);
        let outside = row(2, 5.0, 4.0);
        assert!(region_belongs_to(&inside, &host));
        assert!(!region_belongs_to(&outside, &host));
        let assembly = assemble(&[host], &[inside, outside]);
        assert_eq!(assembly.clips.len(), 1);
        assert_eq!(assembly.unbound, vec![1]);
    }

    /// K5：两条 item 都包含同一条 region ⇒ 取**跨度最小**的那一条。
    #[test]
    fn k5_the_tightest_containing_item_wins() {
        let wide = item(
            "wide",
            0.0,
            10.0,
            vec![take(
                "t",
                geometry(0.0, 10.0, Some(20.0), None, false, false),
            )],
        );
        let tight = item(
            "tight",
            2.0,
            3.0,
            vec![take(
                "t",
                geometry(0.0, 3.0, Some(20.0), None, false, false),
            )],
        );
        let assembly = assemble(&[wide, tight], &[row(7, 2.5, 2.0)]);
        assert_eq!(assembly.clips.len(), 2);
        assert_eq!(assembly.spans["ara-item-tight"], vec![0]);
        assert!(!assembly.spans.contains_key("ara-item-wide"));
    }

    /// K6：跨度与起点都相同的两条 item ⇒ 判**歧义**、不挂、不画。
    ///
    /// 【为什么宁可丢】猜错的代价是写回到**别的** item 上（移动、切分、fade、音量
    /// 互相覆盖且没有一侧报错）。宁可如实报"认不出"，也不猜。
    #[test]
    fn k6_identical_candidates_are_ambiguous_and_stay_unbound() {
        let a = item(
            "a",
            0.0,
            4.0,
            vec![take("t", geometry(0.0, 4.0, Some(9.0), None, false, false))],
        );
        let b = item(
            "b",
            0.0,
            4.0,
            vec![take("t", geometry(0.0, 4.0, Some(9.0), None, false, false))],
        );
        let assembly = assemble(&[a, b], &[row(1, 0.0, 4.0)]);
        assert_eq!(assembly.clips.len(), 2);
        assert_eq!(assembly.unbound, vec![0]);
        assert!(assembly.spans.is_empty());
    }

    /// K7：一条完全落在所有 item 之外的 region ⇒ 进 unbound，**不产生 clip**。
    #[test]
    fn k7_a_region_outside_every_item_produces_no_clip() {
        let host = item(
            "a",
            0.0,
            2.0,
            vec![take("t", geometry(0.0, 2.0, Some(9.0), None, false, false))],
        );
        let assembly = assemble(&[host], &[row(1, 100.0, 1.0)]);
        assert_eq!(assembly.clips.len(), 1);
        assert_eq!(assembly.unbound, vec![0]);
        assert!(assembly.spans.is_empty());
    }

    /// K8：正放 / 倒放 / 循环三种 take 的源窗口与 [TakeGeometry] 逐字段相同。
    #[test]
    fn k8_the_source_window_comes_from_the_shared_geometry_solver() {
        for (reversed, looped, soffs) in [
            (false, false, 0.5),
            (true, false, 0.75),
            (false, true, 0.5),
            (true, true, 0.75),
        ] {
            let host = item(
                "a",
                0.0,
                1.0,
                vec![take(
                    "t",
                    geometry(soffs, 1.0, Some(2.0), None, reversed, looped),
                )],
            );
            let expected = host.takes[0].geometry.forward_source_window().unwrap();
            let assembly = assemble(&[host], &[]);
            let clip = &assembly.clips[0];
            assert!(
                (clip.source_start_sec - expected.start()).abs() < 1e-9,
                "reversed={reversed} looped={looped}: start {} vs {}",
                clip.source_start_sec,
                expected.start()
            );
            assert!(
                (clip.source_end_sec - expected.end()).abs() < 1e-9,
                "reversed={reversed} looped={looped}: end {} vs {}",
                clip.source_end_sec,
                expected.end()
            );
        }
    }

    /// K9：有 SECTION 时周期 = 段长；没有时 = 源总长。
    #[test]
    fn k9_the_loop_period_prefers_the_section_length() {
        let section = SourceSection {
            start_sec: 1.0,
            length_sec: Some(2.5),
        };
        let mut with_section = item(
            "s",
            0.0,
            7.0,
            vec![take(
                "t",
                geometry(0.0, 7.0, Some(9.0), Some(section), false, true),
            )],
        );
        with_section.loop_source = true;
        let mut without = item(
            "p",
            0.0,
            7.0,
            vec![take("t", geometry(0.0, 7.0, Some(9.0), None, false, true))],
        );
        without.loop_source = true;
        let assembly = assemble(&[with_section, without], &[]);
        assert_eq!(assembly.clips[0].loop_period_sec, Some(2.5));
        assert_eq!(assembly.clips[1].loop_period_sec, Some(9.0));
    }

    /// K11：没有任何 region 的 item 仍然产生一条 clip（无音频），不被丢掉。
    #[test]
    fn k11_an_item_without_any_region_still_presents_one_clip() {
        let host = item(
            "silent",
            4.0,
            2.0,
            vec![take("t", geometry(0.0, 2.0, Some(9.0), None, false, false))],
        );
        let assembly = assemble(&[host], &[]);
        assert_eq!(assembly.clips.len(), 1);
        assert_eq!(assembly.clips[0].start_sec, 4.0);
        assert!(
            assembly.clips[0].source_path.is_none(),
            "没有 region ⇒ 没有 ARA 源身份（音频仍在授权路径之外）"
        );
    }

    /// K12：工程长度 = max(item.start + item.length)，与 region 无关。
    #[test]
    fn k12_project_length_follows_the_items() {
        let a = item(
            "a",
            0.0,
            4.0,
            vec![take("t", geometry(0.0, 4.0, Some(9.0), None, false, false))],
        );
        let b = item(
            "b",
            10.0,
            5.0,
            vec![take("t", geometry(0.0, 5.0, Some(9.0), None, false, false))],
        );
        let assembly = assemble(&[a, b], &[row(1, 0.0, 1.0)]);
        let timeline = assembly.timeline(&[track("track-a")], 120.0);
        assert_eq!(timeline.project_sec, 15.0);
        assert_eq!(timeline.bpm, 120.0);
        assert_eq!(timeline.tracks.len(), 1);
        assert_eq!(timeline.clips.len(), 2);
    }

    /// K13：spans 按 region 起点升序，且覆盖全部挂上的 row。
    #[test]
    fn k13_spans_are_sorted_by_region_start() {
        let host = item(
            "a",
            0.0,
            9.0,
            vec![take("t", geometry(0.0, 9.0, Some(3.0), None, false, true))],
        );
        // 故意乱序传入。
        let rows = vec![row(3, 6.0, 3.0), row(1, 0.0, 3.0), row(2, 3.0, 3.0)];
        let assembly = assemble(&[host], &rows);
        assert_eq!(assembly.spans["ara-item-a"], vec![1, 2, 0]);
    }

    /// K14：同一输入装配两次 ⇒ clip id 与顺序完全一致（id 不可变）。
    #[test]
    fn k14_assembling_twice_is_bit_identical() {
        let items = vec![
            item(
                "a",
                0.0,
                4.0,
                vec![take(
                    "t1",
                    geometry(0.0, 4.0, Some(9.0), None, false, false),
                )],
            ),
            item(
                "b",
                4.0,
                4.0,
                vec![take("t2", geometry(1.0, 4.0, Some(9.0), None, false, true))],
            ),
        ];
        let rows = vec![row(1, 0.0, 4.0), row(2, 4.0, 4.0)];
        let first = assemble(&items, &rows);
        let second = assemble(&items, &rows);
        let ids = |a: &Assembly| a.clips.iter().map(|c| c.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&first), ids(&second));
        assert_eq!(first.spans, second.spans);
        assert_eq!(first.clips[0].start_sec, second.clips[0].start_sec);
    }

    /// K15：源总长未知 ⇒ loop_period 为 None，且装配结果**明确标出**它。
    ///
    /// 【为什么这条必须可观测】周期未知时内核会退化成"按 clip 自身长度回绕" ——
    /// 也就是**完全不循环**，后半段静音。而"后半段没声音"在插件里无法与
    /// "宿主没给音频"区分，用户只会以为自己的素材坏了。
    #[test]
    fn k15_an_unknown_loop_period_is_reported_not_guessed() {
        let mut host = item(
            "a",
            0.0,
            9.0,
            vec![take("t", geometry(0.0, 9.0, None, None, false, true))],
        );
        host.loop_source = true;
        let assembly = assemble(&[host], &[]);
        assert_eq!(assembly.clips[0].loop_period_sec, None);
        assert_eq!(assembly.unknown_loop_period, vec!["ara-item-a".to_string()]);
    }

    /// K16：unbound 的下标与实际相符。
    #[test]
    fn k16_unbound_indices_point_at_the_rows_that_matched_nothing() {
        let host = item(
            "a",
            0.0,
            2.0,
            vec![take("t", geometry(0.0, 2.0, Some(9.0), None, false, false))],
        );
        let rows = vec![row(1, 0.0, 1.0), row(2, 50.0, 1.0), row(3, 1.0, 1.0)];
        let assembly = assemble(&[host], &rows);
        assert_eq!(assembly.unbound, vec![1]);
        assert_eq!(assembly.spans["ara-item-a"], vec![0, 2]);
    }

    /// 一个 take 都不报告的 item 也必须能装配（不能因为取 active 而下标越界 panic）。
    #[test]
    fn an_item_without_takes_still_assembles() {
        let mut host = item("a", 0.0, 1.0, vec![]);
        host.name = String::new();
        let assembly = assemble(&[host], &[]);
        assert_eq!(assembly.clips.len(), 1);
        assert_eq!(assembly.clips[0].id, "ara-item-a");
        // `normalize_takes()` 会为"没有 take 的扁平 clip"补一条，因此这里断言
        // 的是"**没有**宿主 take 身份"而不是"take 列表为空"。
        assert!(assembly.clips[0].source_path.is_none());
        assert_eq!(assembly.clips[0].name, "ara-item-a");
    }
}
