//! REAPER 轨道组（folder）的**只读**重建。
//!
//! # 为什么 ARA 侧没有这个东西
//! ARA 的 `ARARegionSequence` 之间没有任何父子边（见 `crate::ara::mapping`），folder
//! 语义在 ARA 里无从表达。而 REAPER 的 folder 也不是独立对象 —— 它由每条轨道上的
//! `I_FOLDERDEPTH` 编码。因此桥接只能落在宿主清单层：读真实轨道清单、重建父子，
//! 再交给时间线当**参数根**（不是音频路由，见 `editor::workspace`）。
//!
//! # 只读，绝不写
//! `ReorderSelectedTracks` 不带 project 参数、操作的是全局/当前工程选择（见本仓
//! `probe/ara/EXECUTION-LEDGER.md` Task 74），所以插件绝不写宿主轨道结构。本模块
//! 只用已核对的只读入口：`CountTracks`、`GetTrack`、`GetMediaTrackInfo_Value`、
//! `GetSetMediaTrackInfo_String`（`setNewValue=false`）、`ValidatePtr2`、
//! `GetProjectStateChangeCount`。
//!
//! 这条边界由**类型**保证而不只靠约定：[`FolderApi`] 只持有上述六个只读函数指针，
//! 没有任何写入口可供调用 —— 想写轨道结构必须先往结构体里加字段，那是一次显式的
//! 代码改动，而不是一次疏忽。
//!
//! # `I_FOLDERDEPTH` 的口径（只认已确证的部分）
//! 它是**本条轨道之后**的深度变化量：
//! - `0`  普通轨道；
//! - `1`  本条轨道是一个 folder 的父轨：其后轨道深一层，直到被收口；
//! - `-1` 本条轨道是其所在 folder 的最后一条（收口一层）；
//! - `-2` / `-3` … 一次收口多层（内层最后一条，且其父也是外层的最后一条）。
//!
//! 嵌套 folder 由重复的 `1` / `-1` 表达，深度栈天然处理。
//!
//! **正数大于 1 一律拒绝。** 本机没有 REAPER，我无法对照确认它的含义；猜错会产出
//! 一棵**错误的父子树**，而错误的参数根会让用户把两条轨的参数编辑混在一起 —— 这比
//! "这个工程暂时不映射 folder" 严重得多。调用方拿到错误后退回"只显示 FX 自己那条
//! 轨"，行为与本次改动之前完全一致。
//!
//! 【未闭合项】要放宽这一条，必须先拿真实 REAPER 采一次样，确认 `I_FOLDERDEPTH > 1`
//! 到底出现在什么结构里（见 `docs/plans/2026-10-08-folder-child-items-to-clips.md`
//! 的 Task 5.6）。在拿到实测之前**不要**凭对称性外推：`-2` 的语义有官方文档支撑，
//! `+2` 没有。放宽的判据是实测样本，不是推理。
use super::{checked, ReaperHost};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};

/// 轨道数上限；超过即拒绝，**不截断**（截断会产出一棵静默错误的树）。
const MAX_TRACKS: usize = 10_000;
/// 嵌套深度上限。
const MAX_FOLDER_DEPTH: usize = 256;
/// 宿主文本（GUID / 轨道名）读取预算。
const TEXT_BUDGET: usize = 8192;

pub(super) type CountTracks = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type GetTrack = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
pub(super) type TrackValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;
pub(super) type TrackText =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, bool) -> bool;
pub(super) type Validate = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> bool;
pub(super) type ChangeCount = unsafe extern "C" fn(*mut c_void) -> i32;

/// 独立于"新建轨道"能力束的只读入口：即使宿主不允许建轨，folder 映射仍然可用。
pub(super) struct FolderApi {
    pub count: CountTracks,
    pub get: GetTrack,
    pub value: TrackValue,
    pub text: TrackText,
    pub validate: Validate,
    pub change: ChangeCount,
}

/// 一条轨道的 folder 相关只读元数据。
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HostTrackNode {
    pub guid: String,
    pub name: String,
    /// 宿主轨道序号（`IP_TRACKNUMBER`）。
    pub order: i32,
    /// `I_FOLDERDEPTH` 原值。
    pub folder_depth: i32,
    /// 宿主轨道句柄（原始地址）。**只在本模块产出的那一刻所在线程内有效**，
    /// 每次使用前必须重新 `ValidatePtr2` —— 与既有 `HostClipTarget` 同一纪律。
    pub(super) track: usize,
}

/// 宿主轨道组的只读重建结果。
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct HostFolderTree {
    pub nodes: Vec<HostTrackNode>,
    parent_of: BTreeMap<String, Option<String>>,
}

impl HostFolderTree {
    /// 轨道 GUID → 父轨道 GUID（`None` = 根级）。GUID 不存在时返回 `None`。
    pub(crate) fn parent_of(&self, guid: &str) -> Option<Option<String>> {
        self.parent_of.get(guid).cloned()
    }

    /// 后代节点（含句柄），按宿主轨道顺序，供清单枚举直接使用。
    pub(crate) fn descendant_nodes(&self, guid: &str) -> Vec<HostTrackNode> {
        self.nodes
            .iter()
            .filter(|node| node.guid != guid && self.is_descendant(&node.guid, guid))
            .cloned()
            .collect()
    }

    fn is_descendant(&self, candidate: &str, ancestor: &str) -> bool {
        let mut cursor = self.parent_of.get(candidate).cloned().flatten();
        let mut guard = 0usize;
        while let Some(current) = cursor {
            if current == ancestor {
                return true;
            }
            guard += 1;
            if guard > MAX_FOLDER_DEPTH {
                // 环（宿主数据损坏）——当作非后代，绝不死循环。
                return false;
            }
            cursor = self.parent_of.get(&current).cloned().flatten();
        }
        false
    }
}

impl ReaperHost {
    /// 只读重建宿主轨道组。必须在宿主 UI/model 线程调用。
    pub(crate) fn folder_tree(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<HostFolderTree, String> {
        if std::thread::current().id() != self.thread {
            return Err("REAPER track inventory queried outside host thread".into());
        }
        let api = self
            .folder
            .as_ref()
            .ok_or("REAPER track inventory API unavailable")?;
        let project = self.project(authorized)?;
        let valid = |pointer: *mut c_void, kind: &CStr| -> Result<(), String> {
            if pointer.is_null()
                || !checked(authorized, || unsafe {
                    (api.validate)(project, pointer, kind.as_ptr())
                })?
            {
                return Err("invalid host track object".into());
            }
            Ok(())
        };
        valid(project, c"ReaProject*")?;
        let before = checked(authorized, || unsafe { (api.change)(project) })?;
        let count = checked(authorized, || unsafe { (api.count)(project) })?;
        if !(0..=MAX_TRACKS as i32).contains(&count) {
            return Err("host track inventory budget exceeded".into());
        }
        let mut nodes = Vec::with_capacity(count as usize);
        for index in 0..count {
            // 每次取用前重验：宿主接口引用不保活任何对象。
            valid(project, c"ReaProject*")?;
            let track = checked(authorized, || unsafe { (api.get)(project, index) })?;
            valid(track, c"MediaTrack*")?;
            let guid = read_text(api, authorized, track, c"GUID", &valid)?;
            let name = read_text(api, authorized, track, c"P_NAME", &valid)?;
            let order = checked(authorized, || unsafe {
                (api.value)(track, c"IP_TRACKNUMBER".as_ptr())
            })?;
            let depth = checked(authorized, || unsafe {
                (api.value)(track, c"I_FOLDERDEPTH".as_ptr())
            })?;
            nodes.push(RawTrackRow {
                guid,
                name,
                order,
                folder_depth: depth,
                track: track as usize,
            });
        }
        // 宿主在枚举期间发生变化：整轮作废，不用半份清单拼树。
        if checked(authorized, || unsafe { (api.change)(project) })? != before {
            return Err("host changed during track inventory".into());
        }
        build_folder_tree(nodes.into_iter())
    }
}

/// 尚未校验的宿主轨道行；字段保持宿主原值，校验与重建都交给 [`build_folder_tree`]。
struct RawTrackRow {
    guid: String,
    name: String,
    order: f64,
    folder_depth: f64,
    track: usize,
}

/// 由轨道序列重建 folder 父子。**纯函数**：宿主调用与算法分离，算法可独立单测。
///
/// 深度变化量作用于**本条轨道之后**的轨道，因此先定归属、再施加本条的变化。
fn build_folder_tree(rows: impl Iterator<Item = RawTrackRow>) -> Result<HostFolderTree, String> {
    let mut nodes = Vec::new();
    let mut parent_of: BTreeMap<String, Option<String>> = BTreeMap::new();
    // 打开的 folder 父轨，最内层在末尾。
    let mut stack: Vec<String> = Vec::new();
    for row in rows {
        if nodes.len() >= MAX_TRACKS {
            return Err("host track inventory budget exceeded".into());
        }
        if row.guid.len() != 38 {
            return Err("invalid host track GUID".into());
        }
        if !row.order.is_finite()
            || row.order.fract() != 0.
            || !(0.0..=MAX_TRACKS as f64).contains(&row.order)
        {
            return Err("invalid host track number".into());
        }
        if !row.folder_depth.is_finite()
            || row.folder_depth.fract() != 0.
            || !(-(MAX_FOLDER_DEPTH as f64)..=1.0).contains(&row.folder_depth)
        {
            // 见模块文档：正数大于 1 的含义未经真实宿主核对，拒绝而不是猜。
            return Err(format!(
                "unsupported host folder depth {} on track {}",
                row.folder_depth, row.guid
            ));
        }
        let folder_depth = row.folder_depth as i32;
        parent_of.insert(row.guid.clone(), stack.last().cloned());
        nodes.push(HostTrackNode {
            guid: row.guid.clone(),
            name: row.name,
            order: row.order as i32,
            folder_depth,
            track: row.track,
        });
        if folder_depth == 1 {
            stack.push(row.guid);
            if stack.len() > MAX_FOLDER_DEPTH {
                return Err("host folder nesting budget exceeded".into());
            }
        } else if folder_depth < 0 {
            let close = (-folder_depth) as usize;
            if close > stack.len() {
                // 收口深度超过已开层数：畸形结构，明确拒绝而不是静默夹到 0。
                return Err("malformed host folder nesting".into());
            }
            stack.truncate(stack.len() - close);
        }
    }
    Ok(HostFolderTree { nodes, parent_of })
}

/// 读一条轨道上的宿主文本（GUID / 名称）；每次调用前重验对象。
fn read_text(
    api: &FolderApi,
    authorized: &impl Fn() -> bool,
    track: *mut c_void,
    field: &CStr,
    valid: &impl Fn(*mut c_void, &CStr) -> Result<(), String>,
) -> Result<String, String> {
    let mut bytes = vec![0_u8; TEXT_BUDGET];
    valid(track, c"MediaTrack*")?;
    if !checked(authorized, || unsafe {
        (api.text)(track, field.as_ptr(), bytes.as_mut_ptr().cast(), false)
    })? {
        return Err("host track identity/name unavailable".into());
    }
    let end = bytes
        .iter()
        .position(|b| *b == 0)
        .ok_or("host track text budget exceeded")?;
    String::from_utf8(bytes[..end].to_vec()).map_err(|_| "invalid host track UTF-8".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 38 字符、形状与真实 REAPER GUID 一致。
    fn guid(seed: u8) -> String {
        format!("{{{seed:08X}-3333-3333-3333-333333333333}}")
    }

    /// 一条轨道行；`depth` 是宿主 `I_FOLDERDEPTH` 原值。
    fn row(seed: u8, order: i32, depth: f64) -> RawTrackRow {
        RawTrackRow {
            guid: guid(seed),
            name: format!("track-{seed}"),
            order: order as f64,
            folder_depth: depth,
            track: seed as usize,
        }
    }

    fn parent_name(tree: &HostFolderTree, seed: u8) -> Option<String> {
        tree.parent_of(&guid(seed)).flatten()
    }

    /// 平铺工程：没有 folder，谁都不该凭空有父级。
    #[test]
    fn flat_project_has_no_parents() {
        let tree =
            build_folder_tree([row(1, 1, 0.), row(2, 2, 0.), row(3, 3, 0.)].into_iter()).unwrap();
        for seed in [1, 2, 3] {
            assert_eq!(parent_name(&tree, seed), None, "seed={seed}");
        }
        assert!(tree.descendant_nodes(&guid(1)).is_empty());
    }

    /// 最常见的形状：父轨 `1`，其后轨道是子轨，最后一条子轨 `-1` 收口。
    #[test]
    fn folder_children_point_at_the_folder_track() {
        let tree = build_folder_tree(
            [
                row(1, 1, 1.),
                row(2, 2, 0.),
                row(3, 3, 0.),
                row(4, 4, -1.),
                row(5, 5, 0.),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(parent_name(&tree, 1), None);
        assert_eq!(parent_name(&tree, 2), Some(guid(1)));
        assert_eq!(parent_name(&tree, 3), Some(guid(1)));
        assert_eq!(parent_name(&tree, 4), Some(guid(1)));
        // 收口之后的轨道回到根级 —— 这是"组边界"的关键断言。
        assert_eq!(parent_name(&tree, 5), None);
    }

    /// 嵌套 folder 由重复的 `1` / `-1` 表达，深度栈必须还原真实层级。
    #[test]
    fn nested_folders_nest() {
        let tree = build_folder_tree(
            [row(1, 1, 1.), row(2, 2, 1.), row(3, 3, -1.), row(4, 4, -1.)].into_iter(),
        )
        .unwrap();
        assert_eq!(parent_name(&tree, 1), None);
        assert_eq!(parent_name(&tree, 2), Some(guid(1)));
        // 内层子轨的父是内层 folder，不是外层 —— 压平会毁掉参数根。
        assert_eq!(parent_name(&tree, 3), Some(guid(2)));
        assert_eq!(parent_name(&tree, 4), Some(guid(1)));
    }

    /// 一条轨道可以同时收口多层（内层最后一条，且其父也是外层的最后一条）。
    #[test]
    fn one_track_can_close_several_levels() {
        let tree =
            build_folder_tree([row(1, 1, 1.), row(2, 2, 1.), row(3, 3, -2.)].into_iter()).unwrap();
        assert_eq!(parent_name(&tree, 2), Some(guid(1)));
        assert_eq!(parent_name(&tree, 3), Some(guid(2)));
        assert!(tree.descendant_nodes(&guid(1)).len() == 2);
    }

    /// 空 folder（自身没有 item）仍然拥有它的收口轨：这正是不展开就"看不见"的那一组。
    #[test]
    fn empty_folder_still_owns_its_closing_track() {
        let tree =
            build_folder_tree([row(1, 1, 1.), row(2, 2, -1.), row(3, 3, 0.)].into_iter()).unwrap();
        assert_eq!(parent_name(&tree, 2), Some(guid(1)));
        assert_eq!(parent_name(&tree, 3), None);
        assert_eq!(tree.descendant_nodes(&guid(1)).len(), 1);
    }

    /// 后代按**宿主轨道顺序**返回：呈现顺序依赖它，不能是栈的访问序。
    #[test]
    fn descendants_follow_host_track_order() {
        let tree = build_folder_tree(
            [row(1, 1, 1.), row(9, 2, 1.), row(3, 3, -1.), row(4, 4, -1.)].into_iter(),
        )
        .unwrap();
        let descendants = tree
            .descendant_nodes(&guid(1))
            .into_iter()
            .map(|node| node.guid)
            .collect::<Vec<_>>();
        assert_eq!(descendants, vec![guid(9), guid(3), guid(4)]);
    }

    /// 收口深度超过已开层数 = 畸形结构：明确拒绝，不静默夹到 0。
    #[test]
    fn closing_more_levels_than_open_is_rejected() {
        let error = build_folder_tree([row(1, 1, -1.)].into_iter()).unwrap_err();
        assert!(error.contains("malformed"), "{error}");
    }

    /// 正数大于 1 的含义未经真实宿主核对：拒绝而不是猜出一棵错误的树。
    #[test]
    fn positive_depth_above_one_is_rejected() {
        let error = build_folder_tree([row(1, 1, 2.)].into_iter()).unwrap_err();
        assert!(error.contains("unsupported host folder depth 2"), "{error}");
    }

    /// 非整数 / 非有限深度同样拒绝（宿主数据损坏时不能靠隐式转换蒙混过关）。
    #[test]
    fn fractional_and_non_finite_depths_are_rejected() {
        assert!(build_folder_tree([row(1, 1, 1.5)].into_iter()).is_err());
        assert!(build_folder_tree([row(1, 1, f64::NAN)].into_iter()).is_err());
        assert!(build_folder_tree([row(1, 1, f64::INFINITY)].into_iter()).is_err());
    }

    /// GUID 形状不对（宿主文本损坏）时不产出半个树。
    #[test]
    fn malformed_guid_is_rejected() {
        let mut broken = row(1, 1, 0.);
        broken.guid = "not-a-guid".into();
        assert!(build_folder_tree([broken].into_iter()).is_err());
    }
}
