//! 宿主GUI清单与ARA播放分配分离；枚举FX真实parent轨道（若它是folder，连同其全部
//! 后代轨道），静音item/空轨不消失。
use super::*;
use std::ffi::CStr;
use std::sync::Arc;

/// 单个 item 的 take 数量上限。REAPER 实际远低于此；设上限只为把畸形/竞态下的
/// `GetMediaItemNumTakes` 返回值挡在枚举循环之外。
const MAX_TAKES: i32 = 1024;

#[derive(Clone)]
pub(crate) struct UiTake {
    pub geometry: super::super::geometry::HostClipGeometry,
    pub name: String,
    /// 宿主 `GetActiveTake` 指向的那一个；每个 item 恰好一个。
    pub active: bool,
    /// 宿主报告的方向。读不出来时为 `false`（**按正放**）。
    ///
    /// 【为什么不再分三态】方向是纯渲染输入，不是需要用户决策的状态：读不到时按正放
    /// 渲染，与 `loop_enabled`（`geometry.rs` 的同名注释）是同一条纪律。此前把"读不到"
    /// 投影成一个面向用户的提示，等于让用户为一个**不可操作**的内部读数负责。
    pub reversed: bool,
}

#[derive(Clone)]
pub(crate) struct UiItem {
    /// active take 的几何。既有消费者（淡化装饰、item→clip 反查）都读它，
    /// 保持"每个 item 一份权威几何"的语义不变。
    pub geometry: super::super::geometry::HostClipGeometry,
    pub name: String,
    pub target: HostClipTarget,
    /// 全部 take（含 active）。宿主没有枚举接口时恰好一个元素。
    pub takes: Vec<UiTake>,
}
#[derive(Clone)]
pub(crate) struct UiTrack {
    pub guid: String,
    pub id: String,
    pub name: String,
    pub order: i32,
    pub items: Vec<UiItem>,
    /// 所属 folder 轨的 GUID（`None` = 根级）。
    ///
    /// 这是**宿主**给出的父子关系（`I_FOLDERDEPTH` 重建），不是插件的私有分组。
    /// 时间线据此设置 `Track.parent_id`，于是参数根、合成开关与分析算法的既有继承
    /// 逻辑零改动即可工作 —— 注意它只决定**参数根**，与 REAPER 的音频路由无关。
    pub parent_guid: Option<String>,
    // 宿主 change 代次字段保留，随 UiTrack 一起快照（当前无读取方）。
    #[allow(dead_code)]
    pub change: i32,
    pub target: HostTrackTarget,
}

impl NativeHost {
    /// 所有查询都在所属UI线程并逐调用授权；没有region也不代表轨道或item已被删除。
    pub(crate) fn ui_track(
        self: &Arc<Self>,
        authorized: &impl Fn() -> bool,
    ) -> Result<UiTrack, String> {
        if std::thread::current().id() != self.thread {
            return Err("UI inventory queried outside host thread".into());
        }
        let project = self.project(authorized)?;
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let track = checked(authorized, || unsafe { (table.parent)(pointer, 1) })?;
        self.ui_track_at(project, track, None, authorized)
    }

    /// FX 所在轨道 +（若它是 folder）其全部后代轨道。
    ///
    /// 【为什么要展开】ARA 的 region sequence 之间没有父子边，folder 语义只能从宿主
    /// 清单层取。FX 挂在 folder 轨上时，folder 轨自身通常没有 item —— 只枚举它会让
    /// 整组"看不见"，用户以为插件坏了。
    ///
    /// folder 结构读不出来时（旧宿主、畸形嵌套、未经核对的深度值）**退回单轨**，
    /// 与本次改动之前完全一致：可选增强不得拖垮整份清单。
    pub(crate) fn ui_folder_tracks(
        self: &Arc<Self>,
        authorized: &impl Fn() -> bool,
    ) -> Result<Vec<UiTrack>, String> {
        let own = self.ui_track(authorized)?;
        let project = self.project(authorized)?;
        let own_guid = own.guid.clone();
        let tree = match self.folder_tree(authorized) {
            Ok(tree) => tree,
            Err(error) => {
                // 退回单轨：与本次改动之前的行为完全一致，不让可选增强拖垮整份清单。
                crate::log_line(&format!(
                    "[reaper-folder] track group unavailable ({error}); showing the FX track only"
                ));
                return Ok(vec![own]);
            }
        };
        let mut tracks = vec![own];
        for node in tree.descendant_nodes(&own_guid) {
            // 父级取自重建出的树，而不是一律挂到 FX 轨上 —— 否则嵌套 folder 会被压平。
            let parent_guid = tree.parent_of(&node.guid).flatten();
            match self.ui_track_at(project, node.track as *mut c_void, parent_guid, authorized) {
                Ok(track) => tracks.push(track),
                // 单条后代读不到不该让整组消失；如实记账并继续。
                Err(error) => crate::log_line(&format!(
                    "[reaper-folder] descendant {} unavailable: {error}",
                    node.guid
                )),
            }
        }
        Ok(tracks)
    }

    /// 本实例能看到的最下方宿主轨道的 `IP_TRACKNUMBER`；用作"新轨道插在它之后"的锚点。
    ///
    /// 【为什么取最大值而不是列表末元素】`ui_folder_tracks` 的顺序由 folder 树重建
    /// 决定，与工程里的实际上下顺序不必然一致。用户说的"最下方"是**工程顺序**上的
    /// 最下方，所以按 `IP_TRACKNUMBER` 取最大 —— 它就是这个顺序。
    ///
    /// 读不出来时返回 `None`，调用方退回工程末尾：锚点只是让插入位置更贴合直觉，
    /// 读不到不该让"添加轨道"失败。
    pub(crate) fn folder_anchor_order(
        self: &Arc<Self>,
        authorized: &impl Fn() -> bool,
    ) -> Option<i32> {
        self.ui_folder_tracks(authorized)
            .ok()?
            .iter()
            .map(|track| track.order)
            .max()
    }

    /// 枚举一条轨道及其 item；project / track 由调用方取得（本函数不做线程与归属推断）。
    pub(crate) fn ui_track_at(
        self: &Arc<Self>,
        project: *mut c_void,
        track: *mut c_void,
        parent_guid: Option<String>,
        authorized: &impl Fn() -> bool,
    ) -> Result<UiTrack, String> {
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        macro_rules! api {
            ($name:expr,$ty:ty) => {{
                let p = checked(authorized, || unsafe {
                    (table.api)(pointer, $name.as_ptr())
                })?;
                if p.is_null() {
                    return Err("REAPER UI inventory API unavailable".into());
                }
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        let item_count = api!(
            c"CountTrackMediaItems",
            unsafe extern "C" fn(*mut c_void) -> i32
        );
        let get_item = api!(
            c"GetTrackMediaItem",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        );
        let active_take = api!(
            c"GetActiveTake",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        );
        let take_name = api!(
            c"GetTakeName",
            unsafe extern "C" fn(*mut c_void) -> *const c_char
        );
        let track_value = api!(c"GetMediaTrackInfo_Value", Value);
        let track_string = api!(c"GetSetMediaTrackInfo_String", Guid);
        let geometry = self.geometry.as_ref().ok_or("host geometry unavailable")?;
        let valid = |p: *mut c_void, kind: &CStr| -> Result<(), String> {
            if p.is_null()
                || !checked(authorized, || unsafe {
                    (geometry.validate)(project, p, kind.as_ptr())
                })?
            {
                return Err("invalid UI inventory object".into());
            }
            Ok(())
        };
        valid(project, c"ReaProject*")?;
        valid(track, c"MediaTrack*")?;
        let change = checked(authorized, || unsafe { (geometry.change)(project) })?;
        let read = |field: &CStr| -> Result<String, String> {
            let mut bytes = vec![0_u8; 8192];
            valid(track, c"MediaTrack*")?;
            if !checked(authorized, || unsafe {
                track_string(track, field.as_ptr(), bytes.as_mut_ptr().cast(), false)
            })? {
                return Err("track identity/name unavailable".into());
            }
            let end = bytes
                .iter()
                .position(|b| *b == 0)
                .ok_or("host track text budget exceeded")?;
            String::from_utf8(bytes[..end].to_vec()).map_err(|_| "invalid host track UTF-8".into())
        };
        let guid = read(c"GUID")?;
        if guid.len() != 38 {
            return Err("invalid host track GUID".into());
        }
        let name = read(c"P_NAME")?;
        let number = checked(authorized, || unsafe {
            track_value(track, c"IP_TRACKNUMBER".as_ptr())
        })?;
        if !number.is_finite() || number.fract() != 0. || !(0.0..=10000.0).contains(&number) {
            return Err("invalid host track number".into());
        }
        let order = number as i32;
        let count = checked(authorized, || unsafe { item_count(track) })?;
        if !(0..=10000).contains(&count) {
            return Err("UI item inventory budget exceeded".into());
        }
        let mut items = Vec::new();
        for index in 0..count {
            valid(track, c"MediaTrack*")?;
            let item = checked(authorized, || unsafe { get_item(track, index) })?;
            valid(item, c"MediaItem*")?;
            // active take 是本实例唯一可能持有 ARA PCM 的那个；其余 take 只取元数据。
            let active = checked(authorized, || unsafe { active_take(item) })?;
            if active.is_null() {
                continue;
            }
            valid(active, c"MediaItem_Take*")?;
            let read_name = |take: *mut c_void| -> Result<String, String> {
                let p = checked(authorized, || unsafe { take_name(take) })?;
                if p.is_null() {
                    return Ok(String::new());
                }
                let value = unsafe { CStr::from_ptr(p) };
                if value.to_bytes().len() > 8192 {
                    return Err("take name budget exceeded".into());
                }
                Ok(value.to_str().map_err(|_| "invalid take UTF-8")?.to_owned())
            };
            let read_take = |take: *mut c_void, is_active: bool| -> Result<UiTake, String> {
                let geometry = self.geometry_for_take(take, authorized)?;
                Ok(UiTake {
                    name: read_name(take)?,
                    // 方向已经在 `geometry_for_take` 里连同 SECTION 边界一起读过一次；
                    // 这里复用，避免对同一个 take 重复调用宿主。
                    reversed: geometry.reversed,
                    geometry,
                    active: is_active,
                })
            };
            // 【为什么先枚举再补 active】宿主清单与 active 指针之间没有原子性保证
            // （用户在枚举期间切 take 会让 `GetActiveTake` 落空）。active 必须一定
            // 在集合里，否则 clip 的 `active_take_id` 会指向一个不存在的 take。
            let mut takes = match self.take_enum() {
                Some(api) => {
                    let total = checked(authorized, || unsafe { (api.count)(item) })?;
                    if !(0..=MAX_TAKES).contains(&total) {
                        return Err("take inventory budget exceeded".into());
                    }
                    let mut takes = Vec::with_capacity(total as usize);
                    for slot in 0..total {
                        valid(item, c"MediaItem*")?;
                        let take = checked(authorized, || unsafe { (api.at)(item, slot) })?;
                        if take.is_null() {
                            continue;
                        }
                        valid(take, c"MediaItem_Take*")?;
                        takes.push(read_take(take, take == active)?);
                    }
                    takes
                }
                None => Vec::new(),
            };
            if !takes.iter().any(|take| take.active) {
                takes.insert(0, read_take(active, true)?);
            }
            let active_index = takes.iter().position(|take| take.active).unwrap_or(0);
            let g = takes[active_index].geometry.clone();
            let name = takes[active_index].name.clone();
            let target = super::write::inventory_target(
                self.clone(),
                project as usize,
                track as usize,
                item as usize,
                active as usize,
                g.clone(),
                authorized,
            )?;
            items.push(UiItem {
                geometry: g,
                name,
                target,
                takes,
            });
        }
        if change != checked(authorized, || unsafe { (geometry.change)(project) })? {
            return Err("host changed during UI inventory".into());
        }
        let target = self.track_target(project as usize, track as usize, authorized)?;
        let name = if name.is_empty() {
            format!("Track {order}")
        } else {
            name
        };
        Ok(UiTrack {
            id: format!("host-track-{guid}"),
            guid,
            name,
            order,
            items,
            parent_guid,
            change,
            target,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::MutexExt;
    /// 不依赖ARA assignment的显示清单必须保留mute item及空轨，不产生音频source读取。
    #[test]
    fn ui_inventory_keeps_muted_item_and_empty_track_without_audio_assignments() {
        let fixture = super::super::MockHost::new();
        fixture.enable_writer();
        fixture.enable_media();
        fixture.clear_markers();
        fixture.inventory_enabled.set(true);
        fixture.set_value("B_MUTE", 1.);
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        assert_eq!(track.items.len(), 1);
        assert!(track.items[0].geometry.muted);
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track.clone());
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(timeline.tracks.len(), 1);
        assert_eq!(timeline.clips.len(), 1);
        assert!(timeline.clips[0].muted);
        assert!(
            timeline.clips[0].source_path.is_none(),
            "显示占位不得未经ARA授权读取文件PCM"
        );
        fixture.inventory_empty.set(true);
        let empty = host.ui_track(&|| true).unwrap();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(empty.guid.clone(), empty);
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(timeline.tracks.len(), 1);
        assert!(timeline.clips.is_empty());
        document.close();
    }

    /// 宿主枚举到的每个 take 都要出现在清单里，且 active 恰好一个。
    ///
    /// 【为什么这条最要紧】清单是用户判断"插件到底看得到什么"的唯一窗口。少列一个
    /// take 会让用户以为宿主丢了内容；把两个 take 认成同一个（身份重复）则会让
    /// active 切换指向错误的 take。
    #[test]
    fn take_inventory_lists_every_take_with_distinct_identities() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        assert_eq!(track.items.len(), 1);
        let takes = &track.items[0].takes;
        assert_eq!(takes.len(), 3, "3 个 take 必须全部出现");
        assert_eq!(
            takes.iter().filter(|take| take.active).count(),
            1,
            "active take 恰好一个"
        );
        let ids: std::collections::BTreeSet<_> = takes
            .iter()
            .map(|take| take.geometry.take_id.clone())
            .collect();
        assert_eq!(ids.len(), 3, "每个 take 的身份必须不同");
        assert!(
            takes
                .iter()
                .all(|take| take.geometry.item_id == track.items[0].geometry.item_id),
            "同一 item 的 take 共享 item 身份"
        );
        // active take 的几何就是 item 的权威几何（既有消费者读它）。
        let active = takes.iter().find(|take| take.active).unwrap();
        assert_eq!(active.geometry.take_id, track.items[0].geometry.take_id);
    }

    /// take 集合随宿主变化时清单必须跟着变（增删 take 都算）。
    #[test]
    fn take_inventory_follows_the_host_when_takes_are_added() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        assert_eq!(host.ui_track(&|| true).unwrap().items[0].takes.len(), 1);
        fixture.enable_takes(3);
        assert_eq!(host.ui_track(&|| true).unwrap().items[0].takes.len(), 4);
    }

    /// 越界的 take 数量必须在枚举循环之外被挡下。
    #[test]
    fn take_inventory_rejects_an_out_of_budget_take_count() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        fixture.take_count_override.set(Some(MAX_TAKES + 1));
        let host = Arc::new(fixture.client());
        let error = host.ui_track(&|| true).err().expect("越界数量必须被拒绝");
        assert!(error.contains("take inventory budget"), "{error}");
    }

    /// 宿主报告的方向要如实透传；读不出来时**按正放**（不是一种需要用户处置的状态）。
    #[test]
    fn take_direction_is_reported_only_when_the_host_answers() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        fixture.section_reports.set(true);
        fixture.take_reversed.set(true);
        let host = Arc::new(fixture.client());
        assert!(host.ui_track(&|| true).unwrap().items[0].takes[0].reversed);

        fixture.take_reversed.set(false);
        assert!(!host.ui_track(&|| true).unwrap().items[0].takes[0].reversed);

        // 宿主不回答（不是 section/reverse 块）→ 按正放，而不是制造一个"未知"态。
        fixture.section_reports.set(false);
        assert!(!host.ui_track(&|| true).unwrap().items[0].takes[0].reversed);
    }

    /// **回归：负 `D_PLAYRATE` 也是倒放的一种表达，必须与方向位等价。**
    ///
    /// 【为什么这是回归测试】REAPER 用**两种**方式表达倒放：负速率、或
    /// `SECTION MODE 1`。独立 App 的判据是 `play_rate < 0 || section_mode > 0`；
    /// 插件此前把速率符号 `.abs()` 掉、只认方向位 —— 于是"只用负速率表达倒放"的
    /// take 被**静默按正放渲染**：输出错的音频，而不是报错。
    #[test]
    fn a_negative_play_rate_marks_the_take_as_reversed() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        // 宿主**不**回答方向位 —— 倒放只由速率符号表达。
        fixture.section_reports.set(false);
        fixture.set_value("D_PLAYRATE", -1.0);
        let host = Arc::new(fixture.client());
        let take = &host.ui_track(&|| true).unwrap().items[0].takes[0];
        assert!(take.reversed, "负速率必须判为倒放");
        assert!(
            take.geometry.play_rate_signed < 0.0,
            "符号必须原样保留，写回时才能不被抹掉"
        );
        // 幅值仍按正数给出：窗口宽度、缓存键这些与方向无关的消费者用它。
        assert!((take.geometry.playback_rate - 1.0).abs() < 1e-9);
    }

    /// 正速率 + 无方向位 ⇒ 正放（默认路径不被上面那条误伤）。
    #[test]
    fn a_positive_play_rate_without_a_section_stays_forward() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        fixture.section_reports.set(false);
        fixture.set_value("D_PLAYRATE", 2.0);
        let host = Arc::new(fixture.client());
        let take = &host.ui_track(&|| true).unwrap().items[0].takes[0];
        assert!(!take.reversed);
        assert!(take.geometry.play_rate_signed > 0.0);
    }

    /// 循环源（`B_LOOPSRC`）是 **item** 属性，必须读进几何并投影到 clip 与每个 take。
    ///
    /// 【为什么要读】内核的 `loop_enabled` 决定源窗口是否对**整份媒体**回绕，并且进渲染
    /// 缓存键。不读就等于"REAPER 里循环了、插件按不循环渲染"—— 又一条静默分叉。
    #[test]
    fn loop_source_is_read_from_the_item_and_projected() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        fixture.set_value("B_LOOPSRC", 1.);
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        assert!(
            track.items[0].geometry.loop_source,
            "B_LOOPSRC is an item attribute and must be read from the item"
        );

        let document = crate::render::document::DocumentSession::new(9877);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        let clip = &timeline.clips[0];
        assert!(
            clip.loop_enabled,
            "loop must reach the flat clip projection"
        );
        assert!(
            clip.takes.iter().all(|take| take.loop_enabled),
            "loop is item-level, so every take of the item carries it"
        );
    }

    /// 清单投影必须按**宿主实际在用的那套轴**取曲率。
    ///
    /// 【为什么值得一条测试】本函数在 `project_ui_fades_locked` **之后**跑
    /// （`ensure_loaded`：先 snapshot 再 inventory）。无条件取旧轴 `D_FADEINDIR` 会把刚
    /// 投影好的新轴曲率覆盖成一个被重映射过的旧值（实测 7.81 上新轴 0.5 读回旧轴是 0），
    /// 表现就是"拖了曲率，滑杆与曲线又跳回另一个值"。
    #[test]
    fn inventory_projects_the_curvature_of_the_host_axis_in_use() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let geometry = track.items[0].geometry.clone();
        assert_ne!(
            geometry.fade_in_dir, geometry.fade_in_dir_new,
            "夹具必须让两套轴取值不同，否则这条测试对缺陷不敏感"
        );
        let document = crate::render::document::DocumentSession::new(9879);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        let clip = &timeline.clips[0];
        // 夹具自报 REAPER 7.81 → 新轴才是权威值。
        assert_eq!(clip.fade_in_dir, geometry.fade_in_dir_new);
        assert_eq!(clip.fade_out_dir, geometry.fade_out_dir_new);
    }

    /// 逐 clip 的宿主媒体状态必须把**成因分开** —— 它们此前共用一句
    /// "等待 REAPER 提供音频（未分配 ARA 区域）"，于是"刚分割了一下"看起来像故障。
    #[test]
    fn host_media_state_separates_the_causes() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9878);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        // 未认领、也不是 folder 父轨 → **在途**（宿主可能还在分配，不是故障）。
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "pending");

        // 有源 → ready。
        payload["clips"][0]["source_path"] = serde_json::json!("C:/pcm/source.wav");
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "ready");

        // 倒放**不再是一种宿主媒体状态**：倒放由本插件渲染（内核装配期翻转），与正放
        // 一样等 ARA 音频 —— 没有源时同样是"在途"，不是一个需要用户处置的状态。
        payload["clips"][0]["source_path"] = serde_json::json!(null);
        payload["clips"][0]["reversed"] = serde_json::json!(true);
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "pending");
    }

    /// 一个已被 ARA 认领的 item，在**带命名空间**的呈现里只能有一条 clip。
    ///
    /// 【为什么这是"同一个 Item 两条 Clip"的回归】`claimed_id` 此前拿**文档级**
    /// （无命名空间）的 `clip_ids` 去比**带命名空间**的编辑器时间线 —— 恒不命中，
    /// 于是每一轮都新建一条无前缀的副本。真 clip 有波形、副本永远"等待宿主提供音频"，
    /// 而副本是后 push 的，画布上盖住前者。
    ///
    /// 【为什么这条以前不存在】`present_host_inventory` 的单测清一色用**空命名空间**，
    /// 而空命名空间下 `prefix()` 是恒等函数 —— bug 天然隐身。
    #[test]
    fn a_claimed_item_presents_exactly_one_clip() {
        let fixture = claimed_host();
        let (document, item_id) = claimed_item(&fixture, 9901);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        timeline
            .clips
            .push(editor_clip(&item_id, Some("ara://source")));
        document.present_host_inventory(&mut timeline, "ui-");
        let ids: Vec<&str> = timeline.clips.iter().map(|clip| clip.id.as_str()).collect();
        assert_eq!(ids, vec![format!("ui-ara-item-{item_id}").as_str()]);
        assert_eq!(
            timeline.clips[0].source_path.as_deref(),
            Some("ara://source"),
            "真 clip 的媒体身份不得被清单刷新剥掉"
        );
        assert_eq!(
            document
                .presentation_dupes
                .load(std::sync::atomic::Ordering::Acquire),
            0,
            "没有副本可收敛"
        );
        document.close();
    }

    /// 身份从 `ara-region-{key}` 升级成 `ara-item-{guid}` 的那一轮，仍然只有一条 clip。
    ///
    /// 【为什么必须单独钉】升级是**换一套过滤规则**的时刻：旧实现按 id 前缀决定去留，
    /// 于是"升级完成、清单尚未枚举到新 GUID"的那一轮会把真 clip 删掉、留下副本。
    #[test]
    fn a_pre_upgrade_region_clip_is_not_duplicated_after_the_upgrade() {
        let fixture = claimed_host();
        let (document, item_id) = claimed_item(&fixture, 9902);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        // 第一轮：clip_ids 还是 key 形态，时间线里是 region 形态的 clip。
        document
            .clip_ids
            .lock_or_recover()
            .insert(7, "ara-region-7".into());
        let mut before = editor_clip(&item_id, Some("ara://source"));
        before.id = "ui-ara-region-7".into();
        timeline.clips.push(before);
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(timeline.clips.len(), 1, "升级前也只有一条");
        // 第二轮：身份升级成 item GUID。
        document
            .clip_ids
            .lock_or_recover()
            .insert(7, format!("ara-item-{item_id}"));
        timeline.clips[0].id = format!("ui-ara-item-{item_id}");
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(
            timeline.clips.len(),
            1,
            "升级那一轮不得同时留下旧形态与副本：{:?}",
            timeline
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<Vec<_>>()
        );
        document.close();
    }

    /// 无命名空间前缀的 ARA clip 在带命名空间的呈现里必须被丢掉。
    ///
    /// 【为什么这是那条"永生副本"的直接回归】旧 retain 的末支是 `true`：任何认不出
    /// 前缀的 id 都**一律保留**。而副本的 id 恰恰没有前缀 —— 于是它一旦被造出来就
    /// 再也不会消失。
    #[test]
    fn a_stray_unprefixed_clip_is_dropped_from_a_namespaced_presentation() {
        let fixture = claimed_host();
        let (document, item_id) = claimed_item(&fixture, 9903);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        timeline
            .clips
            .push(editor_clip(&item_id, Some("ara://source")));
        // 上一代命名留下的副本：同一个 item、无前缀、没有源。
        let mut stray = editor_clip(&item_id, None);
        stray.id = format!("ara-item-{item_id}");
        timeline.clips.push(stray);
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(
            timeline
                .clips
                .iter()
                .map(|clip| clip.id.clone())
                .collect::<Vec<_>>(),
            vec![format!("ui-ara-item-{item_id}")],
            "无前缀的残留必须被剔除"
        );
        document.close();
    }

    /// 连续多轮呈现不会累积出第二条 clip。
    ///
    /// 【为什么是 5 轮】旧实现在**每一轮**都会 push 一条副本，而 retain 从不删它 ——
    /// 一轮就够复现，多轮用来钉住"它不会随刷新次数增长"。
    #[test]
    fn the_duplicate_never_outlives_one_refresh() {
        let fixture = claimed_host();
        let (document, item_id) = claimed_item(&fixture, 9904);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        timeline
            .clips
            .push(editor_clip(&item_id, Some("ara://source")));
        for round in 0..5 {
            document.present_host_inventory(&mut timeline, "ui-");
            assert_eq!(
                timeline.clips.len(),
                1,
                "第 {round} 轮出现第二条 clip：{:?}",
                timeline
                    .clips
                    .iter()
                    .map(|clip| clip.id.clone())
                    .collect::<Vec<_>>()
            );
        }
        document.close();
    }

    /// GUI 定位返回的是**原生** id，调用方补上命名空间后必须能在编辑器时间线里命中。
    ///
    /// 【为什么钉的是"能命中"而不是"带前缀"】`gui_clip_for_host_item` 的契约是
    /// "文档级 id"（调用方统一 `format!("{namespace}{native}")`，见 `host_split.rs`
    /// 与 `clipboard.rs`）。真正会出错的是这条组合的结果对不上时间线 —— 那会让
    /// 分割/粘贴回执永远判"宿主没跟上"。
    #[test]
    fn the_gui_lookup_composes_into_the_presented_id() {
        let fixture = claimed_host();
        let (document, item_id) = claimed_item(&fixture, 9905);
        let native = document
            .gui_clip_for_host_item(&item_id)
            .expect("已认领的 item 必须能被定位");
        let presented = format!("ui-{native}");
        assert_eq!(presented, format!("ui-ara-item-{item_id}"));
        document.close();
    }

    /// 一个"item 已被 ARA 认领、且 region 活着"的最小文档。
    ///
    /// 【为什么把 `MockHost` 交回给调用方】它持有客户端指向的共享状态；helper 里
    /// 丢掉它会让测试结束时访问已释放内存（`STATUS_ACCESS_VIOLATION`），而那个崩溃
    /// 与真正的回归无法区分。
    fn claimed_item(
        fixture: &super::super::MockHost,
        id: u64,
    ) -> (Arc<crate::render::document::DocumentSession>, String) {
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(id);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        document
            .clip_ids
            .lock_or_recover()
            .insert(7, format!("ara-item-{item_id}"));
        document
            .regions
            .lock_or_recover()
            .insert(7, Default::default());
        document
            .ready
            .store(true, std::sync::atomic::Ordering::Release);
        (document, item_id)
    }

    /// 一个"清单里有一条带 item 的轨道"的宿主夹具。
    fn claimed_host() -> Box<super::super::MockHost> {
        let fixture = super::super::MockHost::new();
        fixture.enable_writer();
        fixture.enable_media();
        fixture.clear_markers();
        fixture.inventory_enabled.set(true);
        fixture
    }

    /// 编辑器时间线上的一条 clip（id 带命名空间前缀）。
    fn editor_clip(item_id: &str, source: Option<&str>) -> hifishifter_kernel::state::Clip {
        let mut clip: hifishifter_kernel::state::Clip = serde_json::from_value(serde_json::json!({
            "id": format!("ui-ara-item-{item_id}"),
            "track_id": "ui-host-track",
            "name": "real",
            "start_sec": 0.0,
            "length_sec": 1.0,
            "takes": [{"id": "t1", "source_path": source}],
        }))
        .unwrap();
        clip.normalize_takes();
        clip
    }

    /// "宿主换了 active take"是**可证明**的事实：我们同时有"授权那一刻的 take GUID"
    /// 与"宿主现在报的 take GUID"，两者不同就立刻能说，不必等超时。
    ///
    /// 【为什么这条取代了"等超期再报 `awaiting_region`"】超时是把"没观察到"说成"不会
    /// 来了"，而 `take_switched` 恰恰是唯一不需要等的那一类：它的处置明确（撤销，
    /// 或再编辑一次让 observer 重跑）。
    #[test]
    fn a_switched_take_is_reported_immediately_and_stays_idempotent() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9879);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);

        // 该 item 已被 ARA 认领，授权记录指向**另一个** take GUID（= 用户切了 active take）。
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, "{99999999-9999-9999-9999-999999999999}".into());

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        // 第一轮就该说出来：判据是两条 GUID 不相等，与"等了多久"无关。
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(payload["clips"][0]["host_media_reason"], "take_switched");

        // 幂等：再次呈现仍是 `unavailable` + 同一原因。
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(payload["clips"][0]["host_media_reason"], "take_switched");

        // 拿到音频后必须回到 ready，并**清掉计时** —— 否则下次掉回"在途"会立刻超期。
        payload["clips"][0]["source_path"] = serde_json::json!("C:/pcm/source.wav");
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "ready");
        assert!(document.pending_since.lock_or_recover().is_empty());
    }

    /// ARA 已授权样本访问、但**读取失败**时，原因必须是真因 `source_read_failed`，
    /// 而不是含混的 `awaiting_region`（那会让用户以为"再等等就好"）。
    ///
    /// 【为什么与 `ready` 分开】`ready` 还参与 `ensure_loaded` 的宿主冲突判据，把样本
    /// 读取失败折进 `ready=false` 会被报成"尚未应用：Conflict: host changed"。真因单记
    /// 一份（`DocumentSession::source_failures`），只影响这一条 clip 的原因码。
    #[test]
    fn an_authorized_but_unreadable_source_reports_its_real_reason() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9882);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        document.regions.lock_or_recover().insert(
            7,
            crate::ara::AraPlaybackRegion {
                audio_source_persistent_id: "ara://source".into(),
                audio_modification_persistent_id: "mod".into(),
                ..Default::default()
            },
        );
        // 源被授权、读取失败 ⇒ 立刻报真因，不启动"在途"计时。
        document
            .source_failures
            .lock_or_recover()
            .insert("ara://source".into(), "host read failed".into());

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(
            payload["clips"][0]["host_media_reason"],
            "source_read_failed"
        );

        // 读取成功后条目消失 ⇒ 回到"在途/未认领"的常规判定，不再报读取失败。
        document.source_failures.lock_or_recover().clear();
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "pending");
    }

    /// 被跳过的 playback region 必须进载荷：只 log 会让 MIDI / 零长度 item 静默消失。
    #[test]
    fn skipped_regions_are_reported_in_the_payload() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let document = crate::render::document::DocumentSession::new(9883);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        document
            .skipped_regions
            .lock_or_recover()
            .push((2, "midi".into()));
        document
            .skipped_regions
            .lock_or_recover()
            .push((5, "zero_length".into()));

        let mut payload = serde_json::json!({"clips": []});
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["skipped_regions"]["count"], 2);
        assert_eq!(payload["skipped_regions"]["items"][0]["reason"], "midi");
        assert_eq!(payload["skipped_regions"]["items"][1]["index"], 5);

        // 没有跳过项时**不写**该字段：前端沿用"没有跳过"，而不是被一个空数组扰动。
        document.skipped_regions.lock_or_recover().clear();
        let mut clean = serde_json::json!({"clips": []});
        document.decorate_host_media_locked(&mut clean, "ui-");
        assert!(clean.get("skipped_regions").is_none());
    }

    /// 方向位读不出来（`PCM_Source_GetSectionInfo` 不可用）时，插件**按正放**处理，
    /// 不再制造一个"方向未知"的用户提示 —— 方向是纯渲染输入，读不到不影响可用性。
    #[test]
    fn an_unreadable_direction_is_assumed_forward() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        // `section_reports` 默认 false ⇒ `take_reversed` 落成 false（按正放）。
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        assert!(!track.items[0].takes[0].reversed);
        let document = crate::render::document::DocumentSession::new(9880);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        // 已被认领、也没有"授权指向另一个 take" ⇒ 唯一的诚实结论是"还在等"。
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert!(payload["clips"][0]["reversed_known"].is_null());
        assert_eq!(payload["clips"][0]["host_media"], "pending");

        // 【B4：等多久都不改结论】旧实现在这里转成 `unavailable:awaiting_region` ——
        // 一句我们无法证明的断言，而它在界面上就是"等待宿主提供音频"。
        {
            let mut since = document.pending_since.lock_or_recover();
            since.insert(
                item_id.clone(),
                std::time::Instant::now() - std::time::Duration::from_secs(10),
            );
        }
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "pending");
        assert!(payload["clips"][0].get("host_media_reason").is_none());
        assert_eq!(payload["clips"][0]["wait_seconds"].as_u64(), Some(10));
    }

    /// 多 take 投影进显示 Clip：全部 take 到位、active 一个、**都不带 source_path**。
    #[test]
    fn host_take_set_projects_every_take_without_any_source_path() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        fixture.section_reports.set(true);
        fixture.take_reversed.set(true);
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(timeline.clips.len(), 1);
        let clip = &timeline.clips[0];
        assert_eq!(clip.takes.len(), 3);
        assert!(
            clip.takes.iter().all(|take| take.source_path.is_none()),
            "任何 take 都不得未经 ARA 授权带 source_path"
        );
        assert!(clip.takes.iter().all(|take| take.reversed));
        assert_eq!(clip.takes[0].playback_rate, 0.5);
        assert_eq!(clip.takes[0].channel_mode, 2);
        let active = clip.active_take_id.as_deref();
        assert!(active.is_some());
        assert!(clip
            .takes
            .iter()
            .any(|take| Some(take.id.as_str()) == active));
        assert_eq!(active, Some(clip.takes[0].id.as_str()));
        document.close();
    }

    /// ARA 已授权的那个 take 在清单重建后必须保住授权媒体。
    ///
    /// 【为什么这条测试存在】清单重建从宿主元数据造 take，而宿主元数据按设计不带
    /// `source_path`；重建后 `normalize_takes()` 会把 active take 物化到扁平投影上。
    /// 若不把授权媒体搬过去，每个 ARA 片段都会掉进"等待宿主音频"的斜纹占位。
    #[test]
    fn an_authorized_take_keeps_its_media_through_inventory_rebuild() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let active_take = track.items[0].geometry.take_id.clone();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        // ARA 侧认领了该 item，并记下了授权那一刻的 active take GUID。
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, active_take.clone());
        // 时间线上已有该 clip（ARA 身份 + 授权媒体），但 take 身份还是 ARA 自己的命名。
        let mut timeline: hifishifter_kernel::state::TimelineState =
            serde_json::from_value(serde_json::json!({
                "tracks":[{"id":"ui-host-track","name":"T","order":0}],
                "bpm":120.,"project_sec":1.,
                "clips":[{"id":format!("ui-ara-item-{item_id}"),"track_id":"ui-host-track",
                    "name":"C","start_sec":0.,"length_sec":4.0/44100.,
                    "takes":[{"id":"ara-clip-1-take-1","source_path":"authorized-source",
                        "source_start_sec":0.,"source_end_sec":4.0/44100.}]}],
            }))
            .unwrap();
        document.present_host_inventory(&mut timeline, "ui-");
        let clip = &timeline.clips[0];
        assert_eq!(clip.takes.len(), 3, "宿主清单的三个 take 都要出现");
        assert_eq!(
            clip.source_path.as_deref(),
            Some("authorized-source"),
            "授权媒体必须活过清单重建"
        );
        let granted = clip
            .takes
            .iter()
            .find(|take| take.id.ends_with(&active_take))
            .expect("授权 take 必须在清单里");
        assert_eq!(granted.source_path.as_deref(), Some("authorized-source"));
        assert!(
            clip.takes
                .iter()
                .filter(|take| take.id != granted.id)
                .all(|take| take.source_path.is_none()),
            "非授权 take 必须保持无源"
        );

        // 用户切了 active take（记录还指向一个已经不存在的 take）→ 一个 take 都不许带上
        // 旧采样。宿主清单同时多出一个 take，确保走的是"重建"这条路径。
        let stale = timeline.clips[0].clone();
        fixture.enable_takes(3);
        let grown = host.ui_track(&|| true).unwrap();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(grown.guid.clone(), grown);
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, "{99999999-9999-9999-9999-999999999999}".into());
        let mut again = hifishifter_kernel::state::TimelineState {
            clips: vec![stale],
            ..Default::default()
        };
        document.present_host_inventory(&mut again, "ui-");
        assert_eq!(again.clips[0].takes.len(), 4, "重建必须跟上宿主");
        assert!(
            again.clips[0].source_path.is_none(),
            "active take 换过之后不得沿用旧授权媒体"
        );
        assert!(again.clips[0]
            .takes
            .iter()
            .all(|take| take.source_path.is_none()));
        document.close();
    }

    /// **T22：授权记录缺席不等于撤销授权。**
    ///
    /// 【为什么这是"重开工程后所有 Clip 都在等宿主音频"的回归】`authorized_takes` /
    /// `authorized_sources` 是"某一刻观察到的授权事实"，只有一个写者
    /// （`remap_and_log`），而它要求 `region_items` 与 `ui_tracks` 在同一刻都就绪。
    /// 重开工程时这个条件常常不成立（几何缓存被 `clear_renderers` 的 revision 推进
    /// 作废、`assignments` 还没重发），于是那张表被写成空的。
    ///
    /// 旧实现把"查不到记录"直接当成"未授权"：清单重建出来的 take 一个都不带
    /// `source_path`，`normalize_takes()` 再把扁平投影一起清掉 ⇒ **一次清单轮次剥掉
    /// 全部 clip 的源**。正确解释是"未知"：保留 clip 已有的媒体身份。
    ///
    /// 【与下面那条 take_switch 用例的分界】那边记录**存在**但匹配不上（宿主真的换了
    /// active take）⇒ 必须继续拦。判据是"这张表里有没有这个 key"，不是"匹配没匹配上"。
    #[test]
    fn media_is_not_withheld_when_the_authorization_record_is_missing() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        // 该 item 已被某条活 region 认领 —— 但授权记录**一条都没有**（重开工程的形状）。
        document
            .regions
            .lock_or_recover()
            .insert(7, crate::ara::AraPlaybackRegion::default());
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        assert!(document.authorized_takes.lock_or_recover().is_empty());

        let mut timeline: hifishifter_kernel::state::TimelineState =
            serde_json::from_value(serde_json::json!({
                "tracks":[{"id":"ui-host-track","name":"T","order":0}],
                "bpm":120.,"project_sec":1.,
                "clips":[{"id":format!("ui-ara-item-{item_id}"),"track_id":"ui-host-track",
                    "name":"C","start_sec":0.,"length_sec":4.0/44100.,
                    "takes":[{"id":"ara-clip-1-take-1","source_path":"authorized-source",
                        "source_start_sec":0.,"source_end_sec":4.0/44100.}]}],
            }))
            .unwrap();
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(
            timeline.clips[0].source_path.as_deref(),
            Some("authorized-source"),
            "记录缺席只说明这一刻观察不到，不说明宿主撤销了授权：\
             把它当成撤销会一次性剥掉全部 clip 的源（重开工程后全屏'等待宿主音频'）"
        );

        // 对照：记录**存在**但指向另一个 take ⇒ 仍然不许沿用旧采样。
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, "{00000000-0000-0000-0000-000000000000}".into());
        let mut again = hifishifter_kernel::state::TimelineState {
            clips: vec![timeline.clips[0].clone()],
            ..Default::default()
        };
        document.present_host_inventory(&mut again, "ui-");
        assert!(
            again.clips[0].source_path.is_none(),
            "记录存在却对不上 = 宿主真的换了 take，必须继续拦"
        );
        document.close();
    }

    /// 用户把 active take 换成**同一个文件**的另一个 take 时，授权媒体仍要挂得上。
    ///
    /// 【为什么这条测试存在】媒体嫁接此前只认 take GUID 相等。复制 take / 切换 active
    /// take 会让 GUID 变、内容不变 —— 于是明明有音频的片段显示占位。文件路径相同即证明
    /// "同一份已授权 PCM"，可以安全回退；路径不同（或无路径）时仍然"宁可不挂"。
    #[test]
    fn a_take_switch_to_the_same_file_keeps_the_authorized_media() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        *fixture.source_file_name.borrow_mut() = Some("C:/media/same.wav".into());
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9881);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        document
            .region_items
            .lock_or_recover()
            .insert(7, item_id.clone());
        // GUID 对不上任何一个 take（模拟"切到了别的 take"），但文件路径对得上。
        document
            .authorized_takes
            .lock_or_recover()
            .insert(7, "{99999999-9999-9999-9999-999999999999}".into());
        document
            .authorized_sources
            .lock_or_recover()
            .insert(7, "C:/media/same.wav".into());

        let mut timeline: hifishifter_kernel::state::TimelineState =
            serde_json::from_value(serde_json::json!({
                "tracks":[{"id":"ui-host-track","name":"T","order":0}],
                "bpm":120.,"project_sec":1.,
                "clips":[{"id":format!("ui-ara-item-{item_id}"),"track_id":"ui-host-track",
                    "name":"C","start_sec":0.,"length_sec":4.0/44100.,
                    "takes":[{"id":"ara-clip-1-take-1","source_path":"authorized-source",
                        "source_start_sec":0.,"source_end_sec":4.0/44100.}]}],
            }))
            .unwrap();
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(
            timeline.clips[0].source_path.as_deref(),
            Some("authorized-source"),
            "同文件的 take 切换必须保住授权媒体（文件路径回退）"
        );

        // 路径对不上（宿主换成了另一个文件）→ 不许挂旧采样。
        document
            .authorized_sources
            .lock_or_recover()
            .insert(7, "C:/media/other.wav".into());
        let mut again = hifishifter_kernel::state::TimelineState {
            clips: vec![timeline.clips[0].clone()],
            ..Default::default()
        };
        document.present_host_inventory(&mut again, "ui-");
        assert!(
            again.clips[0]
                .takes
                .iter()
                .all(|take| take.source_path.is_none()),
            "文件不同时不得沿用旧授权媒体"
        );
        document.close();
    }

    /// 宿主改了 take 的声道模式后，**清单重建**必须把新值带到 clip 与 active take 上。
    ///
    /// 【为什么这条测试存在】用户报障："改了声道模式，插件里闪一下又变回旧值"。
    /// 根因是旧的"未变则早退"判据只比 active take id / take id 列表 / `loop_enabled`，
    /// **不含 `channel_mode`** —— 于是重建不发生，界面停在乐观更新的旧值上。现在
    /// `sync_host_takes` 无条件重建，这条测试钉住它不再回退。
    ///
    /// 关键：只测"改完立刻变"不算通过（那是乐观更新）；必须**再跑一次清单呈现**。
    #[test]
    fn a_channel_mode_change_survives_the_inventory_rebuild() {
        let fixture = super::super::MockHost::new();
        fixture.enable_takes(1);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock_or_recover()
            .insert(track.guid.clone(), track);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        // fixture 的 take 声道模式是 2（见 `host_take_set_projects_every_take...`）。
        assert_eq!(timeline.clips[0].channel_mode, 2);

        // 宿主把声道模式改成 3，清单刷新（同一 item、同一 take、同一循环源）。
        fixture.set_value("I_CHANMODE", 3.);
        let refreshed = host.ui_track(&|| true).unwrap();
        document
            .ui_tracks
            .lock_or_recover()
            .insert(refreshed.guid.clone(), refreshed);
        document.present_host_inventory(&mut timeline, "ui-");
        assert_eq!(
            timeline.clips[0].channel_mode, 3,
            "声道模式必须跟着宿主走，不能停在旧值"
        );
        assert!(
            timeline.clips[0]
                .takes
                .iter()
                .all(|take| take.channel_mode == 3),
            "每个 take 都要拿到新的声道模式"
        );
        document.close();
    }
}
