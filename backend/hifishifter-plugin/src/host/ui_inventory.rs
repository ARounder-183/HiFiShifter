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
    /// 宿主报告的方向；`None` = 读不出来（不当作"没倒放"）。
    pub reversed: Option<bool>,
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
    // 宿主 change 代次字段保留，随 UiTrack 一起快照。
    #[allow(dead_code)]
    pub change: i32,
    pub target: HostTrackTarget,
}

impl ReaperHost {
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
                    reversed: self.take_reversed(take, authorized),
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
    /// 不依赖ARA assignment的显示清单必须保留mute item及空轨，不产生音频source读取。
    #[test]
    fn ui_inventory_keeps_muted_item_and_empty_track_without_audio_assignments() {
        let fixture = super::super::ReaperFixture::new();
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
            .lock()
            .unwrap()
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
            .lock()
            .unwrap()
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
        let fixture = super::super::ReaperFixture::new();
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
        let fixture = super::super::ReaperFixture::new();
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
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        fixture.take_count_override.set(Some(MAX_TAKES + 1));
        let host = Arc::new(fixture.client());
        let error = host.ui_track(&|| true).err().expect("越界数量必须被拒绝");
        assert!(error.contains("take inventory budget"), "{error}");
    }

    /// 宿主报告的方向要如实透传；读不出来时保持"未知"，不谎报"没倒放"。
    #[test]
    fn take_direction_is_reported_only_when_the_host_answers() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        fixture.section_reports.set(true);
        fixture.take_reversed.set(true);
        let host = Arc::new(fixture.client());
        assert_eq!(
            host.ui_track(&|| true).unwrap().items[0].takes[0].reversed,
            Some(true)
        );

        fixture.take_reversed.set(false);
        assert_eq!(
            host.ui_track(&|| true).unwrap().items[0].takes[0].reversed,
            Some(false)
        );

        // 宿主不回答（不是 section/reverse 块）→ 未知，而不是"没倒放"。
        fixture.section_reports.set(false);
        assert_eq!(
            host.ui_track(&|| true).unwrap().items[0].takes[0].reversed,
            None
        );
    }

    /// 循环源（`B_LOOPSRC`）是 **item** 属性，必须读进几何并投影到 clip 与每个 take。
    ///
    /// 【为什么要读】内核的 `loop_enabled` 决定源窗口是否对**整份媒体**回绕，并且进渲染
    /// 缓存键。不读就等于"REAPER 里循环了、插件按不循环渲染"—— 又一条静默分叉。
    #[test]
    fn loop_source_is_read_from_the_item_and_projected() {
        let fixture = super::super::ReaperFixture::new();
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
            .lock()
            .unwrap()
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
        let fixture = super::super::ReaperFixture::new();
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
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        timeline.tracks.clear();
        document.present_host_inventory(&mut timeline, "ui-");
        let clip = &timeline.clips[0];
        // 夹具自报 REAPER 7.81 → 新轴才是权威值。
        assert_eq!(clip.fade_in_dir, geometry.fade_in_dir_new);
        assert_eq!(clip.fade_out_dir, geometry.fade_out_dir_new);
    }

    /// 逐 clip 的宿主媒体状态必须把**四种成因分开** —— 它们此前共用一句
    /// "等待 REAPER 提供音频（未分配 ARA 区域）"，于是"刚分割了一下"看起来像故障。
    #[test]
    fn host_media_state_separates_the_causes() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9878);
        document
            .ui_tracks
            .lock()
            .unwrap()
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

        // 倒放被隔离 → reversed（**不是** pending）：ARA 不给反向 PCM，这一条由宿主处理。
        payload["clips"][0]["source_path"] = serde_json::json!(null);
        payload["clips"][0]["reversed"] = serde_json::json!(true);
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "reversed");
    }

    /// "等待 REAPER 完成音频分配"不能是吸收态：等超期后必须转成带**原因码**的
    /// `unavailable`，否则用户对着一个永远转不完的占位，无从判断该做什么。
    ///
    /// 典型触发是 REAPER 的"倒放 Item 为新 Take"：宿主换了 active take，而 ARA 不再
    /// 重发模型 ⇒ `authorized_takes` 永久陈旧 ⇒ 没有 take 拿到 `source_path`。
    #[test]
    fn a_stale_pending_item_becomes_an_actionable_unavailable() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9879);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);

        // 该 item 已被 ARA 认领，授权记录指向**另一个** take GUID（= 用户切了 active take）。
        document
            .region_items
            .lock()
            .unwrap()
            .insert(7, item_id.clone());
        document
            .authorized_takes
            .lock()
            .unwrap()
            .insert(7, "{99999999-9999-9999-9999-999999999999}".into());

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        // 第一轮：刚进入"在途"，不报原因（宿主可能马上就好）。
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "pending");
        assert!(payload["clips"][0].get("host_media_reason").is_none());

        // 把"开始等待的时刻"倒推到阈值之前，模拟"等了很久仍无源"。
        {
            let mut since = document.pending_since.lock().unwrap();
            since.insert(
                item_id.clone(),
                std::time::Instant::now() - std::time::Duration::from_secs(10),
            );
        }
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(payload["clips"][0]["host_media_reason"], "take_switched");

        // 幂等：再次呈现仍是 `unavailable` + 同一原因，不能在 pending/unavailable
        // 之间来回跳（起点被保留，不会重新 `or_insert(now)`）。
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(payload["clips"][0]["host_media_reason"], "take_switched");

        // 拿到音频后必须回到 ready，并**清掉计时** —— 否则下次掉回"在途"会立刻超期。
        payload["clips"][0]["source_path"] = serde_json::json!("C:/pcm/source.wav");
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "ready");
        assert!(document.pending_since.lock().unwrap().is_empty());
    }

    /// 方向位读不出来（`PCM_Source_GetSectionInfo` 不可用）时，插件**既不能**标为倒放，
    /// **也不能**假定为正放。必须如实投影 `reversed_known=false`，并在超期后给出
    /// `direction_unknown` 原因码 —— 再等也不会变好，用户需要知道这一点。
    #[test]
    fn an_unreadable_direction_is_reported_instead_of_assumed_forward() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(0);
        fixture.enable_media();
        // `section_reports` 默认 false ⇒ `take_reversed` 返回 `None`（读不到方向）。
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9880);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);

        let mut payload = serde_json::json!({"clips": [
            {"id": format!("ui-ara-item-{item_id}"), "source_path": null, "reversed": false}
        ]});
        // 已被认领（⇒ 不是 `unclaimed`）、也没有"授权指向另一个 take"（⇒ 不是
        // `take_switched`）—— 于是唯一剩下的成因就是"方向读不出来"。
        document
            .region_items
            .lock()
            .unwrap()
            .insert(7, item_id.clone());
        document.decorate_host_media_locked(&mut payload, "ui-");
        // 读不到方向 ⇒ `reversed_known=false`；未超期时仍是"在途"。
        assert_eq!(payload["clips"][0]["reversed_known"], false);
        assert_eq!(payload["clips"][0]["host_media"], "pending");

        {
            let mut since = document.pending_since.lock().unwrap();
            since.insert(
                item_id.clone(),
                std::time::Instant::now() - std::time::Duration::from_secs(10),
            );
        }
        document.decorate_host_media_locked(&mut payload, "ui-");
        assert_eq!(payload["clips"][0]["host_media"], "unavailable");
        assert_eq!(
            payload["clips"][0]["host_media_reason"],
            "direction_unknown"
        );
    }

    /// 多 take 投影进显示 Clip：全部 take 到位、active 一个、**都不带 source_path**。
    #[test]
    fn host_take_set_projects_every_take_without_any_source_path() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        fixture.section_reports.set(true);
        fixture.take_reversed.set(true);
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock()
            .unwrap()
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
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let active_take = track.items[0].geometry.take_id.clone();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);
        // ARA 侧认领了该 item，并记下了授权那一刻的 active take GUID。
        document
            .region_items
            .lock()
            .unwrap()
            .insert(7, item_id.clone());
        document
            .authorized_takes
            .lock()
            .unwrap()
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
            .lock()
            .unwrap()
            .insert(grown.guid.clone(), grown);
        document
            .authorized_takes
            .lock()
            .unwrap()
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

    /// 用户把 active take 换成**同一个文件**的另一个 take 时，授权媒体仍要挂得上。
    ///
    /// 【为什么这条测试存在】媒体嫁接此前只认 take GUID 相等。复制 take / 切换 active
    /// take 会让 GUID 变、内容不变 —— 于是明明有音频的片段显示占位。文件路径相同即证明
    /// "同一份已授权 PCM"，可以安全回退；路径不同（或无路径）时仍然"宁可不挂"。
    #[test]
    fn a_take_switch_to_the_same_file_keeps_the_authorized_media() {
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(2);
        fixture.enable_media();
        *fixture.source_file_name.borrow_mut() = Some("C:/media/same.wav".into());
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let item_id = track.items[0].geometry.item_id.clone();
        let document = crate::render::document::DocumentSession::new(9881);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track);
        document
            .region_items
            .lock()
            .unwrap()
            .insert(7, item_id.clone());
        // GUID 对不上任何一个 take（模拟"切到了别的 take"），但文件路径对得上。
        document
            .authorized_takes
            .lock()
            .unwrap()
            .insert(7, "{99999999-9999-9999-9999-999999999999}".into());
        document
            .authorized_sources
            .lock()
            .unwrap()
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
            .lock()
            .unwrap()
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
        let fixture = super::super::ReaperFixture::new();
        fixture.enable_takes(1);
        fixture.enable_media();
        let host = Arc::new(fixture.client());
        let track = host.ui_track(&|| true).unwrap();
        let document = crate::render::document::DocumentSession::new(9876);
        document
            .ui_tracks
            .lock()
            .unwrap()
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
            .lock()
            .unwrap()
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
