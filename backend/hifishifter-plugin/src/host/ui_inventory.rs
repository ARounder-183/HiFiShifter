//! 宿主GUI清单与ARA播放分配分离；枚举FX真实parent轨道（若它是folder，连同其全部
//! 后代轨道），静音item/空轨不消失。
use super::*;
use std::ffi::CStr;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct UiItem {
    pub geometry: super::super::geometry::HostClipGeometry,
    pub name: String,
    pub target: HostClipTarget,
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
            let take = checked(authorized, || unsafe { active_take(item) })?;
            if take.is_null() {
                continue;
            }
            valid(take, c"MediaItem_Take*")?;
            let g = self.geometry_for_take(take, authorized)?;
            let p = checked(authorized, || unsafe { take_name(take) })?;
            let name = if p.is_null() {
                String::new()
            } else {
                let value = unsafe { CStr::from_ptr(p) };
                if value.to_bytes().len() > 8192 {
                    return Err("take name budget exceeded".into());
                }
                value.to_str().map_err(|_| "invalid take UTF-8")?.to_owned()
            };
            let target = super::write::inventory_target(
                self.clone(),
                project as usize,
                track as usize,
                item as usize,
                take as usize,
                g.clone(),
                authorized,
            )?;
            items.push(UiItem {
                geometry: g,
                name,
                target,
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
}
