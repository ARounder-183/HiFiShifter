//! 核对官方REAPER SDK的原生宿主适配：只在原始UI/model线程读所属project的真实位置。
//! ABI依据 justinfrankel/reaper-sdk c0eafe87863b2bf69c5c822760f1b32a753b211b，
//! sdk/reaper_vst3_interfaces.h 与 reaper_plugin_functions.h。手写适配，不是SDK原文件。
use crate::editor::connection::UnknownVtbl;
use crate::vst3::{uid_guid, K_RESULT_OK};
use std::ffi::{c_char, c_void};
#[path="reaper_write.rs"]mod write;
pub(crate) use write::{HostClipTarget,HostUndoBlock};
#[path="reaper_media.rs"]mod media;
pub(crate) use media::HostTrackTarget;
pub(crate) use media::CreatedTrack;

const IID: [u32; 4] = [0x79655E36, 0x77EE4267, 0xA573FEF7, 0x4912C27C];
#[repr(C)]
struct HostVtbl {
    base: UnknownVtbl,
    api: unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void,
    parent: unsafe extern "system" fn(*mut c_void, u32) -> *mut c_void,
    extended: unsafe extern "system" fn(
        *mut c_void,
        u32,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> *mut c_void,
}
struct Interface(usize);
impl Drop for Interface {
    fn drop(&mut self) {
        // SAFETY: 成功QI拥有一次引用；FUnknown release只平衡引用，不调用REAPER项目API。
        unsafe {
            let pointer = self.0 as *mut c_void;
            ((**pointer.cast::<*const UnknownVtbl>()).release)(pointer);
        }
    }
}
type Position = unsafe extern "C" fn(*mut c_void) -> f64;
type PlayState = unsafe extern "C" fn(*mut c_void) -> i32;
type Validate = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> bool;
type TakeItem = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type Value = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;
type Guid = unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, bool) -> bool;
type Marker = unsafe extern "C" fn(*mut c_void, i32, *mut f64, *mut f64) -> i32;
type Slope = unsafe extern "C" fn(*mut c_void, i32) -> f64;
type AppVersion = unsafe extern "C" fn() -> *const c_char;
struct Transport {
    position: Position,
    cursor: Position,
    state: PlayState,
}
struct GeometryApi {
    validate: Validate,
    item: TakeItem,
    item_value: Value,
    take_value: Value,
    item_guid: Guid,
    take_guid: Guid,
    count: PlayState,
    marker: Marker,
    slope: Slope,
    change: PlayState,
}
pub(crate) struct ReaperHost {
    _interface: Interface,
    thread: std::thread::ThreadId,
    project: std::sync::atomic::AtomicUsize,
    transport: Option<Transport>,
    geometry: Option<GeometryApi>,
    write:Option<write::WriteApi>,
    history:Option<write::HistoryApi>,
    media:Option<media::MediaApi>,
    extended_media:Option<media::ExtendedMedia>,
    validate: Option<Validate>,
    fade_axes_new: Option<bool>,
}
/// 每次外部调用前后重检；Arc/FUnknown引用不保活project/item/take。
fn checked<T>(authorized: &impl Fn() -> bool, call: impl FnOnce() -> T) -> Result<T, String> {
    if !authorized() {
        return Err("host query authorization revoked".into());
    }
    let value = call();
    if !authorized() {
        return Err("host query authorization revoked by reentry".into());
    }
    Ok(value)
}
impl ReaperHost {
    /// 从真实host context QI；parent(3)是所属project，不使用当前活动project或按轨名猜归属。
    /// # Safety
    /// context是初始化期间可读的活FUnknown；host在该 owning reference期间保持接口/API有效。
    pub unsafe fn from_context(
        context: *mut c_void,
        authorized: impl Fn() -> bool,
    ) -> Option<Self> {
        if context.is_null() || !authorized() {
            return None;
        }
        let mut pointer = std::ptr::null_mut();
        let iid = uid_guid(IID);
        let result = unsafe {
            ((**context.cast::<*const UnknownVtbl>()).query)(context, iid.as_ptr(), &mut pointer)
        };
        if result != K_RESULT_OK || pointer.is_null() {
            crate::log_line(&format!("[reaper-host] QI result={result} interface={}",!pointer.is_null()));
            return None;
        }
        let interface = Interface(pointer as usize);
        crate::log_line("[reaper-host] QI succeeded");
        if !authorized() {
            return None;
        }
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let project = checked(&authorized, || unsafe { (table.parent)(pointer, 3) }).ok()?;
        if project.is_null() {
            // 实测initialize尚未挂接project；保留拥有引用的接口，稍后只沿同一直接parent绑定。
            crate::log_line("[reaper-host] initialization parent(project) pending; retaining host interface");
        }
        let api = |name: &std::ffi::CStr| {
            checked(&authorized, || unsafe {
                (table.api)(pointer, name.as_ptr())
            })
        };
        // SAFETY: 每个名字/类型来自锁定官方头，缺API仅使对应可选能力不可用。
        macro_rules! lookup {
            ($name:expr,$ty:ty) => {{
                let p = api($name).ok()?;
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { std::mem::transmute::<*mut c_void, $ty>(p) })
                }
            }};
        }
        let position = lookup!(c"GetPlayPositionEx", Position);
        let cursor = lookup!(c"GetCursorPositionEx", Position);
        let state = lookup!(c"GetPlayStateEx", PlayState);
        let transport = position
            .zip(cursor)
            .zip(state)
            .map(|((position, cursor), state)| Transport {
                position,
                cursor,
                state,
            });
        let validate = lookup!(c"ValidatePtr2", Validate);
        // 官方GetAppVersion返回静态版本字符串，只有已知版本才选择新/旧淡化轴。
        let version=lookup!(c"GetAppVersion",AppVersion);
        let fade_axes_new=version.and_then(|getter|checked(&authorized,||unsafe {getter()}).ok())
            .filter(|value|!value.is_null()).and_then(|value|unsafe {std::ffi::CStr::from_ptr(value)}.to_str().ok())
            .and_then(new_fade_axes);
        let item = lookup!(c"GetMediaItemTake_Item", TakeItem);
        let item_value = lookup!(c"GetMediaItemInfo_Value", Value);
        let take_value = lookup!(c"GetMediaItemTakeInfo_Value", Value);
        let item_guid = lookup!(c"GetSetMediaItemInfo_String", Guid);
        let take_guid = lookup!(c"GetSetMediaItemTakeInfo_String", Guid);
        let count = lookup!(c"GetTakeNumStretchMarkers", PlayState);
        let marker = lookup!(c"GetTakeStretchMarker", Marker);
        let slope = lookup!(c"GetTakeStretchMarkerSlope", Slope);
        let change = lookup!(c"GetProjectStateChangeCount", PlayState);
        let tracks=match (lookup!(c"InsertTrackInProject",media::InsertTrack),lookup!(c"CountTracks",media::CountTracks),lookup!(c"GetTrack",media::GetTrack),
            lookup!(c"TrackFX_AddByName",media::AddFx),lookup!(c"DeleteTrack",media::DeleteTrack),lookup!(c"CountTrackMediaItems",media::TrackCount),
            lookup!(c"TrackFX_GetCount",media::TrackCount),lookup!(c"TrackFX_GetFXGUID",media::FxGuid)) {
            (Some(insert),Some(count),Some(get),Some(add_fx),Some(delete),Some(item_count),Some(fx_count),Some(fx_guid))=>
                Some(media::NewTrackApi {insert,count,get,add_fx,delete,item_count,fx_count,fx_guid}),_=>None,
        };
        let extended_media=Some(media::ExtendedMedia {picker:lookup!(c"GetUserFileName",media::MultiPicker),tracks});
        let media=match (lookup!(c"PCM_Source_CreateFromFileEx",media::CreateSource),lookup!(c"PCM_Source_Destroy",media::DestroySource),
            lookup!(c"GetMediaSourceLength",media::SourceLength),lookup!(c"AddMediaItemToTrack",media::CreateItem),lookup!(c"AddTakeToMediaItem",media::CreateTake),
            lookup!(c"GetSetMediaItemTakeInfo",media::TakeInfo),lookup!(c"DeleteTrackMediaItem",media::DeleteItem),
            lookup!(c"GetSetMediaTrackInfo_String",Guid),lookup!(c"GetUserFileNameForRead",media::FilePicker)) {
            (Some(create_source),Some(destroy_source),Some(length),Some(create_item),Some(create_take),Some(take_info),Some(delete_item),Some(track_guid),Some(picker))=>
                Some(media::MediaApi {create_source,destroy_source,length,create_item,create_take,take_info,delete_item,track_guid,picker}),_=>None,
        };
        let history=match (lookup!(c"Undo_DoUndo2",write::UndoAction),lookup!(c"Undo_DoRedo2",write::UndoAction),
            lookup!(c"Undo_CanUndo2",write::UndoLabel),lookup!(c"Undo_CanRedo2",write::UndoLabel),
            lookup!(c"Undo_GetCurEntry",write::UndoAction),lookup!(c"Undo_GetNumEntries",write::UndoAction),lookup!(c"Undo_GetEntryDesc",write::UndoEntry)) {
            (Some(undo),Some(redo),Some(can_undo),Some(can_redo),Some(current),Some(count),Some(entry))=>
                Some(write::HistoryApi {undo,redo,can_undo,can_redo,current,count,entry}),_=>None,
        };
        let write=match (
            lookup!(c"SetMediaItemInfo_Value",write::SetValue),lookup!(c"SetMediaItemTakeInfo_Value",write::SetValue),
            lookup!(c"GetMediaItem_Track",write::Track),lookup!(c"MoveMediaItemToTrack",write::Move),
            lookup!(c"Undo_BeginBlock2",write::Begin),lookup!(c"Undo_EndBlock2",write::End),
            lookup!(c"UpdateItemInProject",write::Update),lookup!(c"UpdateArrange",write::Arrange)) {
            (Some(set_item),Some(set_take),Some(item_track),Some(move_item),Some(begin),Some(end),Some(update),Some(arrange))=>
                Some(write::WriteApi {set_item,set_take,item_track,move_item,begin,end,update,arrange}),
            _=>None,
        };
        let geometry = match (
            validate, item, item_value, take_value, item_guid, take_guid, count, marker, slope,
            change,
        ) {
            (
                Some(validate),
                Some(item),
                Some(item_value),
                Some(take_value),
                Some(item_guid),
                Some(take_guid),
                Some(count),
                Some(marker),
                Some(slope),
                Some(change),
            ) => Some(GeometryApi {
                validate,
                item,
                item_value,
                take_value,
                item_guid,
                take_guid,
                count,
                marker,
                slope,
                change,
            }),
            _ => None,
        };
        Some(Self {
            _interface: interface,
            thread: std::thread::current().id(),
            project: std::sync::atomic::AtomicUsize::new(project as usize),
            transport,
            geometry,
            write,
            history,
            media,
            extended_media,
            validate,
            fade_axes_new,
        })
    }
    /// 项目延迟挂接只从同一个接口的直接parent取得；拒绝null，不借API的“当前项目”语义。
    /// 一旦绑定，不因用户切换活动tab而重新绑定；原线程与外部调用授权检查仍然执行。
    fn project(&self,authorized:&impl Fn()->bool)->Result<*mut c_void,String> {
        if std::thread::current().id()!=self.thread {return Err("REAPER project queried outside its model/UI thread".into());}
        let known=self.project.load(std::sync::atomic::Ordering::Acquire);
        if known!=0 {return Ok(known as *mut c_void);}
        let pointer=self._interface.0 as *mut c_void;let table=unsafe {&**pointer.cast::<*const HostVtbl>()};
        let project=checked(authorized,||unsafe {(table.parent)(pointer,3)})?;
        if project.is_null() {return Err("REAPER direct parent project not attached yet".into());}
        self.project.store(project as usize,std::sync::atomic::Ordering::Release);
        crate::log_line("[reaper-host] direct project attached on model/UI thread");Ok(project)
    }
    /// 原UI线程读取延迟补偿的实际听到位置；暂停保持play位置，完全停止才读edit cursor。
    pub fn sample(&self, authorized: impl Fn() -> bool) -> Result<(f64, bool), String> {
        if std::thread::current().id() != self.thread {
            return Err("REAPER transport queried outside its model/UI thread".into());
        }
        let api = self
            .transport
            .as_ref()
            .ok_or("REAPER transport API unavailable")?;
        let project = self.project(&authorized)?;
        let valid = || -> Result<(), String> {
            if let Some(validate) = self.validate {
                if !checked(&authorized, || unsafe {
                    validate(project, project, c"ReaProject*".as_ptr())
                })? {
                    return Err("invalid REAPER project".into());
                }
            }
            Ok(())
        };
        valid()?;
        let state = checked(&authorized, || unsafe { (api.state)(project) })?;
        if state < 0 {
            return Err("invalid REAPER play state".into());
        }
        valid()?;
        let position = checked(&authorized, || unsafe {
            if state & 3 != 0 {
                (api.position)(project)
            } else {
                (api.cursor)(project)
            }
        })?;
        if !position.is_finite() {
            return Err("invalid REAPER project position".into());
        }
        valid()?;
        Ok((position, state & 1 != 0))
    }

    /// UI缓存只读变更token；负数/回绕合法，不假定单调，也不把counter当作take租约。
    pub(crate) fn geometry_revision(&self, authorized: impl Fn()->bool)->Result<i32,String> {
        if std::thread::current().id()!=self.thread {return Err("REAPER geometry queried outside its model/UI thread".into());}
        let api=self.geometry.as_ref().ok_or("REAPER geometry API unavailable")?;
        let project=self.project(&authorized)?;
        // SAFETY: typed API来自核对过的官方头；每次查询前后重新检查调用者许可。
        if !checked(&authorized,||unsafe {(api.validate)(project,project,c"ReaProject*".as_ptr())})? {
            return Err("invalid REAPER project/type ownership".into());
        }
        checked(&authorized,||unsafe {(api.change)(project)})
    }
    /// 只接受直接parent(2) take；唯一ARA绑定由调用方冻结并在每个getter前后重检。
    pub fn geometry(
        &self,
        authorized: impl Fn() -> bool,
    ) -> Result<super::geometry::HostClipGeometry, String> {
        use super::geometry::{HostClipGeometry, HostStretchMarker};
        if std::thread::current().id() != self.thread {
            return Err("REAPER geometry queried outside its model/UI thread".into());
        }
        let api = self
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let project = self.project(&authorized)?;
        let valid = |object, kind: &std::ffi::CStr| -> Result<(), String> {
            if checked(&authorized, || unsafe {
                (api.validate)(project, object, kind.as_ptr())
            })? {
                Ok(())
            } else {
                Err("invalid REAPER project/type ownership".into())
            }
        };
        valid(project, c"ReaProject*")?;
        let before = checked(&authorized, || unsafe { (api.change)(project) })?;
        let take = checked(&authorized, || unsafe { (table.parent)(pointer, 2) })?;
        if take.is_null() {
            return Err("REAPER direct parent take unavailable".into());
        }
        valid(take, c"MediaItem_Take*")?;
        let item = checked(&authorized, || unsafe { (api.item)(take) })?;
        if item.is_null() {
            return Err("REAPER take parent item unavailable".into());
        }
        valid(item, c"MediaItem*")?;
        let guid = |object, getter: Guid| -> Result<String, String> {
            valid(project, c"ReaProject*")?;
            valid(
                object,
                if object == take {
                    c"MediaItem_Take*"
                } else {
                    c"MediaItem*"
                },
            )?;
            // GUID字符串固定38字符+NUL；仅读取GUID，不开放任意长字符串/API写入。
            let mut bytes = [0u8; 64];
            if !checked(&authorized, || unsafe {
                getter(object, c"GUID".as_ptr(), bytes.as_mut_ptr().cast(), false)
            })? {
                return Err("REAPER GUID unavailable".into());
            }
            if bytes[38] != 0
                || bytes[0] != b'{'
                || bytes[37] != b'}'
                || !(1..37).all(|i| {
                    if [9, 14, 19, 24].contains(&i) {
                        bytes[i] == b'-'
                    } else {
                        bytes[i].is_ascii_hexdigit()
                    }
                })
            {
                return Err("invalid REAPER GUID".into());
            }
            Ok(std::str::from_utf8(&bytes[..38])
                .map_err(|_| "invalid REAPER GUID")?
                .into())
        };
        let item_id = guid(item, api.item_guid)?;
        let take_id = guid(take, api.take_guid)?;
        let value = |object, getter: Value, name: &std::ffi::CStr| -> Result<f64, String> {
            valid(project, c"ReaProject*")?;
            valid(
                object,
                if object == take {
                    c"MediaItem_Take*"
                } else {
                    c"MediaItem*"
                },
            )?;
            let value = checked(&authorized, || unsafe { getter(object, name.as_ptr()) })?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!("nonfinite REAPER field {}", name.to_string_lossy()))
            }
        };
        let iv = |name| value(item, api.item_value, name);
        let tv = |name| value(take, api.take_value, name);
        let start_sec = iv(c"D_POSITION")?;
        let duration_sec = iv(c"D_LENGTH")?;
        let source_start_sec = tv(c"D_STARTOFFS")?;
        let playback_rate = tv(c"D_PLAYRATE")?;
        if duration_sec <= 0. || playback_rate <= 0. {
            return Err("invalid REAPER length/playback rate".into());
        }
        let integer = |v: f64| -> Result<i32, String> {
            if v.fract() == 0. && v >= i32::MIN as f64 && v <= i32::MAX as f64 {
                Ok(v as i32)
            } else {
                Err("invalid REAPER integer field".into())
            }
        };
        let boolean = |v: f64| -> Result<bool, String> {
            match v {
                0. => Ok(false),
                1. => Ok(true),
                _ => Err("invalid REAPER boolean field".into()),
            }
        };
        let preserve_pitch = boolean(tv(c"B_PPITCH")?)?;
        let channel_mode = integer(tv(c"I_CHANMODE")?)?;
        let take_pitch = tv(c"D_PITCH")?;
        let item_timebase = integer(iv(c"C_BEATATTACHMODE")?)?;
        let auto_stretch = boolean(iv(c"C_AUTOSTRETCH")?)?;
        let muted = boolean(iv(c"B_MUTE")?)?;
        let length = |name| -> Result<f64, String> {
            let v = iv(name)?;
            if v < 0. {
                Err("invalid REAPER fade length".into())
            } else {
                Ok(v)
            }
        };
        let fade_in_sec = length(c"D_FADEINLEN")?;
        let fade_out_sec = length(c"D_FADEOUTLEN")?;
        let auto_fade_in_sec = length(c"D_FADEINLEN_AUTO")?;
        let auto_fade_out_sec = length(c"D_FADEOUTLEN_AUTO")?;
        let fade_in_shape = iv(c"C_FADEINSHAPE")?;
        let fade_out_shape = iv(c"C_FADEOUTSHAPE")?;
        let fade_in_dir = iv(c"D_FADEINDIR")?;
        let fade_out_dir = iv(c"D_FADEOUTDIR")?;
        let fade_in_dir_new = iv(c"D_FADEINDIR_NEW")?;
        let fade_out_dir_new = iv(c"D_FADEOUTDIR_NEW")?;
        let fade_in_dir2_new = iv(c"D_FADEINDIR2_NEW")?;
        let fade_out_dir2_new = iv(c"D_FADEOUTDIR2_NEW")?;
        let take_valid = || -> Result<(), String> {
            valid(project, c"ReaProject*")?;
            valid(take, c"MediaItem_Take*")
        };
        take_valid()?;
        let count = checked(&authorized, || unsafe { (api.count)(take) })?;
        if !(0..=MAX_MARKERS).contains(&count) {
            return Err("REAPER marker count exceeds metadata budget".into());
        }
        let mut markers = Vec::with_capacity(count as usize);
        for index in 0..count {
            take_valid()?;
            let mut pos = f64::NAN;
            let mut src = f64::NAN;
            let actual = checked(&authorized, || unsafe {
                (api.marker)(take, index, &mut pos, &mut src)
            })?;
            if actual != index || !pos.is_finite() || !src.is_finite() {
                return Err("invalid REAPER stretch marker".into());
            }
            take_valid()?;
            let slope = checked(&authorized, || unsafe { (api.slope)(take, index) })?;
            if !slope.is_finite() {
                return Err("invalid REAPER stretch marker slope".into());
            }
            markers.push(HostStretchMarker {
                item_position_raw: pos,
                source_position_raw: src,
                slope_raw: slope,
            });
        }
        valid(project, c"ReaProject*")?;
        valid(take, c"MediaItem_Take*")?;
        valid(item, c"MediaItem*")?;
        let after = checked(&authorized, || unsafe { (api.change)(project) })?;
        if before != after {
            return Err("REAPER project changed during geometry snapshot".into());
        }
        Ok(HostClipGeometry {
            item_id,
            take_id,
            start_sec,
            source_start_sec,
            duration_sec,
            playback_rate,
            preserve_pitch,
            channel_mode,
            take_pitch,
            item_timebase,
            auto_stretch,
            muted,
            markers,
            fade_in_sec,
            fade_out_sec,
            fade_in_shape,
            fade_out_shape,
            fade_in_dir,
            fade_out_dir,
            fade_in_dir_new,
            fade_out_dir_new,
            fade_in_dir2_new,
            fade_out_dir2_new,
            fade_axes_new:self.fade_axes_new,
            auto_fade_in_sec,
            auto_fade_out_sec,
        })
    }
}
/// 元数据独立有界；不扩大PCM预算，也不根据item窗口删掉合法窗口外marker。
const MAX_MARKERS: i32 = 16_384;

/// 比较官方版本的整数分量（7.100不能按浮点误判为7.10）。
fn new_fade_axes(version:&str)->Option<bool> {
    let numeric=version.split('/').next()?;let (major,minor)=numeric.split_once('.')?;
    let major=major.parse::<u32>().ok()?;
    let minor=minor.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().ok()?;
    Some((major,minor)>=(7,81))
}

#[cfg(test)]
#[path = "reaper_tests.rs"]
mod geometry_tests;
#[cfg(test)]
pub(crate) use geometry_tests::Fixture as ReaperFixture;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    /// 新旧版本边界与未知字符串明确区分，未来minor不能误按小数比较。
    #[test]
    fn fade_axis_version_boundary_does_not_guess_unknown_versions() {
        assert_eq!(new_fade_axes("7.80/x64"),Some(false));assert_eq!(new_fade_axes("7.81/x64"),Some(true));
        assert_eq!(new_fade_axes("7.100+dev1005/x64"),Some(true));assert_eq!(new_fade_axes("8.0"),Some(true));
        assert_eq!(new_fade_axes("unknown"),None);
    }
    struct Project {
        position: f64,
        cursor: f64,
        state: i32,
    }
    #[repr(C)]
    struct Host {
        table: *const HostVtbl,
        refs: AtomicU32,
        project: *mut Project,
    }
    unsafe extern "system" fn query(
        this: *mut c_void,
        iid: *const u8,
        out: *mut *mut c_void,
    ) -> i32 {
        let expected = [
            0x36, 0x5e, 0x65, 0x79, 0xee, 0x77, 0x67, 0x42, 0xa5, 0x73, 0xfe, 0xf7, 0x49, 0x12,
            0xc2, 0x7c,
        ];
        if unsafe { std::slice::from_raw_parts(iid, 16) } != expected {
            return crate::vst3::K_NO_INTERFACE;
        }
        unsafe {
            add(this);
            *out = this;
        }
        K_RESULT_OK
    }
    unsafe extern "system" fn add(this: *mut c_void) -> u32 {
        unsafe { (&*this.cast::<Host>()).refs.fetch_add(1, Ordering::AcqRel) + 1 }
    }
    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        unsafe { (&*this.cast::<Host>()).refs.fetch_sub(1, Ordering::AcqRel) - 1 }
    }
    unsafe extern "C" fn position(project: *mut c_void) -> f64 {
        unsafe { (&*project.cast::<Project>()).position }
    }
    unsafe extern "C" fn cursor(project: *mut c_void) -> f64 {
        unsafe { (&*project.cast::<Project>()).cursor }
    }
    unsafe extern "C" fn state(project: *mut c_void) -> i32 {
        unsafe { (&*project.cast::<Project>()).state }
    }
    unsafe extern "system" fn api(_: *mut c_void, name: *const c_char) -> *mut c_void {
        match unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() {
            b"GetPlayPositionEx" => position as *const () as *mut c_void,
            b"GetCursorPositionEx" => cursor as *const () as *mut c_void,
            b"GetPlayStateEx" => state as *const () as *mut c_void,
            _ => std::ptr::null_mut(),
        }
    }
    unsafe extern "system" fn parent(this: *mut c_void, selector: u32) -> *mut c_void {
        if selector == 3 {
            unsafe { (&*this.cast::<Host>()).project.cast() }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe extern "system" fn extended(
        _: *mut c_void,
        _: u32,
        _: *mut c_void,
        _: *mut c_void,
        _: *mut c_void,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }
    static TABLE: HostVtbl = HostVtbl {
        base: UnknownVtbl {
            query,
            add,
            release,
        },
        api,
        parent,
        extended,
    };
    thread_local! {static REVOKED:std::cell::Cell<bool>=const {std::cell::Cell::new(false)};}
    unsafe extern "C" fn reentrant_state(_: *mut c_void) -> i32 {
        REVOKED.set(true);
        1
    }
    unsafe extern "system" fn reentrant_api(this: *mut c_void, name: *const c_char) -> *mut c_void {
        if unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() == b"GetPlayStateEx" {
            reentrant_state as *const () as *mut c_void
        } else {
            unsafe { api(this, name) }
        }
    }
    /// state读取关闭文档后必须停止，strong host引用不能保活project。
    #[test]
    fn task38a_transport_reentry_revokes_the_next_project_getter() {
        REVOKED.set(false);
        let table = HostVtbl {
            base: UnknownVtbl {
                query,
                add,
                release,
            },
            api: reentrant_api,
            parent,
            extended,
        };
        let mut project = Project {
            position: 4.5,
            cursor: 1.,
            state: 1,
        };
        let mut host = Host {
            table: &table,
            refs: AtomicU32::new(1),
            project: &raw mut project,
        };
        let client =
            unsafe { ReaperHost::from_context((&raw mut host).cast(), || !REVOKED.get()) }.unwrap();
        assert!(
            client.sample(|| !REVOKED.get()).is_err(),
            "重入关闭必须丢弃本次transport结果"
        );
    }
    /// 通过真实QI/函数表适配测试所属project、pause/stop区别、引用平衡和跨线程拒绝。
    #[test]
    fn bound_project_transport_uses_actual_position_and_never_calls_from_a_worker() {
        let mut project = Project {
            position: 4.5,
            cursor: 1.0,
            state: 1,
        };
        let mut host = Host {
            table: &TABLE,
            refs: AtomicU32::new(1),
            project: &raw mut project,
        };
        let client = unsafe { ReaperHost::from_context((&raw mut host).cast(), || true) }.unwrap();
        assert_eq!(client.sample(|| true).unwrap(), (4.5, true));
        assert_eq!(host.refs.load(Ordering::Acquire), 2);
        project.state = 2;
        assert_eq!(client.sample(|| true).unwrap(), (4.5, false));
        project.state = 0;
        assert_eq!(client.sample(|| true).unwrap(), (1.0, false));
        // client中的project是fixture稳定地址；不得换用active project或GetPlayPosition2Ex。
        std::thread::scope(|scope| {
            assert!(scope
                .spawn(|| client.sample(|| true))
                .join()
                .unwrap()
                .is_err());
        });
        drop(client);
        assert_eq!(host.refs.load(Ordering::Acquire), 1);
    }
}
