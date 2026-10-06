//! 原生QI/vtable/raw函数回归；fixture不是实际REAPER parent/API返回的验收证据。
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

thread_local! {static ACTIVE:Cell<usize>=const {Cell::new(0)};}
#[repr(C)]
pub(crate) struct Fixture {
    table: *const HostVtbl,
    refs: AtomicU32,
    take_token: u8,
    item_token: u8,
    track_token:u8,
    writer_enabled:Cell<bool>,
    undo_records:RefCell<Vec<std::ffi::CString>>,undo_position:Cell<i32>,
    media_enabled:Cell<bool>,media_sources:RefCell<Vec<Box<u8>>>,take_source:Cell<usize>,fail_media_take:Cell<bool>,media_path:RefCell<String>,
    extra_tracks:RefCell<Vec<Box<u8>>>,new_fx:RefCell<std::collections::HashSet<usize>>,fx_guid_bytes:[u8;16],picker_paths:RefCell<Vec<String>>,
    pub valid: Cell<bool>,
    connection_host: Cell<bool>,
    calls: RefCell<Vec<String>>,
    revoke_at: Cell<usize>,
    change_at: Cell<usize>,
    missing: Cell<Option<&'static str>>,
    no_take: Cell<bool>,
    no_project:Cell<bool>,
    bad_type: Cell<Option<&'static str>>,
    values: RefCell<BTreeMap<&'static str, f64>>,
    markers: RefCell<Vec<(f64, f64, f64)>>,
    count_override: Cell<Option<i32>>,
    bad_marker: Cell<bool>,
    bad_guid: Cell<bool>,
    change: Cell<i32>,
    pub hook: RefCell<Option<(String, Box<dyn Fn()>)>>,
}
impl Fixture {
    pub fn new() -> Box<Self> {
        let value = Box::new(Self {
            table: &TABLE,
            refs: AtomicU32::new(1),
            take_token: 1,
            item_token: 2,
            track_token:3,writer_enabled:Cell::new(false),
            undo_records:RefCell::new(vec![std::ffi::CString::new("Initial state").unwrap()]),undo_position:Cell::new(0),
            media_enabled:Cell::new(false),media_sources:RefCell::new(Vec::new()),take_source:Cell::new(0),fail_media_take:Cell::new(false),media_path:RefCell::new(String::new()),
            extra_tracks:RefCell::new(Vec::new()),new_fx:RefCell::new(Default::default()),fx_guid_bytes:[4;16],picker_paths:RefCell::new(Vec::new()),
            valid: Cell::new(true),
            connection_host: Cell::new(false),
            calls: RefCell::new(Vec::new()),
            revoke_at: Cell::new(0),
            change_at: Cell::new(0),
            missing: Cell::new(None),
            no_take: Cell::new(false),
            no_project:Cell::new(false),
            bad_type: Cell::new(None),
            values: RefCell::new(BTreeMap::from([
                ("D_POSITION", 1.),
                ("D_LENGTH", 4.),
                ("D_STARTOFFS", 0.),
                ("D_PLAYRATE", 0.5),
                ("B_PPITCH", 1.),
                ("I_CHANMODE", 2.),
                ("D_PITCH", 3.),
                ("C_BEATATTACHMODE", 1.),
                ("C_AUTOSTRETCH", 1.),
                ("B_MUTE", 0.),
                ("B_MUTE_ACTUAL", 0.),
                ("D_FADEINLEN", 0.2),
                ("D_FADEOUTLEN", 0.3),
                ("D_FADEINLEN_AUTO", 0.4),
                ("D_FADEOUTLEN_AUTO", 0.5),
                ("C_FADEINSHAPE", 1.),
                ("C_FADEOUTSHAPE", 6.),
                ("D_FADEINDIR", -0.1),
                ("D_FADEOUTDIR", 0.1),
                ("D_FADEINDIR_NEW", -0.2),
                ("D_FADEOUTDIR_NEW", 0.2),
                ("D_FADEINDIR2_NEW", -0.3),
                ("D_FADEOUTDIR2_NEW", 0.3),
            ])),
            markers: RefCell::new(vec![(-1., -2., 2.5), (6., 3., -2.5)]),
            count_override: Cell::new(None),
            bad_marker: Cell::new(false),
            bad_guid: Cell::new(false),
            change: Cell::new(-9),
            hook: RefCell::new(None),
        });
        ACTIVE.set((&*value as *const Self) as usize);
        value
    }
    pub fn context(&self) -> *mut c_void {
        (self as *const Self as *mut Self).cast()
    }
    fn project(&self) -> *mut c_void {
        self.context()
    }
    fn take(&self) -> *mut c_void {
        (&self.take_token as *const u8 as *mut u8).cast()
    }
    fn item(&self) -> *mut c_void {
        (&self.item_token as *const u8 as *mut u8).cast()
    }
    fn track(&self)->*mut c_void {(&self.track_token as *const u8 as *mut u8).cast()}
    pub fn enable_writer(&self) {self.writer_enabled.set(true);}
    pub fn enable_media(&self) {self.enable_writer();self.media_enabled.set(true);}
    pub fn clear_markers(&self) {self.markers.borrow_mut().clear();}
    pub fn client(&self) -> ReaperHost {
        unsafe { ReaperHost::from_context(self.context(), || self.valid.get()) }.unwrap()
    }
    pub fn reset(&self) {
        self.calls.borrow_mut().clear();
        self.revoke_at.set(0);
        self.change_at.set(0);
        self.valid.set(true);
    }
    pub fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }
    pub fn references(&self) -> u32 {
        self.refs.load(Ordering::Acquire)
    }
    pub fn enable_connection_host(&self) {
        self.connection_host.set(true);
    }
    pub fn set_value(&self, name: &'static str, value: f64) {
        self.values.borrow_mut().insert(name, value);
        self.change.set(self.change.get().wrapping_add(1));
    }
    fn record(&self, name: impl Into<String>) {
        assert!(
            self.valid.get(),
            "raw getter called after prior getter revoked authorization"
        );
        let name = name.into();
        self.calls.borrow_mut().push(name.clone());
        let index = self.calls.borrow().len();
        if self.revoke_at.get() == index {
            self.valid.set(false);
        }
        if self.change_at.get() == index {
            self.change.set(self.change.get().wrapping_add(1));
        }
        let matches = self
            .hook
            .borrow()
            .as_ref()
            .is_some_and(|(label, _)| *label == name);
        if matches {
            let (_, hook) = self.hook.borrow_mut().take().unwrap();
            hook();
        }
    }
    fn geometry(
        &self,
        client: &ReaperHost,
    ) -> Result<crate::host::geometry::HostClipGeometry, String> {
        client.geometry(|| self.valid.get())
    }
}
fn fixture() -> &'static Fixture {
    unsafe { &*(ACTIVE.get() as *const Fixture) }
}
unsafe extern "system" fn query(this: *mut c_void, iid: *const u8, out: *mut *mut c_void) -> i32 {
    let f = unsafe { &*this.cast::<Fixture>() };
    f.record("QI");
    let requested = unsafe { std::slice::from_raw_parts(iid, 16) };
    let host_iid = uid_guid([0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5]);
    if requested != uid_guid(IID) && !(f.connection_host.get() && requested == host_iid) {
        unsafe {
            *out = std::ptr::null_mut();
        }
        return crate::vst3::K_NO_INTERFACE;
    }
    unsafe {
        *out = this;
        add(this);
    }
    K_RESULT_OK
}
unsafe extern "system" fn add(this: *mut c_void) -> u32 {
    unsafe {
        (&*this.cast::<Fixture>())
            .refs
            .fetch_add(1, Ordering::AcqRel)
            + 1
    }
}
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    unsafe {
        (&*this.cast::<Fixture>())
            .refs
            .fetch_sub(1, Ordering::AcqRel)
            - 1
    }
}
unsafe extern "system" fn parent(this: *mut c_void, selector: u32) -> *mut c_void {
    let f = unsafe { &*this.cast::<Fixture>() };
    f.record(format!("parent:{selector}"));
    match selector {
        3 if !f.no_project.get() => f.project(),
        2 if !f.no_take.get() => f.take(),
        1 if f.media_enabled.get()=>f.track(),
        _ => std::ptr::null_mut(),
    }
}
unsafe extern "system" fn api(_: *mut c_void, name: *const c_char) -> *mut c_void {
    let f = fixture();
    let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap();
    f.record(format!("api:{name}"));
    if f.missing.get() == Some(name) {
        return std::ptr::null_mut();
    }
    let p = match name {
        "GetPlayPositionEx" | "GetCursorPositionEx" => position as *const (),
        "GetPlayStateEx" => state as *const (),
        "GetAppVersion" => version as *const (),
        "ValidatePtr2" => validate as *const (),
        "GetMediaItemTake_Item" => item as *const (),
        "GetMediaItemInfo_Value" => item_value as *const (),
        "GetMediaItemTakeInfo_Value" => take_value as *const (),
        "GetSetMediaItemInfo_String" => item_guid as *const (),
        "GetSetMediaItemTakeInfo_String" => take_guid as *const (),
        "GetTakeNumStretchMarkers" => count as *const (),
        "GetTakeStretchMarker" => marker as *const (),
        "GetTakeStretchMarkerSlope" => slope as *const (),
        "GetProjectStateChangeCount" => change as *const (),
        "SetMediaItemInfo_Value" if f.writer_enabled.get()=>set_item as *const (),
        "SetMediaItemTakeInfo_Value" if f.writer_enabled.get()=>set_take as *const (),
        "GetMediaItem_Track" if f.writer_enabled.get()=>item_track as *const (),
        "MoveMediaItemToTrack" if f.writer_enabled.get()=>move_item as *const (),
        "Undo_BeginBlock2" if f.writer_enabled.get()=>undo_begin as *const (),
        "Undo_EndBlock2" if f.writer_enabled.get()=>undo_end as *const (),
        "UpdateItemInProject" if f.writer_enabled.get()=>update_item as *const (),
        "UpdateArrange" if f.writer_enabled.get()=>update_arrange as *const (),
        "Undo_DoUndo2" if f.writer_enabled.get()=>do_undo as *const (),
        "Undo_DoRedo2" if f.writer_enabled.get()=>do_redo as *const (),
        "Undo_CanUndo2" if f.writer_enabled.get()=>can_undo as *const (),
        "Undo_CanRedo2" if f.writer_enabled.get()=>can_redo as *const (),
        "Undo_GetCurEntry" if f.writer_enabled.get()=>undo_current as *const (),
        "Undo_GetEntryDesc" if f.writer_enabled.get()=>undo_entry as *const (),
        "PCM_Source_CreateFromFileEx" if f.media_enabled.get()=>media_source as *const (),
        "PCM_Source_Destroy" if f.media_enabled.get()=>media_destroy as *const (),
        "GetMediaSourceLength" if f.media_enabled.get()=>media_length as *const (),
        "AddMediaItemToTrack" if f.media_enabled.get()=>media_item as *const (),
        "AddTakeToMediaItem" if f.media_enabled.get()=>media_take as *const (),
        "GetSetMediaItemTakeInfo" if f.media_enabled.get()=>media_info as *const (),
        "DeleteTrackMediaItem" if f.media_enabled.get()=>media_delete as *const (),
        "GetSetMediaTrackInfo_String" if f.media_enabled.get()=>media_track_guid as *const (),
        "GetUserFileNameForRead" if f.media_enabled.get()=>media_picker as *const (),
        "GetUserFileName" if f.media_enabled.get()=>multi_picker as *const (),
        "InsertTrackInProject" if f.media_enabled.get()=>insert_track as *const (),
        "CountTracks" if f.media_enabled.get()=>count_tracks as *const (),
        "GetTrack" if f.media_enabled.get()=>get_track as *const (),
        "TrackFX_AddByName" if f.media_enabled.get()=>add_fx as *const (),
        "DeleteTrack" if f.media_enabled.get()=>delete_track as *const (),
        "CountTrackMediaItems" if f.media_enabled.get()=>count_track_items as *const (),
        "TrackFX_GetCount" if f.media_enabled.get()=>fx_count as *const (),
        "TrackFX_GetFXGUID" if f.media_enabled.get()=>fx_guid as *const (),
        _ => std::ptr::null(),
    };
    p as *mut c_void
}
/// 夹具显式采用目标REAPER7.81的新轴，不用缺省字段猜宿主版本。
unsafe extern "C" fn version()->*const c_char {c"7.81/x64".as_ptr()}

/// B_MUTE才是item solo覆盖后的有效静音；原始mute仍开时solo也可让该clip播放。
#[test]
fn effective_item_mute_uses_host_solo_override_not_raw_mute() {
    let fixture=Fixture::new();let client=fixture.client();
    assert!(!fixture.geometry(&client).unwrap().muted);
    fixture.set_value("B_MUTE",1.);fixture.set_value("B_MUTE_ACTUAL",1.);
    assert!(fixture.geometry(&client).unwrap().muted);
    fixture.set_value("B_MUTE",0.);
    assert!(!fixture.geometry(&client).unwrap().muted,"item solo覆盖必须与宿主有效状态一致");
}
unsafe extern "system" fn extended(
    _: *mut c_void,
    _: u32,
    _: *mut c_void,
    _: *mut c_void,
    _: *mut c_void,
) -> *mut c_void {
    panic!("unverified extended opcode must not be queried")
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
unsafe extern "C" fn position(project: *mut c_void) -> f64 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("position");
    2.5
}
unsafe extern "C" fn state(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("state");
    1
}
unsafe extern "C" fn change(project: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(project, f.project());
    f.record("change");
    f.change.get()
}
unsafe extern "C" fn validate(
    project: *mut c_void,
    object: *mut c_void,
    kind: *const c_char,
) -> bool {
    let f = fixture();
    assert_eq!(project, f.project());
    let kind = unsafe { std::ffi::CStr::from_ptr(kind) }.to_str().unwrap();
    f.record(format!("validate:{kind}"));
    if f.bad_type.get() == Some(kind) {
        return false;
    }
    match kind {
        "ReaProject*" => object == f.project(),
        "MediaItem_Take*" => object == f.take(),
        "MediaItem*" => object == f.item(),
        "MediaTrack*"=>object==f.track()||f.extra_tracks.borrow().iter().any(|track|(&**track as *const u8) as usize==object as usize),
        _ => false,
    }
}
unsafe extern "C" fn item(take: *mut c_void) -> *mut c_void {
    let f = fixture();
    assert_eq!(take, f.take());
    f.record("item");
    f.item()
}
fn value(object: *mut c_void, name: *const c_char, is_take: bool) -> f64 {
    let f = fixture();
    assert_eq!(object, if is_take { f.take() } else { f.item() });
    let name = unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap();
    f.record(name);
    f.values.borrow()[name]
}
unsafe extern "C" fn item_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, false)
}
unsafe extern "C" fn take_value(object: *mut c_void, name: *const c_char) -> f64 {
    value(object, name, true)
}
/// 真实typed setter边界夹具：记录宿主写入，不通过插件timeline偷偷模拟成功。
unsafe extern "C" fn set_item(object:*mut c_void,name:*const c_char,value:f64)->bool {
    let f=fixture();assert_eq!(object,f.item());
    let name=unsafe {CStr::from_ptr(name)}.to_str().unwrap();f.record(format!("write-item:{name}"));
    let mut values=f.values.borrow_mut();let key=values.keys().find(|key|**key==name).copied();
    if let Some(key)=key {values.insert(key,value);}f.change.set(f.change.get().wrapping_add(1));true
}
unsafe extern "C" fn set_take(object:*mut c_void,name:*const c_char,value:f64)->bool {
    let f=fixture();assert_eq!(object,f.take());
    let name=unsafe {CStr::from_ptr(name)}.to_str().unwrap();f.record(format!("write-take:{name}"));
    let mut values=f.values.borrow_mut();let key=values.keys().find(|key|**key==name).copied();
    if let Some(key)=key {values.insert(key,value);}f.change.set(f.change.get().wrapping_add(1));true
}
unsafe extern "C" fn item_track(item:*mut c_void)->*mut c_void {let f=fixture();assert_eq!(item,f.item());f.record("item-track");f.track()}
unsafe extern "C" fn move_item(item:*mut c_void,track:*mut c_void)->bool {let f=fixture();assert_eq!(item,f.item());assert_eq!(track,f.track());f.record("move-item");true}
unsafe extern "C" fn undo_begin(project:*mut c_void) {let f=fixture();assert_eq!(project,f.project());f.record("undo-begin");}
unsafe extern "C" fn undo_end(project:*mut c_void,label:*const c_char,flags:i32) {
    let f=fixture();assert_eq!(project,f.project());assert_eq!(flags,-1);f.record("undo-end");
    let mut records=f.undo_records.borrow_mut();records.truncate(f.undo_position.get() as usize+1);
    records.push(unsafe {CStr::from_ptr(label)}.to_owned());f.undo_position.set(records.len() as i32-1);f.change.set(f.change.get().wrapping_add(1));
}
unsafe extern "C" fn undo_current(project:*mut c_void)->i32 {let f=fixture();assert_eq!(project,f.project());f.record("undo-current");f.undo_position.get()}
unsafe extern "C" fn undo_entry(project:*mut c_void,index:i32)->*const c_char {let f=fixture();assert_eq!(project,f.project());f.record(format!("undo-entry:{index}"));if index<0 {return std::ptr::null();}f.undo_records.borrow().get(index as usize).map_or(std::ptr::null(),|s|s.as_ptr())}
unsafe extern "C" fn can_undo(project:*mut c_void)->*const c_char {let f=fixture();assert_eq!(project,f.project());if f.undo_position.get()>0 {unsafe {undo_entry(project,f.undo_position.get())}} else {std::ptr::null()}}
unsafe extern "C" fn can_redo(project:*mut c_void)->*const c_char {let f=fixture();assert_eq!(project,f.project());unsafe {undo_entry(project,f.undo_position.get()+1)}}
unsafe extern "C" fn do_undo(project:*mut c_void)->i32 {let f=fixture();assert_eq!(project,f.project());f.record("undo-action");if f.undo_position.get()>0 {f.undo_position.set(f.undo_position.get()-1);f.change.set(f.change.get().wrapping_add(1));1} else {0}}
unsafe extern "C" fn do_redo(project:*mut c_void)->i32 {let f=fixture();assert_eq!(project,f.project());f.record("redo-action");if f.undo_position.get()+1<f.undo_records.borrow().len() as i32 {f.undo_position.set(f.undo_position.get()+1);f.change.set(f.change.get().wrapping_add(1));1} else {0}}
unsafe extern "C" fn media_source(path:*const c_char,force:bool)->*mut c_void {
    let f=fixture();assert!(force);f.record("media-source-create");*f.media_path.borrow_mut()=unsafe {CStr::from_ptr(path)}.to_str().unwrap().into();
    let mut source=Box::new(7_u8);let pointer=(&mut *source as *mut u8).cast();f.media_sources.borrow_mut().push(source);pointer
}
unsafe extern "C" fn media_destroy(pointer:*mut c_void) {let f=fixture();f.record("media-source-destroy");let mut sources=f.media_sources.borrow_mut();
    let index=sources.iter().position(|source|(&**source as *const u8) as usize==pointer as usize).expect("source must be owned and destroyed once");sources.remove(index);}
unsafe extern "C" fn media_length(_source:*mut c_void,qn:*mut bool)->f64 {fixture().record("media-length");unsafe {*qn=false};2.}
unsafe extern "C" fn media_item(track:*mut c_void)->*mut c_void {let f=fixture();assert_eq!(track,f.track());f.record("media-item-create");f.item()}
unsafe extern "C" fn media_take(item:*mut c_void)->*mut c_void {let f=fixture();assert_eq!(item,f.item());f.record("media-take-create");if f.fail_media_take.get() {std::ptr::null_mut()} else {f.take()}}
unsafe extern "C" fn media_info(take:*mut c_void,name:*const c_char,new:*mut c_void)->*mut c_void {let f=fixture();assert_eq!(take,f.take());assert_eq!(unsafe {CStr::from_ptr(name)},c"P_SOURCE");
    if new.is_null() {f.record("media-source-read");f.take_source.get() as *mut c_void} else {f.record("media-source-attach");f.take_source.replace(new as usize) as *mut c_void}}
unsafe extern "C" fn media_delete(track:*mut c_void,item:*mut c_void)->bool {let f=fixture();assert_eq!(track,f.track());assert_eq!(item,f.item());f.record("media-item-delete");let source=f.take_source.replace(0);if source!=0 {unsafe {media_destroy(source as *mut c_void)};}true}
unsafe extern "C" fn media_track_guid(track:*mut c_void,name:*const c_char,buffer:*mut c_char,set:bool)->bool {
    let f=fixture();if set {assert_eq!(unsafe {CStr::from_ptr(name)},c"P_NAME");f.record("track-name");return true;}
    assert_eq!(unsafe {CStr::from_ptr(name)},c"GUID");f.record("track-guid");
    let guid=if track==f.track() {std::ffi::CString::new("{33333333-3333-3333-3333-333333333333}").unwrap()} else {
        let tracks=f.extra_tracks.borrow();let index=tracks.iter().position(|t|(&**t as *const u8) as usize==track as usize).unwrap();
        std::ffi::CString::new(format!("{{44444444-4444-4444-4444-{:012}}}",index+1)).unwrap()
    };unsafe {std::ptr::copy_nonoverlapping(guid.as_ptr(),buffer,guid.to_bytes_with_nul().len());}true
}
unsafe extern "C" fn media_picker(_path:*mut c_char,_title:*const c_char,_ext:*const c_char)->bool {fixture().record("media-picker");false}
unsafe extern "C" fn multi_picker(mode:i32,_caption:*const c_char,_initial:*const c_char,_extensions:*const c_char,buffer:*mut c_char,size:i32)->bool {
    let f=fixture();assert!((1..=2).contains(&mode));f.record("media-multi-picker");let paths=f.picker_paths.borrow();if paths.is_empty() {return false;}
    let value=std::ffi::CString::new(paths.join("|")).unwrap();assert!(value.to_bytes_with_nul().len()<size as usize);
    unsafe {std::ptr::copy_nonoverlapping(value.as_ptr(),buffer,value.to_bytes_with_nul().len());}true
}
unsafe extern "C" fn count_tracks(project:*mut c_void)->i32 {let f=fixture();assert_eq!(project,f.project());f.record("count-tracks");1+f.extra_tracks.borrow().len() as i32}
unsafe extern "C" fn get_track(project:*mut c_void,index:i32)->*mut c_void {let f=fixture();assert_eq!(project,f.project());f.record(format!("get-track:{index}"));if index==0 {f.track()} else {f.extra_tracks.borrow().get(index as usize-1).map_or(std::ptr::null_mut(),|track|(&**track as *const u8 as *mut u8).cast())}}
unsafe extern "C" fn insert_track(project:*mut c_void,index:i32,flags:i32) {let f=fixture();assert_eq!(project,f.project());assert_eq!(index,1+f.extra_tracks.borrow().len() as i32);assert_eq!(flags,0);f.record("insert-track");f.extra_tracks.borrow_mut().push(Box::new(9));f.change.set(f.change.get().wrapping_add(1));}
unsafe extern "C" fn add_fx(track:*mut c_void,name:*const c_char,record:bool,index:i32)->i32 {let f=fixture();assert_eq!(unsafe {CStr::from_ptr(name)},c"VST3:HiFiShifter");assert!(!record);assert_eq!(index,-1000);f.record("add-hfs-first-fx");f.new_fx.borrow_mut().insert(track as usize);0}
unsafe extern "C" fn delete_track(track:*mut c_void) {let f=fixture();f.record("delete-created-track");let mut tracks=f.extra_tracks.borrow_mut();let index=tracks.iter().position(|t|(&**t as *const u8) as usize==track as usize).unwrap();tracks.remove(index);f.new_fx.borrow_mut().remove(&(track as usize));f.change.set(f.change.get().wrapping_add(1));}
unsafe extern "C" fn count_track_items(_track:*mut c_void)->i32 {0}
unsafe extern "C" fn fx_count(track:*mut c_void)->i32 {i32::from(fixture().new_fx.borrow().contains(&(track as usize)))}
unsafe extern "C" fn fx_guid(track:*mut c_void,index:i32)->*const u8 {let f=fixture();assert_eq!(index,0);if f.new_fx.borrow().contains(&(track as usize)) {f.fx_guid_bytes.as_ptr()} else {std::ptr::null()}}

#[test]
fn host_media_new_track_uses_project_guid_difference_and_first_plugin_and_rolls_back_empty_failure() {
    let f=Fixture::new();f.enable_media();let host=std::sync::Arc::new(f.client());
    {let _track=host.create_audio_track("元音",&||true).unwrap();assert_eq!(f.extra_tracks.borrow().len(),1);assert!(f.calls().contains(&"add-hfs-first-fx".into()));}
    assert!(f.extra_tracks.borrow().is_empty());assert!(f.calls().contains(&"delete-created-track".into()));
    let mut track=host.create_audio_track("保留",&||true).unwrap();track.commit();drop(track);assert_eq!(f.extra_tracks.borrow().len(),1);
}
#[test]
fn host_media_multi_picker_preserves_complete_unicode_paths_and_cancel_is_empty() {
    let f=Fixture::new();f.enable_media();let host=std::sync::Arc::new(f.client());
    assert!(host.pick_audio_paths(true,&||true).unwrap().is_empty());
    let paths=vec![r"E:\声音\第一段.wav".to_owned(),r"E:\声音\第二段.wav".to_owned()];*f.picker_paths.borrow_mut()=paths.clone();
    assert_eq!(host.pick_audio_paths(true,&||true).unwrap(),paths);
}

/// 真实typed媒体对象所有权合同；不是假造ARA PCM或实际REAPER导入验收。
#[test]
fn host_media_import_preserves_utf8_and_transfers_source_ownership() {
    let f=Fixture::new();f.enable_media();let host=std::sync::Arc::new(f.client());let target=host.direct_track_target(&||true).unwrap();
    let path=r"E:\素材\辅音和元音.wav";let guid=target.import_audio(path,3.,&||true).unwrap();
    assert_eq!(guid,"{11111111-1111-1111-1111-111111111111}");assert_eq!(f.media_path.borrow().as_str(),path);
    assert_eq!(f.values.borrow()["D_POSITION"],3.);assert_eq!(f.values.borrow()["D_LENGTH"],2.);
    assert_eq!(f.media_sources.borrow().len(),1);assert!(!f.calls().contains(&"media-source-destroy".into()));
    unsafe {media_delete(f.track(),f.item())};assert!(f.media_sources.borrow().is_empty());
}
#[test]
fn host_media_import_failure_removes_only_created_item_and_frees_unattached_source() {
    let f=Fixture::new();f.enable_media();f.fail_media_take.set(true);let host=std::sync::Arc::new(f.client());let target=host.direct_track_target(&||true).unwrap();
    assert!(target.import_audio(r"E:\clip.wav",0.,&||true).unwrap_err().contains("take creation"));
    assert!(f.calls().contains(&"media-item-delete".into()));assert_eq!(f.calls().iter().filter(|s|s.as_str()=="media-source-destroy").count(),1);
    assert!(f.media_sources.borrow().is_empty());
}
unsafe extern "C" fn update_item(item:*mut c_void) {let f=fixture();assert_eq!(item,f.item());f.record("update-item");}
unsafe extern "C" fn update_arrange() {fixture().record("update-arrange");}

/// 已捕获地址被宿主重用后，GUID核对失败必须发生在任何setter之前。
#[test]
fn host_edit_reused_item_identity_never_reaches_setter() {
    let fixture=Fixture::new();fixture.enable_writer();let client=std::sync::Arc::new(fixture.client());
    let target=client.clip_target(||true).unwrap();fixture.reset();fixture.bad_guid.set(true);
    assert!(target.set_item(c"D_POSITION",3.,&||true).unwrap_err().contains("identity"));
    assert!(!fixture.calls().iter().any(|name|name.starts_with("write-")));
}

/// 第一笔写入可触发关闭/路由撤销；后续setter停止，Undo块仍在原project收尾。
#[test]
fn host_edit_reentry_revokes_later_writes_and_finishes_undo() {
    let fixture=Fixture::new();fixture.enable_writer();let client=std::sync::Arc::new(fixture.client());
    let live=std::rc::Rc::new(Cell::new(true));let target=client.clip_target(||live.get()).unwrap();
    let block=target.begin_undo(&||live.get()).unwrap();fixture.reset();let flag=live.clone();
    *fixture.hook.borrow_mut()=Some(("write-item:D_POSITION".into(),Box::new(move||flag.set(false))));
    assert!(target.set_item(c"D_POSITION",3.,&||live.get()).is_err());
    assert!(target.set_take(c"D_PLAYRATE",2.,&||live.get()).is_err());drop(block);
    let calls=fixture.calls();assert_eq!(calls.iter().filter(|name|name.starts_with("write-")).count(),1);
    assert!(calls.contains(&"undo-end".into()));assert!(!calls.contains(&"write-take:D_PLAYRATE".into()));
}
fn guid(
    object: *mut c_void,
    name: *const c_char,
    buffer: *mut c_char,
    write: bool,
    is_take: bool,
) -> bool {
    let f = fixture();
    assert_eq!(object, if is_take { f.take() } else { f.item() });
    assert!(!write, "GUID must remain read-only");
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }, c"GUID");
    f.record(if is_take { "take_guid" } else { "item_guid" });
    let text = if f.bad_guid.get() {
        c"bad"
    } else if is_take {
        c"{22222222-2222-2222-2222-222222222222}"
    } else {
        c"{11111111-1111-1111-1111-111111111111}"
    };
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, text.to_bytes_with_nul().len());
    }
    true
}
unsafe extern "C" fn item_guid(o: *mut c_void, n: *const c_char, b: *mut c_char, w: bool) -> bool {
    guid(o, n, b, w, false)
}
unsafe extern "C" fn take_guid(o: *mut c_void, n: *const c_char, b: *mut c_char, w: bool) -> bool {
    guid(o, n, b, w, true)
}
unsafe extern "C" fn count(take: *mut c_void) -> i32 {
    let f = fixture();
    assert_eq!(take, f.take());
    f.record("count");
    f.count_override
        .get()
        .unwrap_or(f.markers.borrow().len() as i32)
}
unsafe extern "C" fn marker(take: *mut c_void, index: i32, pos: *mut f64, src: *mut f64) -> i32 {
    let f = fixture();
    assert_eq!(take, f.take());
    f.record(format!("marker:{index}"));
    if f.bad_marker.get() {
        return -1;
    }
    let (p, s, _) = f.markers.borrow()[index as usize];
    unsafe {
        *pos = p;
        *src = s;
    }
    index
}
unsafe extern "C" fn slope(take: *mut c_void, index: i32) -> f64 {
    let f = fixture();
    assert_eq!(take, f.take());
    f.record(format!("slope:{index}"));
    f.markers.borrow()[index as usize].2
}

#[test]
fn initialization_without_project_retains_interface_and_binds_only_the_later_direct_parent() {
    let f=Fixture::new();f.no_project.set(true);let client=f.client();assert_eq!(f.references(),2);
    f.reset();assert!(client.sample(||f.valid.get()).unwrap_err().contains("not attached"));
    assert_eq!(f.calls(),["parent:3"],"不能把null传给API而暗中使用当前活动project");
    f.no_project.set(false);f.reset();assert_eq!(client.sample(||f.valid.get()).unwrap(),(2.5,true));
    assert!(f.calls().contains(&"parent:3".into()));
    f.reset();client.sample(||f.valid.get()).unwrap();assert!(!f.calls().contains(&"parent:3".into()),"绑定后不跟随活动tab重新选项目");
    drop(client);assert_eq!(f.references(),1);
}
#[test]
fn deferred_project_query_reentry_cannot_bind_or_call_the_next_api() {
    let f=Fixture::new();f.no_project.set(true);let client=f.client();f.no_project.set(false);f.reset();f.revoke_at.set(1);
    assert!(client.sample(||f.valid.get()).is_err());assert_eq!(f.calls(),["parent:3"]);
    assert_eq!(client.project.load(Ordering::Acquire),0,"外部调用撤销许可后不保存刚返回的parent");
    drop(client);assert_eq!(f.references(),1);
}
#[test]
fn task38a_native_geometry_preserves_direct_identity_new_fades_and_outside_raw_markers() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    let g = f.geometry(&client).unwrap();
    assert_eq!(
        (
            g.start_sec,
            g.duration_sec,
            g.source_start_sec,
            g.playback_rate
        ),
        (1., 4., 0., 0.5)
    );
    assert_eq!(
        (
            g.preserve_pitch,
            g.channel_mode,
            g.take_pitch,
            g.item_timebase,
            g.auto_stretch
        ),
        (true, 2, 3., 1, true)
    );
    assert_eq!(g.item_id, "{11111111-1111-1111-1111-111111111111}");
    assert_eq!(g.take_id, "{22222222-2222-2222-2222-222222222222}");
    assert_eq!(
        (
            g.fade_in_sec,
            g.fade_out_sec,
            g.auto_fade_in_sec,
            g.auto_fade_out_sec
        ),
        (0.2, 0.3, 0.4, 0.5)
    );
    assert_eq!(
        (
            g.fade_in_shape,
            g.fade_out_shape,
            g.fade_in_dir,
            g.fade_out_dir
        ),
        (1., 6., -0.1, 0.1)
    );
    assert_eq!(
        (
            g.fade_in_dir_new,
            g.fade_out_dir_new,
            g.fade_in_dir2_new,
            g.fade_out_dir2_new
        ),
        (-0.2, 0.2, -0.3, 0.3)
    );
    assert_eq!(
        g.markers
            .iter()
            .map(|m| (m.item_position_raw, m.source_position_raw, m.slope_raw))
            .collect::<Vec<_>>(),
        [(-1., -2., 2.5), (6., 3., -2.5)]
    );
    assert!(f.calls.borrow().contains(&"parent:2".into()));
    assert_eq!(f.refs.load(Ordering::Acquire), 2);
    drop(client);
    assert_eq!(f.refs.load(Ordering::Acquire), 1);
}

#[test]
fn task38a_missing_geometry_apis_and_take_never_disable_transport() {
    for name in [
        "ValidatePtr2",
        "GetMediaItemTake_Item",
        "GetMediaItemInfo_Value",
        "GetMediaItemTakeInfo_Value",
        "GetSetMediaItemInfo_String",
        "GetSetMediaItemTakeInfo_String",
        "GetTakeNumStretchMarkers",
        "GetTakeStretchMarker",
        "GetTakeStretchMarkerSlope",
        "GetProjectStateChangeCount",
    ] {
        let f = Fixture::new();
        f.missing.set(Some(name));
        let client = f.client();
        f.reset();
        assert!(f.geometry(&client).unwrap_err().contains("API unavailable"));
        assert_eq!(client.sample(|| f.valid.get()).unwrap(), (2.5, true));
    }
    let f = Fixture::new();
    f.no_take.set(true);
    let client = f.client();
    assert!(f.geometry(&client).unwrap_err().contains("parent take"));
    assert!(client.sample(|| f.valid.get()).is_ok());
    f.no_take.set(false);
    for kind in ["ReaProject*", "MediaItem_Take*", "MediaItem*"] {
        f.bad_type.set(Some(kind));
        assert!(f.geometry(&client).is_err());
    }
    drop(client);
    let f = Fixture::new();
    f.missing.set(Some("GetPlayStateEx"));
    let client = f.client();
    assert!(client.sample(|| f.valid.get()).is_err());
    assert!(f.geometry(&client).is_ok());
}

#[test]
fn task38a_bad_fields_guid_and_marker_count_are_bounded_and_explicit() {
    for (name, value) in [
        ("D_POSITION", f64::NAN),
        ("D_LENGTH", 0.),
        ("D_PLAYRATE", 0.),
        ("B_PPITCH", 2.),
        ("I_CHANMODE", 1.5),
        ("D_FADEINLEN", -1.),
    ] {
        let f = Fixture::new();
        f.values.borrow_mut().insert(name, value);
        let client = f.client();
        assert!(f.geometry(&client).is_err(), "{name}");
    }
    let f = Fixture::new();
    let client = f.client();
    f.bad_guid.set(true);
    assert!(f.geometry(&client).is_err());
    f.bad_guid.set(false);
    for count in [-1, MAX_MARKERS + 1, i32::MAX] {
        f.count_override.set(Some(count));
        f.reset();
        assert!(f.geometry(&client).is_err());
        assert!(!f.calls.borrow().iter().any(|s| s.starts_with("marker:")));
    }
    f.count_override.set(None);
    f.bad_marker.set(true);
    assert!(f.geometry(&client).is_err());
    f.bad_marker.set(false);
    f.markers.borrow_mut()[0].2 = f64::INFINITY;
    assert!(f.geometry(&client).is_err());
    *f.markers.borrow_mut() = vec![(f64::NAN, 0., 0.)];
    assert!(f.geometry(&client).is_err());
    *f.markers.borrow_mut() = vec![(0., f64::NAN, 0.)];
    assert!(f.geometry(&client).is_err());
    f.markers.borrow_mut().clear();
    assert!(f.geometry(&client).unwrap().markers.is_empty());
    *f.markers.borrow_mut() = vec![(0., 0., 0.); MAX_MARKERS as usize];
    assert_eq!(
        f.geometry(&client).unwrap().markers.len(),
        MAX_MARKERS as usize
    );
}

#[test]
fn task38a_each_raw_geometry_getter_reentry_stops_before_the_next_getter() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    f.geometry(&client).unwrap();
    let count = f.calls.borrow().len();
    for index in 1..=count {
        f.reset();
        f.revoke_at.set(index);
        assert!(f.geometry(&client).is_err(), "getter {index}");
        assert_eq!(f.calls.borrow().len(), index);
    }
    f.reset();
    f.revoke_at.set(1);
    assert!(client.sample(|| f.valid.get()).is_err());
    assert_eq!(&*f.calls.borrow(), &["validate:ReaProject*"]);
}

#[test]
fn task38a_project_change_integer_discards_whole_snapshot_without_retry() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    f.geometry(&client).unwrap();
    let baseline = f.calls.borrow().clone();
    let index = baseline.iter().position(|v| v == "D_LENGTH").unwrap() + 1;
    f.reset();
    f.change_at.set(index);
    assert!(f.geometry(&client).unwrap_err().contains("project changed"));
    assert_eq!(*f.calls.borrow(), baseline);
    f.reset();
    f.change.set(i32::MIN);
    assert!(
        f.geometry(&client).is_ok(),
        "integer is compared for equality, not monotonicity/sign"
    );
}

#[test]
fn task38a_native_take_invalidation_is_checked_before_another_take_getter() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    *f.hook.borrow_mut() = Some((
        "item_guid".into(),
        Box::new(|| {
            fixture().bad_type.set(Some("MediaItem_Take*"));
        }),
    ));
    assert!(f.geometry(&client).is_err());
    assert!(
        !f.calls.borrow().contains(&"take_guid".into()),
        "owning extension reference does not keep a deleted take alive"
    );
}

#[test]
fn task38a_query_initialization_reentry_balances_qi_and_stops_lookup() {
    let f = Fixture::new();
    let client = f.client();
    let count = f.calls.borrow().len();
    drop(client);
    for index in 1..=count {
        f.reset();
        f.revoke_at.set(index);
        assert!(unsafe { ReaperHost::from_context(f.context(), || f.valid.get()) }.is_none());
        assert_eq!(f.calls.borrow().len(), index);
        assert_eq!(f.refs.load(Ordering::Acquire), 1);
    }
}

#[test]
fn task38a_geometry_and_transport_reject_worker_threads_without_host_calls() {
    let f = Fixture::new();
    let client = f.client();
    f.reset();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(client.geometry(|| true).is_err());
                assert!(client.sample(|| true).is_err());
            })
            .join()
            .unwrap();
    });
    assert!(f.calls.borrow().is_empty());
    f.valid.set(false);
    assert!(f.geometry(&client).is_err());
    assert!(f.calls.borrow().is_empty());
}
