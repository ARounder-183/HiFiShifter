//! 官方REAPER媒体创建接口：宿主创建item/take/source，PCM与region仍等真实ARA回流授权。
use super::{ReaperHost,Guid,checked,HostVtbl};
use std::ffi::{c_char,c_void,CStr,CString};
use std::sync::Arc;

pub(super) type CreateSource=unsafe extern "C" fn(*const c_char,bool)->*mut c_void;
pub(super) type DestroySource=unsafe extern "C" fn(*mut c_void);
pub(super) type SourceLength=unsafe extern "C" fn(*mut c_void,*mut bool)->f64;
pub(super) type CreateItem=unsafe extern "C" fn(*mut c_void)->*mut c_void;
pub(super) type CreateTake=unsafe extern "C" fn(*mut c_void)->*mut c_void;
pub(super) type TakeInfo=unsafe extern "C" fn(*mut c_void,*const c_char,*mut c_void)->*mut c_void;
pub(super) type DeleteItem=unsafe extern "C" fn(*mut c_void,*mut c_void)->bool;
pub(super) type FilePicker=unsafe extern "C" fn(*mut c_char,*const c_char,*const c_char)->bool;
pub(super) struct MediaApi {pub create_source:CreateSource,pub destroy_source:DestroySource,pub length:SourceLength,
    pub create_item:CreateItem,pub create_take:CreateTake,pub take_info:TakeInfo,pub delete_item:DeleteItem,
    pub track_guid:Guid,pub picker:FilePicker}

/// 目标轨道由直接parent或已绑定item的真实parent证明，不接受JS地址或活动工程编号。
#[derive(Clone)]
pub(crate) struct HostTrackTarget {host:Arc<ReaperHost>,project:usize,track:usize,guid:String}
/// Source_Create返回的未附着源由调用者拥有；附着后立即撤销此析构责任。
struct OwnedSource {pointer:*mut c_void,destroy:DestroySource}
impl Drop for OwnedSource {fn drop(&mut self) {if !self.pointer.is_null() {unsafe {(self.destroy)(self.pointer)};}}}
/// 只回滚自己新建且GUID仍相同的item；地址重用或项目销毁时不误删用户数据。
struct CreatedItem {target:HostTrackTarget,item:usize,guid:String,committed:bool}
impl CreatedItem {
    fn verify(&self,authorized:&impl Fn()->bool)->Result<(),String> {
        self.target.verify(authorized)?;let api=self.target.host.geometry.as_ref().unwrap();
        if !checked(authorized,||unsafe {(api.validate)(self.target.project as *mut c_void,self.item as *mut c_void,c"MediaItem*".as_ptr())})?
            ||read_guid(api.item_guid,self.item as *mut c_void,authorized)?!=self.guid
            ||checked(authorized,||unsafe {(self.target.host.write.as_ref().unwrap().item_track)(self.item as *mut c_void)})? as usize!=self.target.track {
            return Err("created item identity/parent changed during import".into());
        }Ok(())
    }
}
impl Drop for CreatedItem {
    fn drop(&mut self) {
        if self.committed||std::thread::current().id()!=self.target.host.thread {return;}
        let Some(media)=&self.target.host.media else {return;};let Some(api)=&self.target.host.geometry else {return;};
        let project=self.target.project as *mut c_void;let item=self.item as *mut c_void;
        if self.verify(&||true).is_ok()&&unsafe {(api.validate)(project,item,c"MediaItem*".as_ptr())} {
            unsafe {(media.delete_item)(self.target.track as *mut c_void,item)};
        }
    }
}

fn read_guid(getter:Guid,object:*mut c_void,authorized:&impl Fn()->bool)->Result<String,String> {
    let mut bytes=[0_u8;64];
    if !checked(authorized,||unsafe {getter(object,c"GUID".as_ptr(),bytes.as_mut_ptr().cast(),false)})?
        ||bytes[0]!=b'{'||bytes[37]!=b'}'||bytes[38]!=0 {return Err("invalid host import GUID".into());}
    if !(1..37).all(|i|if [9,14,19,24].contains(&i) {bytes[i]==b'-'} else {bytes[i].is_ascii_hexdigit()}) {return Err("invalid host import GUID".into());}
    String::from_utf8(bytes[..38].to_vec()).map_err(|_|"invalid GUID encoding".into())
}

impl ReaperHost {
    pub(crate) fn can_import_audio(&self)->bool {self.media.is_some()&&self.has_project_history()}
    /// 原生文件选择器仅打开媒体文件；取消不创建item或Undo块，UTF-8路径不经过ANSI转换。
    pub(crate) fn pick_audio(&self,authorized:&impl Fn()->bool)->Result<Option<String>,String> {
        self.project(authorized)?;let media=self.media.as_ref().ok_or("REAPER media API unavailable")?;
        let mut bytes=vec![0_u8;4096];
        if !checked(authorized,||unsafe {(media.picker)(bytes.as_mut_ptr().cast(),c"Import audio into HiFiShifter".as_ptr(),c"wav".as_ptr())})? {return Ok(None);}
        let end=bytes.iter().position(|b|*b==0).ok_or("host file picker path exceeds buffer")?;
        String::from_utf8(bytes[..end].to_vec()).map(Some).map_err(|_|"host file path is not UTF-8".into())
    }
    /// 空ARA轨道的首次导入也沿直接parent(1)，不伪造track_main或根据轨名找目标。
    pub(crate) fn direct_track_target(self:&Arc<Self>,authorized:&impl Fn()->bool)->Result<HostTrackTarget,String> {
        let project=self.project(authorized)?;let pointer=self._interface.0 as *mut c_void;
        let table=unsafe {&**pointer.cast::<*const HostVtbl>()};
        let track=checked(authorized,||unsafe {(table.parent)(pointer,1)})?;
        self.track_target(project as usize,track as usize,authorized)
    }
    pub(super) fn track_target(self:&Arc<Self>,project:usize,track:usize,authorized:&impl Fn()->bool)->Result<HostTrackTarget,String> {
        if !self.can_import_audio() {return Err("REAPER audio import API unavailable".into());}
        let api=self.geometry.as_ref().unwrap();
        if project==0||track==0||!checked(authorized,||unsafe {(api.validate)(project as *mut c_void,track as *mut c_void,c"MediaTrack*".as_ptr())})? {return Err("invalid import target track".into());}
        let guid=read_guid(self.media.as_ref().unwrap().track_guid,track as *mut c_void,authorized)?;
        let target=HostTrackTarget {host:self.clone(),project,track,guid};target.verify(authorized)?;Ok(target)
    }
}
impl HostTrackTarget {
    fn verify(&self,authorized:&impl Fn()->bool)->Result<(),String> {
        if std::thread::current().id()!=self.host.thread {return Err("host audio import off UI thread".into());}
        let api=self.host.geometry.as_ref().ok_or("host geometry API missing")?;
        for (pointer,kind) in [(self.project,c"ReaProject*"),(self.track,c"MediaTrack*")] {
            if !checked(authorized,||unsafe {(api.validate)(self.project as *mut c_void,pointer as *mut c_void,kind.as_ptr())})? {return Err("import project/track no longer valid".into());}
        }
        if read_guid(self.host.media.as_ref().unwrap().track_guid,self.track as *mut c_void,authorized)?!=self.guid {return Err("import track identity changed".into());}Ok(())
    }
    /// 新source尚未被任何项目引用；用GetSetMediaItemTakeInfo转移所有权并按官方约定销毁旧源。
    pub(crate) fn import_audio(&self,path:&str,start_sec:f64,authorized:&impl Fn()->bool)->Result<String,String> {
        self.verify(authorized)?;
        if !std::path::Path::new(path).is_absolute()||path.len()>32767||!start_sec.is_finite()||start_sec<0. {return Err("absolute audio path and nonnegative finite start required".into());}
        let path=CString::new(path).map_err(|_|"audio path contains NUL")?;let media=self.host.media.as_ref().unwrap();let api=self.host.geometry.as_ref().unwrap();let write=self.host.write.as_ref().unwrap();
        if !authorized() {return Err("import authorization revoked".into());}
        let pointer=unsafe {(media.create_source)(path.as_ptr(),true)};
        if pointer.is_null() {return Err("REAPER could not open audio source".into());}
        let mut source=OwnedSource {pointer,destroy:media.destroy_source};self.verify(authorized)?;
        let mut quarter_notes=false;let length=checked(authorized,||unsafe {(media.length)(source.pointer,&mut quarter_notes)})?;
        if quarter_notes||!length.is_finite()||length<=0. {return Err("source must be nonempty time-based audio, not MIDI".into());}
        let item=checked(authorized,||unsafe {(media.create_item)(self.track as *mut c_void)})?;
        if item.is_null() {return Err("host item creation failed".into());}
        let guid=read_guid(api.item_guid,item,authorized)?;
        let mut created=CreatedItem {target:self.clone(),item:item as usize,guid:guid.clone(),committed:false};
        created.verify(authorized)?;let take=checked(authorized,||unsafe {(media.create_take)(item)})?;
        if take.is_null() {return Err("host take creation failed".into());}
        let take_guid=read_guid(api.take_guid,take,authorized)?;
        let verify=||->Result<(),String>{
            created.verify(authorized)?;
            if !checked(authorized,||unsafe {(api.validate)(self.project as *mut c_void,take,c"MediaItem_Take*".as_ptr())})?
                ||checked(authorized,||unsafe {(api.item)(take)})?!=item||read_guid(api.take_guid,take,authorized)?!=take_guid {
                return Err("created take identity/parent changed during import".into());
            }Ok(())
        };
        for (name,value) in [(c"D_POSITION",start_sec),(c"D_LENGTH",length),(c"B_LOOPSRC",0.)] {
            verify()?;
            if !checked(authorized,||unsafe {(write.set_item)(item,name.as_ptr(),value)})? {return Err("host rejected imported item geometry".into());}
        }
        let old=checked(authorized,||unsafe {(media.take_info)(take,c"P_SOURCE".as_ptr(),std::ptr::null_mut())})?;
        verify()?;
        // 此setter无bool失败返回；调用发生即把新源交给宿主，随后租约失效也不能双重释放。
        unsafe {(media.take_info)(take,c"P_SOURCE".as_ptr(),source.pointer)};source.pointer=std::ptr::null_mut();
        if !old.is_null() {unsafe {(media.destroy_source)(old)};}
        verify()?;
        checked(authorized,||unsafe {(write.update)(item)})?;
        checked(authorized,||unsafe {(write.arrange)()})?;
        created.committed=true;Ok(guid)
    }
}
