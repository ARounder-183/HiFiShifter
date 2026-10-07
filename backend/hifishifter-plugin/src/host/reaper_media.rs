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
pub(super) type MultiPicker=unsafe extern "C" fn(i32,*const c_char,*const c_char,*const c_char,*mut c_char,i32)->bool;
pub(super) type InsertTrack=unsafe extern "C" fn(*mut c_void,i32,i32);
pub(super) type CountTracks=unsafe extern "C" fn(*mut c_void)->i32;
pub(super) type GetTrack=unsafe extern "C" fn(*mut c_void,i32)->*mut c_void;
pub(super) type AddFx=unsafe extern "C" fn(*mut c_void,*const c_char,bool,i32)->i32;
pub(super) type DeleteTrack=unsafe extern "C" fn(*mut c_void);
pub(super) type TrackCount=unsafe extern "C" fn(*mut c_void)->i32;
pub(super) type FxGuid=unsafe extern "C" fn(*mut c_void,i32)->*const u8;
pub(super) struct NewTrackApi {pub insert:InsertTrack,pub count:CountTracks,pub get:GetTrack,pub add_fx:AddFx,pub delete:DeleteTrack,pub item_count:TrackCount,pub fx_count:TrackCount,pub fx_guid:FxGuid}
pub(super) struct MediaApi {pub create_source:CreateSource,pub destroy_source:DestroySource,pub length:SourceLength,
    pub create_item:CreateItem,pub create_take:CreateTake,pub take_info:TakeInfo,pub delete_item:DeleteItem,
    pub track_guid:Guid,pub picker:FilePicker}
pub(super) struct ExtendedMedia {pub picker:Option<MultiPicker>,pub tracks:Option<NewTrackApi>}

/// 目标轨道由直接parent或已绑定item的真实parent证明，不接受JS地址或活动工程编号。
#[derive(Clone)]
pub(crate) struct HostTrackTarget {host:Arc<ReaperHost>,project:usize,track:usize,guid:String}
/// Source_Create返回的未附着源由调用者拥有；附着后立即撤销此析构责任。
struct OwnedSource {pointer:*mut c_void,destroy:DestroySource}
impl Drop for OwnedSource {fn drop(&mut self) {if !self.pointer.is_null() {unsafe {(self.destroy)(self.pointer)};}}}
/// 只回滚自己新建且GUID仍相同的item；地址重用或项目销毁时不误删用户数据。
struct CreatedItem {target:HostTrackTarget,item:usize,guid:String,state_guid:Option<String>,committed:bool}
/// 失败时只有仍为空且仅含自己新增FX的轨道才回滚，不能删除重入添加的用户内容。
pub(crate) struct CreatedTrack {pub target:HostTrackTarget,fx:Option<[u8;16]>,committed:bool}
impl CreatedTrack {pub(crate) fn commit(&mut self) {self.committed=true;}}
impl Drop for CreatedTrack {
    fn drop(&mut self) {
        if self.committed||self.target.verify(&||true).is_err() {return;}
        let Some(api)=self.target.host.extended_media.as_ref().and_then(|m|m.tracks.as_ref()) else {return;};
        let track=self.target.track as *mut c_void;
        if unsafe {(api.item_count)(track)}!=0 {return;}
        let count=unsafe {(api.fx_count)(track)};
        let expected=if let Some(guid)=self.fx {count==1&&fx_guid(api,track,0).is_some_and(|current|current==guid)} else {count==0};
        if expected {unsafe {(api.delete)(track)};}
    }
}
fn fx_guid(api:&NewTrackApi,track:*mut c_void,index:i32)->Option<[u8;16]> {
    let pointer=unsafe {(api.fx_guid)(track,index)};if pointer.is_null() {return None;}
    let mut value=[0_u8;16];unsafe {std::ptr::copy_nonoverlapping(pointer,value.as_mut_ptr(),16)};Some(value)
}
impl CreatedItem {
    fn verify(&self,authorized:&impl Fn()->bool)->Result<(),String> {
        self.target.verify(authorized)?;let api=self.target.host.geometry.as_ref().unwrap();
        if !checked(authorized,||unsafe {(api.validate)(self.target.project as *mut c_void,self.item as *mut c_void,c"MediaItem*".as_ptr())})? {
            return Err("created item no longer valid".into());
        }
        let guid=read_guid(api.item_guid,self.item as *mut c_void,authorized)?;
        if (guid!=self.guid&&self.state_guid.as_ref()!=Some(&guid))
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
    /// 媒体剪贴板要求完整创建/删除/历史和状态API；不借clipEditing宣称全部命令可用。
    pub(crate) fn can_clipboard_items(&self)->bool {
        self.can_import_audio()&&self.item_state.is_some()&&self.split.is_some()
    }
    /// 只生成宿主GUID和解析状态，不创建对象；整批可在Undo/首个setter之前完成预检。
    pub(crate) fn prepare_copied_item(&self,chunk:&str,position:f64,authorized:&impl Fn()->bool)->Result<super::RewrittenItem,String> {
        self.project(authorized)?;
        let api=self.item_state.as_ref().ok_or("host item state API unavailable")?;
        super::item_chunk::rewrite_item(chunk,position,|| {
            let mut guid=super::item_chunk::NativeGuid::default();
            checked(authorized,||unsafe {(api.generate)(&mut guid)})?;
            let mut bytes=[0_u8;64];
            checked(authorized,||unsafe {(api.stringify)(&guid,bytes.as_mut_ptr().cast())})?;
            let end=bytes.iter().position(|value|*value==0).ok_or("unterminated generated GUID")?;
            String::from_utf8(bytes[..end].to_vec()).map_err(|_|"generated GUID is not UTF-8".into())
        })
    }
    pub(crate) fn can_import_audio(&self)->bool {self.media.is_some()&&self.has_project_history()}
    pub(crate) fn can_create_audio_track(&self)->bool {self.can_import_audio()&&self.extended_media.as_ref().is_some_and(|m|m.tracks.is_some())}
    /// 现代宿主选择器支持多选，返回完整UTF-8路径列表；旧宿主仍回退单文件。
    pub(crate) fn pick_audio_paths(&self,multiple:bool,authorized:&impl Fn()->bool)->Result<Vec<String>,String> {
        self.project(authorized)?;
        if let Some(picker)=self.extended_media.as_ref().and_then(|m|m.picker) {
            let mut bytes=vec![0_u8;65536];
            if !checked(authorized,||unsafe {picker(if multiple {2} else {1},c"Import audio into HiFiShifter".as_ptr(),c".wav".as_ptr(),
                c"Audio files|*.wav;*.flac;*.mp3;*.aiff;*.aif;*.ogg;*.opus;*.m4a|All files|*.*".as_ptr(),bytes.as_mut_ptr().cast(),bytes.len() as i32)})? {return Ok(Vec::new());}
            let end=bytes.iter().position(|b|*b==0).ok_or("host picker path budget exceeded")?;
            let text=std::str::from_utf8(&bytes[..end]).map_err(|_|"host picker path is not UTF-8")?;
            let paths=text.split('|').filter(|path|!path.is_empty()).map(str::to_owned).collect::<Vec<_>>();
            if paths.len()>512||paths.iter().any(|path|!std::path::Path::new(path).is_absolute()) {return Err("host picker returned invalid path batch".into());}return Ok(paths);
        }
        Ok(self.pick_audio(authorized)?.into_iter().collect())
    }
    /// 新轨身份取所属project创建前后的唯一GUID差集；重入额外增删时失败，不按位置猜新轨。
    pub(crate) fn create_audio_track(self:&Arc<Self>,name:&str,authorized:&impl Fn()->bool)->Result<CreatedTrack,String> {
        if !self.can_create_audio_track() {return Err("REAPER new-track import API unavailable".into());}
        let api=self.extended_media.as_ref().unwrap().tracks.as_ref().unwrap();let project=self.project(authorized)?;
        let enumerate=||->Result<Vec<(String,usize)>,String>{
            let token=self.geometry_revision(authorized)?;let count=checked(authorized,||unsafe {(api.count)(project)})?;
            if !(0..=10000).contains(&count) {return Err("host track count budget exceeded".into());}
            let mut result=Vec::new();for index in 0..count {
                let track=checked(authorized,||unsafe {(api.get)(project,index)})?;
                if track.is_null() {return Err("host track enumeration changed".into());}
                let target=self.track_target(project as usize,track as usize,authorized)?;result.push((target.guid,target.track));
            }
            if token!=self.geometry_revision(authorized)? {return Err("host tracks changed during enumeration".into());}Ok(result)
        };
        let before=enumerate()?;checked(authorized,||unsafe {(api.insert)(project,before.len() as i32,0)})?;let after=enumerate()?;
        let old=before.iter().map(|(guid,_)|guid).collect::<std::collections::HashSet<_>>();
        let added=after.iter().filter(|(guid,_)|!old.contains(guid)).collect::<Vec<_>>();
        if after.len()!=before.len()+1||added.len()!=1||before.iter().any(|(guid,_)|!after.iter().any(|(known,_)|known==guid)) {return Err("new track creation reentered; undo the host operation to cancel".into());}
        let target=self.track_target(project as usize,added[0].1,authorized)?;let mut created=CreatedTrack {target,fx:None,committed:false};
        let mut name=CString::new(name).map_err(|_|"track name contains NUL")?.into_bytes_with_nul();
        created.target.verify(authorized)?;
        if !checked(authorized,||unsafe {(self.media.as_ref().unwrap().track_guid)(created.target.track as *mut c_void,c"P_NAME".as_ptr(),name.as_mut_ptr().cast(),true)})? {return Err("host track naming failed".into());}
        let index=checked(authorized,||unsafe {(api.add_fx)(created.target.track as *mut c_void,c"VST3:HiFiShifter".as_ptr(),false,-1000)})?;
        if index!=0 {return Err("new audio track could not attach HiFiShifter as first FX".into());}
        created.fx=fx_guid(api,created.target.track as *mut c_void,index);if created.fx.is_none() {return Err("new HiFiShifter FX identity unavailable".into());}
        created.target.verify(authorized)?;Ok(created)
    }
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
    /// 从已重建身份的真实item状态创建；SOURCE/take/FX所有权由宿主state loader管理。
    pub(crate) fn create_copied_item(&self,state:&super::RewrittenItem,authorized:&impl Fn()->bool)->Result<crate::host::geometry::HostClipGeometry,String> {
        self.verify(authorized)?;
        if !self.host.can_clipboard_items() {return Err("host clip clipboard capability unavailable".into());}
        let bytes=CString::new(state.text.as_str()).map_err(|_|"item state contains NUL")?;
        let media=self.host.media.as_ref().unwrap();let api=self.host.geometry.as_ref().unwrap();
        let state_api=self.host.item_state.as_ref().unwrap();let write=self.host.write.as_ref().unwrap();
        let item=checked(authorized,||unsafe {(media.create_item)(self.track as *mut c_void)})?;
        if item.is_null() {return Err("host copied item creation failed".into());}
        let guid=read_guid(api.item_guid,item,authorized)?;
        let mut created=CreatedItem {target:self.clone(),item:item as usize,guid,state_guid:Some(state.item_guid.clone()),committed:false};
        created.verify(authorized)?;
        if !checked(authorized,||unsafe {(state_api.set)(item,bytes.as_ptr(),false)})? {return Err("host rejected copied item state".into());}
        created.verify(authorized)?;
        if read_guid(api.item_guid,item,authorized)?!=state.item_guid {return Err("copied item identity differs from prepared GUID".into());}
        let take=checked(authorized,||unsafe {(self.host.split.as_ref().unwrap().active_take)(item)})?;
        if take.is_null() {return Err("copied item has no active take".into());}
        let geometry=self.host.geometry_for_take(take,authorized)?;
        if geometry.item_id!=state.item_guid||!state.take_guids.contains(&geometry.take_id) {
            return Err("copied active take identity differs from prepared state".into());
        }
        created.verify(authorized)?;
        checked(authorized,||unsafe {(write.update)(item)})?;
        checked(authorized,||unsafe {(write.arrange)()})?;
        created.committed=true;Ok(geometry)
    }
    /// parent轨道的GUID用于GUI去重，不用名称或轨道序号猜对象。
    pub(crate) fn inventory_guid(&self)->&str {&self.guid}
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
        let mut created=CreatedItem {target:self.clone(),item:item as usize,guid:guid.clone(),state_guid:None,committed:false};
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
