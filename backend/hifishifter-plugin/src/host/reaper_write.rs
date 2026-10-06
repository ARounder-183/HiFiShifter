//! 核对锁定官方REAPER API的主线程片段写入口；GUID来自真实直接take绑定，不猜活动工程。
use super::{ReaperHost,Guid,checked,HostVtbl};
use crate::host::geometry::HostClipGeometry;
use std::ffi::{c_char,c_void,CStr};
use std::sync::Arc;

pub(super) type SetValue=unsafe extern "C" fn(*mut c_void,*const c_char,f64)->bool;
pub(super) type Track=unsafe extern "C" fn(*mut c_void)->*mut c_void;
pub(super) type Move=unsafe extern "C" fn(*mut c_void,*mut c_void)->bool;
pub(super) type Begin=unsafe extern "C" fn(*mut c_void);
pub(super) type End=unsafe extern "C" fn(*mut c_void,*const c_char,i32);
pub(super) type Update=unsafe extern "C" fn(*mut c_void);
pub(super) type Arrange=unsafe extern "C" fn();

pub(super) struct WriteApi {
    pub set_item:SetValue,pub set_take:SetValue,pub item_track:Track,pub move_item:Move,
    pub begin:Begin,pub end:End,pub update:Update,pub arrange:Arrange,
}

/// 地址只在所属UI线程使用；拥有host接口引用不等同item/take生命周期，每次写前都验证。
#[derive(Clone)]
pub(crate) struct HostClipTarget {
    host:Arc<ReaperHost>,project:usize,item:usize,take:usize,track:usize,
    pub geometry:HostClipGeometry,
}

/// 一个真实宿主Undo块；关闭视图/错误也收尾。析构不在错误线程调用宿主API。
pub(crate) struct HostUndoBlock {host:Arc<ReaperHost>,project:usize}
impl Drop for HostUndoBlock {
    fn drop(&mut self) {
        if std::thread::current().id()!=self.host.thread {return;}
        let Some(api)=&self.host.geometry else {return;};let Some(write)=&self.host.write else {return;};
        let project=self.project as *mut c_void;
        if unsafe {(api.validate)(project,project,c"ReaProject*".as_ptr())} {
            unsafe {(write.end)(project,c"HiFiShifter clip edit".as_ptr(),-1);}
        }
    }
}

impl ReaperHost {
    /// 可选写能力与只读几何独立；缺setter时GUI保持只读，不调用未查到的函数。
    pub(crate) fn can_edit_clips(&self)->bool {self.write.is_some()&&self.geometry.is_some()}
    /// 冻结唯一直接take身份；调用方先验证该host对应真实assigned region。
    pub(crate) fn clip_target(self:&Arc<Self>,authorized:impl Fn()->bool)->Result<HostClipTarget,String> {
        if !self.can_edit_clips() {return Err("REAPER clip write API unavailable".into());}
        let geometry=self.geometry(&authorized)?;
        let project=self.project(&authorized)?;let pointer=self._interface.0 as *mut c_void;
        let table=unsafe {&**pointer.cast::<*const HostVtbl>()};
        let take=checked(&authorized,||unsafe {(table.parent)(pointer,2)})?;
        let api=self.geometry.as_ref().unwrap();
        let item=checked(&authorized,||unsafe {(api.item)(take)})?;
        let track=checked(&authorized,||unsafe {(self.write.as_ref().unwrap().item_track)(item)})?;
        let target=HostClipTarget {host:self.clone(),project:project as usize,item:item as usize,take:take as usize,track:track as usize,geometry};
        target.verify(&authorized)?;Ok(target)
    }
}

impl HostClipTarget {
    /// 同批所有对象须来自同一project，不以“当前活动项目”或相同轨名推断。
    pub(crate) fn same_project(&self,other:&Self)->bool {self.project==other.project}
    pub(crate) fn track_key(&self)->usize {self.track}
    fn api(&self)->(&super::GeometryApi,&WriteApi) {(self.host.geometry.as_ref().unwrap(),self.host.write.as_ref().unwrap())}
    fn valid(&self,pointer:usize,kind:&CStr,authorized:&impl Fn()->bool)->Result<(),String> {
        if pointer==0 {return Err("missing host edit object".into());}
        if !checked(authorized,||unsafe {(self.api().0.validate)(self.project as *mut c_void,pointer as *mut c_void,kind.as_ptr())})? {
            return Err("host edit object no longer belongs to project".into());
        }Ok(())
    }
    fn guid(&self,pointer:usize,getter:Guid,expected:&str,authorized:&impl Fn()->bool)->Result<(),String> {
        if expected.len()!=38 {return Err("invalid captured host GUID".into());}
        let mut bytes=[0_u8;64];
        if !checked(authorized,||unsafe {getter(pointer as *mut c_void,c"GUID".as_ptr(),bytes.as_mut_ptr().cast(),false)})?
            || bytes[..expected.len()]!=*expected.as_bytes() || bytes[expected.len()]!=0 {
            return Err("host item/take identity changed before write".into());
        }Ok(())
    }
    /// 即使地址被重用也不能写新对象。自身setter造成ARA模型代次推进不等同租约失效。
    pub(crate) fn verify(&self,authorized:&impl Fn()->bool)->Result<(),String> {
        if std::thread::current().id()!=self.host.thread {return Err("host edit off UI thread".into());}
        self.valid(self.project,c"ReaProject*",authorized)?;
        self.valid(self.item,c"MediaItem*",authorized)?;self.valid(self.take,c"MediaItem_Take*",authorized)?;
        let api=self.api().0;
        if checked(authorized,||unsafe {(api.item)(self.take as *mut c_void)})? as usize!=self.item {
            return Err("host take parent changed".into());
        }
        self.guid(self.item,api.item_guid,&self.geometry.item_id,authorized)?;
        self.guid(self.take,api.take_guid,&self.geometry.take_id,authorized)
    }
    /// 批次开始前调用；拒绝跨工程/失效对象，不在document锁内调用Undo API。
    pub(crate) fn begin_undo(&self,authorized:&impl Fn()->bool)->Result<HostUndoBlock,String> {
        self.verify(authorized)?;
        let block=HostUndoBlock {host:self.host.clone(),project:self.project};
        checked(authorized,||unsafe {(self.api().1.begin)(self.project as *mut c_void)})?;Ok(block)
    }
    /// 仅明确白名单的数值字段可写；重入撤销租约后停止后续setter。
    pub(crate) fn set_item(&self,name:&CStr,value:f64,authorized:&impl Fn()->bool)->Result<(),String> {
        self.verify(authorized)?;
        if !matches!(name.to_bytes(),b"D_POSITION"|b"D_LENGTH"|b"B_MUTE"|b"B_LOOPSRC"|b"D_SNAPOFFSET"|b"D_FADEINLEN"|b"D_FADEOUTLEN"|b"D_FADEINLEN_AUTO"|b"D_FADEOUTLEN_AUTO") {return Err("unsupported host item field".into());}
        if !value.is_finite() {return Err("nonfinite host item edit".into());}
        if !checked(authorized,||unsafe {(self.api().1.set_item)(self.item as *mut c_void,name.as_ptr(),value)})? {
            return Err(format!("REAPER rejected item field {}",name.to_string_lossy()));
        }Ok(())
    }
    pub(crate) fn set_take(&self,name:&CStr,value:f64,authorized:&impl Fn()->bool)->Result<(),String> {
        self.verify(authorized)?;
        if !matches!(name.to_bytes(),b"D_STARTOFFS"|b"D_PLAYRATE"|b"B_PPITCH"|b"I_CHANMODE") {return Err("unsupported host take field".into());}
        if !value.is_finite() {return Err("nonfinite host take edit".into());}
        if !checked(authorized,||unsafe {(self.api().1.set_take)(self.take as *mut c_void,name.as_ptr(),value)})? {
            return Err(format!("REAPER rejected take field {}",name.to_string_lossy()));
        }Ok(())
    }
    /// 目标轨道由另一真实region的直接绑定证明，不接受任意JS指针/轨道编号。
    pub(crate) fn move_to(&self,destination:&Self,authorized:&impl Fn()->bool)->Result<(),String> {
        self.verify(authorized)?;destination.verify(authorized)?;
        if !self.same_project(destination) {return Err("cannot move item across host projects".into());}
        self.valid(destination.track,c"MediaTrack*",authorized)?;
        if !checked(authorized,||unsafe {(self.api().1.move_item)(self.item as *mut c_void,destination.track as *mut c_void)})? {
            return Err("REAPER rejected item track move".into());
        }Ok(())
    }
    /// 触发宿主刷新和真实ARA回流，不直接篡改插件timeline几何。
    pub(crate) fn update(&self,authorized:&impl Fn()->bool)->Result<(),String> {
        self.verify(authorized)?;
        checked(authorized,||unsafe {(self.api().1.update)(self.item as *mut c_void)})?;
        checked(authorized,||unsafe {(self.api().1.arrange)()})
    }
}
