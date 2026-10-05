//! 核对官方REAPER SDK的原生宿主适配：只在原始UI/model线程读所属project的真实位置。
//! ABI依据 justinfrankel/reaper-sdk c0eafe87863b2bf69c5c822760f1b32a753b211b，
//! sdk/reaper_vst3_interfaces.h 与 reaper_plugin_functions.h。手写适配，不是SDK原文件。
use crate::editor::connection::UnknownVtbl;
use crate::vst3::{uid_guid,K_RESULT_OK};
use std::ffi::{c_void,c_char};

const IID:[u32;4]=[0x79655E36,0x77EE4267,0xA573FEF7,0x4912C27C];
#[repr(C)]
struct HostVtbl {
    base:UnknownVtbl,
    api:unsafe extern "system" fn(*mut c_void,*const c_char)->*mut c_void,
    parent:unsafe extern "system" fn(*mut c_void,u32)->*mut c_void,
    extended:unsafe extern "system" fn(*mut c_void,u32,*mut c_void,*mut c_void,*mut c_void)->*mut c_void,
}
struct Interface(usize);
impl Drop for Interface {
    fn drop(&mut self) {
        // SAFETY: 成功QI拥有一次引用；FUnknown release只平衡引用，不调用REAPER项目API。
        unsafe {let pointer=self.0 as *mut c_void;((**pointer.cast::<*const UnknownVtbl>()).release)(pointer);}
    }
}
type Position=unsafe extern "C" fn(*mut c_void)->f64;
type PlayState=unsafe extern "C" fn(*mut c_void)->i32;
pub(crate) struct ReaperHost {
    _interface:Interface,thread:std::thread::ThreadId,project:usize,
    position:Position,cursor:Position,state:PlayState,
}
impl ReaperHost {
    /// 从真实host context QI；parent(3)是所属project，不使用当前活动project或按轨名猜归属。
    /// # Safety
    /// context是初始化期间可读的活FUnknown；host在该 owning reference期间保持接口/API有效。
    pub unsafe fn from_context(context:*mut c_void)->Option<Self> {
        if context.is_null() {return None;}
        let mut pointer=std::ptr::null_mut();let iid=uid_guid(IID);
        let result=unsafe {((**context.cast::<*const UnknownVtbl>()).query)(context,iid.as_ptr(),&mut pointer)};
        if result!=K_RESULT_OK||pointer.is_null() {return None;}
        let interface=Interface(pointer as usize);
        let table=unsafe {&**pointer.cast::<*const HostVtbl>()};
        let project=unsafe {(table.parent)(pointer,3)};
        let position=unsafe {(table.api)(pointer,c"GetPlayPositionEx".as_ptr())};
        let cursor=unsafe {(table.api)(pointer,c"GetCursorPositionEx".as_ptr())};
        let state=unsafe {(table.api)(pointer,c"GetPlayStateEx".as_ptr())};
        if project.is_null()||position.is_null()||cursor.is_null()||state.is_null() {return None;}
        // SAFETY: 上述函数名与签名来自同一官方头；不猜未知API的函数类型。
        Some(Self {_interface:interface,thread:std::thread::current().id(),project:project as usize,
            position:unsafe {std::mem::transmute::<*mut c_void,Position>(position)},
            cursor:unsafe {std::mem::transmute::<*mut c_void,Position>(cursor)},
            state:unsafe {std::mem::transmute::<*mut c_void,PlayState>(state)}})
    }
    /// 原UI线程读取延迟补偿的实际听到位置；暂停保持play位置，完全停止才读edit cursor。
    pub fn sample(&self)->Result<(f64,bool),String> {
        if std::thread::current().id()!=self.thread {return Err("REAPER transport queried outside its model/UI thread".into());}
        let project=self.project as *mut c_void;
        // SAFETY: 生命周期/线程由 owning host reference与调用者的活组件路由约束。
        let state=unsafe {(self.state)(project)};
        if state<0 {return Err("invalid REAPER play state".into());}
        let position=unsafe {if state & 3!=0 {(self.position)(project)} else {(self.cursor)(project)}};
        if !position.is_finite() {return Err("invalid REAPER project position".into());}
        Ok((position,state & 1!=0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32,Ordering};
    struct Project {position:f64,cursor:f64,state:i32}
    #[repr(C)]
    struct Host {table:*const HostVtbl,refs:AtomicU32,project:*mut Project}
    unsafe extern "system" fn query(this:*mut c_void,iid:*const u8,out:*mut *mut c_void)->i32 {
        let expected=[0x36,0x5e,0x65,0x79,0xee,0x77,0x67,0x42,0xa5,0x73,0xfe,0xf7,0x49,0x12,0xc2,0x7c];
        if unsafe {std::slice::from_raw_parts(iid,16)}!=expected {return crate::vst3::K_NO_INTERFACE;}
        unsafe {add(this);*out=this;}K_RESULT_OK
    }
    unsafe extern "system" fn add(this:*mut c_void)->u32 {unsafe {(&*this.cast::<Host>()).refs.fetch_add(1,Ordering::AcqRel)+1}}
    unsafe extern "system" fn release(this:*mut c_void)->u32 {unsafe {(&*this.cast::<Host>()).refs.fetch_sub(1,Ordering::AcqRel)-1}}
    unsafe extern "C" fn position(project:*mut c_void)->f64 {unsafe {(&*project.cast::<Project>()).position}}
    unsafe extern "C" fn cursor(project:*mut c_void)->f64 {unsafe {(&*project.cast::<Project>()).cursor}}
    unsafe extern "C" fn state(project:*mut c_void)->i32 {unsafe {(&*project.cast::<Project>()).state}}
    unsafe extern "system" fn api(_: *mut c_void,name:*const c_char)->*mut c_void {
        match unsafe {std::ffi::CStr::from_ptr(name)}.to_bytes() {
            b"GetPlayPositionEx"=>position as *const () as *mut c_void,b"GetCursorPositionEx"=>cursor as *const () as *mut c_void,
            b"GetPlayStateEx"=>state as *const () as *mut c_void,_=>std::ptr::null_mut(),
        }
    }
    unsafe extern "system" fn parent(this:*mut c_void,selector:u32)->*mut c_void {
        if selector==3 {unsafe {(&*this.cast::<Host>()).project.cast()}} else {std::ptr::null_mut()}
    }
    unsafe extern "system" fn extended(_: *mut c_void,_:u32,_:*mut c_void,_:*mut c_void,_:*mut c_void)->*mut c_void {std::ptr::null_mut()}
    static TABLE:HostVtbl=HostVtbl {base:UnknownVtbl {query,add,release},api,parent,extended};
    /// 通过真实QI/函数表适配测试所属project、pause/stop区别、引用平衡和跨线程拒绝。
    #[test]
    fn bound_project_transport_uses_actual_position_and_never_calls_from_a_worker() {
        let mut project=Project {position:4.5,cursor:1.0,state:1};
        let mut host=Host {table:&TABLE,refs:AtomicU32::new(1),project:&raw mut project};
        let client=unsafe {ReaperHost::from_context((&raw mut host).cast())}.unwrap();
        assert_eq!(client.sample().unwrap(),(4.5,true));assert_eq!(host.refs.load(Ordering::Acquire),2);
        project.state=2;assert_eq!(client.sample().unwrap(),(4.5,false));
        project.state=0;assert_eq!(client.sample().unwrap(),(1.0,false));
        // client中的project是fixture稳定地址；不得换用active project或GetPlayPosition2Ex。
        std::thread::scope(|scope|{assert!(scope.spawn(||client.sample()).join().unwrap().is_err());});
        drop(client);assert_eq!(host.refs.load(Ordering::Acquire),1);
    }
}
