//! 经真实VST3 connection消息关联实例；只接受本进程登记的活令牌，不猜最后一轨。
use crate::render::extension::ExtensionOwner;
use crate::vst3::{uid_guid,K_RESULT_OK,TResult};
use crate::editor::connection::UnknownVtbl;
use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::sync::atomic::{AtomicU64,Ordering};
fn routes() -> &'static Mutex<HashMap<String,Weak<ExtensionOwner>>> {
    static ROUTES:OnceLock<Mutex<HashMap<String,Weak<ExtensionOwner>>>>=OnceLock::new();
    ROUTES.get_or_init(Default::default)
}
pub(crate) struct RouteLease { token:String }
impl RouteLease {
    /// 登记组件自己的owner；租约析构即撤销，额外Arc不能让已销毁组件再次被连接。
    pub fn new(owner:&Arc<ExtensionOwner>) -> Self {
        static SEQUENCE:AtomicU64=AtomicU64::new(1);
        let input=format!("{}-{}-{:?}",std::process::id(),SEQUENCE.fetch_add(1,Ordering::Relaxed),
            std::time::SystemTime::now());
        let token=blake3::hash(input.as_bytes()).to_hex().to_string();
        routes().lock().unwrap_or_else(|e|e.into_inner()).insert(token.clone(),Arc::downgrade(owner));
        Self {token}
    }
    pub fn token(&self)->&str { &self.token }
}
impl Drop for RouteLease {
    fn drop(&mut self) { routes().lock().unwrap_or_else(|e|e.into_inner()).remove(&self.token); }
}
#[derive(Default)]
struct Binding {token:Option<String>,generation:u64}
const IID_COMPONENT_HANDLER2:[u32;4]=[0xF040B4B3,0xA36045EC,0xABCDC045,0xB4D5A2CC];
#[repr(C)] struct Handler2Vtbl {
    base:UnknownVtbl,
    set_dirty:unsafe extern "system" fn(*mut c_void,u8)->TResult,
}
#[derive(Default)]
pub(crate) struct EditorLink { binding:Mutex<Binding>, handler2:Mutex<usize>, dirty:std::sync::atomic::AtomicBool }
fn release_handler(pointer:usize) {
    if pointer!=0 {unsafe {let pointer=pointer as *mut c_void;((**pointer.cast::<*const UnknownVtbl>()).release)(pointer);}}
}
impl Drop for EditorLink {fn drop(&mut self) {release_handler(*self.handler2.get_mut().unwrap_or_else(|e|e.into_inner()));}}
impl EditorLink {
    /// 保存宿主IComponentHandler2的独立COM引用；不能把裸IComponentHandler指针跨线程调用。
    pub(crate) fn set_component_handler(&self,handler:*mut c_void)->Result<(),String> {
        let next=if handler.is_null() {0} else {
            let mut out=std::ptr::null_mut();let iid=uid_guid(IID_COMPONENT_HANDLER2);
            let result=unsafe {((**handler.cast::<*const UnknownVtbl>()).query)(handler,iid.as_ptr(),&mut out)};
            if result!=K_RESULT_OK||out.is_null() {0} else {out as usize}
        };
        let old={let mut current=self.handler2.lock().unwrap_or_else(|e|e.into_inner());std::mem::replace(&mut *current,next)};
        crate::log_line(&format!("[undo] IComponentHandler2 available={}",next!=0));
        release_handler(old);Ok(())
    }
    /// actor/UI写入只置位；真实handler回调由插件UI timer调用。
    pub(crate) fn mark_dirty(&self) {self.dirty.store(true,Ordering::Release);}
    pub(crate) fn flush_dirty(&self) {
        if !self.dirty.swap(false,Ordering::AcqRel) {return;}
        let pointer=*self.handler2.lock().unwrap_or_else(|e|e.into_inner());if pointer==0 {return;}
        let pointer=pointer as *mut c_void;let vtbl=unsafe {&*(*(pointer as *const *const Handler2Vtbl))};
        let result=unsafe {(vtbl.set_dirty)(pointer,1)};
        crate::log_line(&format!("[undo] IComponentHandler2::setDirty result={result}"));
    }
    /// PID与令牌都匹配真实本地组件才绑定；宿主代理传递消息但不能替代归属证据。
    pub fn bind(&self,pid:i64,token:&str)->Result<(),String> {
        if pid!=std::process::id() as i64 || token.len()!=64 { return Err("editor route is not local".into()); }
        let owner=routes().lock().unwrap_or_else(|e|e.into_inner()).get(token).and_then(Weak::upgrade);
        if owner.is_none() { return Err("editor route expired or unknown".into()); }
        let mut binding=self.binding.lock().unwrap_or_else(|e|e.into_inner());
        binding.generation=binding.generation.checked_add(1).ok_or("editor route generation exhausted")?;
        binding.token=Some(token.to_owned());
        Ok(())
    }
    pub fn clear(&self) {let mut binding=self.binding.lock().unwrap_or_else(|e|e.into_inner());binding.token=None;binding.generation=binding.generation.saturating_add(1);}
    /// 每次请求重新验证租约，杜绝组件已销毁但editor旧weak仍可升级的情况。
    pub fn owner(&self)->Result<Arc<ExtensionOwner>,String> {
        let token=self.binding.lock().unwrap_or_else(|e|e.into_inner()).token.clone().ok_or("FX editor not connected to its processor yet")?;
        routes().lock().unwrap_or_else(|e|e.into_inner()).get(&token).and_then(Weak::upgrade).filter(|owner|!owner.is_closed())
            .ok_or_else(||"FX processor closed".into())
    }
    /// 排队和执行各核对一次真实文档与绑定代次；clear/rebind不能为旧请求借新的组件入口。
    pub(super) fn authorize(&self,document:&Arc<crate::render::document::DocumentSession>)->Result<u64,String> {
        let binding=self.binding.lock().unwrap_or_else(|e|e.into_inner());
        let token=binding.token.as_ref().ok_or("FX editor not connected to its processor yet")?;
        let owner=routes().lock().unwrap_or_else(|e|e.into_inner()).get(token).and_then(Weak::upgrade).ok_or("FX processor closed")?;
        if !Arc::ptr_eq(&owner.editor_document()?,document) {return Err("editor route belongs to another document".into());}
        Ok(binding.generation)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    #[test]
    fn actual_route_tokens_never_select_another_live_instance() {
        let a=Arc::new(ExtensionOwner::default()); let b=Arc::new(ExtensionOwner::default());
        let ra=RouteLease::new(&a); let rb=RouteLease::new(&b);
        let la=EditorLink::default(); let lb=EditorLink::default();
        la.bind(std::process::id() as i64,ra.token()).unwrap();
        lb.bind(std::process::id() as i64,rb.token()).unwrap();
        assert!(Arc::ptr_eq(&la.owner().unwrap(),&a));
        assert!(Arc::ptr_eq(&lb.owner().unwrap(),&b));
        assert!(la.bind(-1,rb.token()).is_err());
        assert!(Arc::ptr_eq(&la.owner().unwrap(),&a));
        drop(ra); // a仍有Arc，租约撤销必须独立起效。
        assert!(la.owner().is_err());
        assert!(Arc::ptr_eq(&lb.owner().unwrap(),&b));
        lb.clear(); assert!(lb.owner().is_err());
    }

    #[repr(C)] struct FakeHandler { vtbl:*const Handler2Vtbl, refs:AtomicU32, dirty:AtomicU32 }
    unsafe extern "system" fn fake_query(this:*mut c_void,_iid:*const u8,out:*mut *mut c_void)->TResult {unsafe {*out=this;(*this.cast::<FakeHandler>()).refs.fetch_add(1,Ordering::AcqRel);};K_RESULT_OK}
    unsafe extern "system" fn fake_add(this:*mut c_void)->u32 {unsafe {(*this.cast::<FakeHandler>()).refs.fetch_add(1,Ordering::AcqRel)+1}}
    unsafe extern "system" fn fake_release(this:*mut c_void)->u32 {unsafe {(*this.cast::<FakeHandler>()).refs.fetch_sub(1,Ordering::AcqRel)-1}}
    unsafe extern "system" fn fake_dirty(this:*mut c_void,value:u8)->TResult {unsafe {if value!=0 {(*this.cast::<FakeHandler>()).dirty.fetch_add(1,Ordering::AcqRel);}}K_RESULT_OK}
    static FAKE_HANDLER_VTBL:Handler2Vtbl=Handler2Vtbl {base:UnknownVtbl {query:fake_query,add:fake_add,release:fake_release},set_dirty:fake_dirty};

    /// handler查询/引用/dirty调用必须都发生在显式UI flush阶段，清理不泄漏宿主引用。
    #[test]
    fn component_handler_dirty_bridge_retains_and_flushes_on_ui_boundary() {
        let handler=Box::new(FakeHandler {vtbl:&FAKE_HANDLER_VTBL,refs:AtomicU32::new(1),dirty:AtomicU32::new(0)});
        let pointer=(&*handler as *const FakeHandler) as *mut c_void;let link=EditorLink::default();
        link.set_component_handler(pointer).unwrap();link.mark_dirty();assert_eq!(handler.dirty.load(Ordering::Acquire),0);
        link.flush_dirty();assert_eq!(handler.dirty.load(Ordering::Acquire),1);link.flush_dirty();assert_eq!(handler.dirty.load(Ordering::Acquire),1);
        let _=link.set_component_handler(std::ptr::null_mut());assert!(handler.refs.load(Ordering::Acquire)>=1);drop(link);drop(handler);
    }
}
