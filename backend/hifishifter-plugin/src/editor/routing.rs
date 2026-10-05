//! 经真实VST3 connection消息关联实例；只接受本进程登记的活令牌，不猜最后一轨。
use crate::render::extension::ExtensionOwner;
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
#[derive(Default)]
pub(crate) struct EditorLink { binding:Mutex<Binding> }
impl EditorLink {
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
}
