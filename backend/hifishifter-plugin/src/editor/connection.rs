//! 锁定SDK的IConnectionPoint/IHostApplication/IMessage ABI；消息允许宿主放置代理。
use crate::vst3::{iid_matches,uid_guid,TResult,K_RESULT_OK,K_RESULT_FALSE,K_INVALID_ARGUMENT};
use std::ffi::{c_char,c_void,CStr};
use std::sync::Mutex;
pub(crate) const IID_CONNECTION:[u32;4]=[0x70A4156F,0x6E6E4026,0x989148BF,0xAA60D8D1];
const IID_HOST:[u32;4]=[0x58E595CC,0xDB2D4969,0x8B6AAF8C,0x36A664E5];
const IID_MESSAGE:[u32;4]=[0x936F033B,0xC6C047DB,0xBB0882F8,0x13C1E613];
const ROUTE_ID:&CStr=c"HiFiShifter.Editor.Route.v1";
const REQUEST_ID:&CStr=c"HiFiShifter.Editor.RequestRoute.v1";
#[repr(C)]
pub(crate) struct UnknownVtbl {
    pub query:unsafe extern "system" fn(*mut c_void,*const u8,*mut *mut c_void)->TResult,
    pub add:unsafe extern "system" fn(*mut c_void)->u32,
    pub release:unsafe extern "system" fn(*mut c_void)->u32,
}
#[repr(C)]
pub(crate) struct ConnectionVtbl {
    pub base:UnknownVtbl,
    pub connect:unsafe extern "system" fn(*mut c_void,*mut c_void)->TResult,
    pub disconnect:unsafe extern "system" fn(*mut c_void,*mut c_void)->TResult,
    pub notify:unsafe extern "system" fn(*mut c_void,*mut c_void)->TResult,
}
#[repr(C)]
struct HostVtbl {
    base:UnknownVtbl,
    name:unsafe extern "system" fn(*mut c_void,*mut u16)->TResult,
    create:unsafe extern "system" fn(*mut c_void,*const u8,*const u8,*mut *mut c_void)->TResult,
}
#[repr(C)]
struct MessageVtbl {
    base:UnknownVtbl,
    id:unsafe extern "system" fn(*mut c_void)->*const c_char,
    set_id:unsafe extern "system" fn(*mut c_void,*const c_char),
    attributes:unsafe extern "system" fn(*mut c_void)->*mut c_void,
}
#[repr(C)]
struct AttributesVtbl {
    base:UnknownVtbl,
    set_int:unsafe extern "system" fn(*mut c_void,*const c_char,i64)->TResult,
    get_int:unsafe extern "system" fn(*mut c_void,*const c_char,*mut i64)->TResult,
    set_float:unsafe extern "system" fn(*mut c_void,*const c_char,f64)->TResult,
    get_float:unsafe extern "system" fn(*mut c_void,*const c_char,*mut f64)->TResult,
    set_string:unsafe extern "system" fn(*mut c_void,*const c_char,*const u16)->TResult,
    get_string:unsafe extern "system" fn(*mut c_void,*const c_char,*mut u16,u32)->TResult,
    set_binary:unsafe extern "system" fn(*mut c_void,*const c_char,*const c_void,u32)->TResult,
    get_binary:unsafe extern "system" fn(*mut c_void,*const c_char,*mut *const c_void,*mut u32)->TResult,
}
struct Owned(usize);
impl Owned {
    unsafe fn retain(pointer:*mut c_void)->Self {
        if !pointer.is_null() { unsafe { ((**pointer.cast::<*const UnknownVtbl>()).add)(pointer); } }
        Self(pointer as usize)
    }
    fn ptr(&self)->*mut c_void { self.0 as *mut c_void }
}
impl Drop for Owned {
    fn drop(&mut self) { let ptr=self.ptr(); if !ptr.is_null() { unsafe { ((**ptr.cast::<*const UnknownVtbl>()).release)(ptr); } } }
}
#[derive(Default)]
pub(crate) struct ConnectionState { host:Mutex<usize>, peer:Mutex<usize> }
impl ConnectionState {
    /// 宿主提供message对象，不能把Rust Arc裸指针当跨代理协议。
    pub unsafe fn initialize(&self,context:*mut c_void) {
        if context.is_null() { return; }
        let mut host=std::ptr::null_mut();
        let iid=uid_guid(IID_HOST);
        let result=unsafe { ((**context.cast::<*const UnknownVtbl>()).query)(context,iid.as_ptr(),&mut host) };
        if result==K_RESULT_OK && !host.is_null() {
            let old=std::mem::replace(&mut *self.host.lock().unwrap_or_else(|e|e.into_inner()),host as usize);
            drop(Owned(old));
        }
    }
    pub unsafe fn connect(&self,peer:*mut c_void)->TResult {
        if peer.is_null() { return K_INVALID_ARGUMENT; }
        let owned=unsafe { Owned::retain(peer) };
        let old=std::mem::replace(&mut *self.peer.lock().unwrap_or_else(|e|e.into_inner()),owned.0);
        std::mem::forget(owned); drop(Owned(old)); K_RESULT_OK
    }
    pub fn disconnect(&self,peer:*mut c_void)->TResult {
        let old={ let mut known=self.peer.lock().unwrap_or_else(|e|e.into_inner());
            if *known!=peer as usize { return K_RESULT_FALSE; } std::mem::take(&mut *known) };
        drop(Owned(old)); K_RESULT_OK
    }
    pub fn close(&self) {
        let peer=std::mem::take(&mut *self.peer.lock().unwrap_or_else(|e|e.into_inner()));
        let host=std::mem::take(&mut *self.host.lock().unwrap_or_else(|e|e.into_inner()));
        drop(Owned(peer)); drop(Owned(host));
    }
    /// 在无内部锁时调用宿主，notify可同步重入另一个connection并请求回信。
    pub fn send(&self,token:Option<&str>)->TResult {
        let host={ let host=self.host.lock().unwrap_or_else(|e|e.into_inner()); unsafe { Owned::retain(*host as *mut c_void) } };
        let peer={ let peer=self.peer.lock().unwrap_or_else(|e|e.into_inner()); unsafe { Owned::retain(*peer as *mut c_void) } };
        if host.0==0 || peer.0==0 { return K_RESULT_FALSE; }
        unsafe {
            let table=&**host.ptr().cast::<*const HostVtbl>();
            let iid=uid_guid(IID_MESSAGE); let mut message=std::ptr::null_mut();
            if (table.create)(host.ptr(),iid.as_ptr(),iid.as_ptr(),&mut message)!=K_RESULT_OK || message.is_null() { return K_RESULT_FALSE; }
            let _message=Owned(message as usize);
            let message_table=&**message.cast::<*const MessageVtbl>();
            (message_table.set_id)(message,if token.is_some() { ROUTE_ID.as_ptr() } else { REQUEST_ID.as_ptr() });
            if let Some(token)=token {
                let attrs=(message_table.attributes)(message);
                if attrs.is_null() { return K_RESULT_FALSE; }
                let table=&**attrs.cast::<*const AttributesVtbl>();
                let wide:Vec<u16>=token.encode_utf16().chain(Some(0)).collect();
                if (table.set_string)(attrs,c"hfs_editor_route".as_ptr(),wide.as_ptr())!=K_RESULT_OK ||
                    (table.set_int)(attrs,c"hfs_process_id".as_ptr(),std::process::id() as i64)!=K_RESULT_OK { return K_RESULT_FALSE; }
            }
            ((**peer.ptr().cast::<*const ConnectionVtbl>()).notify)(peer.ptr(),message)
        }
    }
}
impl Drop for ConnectionState { fn drop(&mut self) { self.close(); } }

pub(crate) enum Message { RequestRoute, Route { pid:i64, token:String } }
/// 读取宿主拥有的活message；严格按UTF16字节容量和NUL边界处理属性。
pub(crate) unsafe fn read(message:*mut c_void)->Result<Message,String> {
    if message.is_null() { return Err("null VST3 message".into()); }
    let table=unsafe { &**message.cast::<*const MessageVtbl>() };
    let id=unsafe { (table.id)(message) };
    if id.is_null() { return Err("null message ID".into()); }
    let id=unsafe { CStr::from_ptr(id) };
    if id==REQUEST_ID { return Ok(Message::RequestRoute); }
    if id!=ROUTE_ID { return Err("unknown VST3 message".into()); }
    let attrs=unsafe { (table.attributes)(message) };
    if attrs.is_null() { return Err("route attributes missing".into()); }
    let table=unsafe { &**attrs.cast::<*const AttributesVtbl>() };
    let mut pid=0_i64; let mut wide=[0_u16;128];
    if unsafe { (table.get_int)(attrs,c"hfs_process_id".as_ptr(),&mut pid) }!=K_RESULT_OK ||
        unsafe { (table.get_string)(attrs,c"hfs_editor_route".as_ptr(),wide.as_mut_ptr(),256) }!=K_RESULT_OK { return Err("route attributes invalid".into()); }
    let length=wide.iter().position(|c|*c==0).ok_or("unterminated route token")?;
    Ok(Message::Route { pid,token:String::from_utf16(&wide[..length]).map_err(|e|e.to_string())? })
}
/// 判断IID仅在真实queryInterface入口使用，不把其它VST对象误当connection。
pub(crate) unsafe fn is_connection(iid:*const u8)->bool { unsafe { iid_matches(iid,IID_CONNECTION) } }
