//! Optional host playback-request client.

use ara2_bridge_core::{AraBool, AraError, SizedInput};
use ara2_bridge_sys::*;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::sync::{Arc,atomic::{AtomicBool,Ordering}};

type Void = unsafe extern "C" fn(ARAPlaybackControllerHostRef);
type Position = unsafe extern "C" fn(ARAPlaybackControllerHostRef, f64);
type Cycle = unsafe extern "C" fn(ARAPlaybackControllerHostRef, f64, f64);
type Enable = unsafe extern "C" fn(ARAPlaybackControllerHostRef, ARABool);

/// Optional host playback-control request service.
pub struct PlaybackAccess<'host> {
    host_ref: ARAPlaybackControllerHostRef,
    start: Void,
    stop: Void,
    position: Position,
    cycle: Cycle,
    enable: Enable,
    _lifetime: PhantomData<&'host ()>,
    endpoint: Arc<PlaybackEndpoint>,
}

struct PlaybackEndpoint {host:usize,start:Void,stop:Void,position:Position,alive:AtomicBool,thread:std::thread::ThreadId}
/// 生命周期受HostClients撤销、且只允许原模型线程调用的宿主播放请求租约。
#[derive(Clone)]
pub struct PlaybackRequestHandle {endpoint:Arc<PlaybackEndpoint>}
impl PlaybackRequestHandle {
    fn check(&self)->Result<(),AraError> {
        if !self.endpoint.alive.load(Ordering::Acquire) {return Err(AraError::InvalidState("host playback service revoked"));}
        if self.endpoint.thread!=std::thread::current().id() {return Err(AraError::InvalidState("host playback request off model thread"));}
        Ok(())
    }
    /// 请求宿主开始播放，不伪造实际播放状态。
    pub fn start(&self)->Result<(),AraError> {self.check()?;
        // SAFETY: 原线程且租约尚未撤销；HostClients直到模型销毁都保持宿主ref有效。
        unsafe {(self.endpoint.start)(self.endpoint.host as ARAPlaybackControllerHostRef)};Ok(())}
    /// 请求宿主暂停/停止，不在后台或音频线程执行。
    pub fn stop(&self)->Result<(),AraError> {self.check()?;
        // SAFETY: 与start相同的宿主线程及生命周期约束。
        unsafe {(self.endpoint.stop)(self.endpoint.host as ARAPlaybackControllerHostRef)};Ok(())}
    /// 请求宿主定位到有限秒位置。
    pub fn set_position(&self,position:f64)->Result<(),AraError> {self.check()?;
        if !position.is_finite() {return Err(AraError::InvalidArgument("playback position must be finite"));}
        // SAFETY: 与start相同；位置已经验证为有限值。
        unsafe {(self.endpoint.position)(self.endpoint.host as ARAPlaybackControllerHostRef,position)};Ok(())}
}
impl Drop for PlaybackAccess<'_> {
    fn drop(&mut self) {self.endpoint.alive.store(false,Ordering::Release);}
}

impl<'host> PlaybackAccess<'host> {
    pub(crate) unsafe fn from_raw(
        host_ref: ARAPlaybackControllerHostRef,
        interface: *const ARAPlaybackControllerInterface,
    ) -> Result<Option<Self>, AraError> {
        if interface.is_null() {
            return Ok(None);
        }
        if host_ref.is_null() {
            return Err(AraError::Abi("playback host reference is null"));
        }
        // SAFETY: caller supplies readable optional interface storage for the lifetime.
        let input = unsafe { SizedInput::from_ptr(interface) }?;
        macro_rules! required {
            ($field:ident, $type:ty, $extent:ident, $error:literal) => {{
                // SAFETY: generated offset/type/extent identify this callback field.
                unsafe {
                    input.copy_field::<Option<$type>>(
                        offset_of!(ARAPlaybackControllerInterface, $field),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
                .ok_or(AraError::Abi($error))?
            }};
        }
        let start=required!(requestStartPlayback,Void,ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_START_PLAYBACK,"start-playback callback is null");
        let stop=required!(requestStopPlayback,Void,ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_STOP_PLAYBACK,"stop-playback callback is null");
        let position=required!(requestSetPlaybackPosition,Position,ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_SET_PLAYBACK_POSITION,"set-position callback is null");
        Ok(Some(Self {
            endpoint:Arc::new(PlaybackEndpoint {host:host_ref as usize,start,stop,position,alive:AtomicBool::new(true),thread:std::thread::current().id()}),
            host_ref,
            start: required!(
                requestStartPlayback,
                Void,
                ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_START_PLAYBACK,
                "start-playback callback is null"
            ),
            stop: required!(
                requestStopPlayback,
                Void,
                ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_STOP_PLAYBACK,
                "stop-playback callback is null"
            ),
            position: required!(
                requestSetPlaybackPosition,
                Position,
                ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_SET_PLAYBACK_POSITION,
                "set-position callback is null"
            ),
            cycle: required!(
                requestSetCycleRange,
                Cycle,
                ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_SET_CYCLE_RANGE,
                "set-cycle callback is null"
            ),
            enable: required!(
                requestEnableCycle,
                Enable,
                ARAPLAYBACK_CONTROLLER_INTERFACE_REQUEST_ENABLE_CYCLE,
                "enable-cycle callback is null"
            ),
            _lifetime: PhantomData,
        }))
    }

    /// Requests playback start.
    pub fn start(&self) {
        // SAFETY: callback and host ref were validated during construction.
        unsafe { (self.start)(self.host_ref) };
    }
    /// 借出可存储租约；原HostClients销毁时自动撤销，不能延长宿主生命周期。
    pub fn request_handle(&self)->PlaybackRequestHandle {PlaybackRequestHandle {endpoint:self.endpoint.clone()}}

    /// Requests playback stop.
    pub fn stop(&self) {
        // SAFETY: callback and host ref were validated during construction.
        unsafe { (self.stop)(self.host_ref) };
    }

    /// Requests a new playback position in seconds.
    pub fn set_position(&self, position: f64) -> Result<(), AraError> {
        if !position.is_finite() {
            return Err(AraError::InvalidArgument(
                "playback position must be finite",
            ));
        }
        // SAFETY: callback and host ref were validated during construction.
        unsafe { (self.position)(self.host_ref, position) };
        Ok(())
    }

    /// Requests a finite, nonnegative cycle range.
    pub fn set_cycle(&self, start: f64, duration: f64) -> Result<(), AraError> {
        if !start.is_finite() || !duration.is_finite() || duration < 0.0 {
            return Err(AraError::InvalidArgument("cycle range is invalid"));
        }
        // SAFETY: callback and host ref were validated during construction.
        unsafe { (self.cycle)(self.host_ref, start, duration) };
        Ok(())
    }

    /// Requests enabling or disabling cycle playback.
    pub fn enable_cycle(&self, enabled: bool) {
        // SAFETY: callback and host ref were validated during construction.
        unsafe { (self.enable)(self.host_ref, AraBool::from(enabled).into_raw()) };
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;
    struct HostState {playing:bool,position:f64}
    unsafe extern "C" fn start(host:ARAPlaybackControllerHostRef) {
        // SAFETY: 本夹具的opaque ref一直指向存活HostState。
        unsafe {(*host.cast::<HostState>()).playing=true;}
    }
    unsafe extern "C" fn stop(host:ARAPlaybackControllerHostRef) {
        // SAFETY: 与start相同，整个测试期间存活。
        unsafe {(*host.cast::<HostState>()).playing=false;}
    }
    unsafe extern "C" fn position(host:ARAPlaybackControllerHostRef,value:f64) {
        // SAFETY: 与start相同，位置由产品入口验证。
        unsafe {(*host.cast::<HostState>()).position=value;}
    }
    unsafe extern "C" fn cycle(_:ARAPlaybackControllerHostRef,_:f64,_:f64) {}
    unsafe extern "C" fn enable(_:ARAPlaybackControllerHostRef,_:ARABool) {}
    #[test]
    fn request_handle_controls_only_live_host_on_original_thread() {
        let mut host=HostState {playing:false,position:0.};
        let interface=ARAPlaybackControllerInterface {structSize:std::mem::size_of::<ARAPlaybackControllerInterface>(),
            requestStartPlayback:Some(start),requestStopPlayback:Some(stop),requestSetPlaybackPosition:Some(position),
            requestSetCycleRange:Some(cycle),requestEnableCycle:Some(enable)};
        // SAFETY: 完整接口及opaque ref存活到client销毁，回调签名与锁定ARA ABI一致。
        let client=unsafe {PlaybackAccess::from_raw((&raw mut host).cast(),&interface)}.unwrap().unwrap();
        let request=client.request_handle();
        request.start().unwrap();assert!(host.playing);
        request.set_position(2.).unwrap();assert_eq!(host.position,2.);
        request.stop().unwrap();assert!(!host.playing);
        assert!(request.set_position(f64::NAN).is_err());assert_eq!(host.position,2.);
        let other=request.clone();assert!(std::thread::spawn(move||other.start().is_err()).join().unwrap());assert!(!host.playing);
        drop(client);assert!(request.start().is_err());assert!(!host.playing);
    }
}
