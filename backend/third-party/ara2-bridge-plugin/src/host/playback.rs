//! Optional host playback-request client.

use ara2_bridge_core::{AraBool, AraError, SizedInput};
use ara2_bridge_sys::*;
use std::marker::PhantomData;
use std::mem::offset_of;

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
        Ok(Some(Self {
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
