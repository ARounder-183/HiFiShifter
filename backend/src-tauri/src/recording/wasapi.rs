//! Native WASAPI capture engine (Windows).
//!
//! Why not cpal?
//!
//! * cpal's WASAPI loopback never inspects `AUDCLNT_BUFFERFLAGS_SILENT`.
//!   When the endpoint is idle the audio engine marks packets as silent and
//!   leaves the buffer contents *undefined*; writing those bytes produces the
//!   hiss/rustle users hear with cpal but not with other recorders.
//! * Application-specific capture requires the process-loopback activation
//!   API (`ActivateAudioInterfaceAsync` + `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS`,
//!   Windows 10 build 20348+), which cpal does not expose. On older builds we
//!   fall back to capturing the system mix while muting every other session
//!   (the same technique used by OBS-style application capture).
//! * Asking the engine to convert (AUTOCONVERTPCM + SRC_DEFAULT_QUALITY)
//!   yields the exact sample rate/channels the user configured without extra
//!   software resampling. If a device refuses that, we fall back to its mix
//!   format and convert in software.

use super::capture::{
    convert_channels, AppAudioInfo, AudioDeviceInfo, CaptureContext, CapturePlan, RateConverter,
};
use std::collections::HashMap;
use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::Duration;
use windows::core::{implement, Interface, IUnknown, PCSTR, PWSTR};
use windows::Win32::Foundation::{CloseHandle, E_POINTER, HANDLE, RPC_E_CHANGED_MODE};
use windows::Win32::Media::Audio::{
    self, ActivateAudioInterfaceAsync, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
    AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    AudioSessionStateActive, AudioSessionStateExpired, AudioSessionStateInactive,
    DEVICE_STATE_ACTIVE,
    IActivateAudioInterfaceAsyncOperation, IActivateAudioInterfaceCompletionHandler,
    IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, IAudioSessionControl2, IAudioSessionManager2,
    ISimpleAudioVolume, IMMDevice, IMMDeviceEnumerator,
    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
    WAVEFORMATEX, WAVEFORMATEXTENSIBLE, WAVEFORMATEXTENSIBLE_0, eConsole, eRender,
};
use windows::Win32::Media::KernelStreaming::{
    KSDATAFORMAT_SUBTYPE_PCM, SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT,
    WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::{
    KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED, STGM_READ, StructuredStorage,
};
use windows::Win32::System::Threading::{
    CreateEventA, OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
    QueryFullProcessImageNameW, WaitForSingleObject,
};
use windows::Win32::System::Variant::VT_LPWSTR;
use windows::Win32::Devices::Properties;

/// Runs system-sound (loopback) capture until the stop signal is set.
pub fn run_loopback_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let CapturePlan::Loopback {
        device_id,
        sample_rate,
        channels,
    } = plan
    else {
        unreachable!()
    };
    unsafe {
        let _com = ComGuard::init()?;
        let device = get_endpoint_device(&device_id)?;
        let client: IAudioClient = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
        run_client_capture(client, sample_rate, channels, ctx.as_ref(), ready_tx)
    }
}

/// Runs application-specific capture until the stop signal is set.
pub fn run_app_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let CapturePlan::Application {
        pid,
        sample_rate,
        channels,
        ..
    } = plan
    else {
        unreachable!()
    };
    unsafe {
        let _com = ComGuard::init()?;
        match activate_process_loopback(pid) {
            Ok(client) => {
                // Preferred path (Windows 10 build 20348+): the OS itself
                // filters the loopback stream to the target process.
                run_client_capture(client, sample_rate, channels, ctx.as_ref(), ready_tx)
            }
            Err(primary_err) => {
                eprintln!(
                    "[recording] process loopback unavailable, using session-muting fallback: {primary_err}"
                );
                run_app_capture_fallback(pid, sample_rate, channels, ctx, ready_tx)
            }
        }
    }
}

/// Shared capture lifecycle: initialize, start, drain, stop.
unsafe fn run_client_capture(
    client: IAudioClient,
    sample_rate: u32,
    channels: u16,
    ctx: &CaptureContext,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let (capture, event, output) = init_capture_client(&client, sample_rate, channels)?;
    client
        .Start()
        .map_err(|e| format!("recording_error_play_input:{e}"))?;
    let _ = ready_tx.send(Ok(()));
    let result = run_capture_loop(capture, event, ctx, output);
    let _ = client.Stop();
    let _ = CloseHandle(event);
    result
}

// ---------------------------------------------------------------------------
// Process loopback activation
// ---------------------------------------------------------------------------

#[implement(IActivateAudioInterfaceCompletionHandler)]
struct ActivationHandler {
    tx: mpsc::Sender<windows::core::Result<IUnknown>>,
}

impl IActivateAudioInterfaceCompletionHandler_Impl for ActivationHandler_Impl {
    fn ActivateCompleted(
        &self,
        operation: Option<&IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        let this = &self.this;
        let Some(operation) = operation else {
            let _ = this
                .tx
                .send(Err(windows::core::Error::from_hresult(E_POINTER)));
            return Ok(());
        };
        let mut result: windows::core::HRESULT = windows::core::HRESULT(0);
        let mut unknown: Option<IUnknown> = None;
        unsafe {
            operation.GetActivateResult(&mut result, &mut unknown)?;
        }
        if result.is_ok() {
            match unknown {
                Some(interface) => {
                    let _ = this.tx.send(Ok(interface));
                }
                None => {
                    let _ = this
                        .tx
                        .send(Err(windows::core::Error::from_hresult(E_POINTER)));
                }
            }
        } else {
            let _ = this.tx.send(Err(windows::core::Error::from_hresult(result)));
        }
        Ok(())
    }
}

unsafe fn activate_process_loopback(pid: u32) -> Result<IAudioClient, String> {
    let params = AUDIOCLIENT_ACTIVATION_PARAMS {
        ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
            ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                TargetProcessId: pid,
                ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            },
        },
    };
    // The activation parameters are passed as a VT_BLOB PROPVARIANT. The
    // pointer must stay valid until the async activation completes, so both
    // values live in this frame until `recv_timeout` returns. We build the
    // raw layout directly (instead of `PROPVARIANT::from_raw`) because the
    // wrapper's `Drop` would call PropVariantClear on our stack blob.
    let raw_prop = windows::core::imp::PROPVARIANT {
        Anonymous: windows::core::imp::PROPVARIANT_0 {
            Anonymous: windows::core::imp::PROPVARIANT_0_0 {
                vt: 65u16, // VT_BLOB
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: windows::core::imp::PROPVARIANT_0_0_0 {
                    blob: windows::core::imp::BLOB {
                        cbSize: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
                        pBlobData: &params as *const _ as *mut _,
                    },
                },
            },
        },
    };
    let (tx, rx) = mpsc::channel::<windows::core::Result<IUnknown>>();
    let handler: IActivateAudioInterfaceCompletionHandler = ActivationHandler { tx }.into();
    let operation = ActivateAudioInterfaceAsync(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        &IAudioClient::IID,
        Some(&raw_prop as *const _ as *const windows::core::PROPVARIANT),
        &handler,
    )
    .map_err(|e| format!("recording_error_app_activation:{e}"))?;
    // Keep the operation alive until the handler has run.
    let _operation = operation;

    let unknown = rx
        .recv_timeout(Duration::from_secs(6))
        .map_err(|_| "recording_error_app_activation_timeout".to_string())?
        .map_err(|e| format!("recording_error_app_activation:{e}"))?;
    unknown
        .cast()
        .map_err(|e| format!("recording_error_app_activation:{e}"))
}

// ---------------------------------------------------------------------------
// Capture client initialization
// ---------------------------------------------------------------------------

struct MixFormatGuard(*mut WAVEFORMATEX);

impl Drop for MixFormatGuard {
    fn drop(&mut self) {
        unsafe {
            CoTaskMemFree(Some(self.0 as *const _));
        }
    }
}

unsafe fn build_float_format(sample_rate: u32, channels: u16) -> WAVEFORMATEXTENSIBLE {
    let block_align = channels * 4;
    let channel_mask = if channels == 1 {
        SPEAKER_FRONT_CENTER
    } else {
        SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
    };
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
            nChannels: channels,
            nSamplesPerSec: sample_rate,
            nAvgBytesPerSec: sample_rate * block_align as u32,
            nBlockAlign: block_align,
            wBitsPerSample: 32,
            cbSize: (std::mem::size_of::<WAVEFORMATEXTENSIBLE>()
                - std::mem::size_of::<WAVEFORMATEX>()) as u16,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: 32 },
        dwChannelMask: channel_mask,
        SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    }
}

enum NativeDecoder {
    F32,
    I16,
    I32,
    I64,
    U8,
}

impl NativeDecoder {
    unsafe fn decode(&self, data: *const u8, count: usize) -> Vec<f32> {
        match self {
            NativeDecoder::F32 => {
                slice::from_raw_parts(data as *const f32, count).to_vec()
            }
            NativeDecoder::I16 => slice::from_raw_parts(data as *const i16, count)
                .iter()
                .map(|v| *v as f32 / 32768.0)
                .collect(),
            NativeDecoder::I32 => slice::from_raw_parts(data as *const i32, count)
                .iter()
                .map(|v| *v as f32 / 2_147_483_648.0)
                .collect(),
            NativeDecoder::I64 => slice::from_raw_parts(data as *const i64, count)
                .iter()
                .map(|v| (*v as f64 / 9_223_372_036_854_775_808.0) as f32)
                .collect(),
            NativeDecoder::U8 => slice::from_raw_parts(data as *const u8, count)
                .iter()
                .map(|v| (*v as f32 - 128.0) / 128.0)
                .collect(),
        }
    }
}

unsafe fn decode_mix_format(
    ptr: *const WAVEFORMATEX,
) -> Option<(NativeDecoder, u16, u32)> {
    let format = &*ptr;
    let channels = format.nChannels;
    let rate = format.nSamplesPerSec;
        let decoder = match (format.wBitsPerSample, format.wFormatTag as u32) {
        (8, Audio::WAVE_FORMAT_PCM) => NativeDecoder::U8,
        (16, Audio::WAVE_FORMAT_PCM) => NativeDecoder::I16,
        (32, WAVE_FORMAT_IEEE_FLOAT) => NativeDecoder::F32,
        (n_bits, WAVE_FORMAT_EXTENSIBLE) => {
            let extensible = ptr::read_unaligned(ptr as *const WAVEFORMATEXTENSIBLE);
            let sub_format = unsafe { ptr::addr_of!(extensible.SubFormat).read_unaligned() };
            if sub_format == KSDATAFORMAT_SUBTYPE_PCM {
                match n_bits {
                    8 => NativeDecoder::U8,
                    16 => NativeDecoder::I16,
                    32 => NativeDecoder::I32,
                    64 => NativeDecoder::I64,
                    _ => return None,
                }
            } else if n_bits == 32 && sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {
                NativeDecoder::F32
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some((decoder, channels, rate))
}

/// What the capture loop must do with each raw WASAPI packet.
enum CaptureOutput {
    /// Engine delivers f32 at the requested rate/channels.
    NativeF32 { channels: u16 },
    /// Engine delivers the endpoint mix format; convert in software.
    Converted {
        decoder: NativeDecoder,
        channels_in: u16,
        channels_out: u16,
        resampler: Option<RateConverter>,
    },
}

impl CaptureOutput {
    unsafe fn push(
        &mut self,
        data: *mut u8,
        frames: usize,
        silent: bool,
        ctx: &CaptureContext,
    ) {
        match self {
            CaptureOutput::NativeF32 { channels } => {
                let count = frames * *channels as usize;
                if silent {
                    ctx.push(&vec![0.0f32; count]);
                } else {
                    ctx.push(slice::from_raw_parts(data as *const f32, count));
                }
            }
            CaptureOutput::Converted {
                decoder,
                channels_in,
                channels_out,
                resampler,
            } => {
                let count = frames * *channels_in as usize;
                let decoded = if silent {
                    vec![0.0f32; count]
                } else {
                    decoder.decode(data, count)
                };
                let converted =
                    convert_channels(&decoded, *channels_in as usize, *channels_out as usize);
                if let Some(resampler) = resampler {
                    if let Ok(out) = resampler.push_interleaved(&converted, *channels_out as usize) {
                        ctx.push(&out);
                    }
                } else {
                    ctx.push(&converted);
                }
            }
        }
    }

    /// Flushes any samples still sitting in the software resampler's input
    /// buffer when capture stops, so the final ~20 ms of audio is not lost.
    fn finish(&mut self, ctx: &CaptureContext) {
        if let CaptureOutput::Converted {
            channels_out,
            resampler: Some(converter),
            ..
        } = self
        {
            let out = converter.flush_interleaved(*channels_out as usize, *channels_out as usize);
            if !out.is_empty() {
                ctx.push(&out);
            }
        }
    }
}

unsafe fn init_capture_client(
    client: &IAudioClient,
    sample_rate: u32,
    channels: u16,
) -> Result<(IAudioCaptureClient, HANDLE, CaptureOutput), String> {
    // Preferred path: let the audio engine convert to the requested format.
    let requested = build_float_format(sample_rate, channels);
    let convert_flags = AUDCLNT_STREAMFLAGS_LOOPBACK
        | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
        | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    let output = if client
        .Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            convert_flags,
            0,
            0,
            &requested.Format,
            None,
        )
        .is_ok()
    {
        CaptureOutput::NativeF32 { channels }
    } else {
        // Fallback: use the endpoint mix format and convert in software.
        let mix = client
            .GetMixFormat()
            .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
        let _mix_guard = MixFormatGuard(mix);
        let (decoder, mix_channels, mix_rate) =
            decode_mix_format(mix).ok_or_else(|| {
                "recording_error_wasapi_init:unsupported mix format".to_string()
            })?;
        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                0,
                0,
                mix,
                None,
            )
            .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
        let resampler = if mix_rate != sample_rate {
            Some(RateConverter::new(
                mix_rate,
                sample_rate,
                channels as usize,
                channels as usize,
            )?)
        } else {
            None
        };
        CaptureOutput::Converted {
            decoder,
            channels_in: mix_channels,
            channels_out: channels,
            resampler,
        }
    };

    let capture = client
        .GetService::<IAudioCaptureClient>()
        .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
    let event = CreateEventA(None, false, false, PCSTR(ptr::null()))
        .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
    client
        .SetEventHandle(event)
        .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
    Ok((capture, event, output))
}

unsafe fn run_capture_loop(
    capture: IAudioCaptureClient,
    event: HANDLE,
    ctx: &CaptureContext,
    mut output: CaptureOutput,
) -> Result<(), String> {
    let mut consecutive_errors = 0u32;
    loop {
        if ctx.stop.load(Ordering::Relaxed) {
            break;
        }
        // Event-driven with a polling fallback: some older systems never
        // signal the loopback event, and the 100 ms timeout keeps the loop
        // responsive for stop requests as well.
        WaitForSingleObject(event, 100);

        loop {
            let mut data: *mut u8 = ptr::null_mut();
            let mut frames: u32 = 0;
            let mut flags: u32 = 0;
            if let Err(err) =
                capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)
            {
                consecutive_errors += 1;
                if consecutive_errors > 50 {
                    return Err(format!("recording_error_wasapi_read:{err}"));
                }
                break;
            }
            consecutive_errors = 0;
            if frames == 0 {
                break;
            }
            // THE hiss fix: when the engine marks a packet SILENT the buffer
            // bytes are undefined. cpal writes them anyway; we write zeros.
            let silent = flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
            output.push(data, frames as usize, silent, ctx);
            if capture.ReleaseBuffer(frames).is_err() {
                break;
            }
        }
    }
    output.finish(ctx);
    Ok(())
}

// ---------------------------------------------------------------------------
// Session-muting fallback for Windows builds older than 20348
// ---------------------------------------------------------------------------

struct MuteGuard {
    sessions: Vec<(ISimpleAudioVolume, bool)>,
}

impl Drop for MuteGuard {
    fn drop(&mut self) {
        for (volume, previous) in &self.sessions {
            unsafe {
                let _ = volume.SetMute(*previous, ptr::null());
            }
        }
    }
}

unsafe fn run_app_capture_fallback(
    pid: u32,
    sample_rate: u32,
    channels: u16,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let enumerator = create_enumerator()
        .map_err(|e| format!("recording_error_enumeration:{e}"))?;
    let device = enumerator
        .GetDefaultAudioEndpoint(eRender, eConsole)
        .map_err(|e| format!("recording_error_loopback_not_found:{e}"))?;
    let manager: IAudioSessionManager2 = device
        .Activate(CLSCTX_ALL, None)
        .map_err(|e| format!("recording_error_app_activation:{e}"))?;

    let mut mute_guard = MuteGuard {
        sessions: Vec::new(),
    };
    let mut target_found = false;
    if let Ok(sessions) = manager.GetSessionEnumerator() {
        let count = sessions.GetCount().unwrap_or(0).max(0);
        for index in 0..count {
            if let Ok(session) = sessions.GetSession(index) {
                let Ok(session2) = session.cast::<IAudioSessionControl2>() else {
                    continue;
                };
                let session_pid = session2.GetProcessId().unwrap_or(0);
                if session_pid == pid {
                    target_found = true;
                    continue;
                }
                if session_pid == 0 {
                    continue;
                }
                if let Ok(volume) = session.cast::<ISimpleAudioVolume>() {
                    let previous = volume.GetMute().map(|value| value.as_bool()).unwrap_or(false);
                    if !previous {
                        let _ = volume.SetMute(true, ptr::null());
                    }
                    mute_guard.sessions.push((volume, previous));
                }
            }
        }
    }
    if !target_found {
        return Err("recording_error_app_not_found".to_string());
    }

    let client: IAudioClient = device
        .Activate(CLSCTX_ALL, None)
        .map_err(|e| format!("recording_error_wasapi_init:{e}"))?;
    let (capture, event, output) = match init_capture_client(&client, sample_rate, channels) {
        Ok(value) => value,
        Err(err) => return Err(err),
    };
    client
        .Start()
        .map_err(|e| format!("recording_error_play_input:{e}"))?;
    let _ = ready_tx.send(Ok(()));
    let result = run_capture_loop(capture, event, ctx.as_ref(), output);
    let _ = client.Stop();
    let _ = CloseHandle(event);
    drop(mute_guard);
    result
}

// ---------------------------------------------------------------------------
// Endpoint / session enumeration
// ---------------------------------------------------------------------------

unsafe fn create_enumerator() -> windows::core::Result<IMMDeviceEnumerator> {
    CoCreateInstance::<_, IMMDeviceEnumerator>(&Audio::MMDeviceEnumerator, None, CLSCTX_ALL)
}

unsafe fn device_id_string(device: &IMMDevice) -> Option<String> {
    let id = device.GetId().ok()?;
    let text = pwstr_to_string(id);
    CoTaskMemFree(Some(id.as_ptr() as *const _));
    text
}

unsafe fn pwstr_to_string(ptr: PWSTR) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let mut len = 0usize;
    while *ptr.as_ptr().add(len) != 0 {
        len += 1;
    }
    let wide = slice::from_raw_parts(ptr.as_ptr(), len);
    let os_string = OsString::from_wide(wide);
    Some(os_string.to_string_lossy().into_owned())
}

unsafe fn device_friendly_name(device: &IMMDevice) -> Option<String> {
    let property_store = device.OpenPropertyStore(STGM_READ).ok()?;
    let mut property_value = property_store
        .GetValue(&Properties::DEVPKEY_Device_FriendlyName as *const _ as *const _)
        .ok()?;
    let prop_variant = &property_value.as_raw().Anonymous.Anonymous;
    if prop_variant.vt != VT_LPWSTR.0 {
        return None;
    }
    let ptr_utf16 = *(&prop_variant.Anonymous as *const _ as *const *const u16);
    let text = pwstr_to_string(PWSTR(ptr_utf16 as *mut u16));
    StructuredStorage::PropVariantClear(&mut property_value).ok();
    text
}

unsafe fn get_endpoint_device(device_id: &str) -> Result<IMMDevice, String> {
    let enumerator = create_enumerator()
        .map_err(|e| format!("recording_error_enumeration:{e}"))?;
    let id = device_id.strip_prefix("loopback:").unwrap_or(device_id);
    if id.is_empty() || id == "default" {
        return enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(|e| format!("recording_error_loopback_not_found:{e}"));
    }
    let wide: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
    if let Ok(device) = enumerator.GetDevice(windows::core::PCWSTR(wide.as_ptr())) {
        return Ok(device);
    }
    // Legacy configs store device names instead of endpoint ids.
    if let Ok(collection) = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) {
        if let Ok(count) = collection.GetCount() {
            for index in 0..count {
                if let Ok(device) = collection.Item(index) {
                    if device_friendly_name(&device).as_deref() == Some(id) {
                        return Ok(device);
                    }
                }
            }
        }
    }
    Err("recording_error_loopback_not_found".to_string())
}

/// Lists output endpoints as selectable loopback sources.
pub fn enumerate_loopback_devices() -> Vec<AudioDeviceInfo> {
    let Ok(com) = ComGuard::init() else {
        return Vec::new();
    };
    let _com = com;
    unsafe {
        let Ok(enumerator) = create_enumerator() else {
            return Vec::new();
        };
        let default_id = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .ok()
            .and_then(|device| device_id_string(&device));
        let mut devices = vec![AudioDeviceInfo {
            id: "loopback:default".to_string(),
            name: "System Default Output".to_string(),
            kind: "output".to_string(),
            is_default: true,
            is_loopback: true,
        }];
        if let Ok(collection) = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) {
            if let Ok(count) = collection.GetCount() {
                for index in 0..count {
                    if let Ok(device) = collection.Item(index) {
                        let id = device_id_string(&device).unwrap_or_default();
                        if id.is_empty() {
                            continue;
                        }
                        let name = device_friendly_name(&device)
                            .unwrap_or_else(|| "Unknown".to_string());
                        devices.push(AudioDeviceInfo {
                            id: format!("loopback:{id}"),
                            name,
                            kind: "output".to_string(),
                            is_default: default_id.as_deref() == Some(id.as_str()),
                            is_loopback: true,
                        });
                    }
                }
            }
        }
        devices
    }
}

fn process_exe_name(pid: u32) -> Option<String> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buffer = [0u16; 1024];
        let mut size = buffer.len() as u32;
        let result = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut size,
        );
        let _ = CloseHandle(handle);
        if result.is_err() {
            return None;
        }
        let path = String::from_utf16_lossy(&buffer[..size as usize]);
        Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    }
}

/// Lists applications with audio sessions on the default render endpoint.
pub fn enumerate_applications() -> Vec<AppAudioInfo> {
    let Ok(com) = ComGuard::init() else {
        return Vec::new();
    };
    let _com = com;
    unsafe {
        let Ok(enumerator) = create_enumerator() else {
            return Vec::new();
        };
        let Ok(device) = enumerator.GetDefaultAudioEndpoint(eRender, eConsole) else {
            return Vec::new();
        };
        let Ok(manager) = device.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) else {
            return Vec::new();
        };
        let Ok(sessions) = manager.GetSessionEnumerator() else {
            return Vec::new();
        };
        let mut apps: HashMap<u32, (String, String, bool)> = HashMap::new();
        let count = sessions.GetCount().unwrap_or(0).max(0);
        for index in 0..count {
            let Ok(session) = sessions.GetSession(index) else {
                continue;
            };
            let Ok(session2) = session.cast::<IAudioSessionControl2>() else {
                continue;
            };
            let Ok(pid) = session2.GetProcessId() else {
                continue;
            };
            if pid == 0 {
                continue;
            }
            let state = session2.GetState().unwrap_or(AudioSessionStateInactive);
            if state == AudioSessionStateExpired {
                continue;
            }
            let display = session2
                .GetDisplayName()
                .ok()
                .and_then(|ptr| pwstr_to_string(ptr))
                .unwrap_or_default();
            let process_name = process_exe_name(pid).unwrap_or_default();
            let name = if !display.trim().is_empty() {
                display.trim().to_string()
            } else if !process_name.is_empty() {
                process_name.clone()
            } else {
                format!("Application (PID {pid})")
            };
            let entry = apps
                .entry(pid)
                .or_insert_with(|| (name.clone(), process_name.clone(), false));
            if state == AudioSessionStateActive {
                entry.2 = true;
            }
        }
        let mut result: Vec<AppAudioInfo> = apps
            .into_iter()
            .map(|(pid, (name, process_name, is_active))| AppAudioInfo {
                id: format!("pid:{pid}"),
                name,
                process_name,
                pid,
                is_active,
            })
            .collect();
        result.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        result
    }
}

// ---------------------------------------------------------------------------
// COM lifecycle
// ---------------------------------------------------------------------------

struct ComGuard {
    uninitialize: bool,
}

impl ComGuard {
    fn init() -> Result<Self, String> {
        unsafe {
            let result = CoInitializeEx(None, COINIT_MULTITHREADED);
            if result.is_ok() {
                Ok(ComGuard {
                    uninitialize: result == windows::core::HRESULT(0),
                })
            } else if result == RPC_E_CHANGED_MODE {
                // Already initialized in another apartment mode; proceed
                // without balancing the call.
                Ok(ComGuard {
                    uninitialize: false,
                })
            } else {
                Err(format!("recording_error_com_init:{result}"))
            }
        }
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.uninitialize {
            unsafe {
                CoUninitialize();
            }
        }
    }
}
