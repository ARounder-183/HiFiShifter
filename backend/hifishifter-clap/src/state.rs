//! CLAP state：把宿主的不透明字节块交给 core 的 `ComponentState` 容器。
//!
//! 【为什么 core 认识字节格式】`ComponentState` 是产品存档格式，与插件格式无关；
//! CLAP 与 VST3 共用同一份魔数/版本/三段载荷，跨格式复制粘贴实例仍然可读。
//!
//! 【为什么流读写必须循环】`clap/stream.h` 明确写了宿主可以限制单次读/写的字节数，
//! 并提示「You need to use a loop to ensure that you read or write the entirety of your
//! state」。单次 `write` 只写出一部分就返回 true 的相反面是**保存失败**；短读立刻
//! 退出则会把尾部截掉，重载后编辑凭空少一截。两者在宿主日志里都只是"存档失败"，
//! 没有别的线索，因此这里把短读/短写都当成正常进度继续推进。

use crate::abi::clap_plugin_t;
use hifishifter_plugin_core::component_state::ComponentState;
use hifishifter_plugin_core::format::FormatState;
use std::ffi::c_void;
use std::sync::Mutex;

/// 组件 state 的硬上限；与 `ComponentState` 的 64 MiB 约定一致。
///
/// 【为什么需要上限】宿主若一直返回同一个正数而从不给 EOF，循环就永不结束；
/// 音频插件不能把宿主线程挂死。超过上限按明确错误返回，而不是静默截断。
const MAX_STATE_BYTES: usize = 64 * 1024 * 1024;

/// 一个 CLAP 实例的持久状态。
#[derive(Default)]
pub struct InstanceState(Mutex<Vec<u8>>);

impl FormatState for InstanceState {
    fn load(&self, bytes: &[u8]) -> Result<(), String> {
        // 验证容器可读，再保存原始字节：宿主拿到的与存进来的逐字相同。
        ComponentState::decode(bytes)?;
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = bytes.to_vec();
        Ok(())
    }
    fn save(&self) -> Result<Vec<u8>, String> {
        let stored = self.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if stored.is_empty() {
            // 【为什么新实例也要写出字节】宿主把"0 字节"理解成"这个插件没有 state"，
            // 而解码端又会把零字节当成裸 `EditState`（历史分支）。同一个工程在 CLAP
            // 与 VST3 之间来回打开时，两边就会给出不同的字节块。写一个合法的空容器，
            // 两个格式对"空状态"的表示就一致了。
            return Ok(ComponentState::default().encode());
        }
        Ok(stored)
    }
}

/// `clap_ostream_t`（`.ctx` 必须在函数指针之前，见 clap/stream.h）。
#[repr(C)]
pub struct clap_ostream_t {
    pub ctx: *mut c_void,
    pub write: Option<unsafe extern "C" fn(*const clap_ostream_t, *const c_void, u64) -> i64>,
}
/// `clap_istream_t`。
#[repr(C)]
pub struct clap_istream_t {
    pub ctx: *mut c_void,
    pub read: Option<unsafe extern "C" fn(*const clap_istream_t, *mut c_void, u64) -> i64>,
}
/// `clap_plugin_state_t`。
#[repr(C)]
pub struct clap_plugin_state_t {
    pub save: Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_ostream_t) -> bool>,
    pub load: Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_istream_t) -> bool>,
}

/// 读完宿主输入流；`0` = EOF，负数 = IO 错误，正数 = 本次实际读到的字节数。
///
/// # Safety
/// `stream` 与其 `ctx` 必须在整个调用期间有效；`read` 回调须遵守 CLAP 的流契约。
pub unsafe fn read_all(stream: *const clap_istream_t) -> Result<Vec<u8>, String> {
    let Some(stream) = (unsafe { stream.as_ref() }) else {
        return Err("null CLAP input stream".into());
    };
    let Some(read) = stream.read else {
        return Err("CLAP input stream has no read callback".into());
    };
    let mut out = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let count = unsafe { read(stream, chunk.as_mut_ptr().cast(), chunk.len() as u64) };
        if count < 0 {
            return Err("CLAP input stream reported a read error".into());
        }
        if count == 0 {
            // 只有宿主明确给 0 才是 EOF；短读是正常进度，不能当成结束。
            break;
        }
        let count = (count as usize).min(chunk.len());
        if out.len().saturating_add(count) > MAX_STATE_BYTES {
            return Err("CLAP state exceeds the 64 MiB limit".into());
        }
        out.extend_from_slice(&chunk[..count]);
    }
    Ok(out)
}

/// 写满宿主输出流；每次只推进 `write` 实际接受的字节数。
///
/// # Safety
/// `stream` 与其 `ctx` 必须在整个调用期间有效；`write` 回调须遵守 CLAP 的流契约。
pub unsafe fn write_all(stream: *const clap_ostream_t, bytes: &[u8]) -> bool {
    let Some(stream) = (unsafe { stream.as_ref() }) else {
        return false;
    };
    let Some(write) = stream.write else {
        return false;
    };
    let mut offset = 0usize;
    while offset < bytes.len() {
        let remaining = &bytes[offset..];
        let count = unsafe { write(stream, remaining.as_ptr().cast(), remaining.len() as u64) };
        // 0 表示毫无进展，负数是写错误；两者都必须判失败，否则会死循环或丢尾部。
        if count <= 0 {
            return false;
        }
        offset += (count as usize).min(remaining.len());
    }
    true
}

pub static STATE_EXT: clap_plugin_state_t = clap_plugin_state_t {
    save: Some(state_save),
    load: Some(state_load),
};

unsafe extern "C" fn state_save(
    plugin: *const clap_plugin_t,
    stream: *const clap_ostream_t,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(instance) = (unsafe { crate::instance(plugin) }) else {
            return false;
        };
        let Ok(bytes) = instance.state.save() else {
            return false;
        };
        unsafe { write_all(stream, &bytes) }
    })
}

unsafe extern "C" fn state_load(
    plugin: *const clap_plugin_t,
    stream: *const clap_istream_t,
) -> bool {
    hifishifter_plugin_core::guard_ffi(false, || {
        let Some(instance) = crate::instance(plugin) else {
            return false;
        };
        let bytes = match unsafe { read_all(stream) } {
            Ok(bytes) => bytes,
            Err(error) => {
                log::warn!("[clap] state.load rejected an unreadable stream: {error}");
                return false;
            }
        };
        instance.state.load(&bytes).is_ok()
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for clap_plugin_state_t {}
unsafe impl Send for clap_plugin_state_t {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instance;

    /// 测试宿主：把每次读/写限制在 `max_chunk` 字节内，并可在指定调用次数上报错。
    struct TestStream {
        data: Vec<u8>,
        max_chunk: usize,
        read_offset: usize,
        fail_on_call: Option<usize>,
        calls: usize,
    }

    unsafe extern "C" fn short_write(
        stream: *const clap_ostream_t,
        buffer: *const c_void,
        size: u64,
    ) -> i64 {
        let state = unsafe { &mut *(*stream).ctx.cast::<TestStream>() };
        state.calls += 1;
        if state.fail_on_call == Some(state.calls) {
            return -1;
        }
        let n = (size as usize).min(state.max_chunk);
        if n == 0 {
            return 0;
        }
        let src = unsafe { std::slice::from_raw_parts(buffer.cast::<u8>(), n) };
        state.data.extend_from_slice(src);
        n as i64
    }

    unsafe extern "C" fn short_read(
        stream: *const clap_istream_t,
        buffer: *mut c_void,
        size: u64,
    ) -> i64 {
        let state = unsafe { &mut *(*stream).ctx.cast::<TestStream>() };
        state.calls += 1;
        if state.fail_on_call == Some(state.calls) {
            return -1;
        }
        let remaining = state.data.len().saturating_sub(state.read_offset);
        if remaining == 0 {
            return 0;
        }
        let n = (size as usize).min(state.max_chunk).min(remaining);
        let dst = unsafe { std::slice::from_raw_parts_mut(buffer.cast::<u8>(), n) };
        dst.copy_from_slice(&state.data[state.read_offset..state.read_offset + n]);
        state.read_offset += n;
        n as i64
    }

    fn container() -> Vec<u8> {
        ComponentState {
            edits: vec![7; 517],
            capture: Some(br#"{"takes":[{"id":"a"}]}"#.to_vec()),
            capture_edits: Some(br#"{"notes":[]}"#.to_vec()),
        }
        .encode()
    }

    fn ostream(state: &mut TestStream) -> clap_ostream_t {
        clap_ostream_t {
            ctx: (state as *mut TestStream).cast(),
            write: Some(short_write),
        }
    }
    fn istream(state: &mut TestStream) -> clap_istream_t {
        clap_istream_t {
            ctx: (state as *mut TestStream).cast(),
            read: Some(short_read),
        }
    }

    #[test]
    fn partial_writes_are_retried_until_the_whole_container_is_written() {
        let bytes = container();
        let mut state = TestStream {
            data: Vec::new(),
            max_chunk: 7,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let stream = ostream(&mut state);
        assert!(unsafe { write_all(&stream, &bytes) });
        assert_eq!(state.data, bytes, "短写宿主必须收到完整容器");
        assert!(state.calls > 1, "单块写完说明没有真正模拟短写宿主");
    }

    #[test]
    fn partial_reads_do_not_truncate_the_container() {
        let bytes = container();
        let mut state = TestStream {
            data: bytes.clone(),
            max_chunk: 5,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let stream = istream(&mut state);
        let read = unsafe { read_all(&stream) }.expect("short reads must still reach EOF");
        assert_eq!(read, bytes, "短读不能被误判成 EOF");
        assert!(state.calls > 1);
    }

    #[test]
    fn a_write_error_fails_the_save_instead_of_accepting_a_prefix() {
        let bytes = container();
        let mut state = TestStream {
            data: Vec::new(),
            max_chunk: 64,
            read_offset: 0,
            fail_on_call: Some(2),
            calls: 0,
        };
        let stream = ostream(&mut state);
        assert!(!unsafe { write_all(&stream, &bytes) });
    }

    #[test]
    fn a_read_error_fails_the_load_instead_of_accepting_a_prefix() {
        let mut state = TestStream {
            data: container(),
            max_chunk: 64,
            read_offset: 0,
            fail_on_call: Some(2),
            calls: 0,
        };
        let stream = istream(&mut state);
        let error = unsafe { read_all(&stream) }.unwrap_err();
        assert!(error.contains("read error"), "{error}");
    }

    #[test]
    fn a_zero_length_write_is_an_error_not_an_infinite_loop() {
        let mut state = TestStream {
            data: Vec::new(),
            max_chunk: 0,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let stream = ostream(&mut state);
        assert!(!unsafe { write_all(&stream, b"payload") });
    }

    #[test]
    fn an_empty_stream_is_a_valid_end_of_state() {
        let mut state = TestStream {
            data: Vec::new(),
            max_chunk: 64,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let stream = istream(&mut state);
        assert_eq!(unsafe { read_all(&stream) }.unwrap(), Vec::<u8>::new());
    }

    /// 端到端：真实的 `clap_plugin_t` → `state_save` → 短写流 → 短读流 →
    /// `state_load`，验证 C 边界上的部分读写不会破坏跨格式存档容器。
    #[test]
    fn state_round_trips_through_the_c_abi_with_short_streams() {
        let instance = Box::into_raw(Box::new(Instance {
            state: InstanceState::default(),
            audio: crate::audio::AudioState::default(),
            host: std::ptr::null(),
            plugin: std::ptr::null(),
            ara: None,
            initialized: std::sync::atomic::AtomicBool::new(false),
            active: std::sync::atomic::AtomicBool::new(false),
            processing: std::sync::atomic::AtomicBool::new(false),
            in_process: std::sync::atomic::AtomicBool::new(false),
        }));
        let plugin = clap_plugin_t {
            desc: std::ptr::null(),
            plugin_data: instance.cast(),
            init: None,
            destroy: None,
            activate: None,
            deactivate: None,
            start_processing: None,
            stop_processing: None,
            reset: None,
            process: None,
            get_extension: None,
            on_main_thread: None,
        };
        let plugin_ptr: *const clap_plugin_t = &raw const plugin;
        let original = container();

        let mut source = TestStream {
            data: original.clone(),
            max_chunk: 5,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let input = istream(&mut source);
        assert!(unsafe { state_load(plugin_ptr, &input) });

        let mut sink = TestStream {
            data: Vec::new(),
            max_chunk: 7,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let output = ostream(&mut sink);
        assert!(unsafe { state_save(plugin_ptr, &output) });
        assert_eq!(sink.data, original, "C 边界往返必须逐字保真");

        unsafe { drop(Box::from_raw(instance)) };
    }

    #[test]
    fn component_state_round_trips_through_the_clap_state_contract() {
        let container = ComponentState {
            edits: vec![1, 2, 3],
            capture: Some(b"{\"takes\":[]}".to_vec()),
            capture_edits: None,
        };
        let bytes = container.encode();
        let state = InstanceState::default();
        assert!(state.load(&bytes).is_ok());
        assert_eq!(state.save().unwrap(), bytes);
    }
    #[test]
    fn a_fresh_instance_saves_a_valid_empty_container() {
        let state = InstanceState::default();
        let bytes = state.save().unwrap();
        assert!(
            !bytes.is_empty(),
            "a fresh instance must still write a container"
        );
        assert!(ComponentState::decode(&bytes).is_ok());
        state.load(&bytes).unwrap();
        assert_eq!(state.save().unwrap(), bytes);
    }

    #[test]
    fn a_bare_legacy_blob_is_still_accepted() {
        let state = InstanceState::default();
        assert!(state.load(b"legacy edit state").is_ok());
        assert_eq!(state.save().unwrap(), b"legacy edit state");
    }
}
