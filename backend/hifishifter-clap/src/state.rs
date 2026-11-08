//! CLAP state：宿主的不透明字节块 ⇄ 共享组件存档（`ComponentState`）。
//!
//! 【为什么外壳不再自己存字节】VST3 的 `getState` / `setState` 与 CLAP 的 `clap.state`
//! 是同一件事：把本实例的编辑编码成宿主能保管的字节块，再读回来。此前 CLAP 侧把宿主
//! 编来的字节**原样缓存**再原样写回 —— 保存时不编码当前编辑，加载时也不把编辑合入
//! 会话，于是 CLAP 实例的存档是一条只有透传、没有内容的通路（重开工程后编辑全丢，
//! 而日志里一切正常）。现在两条格式都走 `ExtensionOwner::encode_state` /
//! `restore_state`：存档格式（编辑 + 捕获清单 + 捕获曲线）与"什么时候合入"的规则
//! 只有一份，跨格式复制粘贴实例也仍然可读。
//!
//! 【为什么流读写必须循环】`clap/stream.h` 明确写了宿主可以限制单次读/写的字节数，
//! 并提示「You need to use a loop to ensure that you read or write the entirety of your
//! state」。单次 `write` 只写出一部分就返回 true 的相反面是**保存失败**；短读立刻
//! 退出则会把尾部截掉，重载后编辑凭空少一截。两者在宿主日志里都只是"存档失败"，
//! 没有别的线索，因此这里把短读/短写都当成正常进度继续推进。

use crate::abi::clap_plugin_t;
use std::ffi::c_void;

/// 组件 state 的硬上限；与 `ComponentState` 的 64 MiB 约定一致。
///
/// 【为什么需要上限】宿主若一直返回同一个正数而从不给 EOF，循环就永不结束；
/// 音频插件不能把宿主线程挂死。超过上限按明确错误返回，而不是静默截断。
const MAX_STATE_BYTES: usize = 64 * 1024 * 1024;
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
        // 【为什么保存必须现场编码】宿主保存的是"此刻的实例"。回放上一次加载到的字节
        // 只会存下一份陈旧载荷 —— 用户在这次会话里画的每一条曲线都不会进工程。
        // 编码本身包含捕获清单与捕获曲线（与 VST3 的 getState 同一份实现）。
        let Ok(bytes) = instance.owner.encode_state() else {
            log::error!("[clap] state.save failed to encode the component state");
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
        // 【失败不阻断加载】与 VST3 的 setState 同一条纪律：一份读不出来的旧存档只该
        // 丢掉那份编辑，不该让整份工程加载失败。真因记进日志与恢复读数。
        if let Err(error) = instance.owner.restore_state(&bytes) {
            log::error!("[clap] state.load ignored: {error}; the host document load continues");
        }
        true
    })
}

// SAFETY: 宿主只读的静态扩展表。
unsafe impl Sync for clap_plugin_state_t {}
unsafe impl Send for clap_plugin_state_t {}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instance;
    use hifishifter_plugin_core::component_state::ComponentState;

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

    /// 一个与宿主给的形状一致的实例记录；调用方负责 `release_instance`。
    fn instance() -> *const clap_plugin_t {
        let owner =
            std::sync::Arc::new(hifishifter_plugin::render::extension::ExtensionOwner::default());
        let route = hifishifter_plugin::editor::RouteLease::new(&owner);
        let instance = Box::into_raw(Box::new(Instance {
            audio: crate::audio::AudioState::default(),
            host: std::ptr::null(),
            plugin: std::ptr::null(),
            owner,
            ara: None,
            initialized: std::sync::atomic::AtomicBool::new(false),
            active: std::sync::atomic::AtomicBool::new(false),
            processing: std::sync::atomic::AtomicBool::new(false),
            in_process: std::sync::atomic::AtomicBool::new(false),
            route,
            timer_id: std::sync::atomic::AtomicU32::new(u32::MAX),
        }));
        let plugin = Box::into_raw(Box::new(clap_plugin_t {
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
        }));
        plugin as *const clap_plugin_t
    }

    unsafe fn release_instance(plugin: *const clap_plugin_t) {
        let plugin = plugin.cast_mut();
        let data = unsafe { (*plugin).plugin_data };
        unsafe { drop(Box::from_raw(data.cast::<Instance>())) };
        unsafe { drop(Box::from_raw(plugin)) };
    }

    #[test]
    fn partial_writes_are_retried_until_the_whole_container_is_written() {
        let bytes = b"a container longer than one chunk".to_vec();
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
        let bytes = b"a container longer than one chunk".to_vec();
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
        let mut state = TestStream {
            data: Vec::new(),
            // 块长必须小于载荷，否则第一次调用就把整块写完，注入的错误永远轮不到。
            max_chunk: 8,
            read_offset: 0,
            fail_on_call: Some(2),
            calls: 0,
        };
        let stream = ostream(&mut state);
        assert!(!unsafe { write_all(&stream, b"payload payload payload") });
    }

    #[test]
    fn a_read_error_fails_the_load_instead_of_accepting_a_prefix() {
        let mut state = TestStream {
            data: b"payload payload payload".to_vec(),
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

    /// 端到端：真实的 `clap_plugin_t` → `state_save` → 短写流 → `state_load`。
    ///
    /// 【为什么这条用例存在】它钉的是"存档走共享编解码器"这件事本身：旧实现把宿主
    /// 编来的字节原样缓存，于是**从没加载过**的实例保存出来的是一份空容器；现在保存
    /// 必须现场编码，输出必须能被 `ComponentState::decode` 读回，并且再过一遍加载 →
    /// 保存后逐字稳定（宿主会反复保存同一份工程）。
    #[test]
    fn state_round_trips_through_the_c_abi_with_short_streams() {
        let plugin = instance();
        let mut sink = TestStream {
            data: Vec::new(),
            max_chunk: 7,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let output = ostream(&mut sink);
        assert!(unsafe { state_save(plugin, &output) });
        let saved = ComponentState::decode(&sink.data).expect("保存的载荷必须是可读的组件存档");
        assert!(!saved.edits.is_empty(), "存档必须带一份编辑载荷");

        // ① 连续两次保存必须逐字相同：宿主会反复保存同一份工程，而"两次不同"
        // 意味着存档里混进了与内容无关的噪声（旧实现正是每次写一份空容器）。
        let mut second = TestStream {
            data: Vec::new(),
            max_chunk: 7,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let output = ostream(&mut second);
        assert!(unsafe { state_save(plugin, &output) });
        assert_eq!(second.data, sink.data, "同一次会话的重复保存必须逐字稳定");

        // ② 读回来之后仍能保存出一份可读载荷（恢复 → 再保存不产生坏容器）。
        let mut source = TestStream {
            data: sink.data.clone(),
            max_chunk: 5,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let input = istream(&mut source);
        assert!(unsafe { state_load(plugin, &input) });
        let mut third = TestStream {
            data: Vec::new(),
            max_chunk: 7,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let output = ostream(&mut third);
        assert!(unsafe { state_save(plugin, &output) });
        assert!(
            ComponentState::decode(&third.data).is_ok(),
            "恢复之后保存出来的载荷仍必须是可读容器"
        );

        unsafe { release_instance(plugin) };
    }

    /// 加载必须把捕获清单交回捕获存储 —— 这是"重开工程 take 还在"的唯一通路。
    #[test]
    fn a_load_restores_the_capture_manifest_through_the_shared_path() {
        let source = instance();
        let manifest = unsafe { crate::instance(source) }
            .unwrap()
            .owner
            .capture()
            .manifest();
        let container = ComponentState {
            edits: b"{}".to_vec(),
            capture: Some(serde_json::to_vec(&manifest).unwrap()),
            capture_edits: None,
        }
        .encode();

        let target = instance();
        let mut stream = TestStream {
            data: container,
            max_chunk: 3,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let input = istream(&mut stream);
        assert!(unsafe { state_load(target, &input) });
        assert_eq!(
            unsafe { crate::instance(target) }
                .unwrap()
                .owner
                .capture()
                .manifest(),
            manifest,
            "清单必须经共享恢复路径回到捕获存储"
        );

        unsafe { release_instance(source) };
        unsafe { release_instance(target) };
    }

    /// 一份读不出来的旧载荷只该丢掉那份编辑，不该让宿主的工程加载失败。
    #[test]
    fn an_unreadable_payload_does_not_fail_the_host_load() {
        let plugin = instance();
        let mut stream = TestStream {
            data: b"this is not a component state".to_vec(),
            max_chunk: 64,
            read_offset: 0,
            fail_on_call: None,
            calls: 0,
        };
        let input = istream(&mut stream);
        assert!(
            unsafe { state_load(plugin, &input) },
            "读不出来的存档不能让宿主认为整份工程加载失败"
        );
        unsafe { release_instance(plugin) };
    }
}
