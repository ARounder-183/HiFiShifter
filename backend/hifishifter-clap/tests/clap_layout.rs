//! CLAP 1.1.9 手写 `#[repr(C)]` 的原生 ABI oracle。
//!
//! 【为什么必须有这一条】Rust 侧的声明与 `scripts/clap/probe.c` 是同一份假设的两份
//! 拷贝：两者若以同一种方式写错（多一个字段、少一层包装、把 `bool` 当成 `i32`），
//! 黑盒探针照样 `dlopen` 成功、`process` 照样直通，而**真实宿主**按官方头文件布局
//! 传进来的结构会被整体错位解读。唯一能打破这个闭环的是让钉住的 SDK 头文件自己算
//! 出 size / align / offset，再逐项与 Rust 的 `size_of` / `align_of` / `offset_of!`
//! 比对。
//!
//! 【为什么是集成测试而不是构建脚本】布局判据属于"产物能不能被宿主加载"的验收，
//! 与 `clap_exports.rs` 同层；构建脚本无法读回 Rust 类型的 `offset_of!`。
//!
//! 【环境】`HIFISHIFTER_CLAP_SDK_DIR` 由 `scripts/env/sdk-env.ps1` 的
//! `Set-SdkEnvironment` 提供；原生编译器在 Windows 用 `cl.exe`，Unix 用 `cc`
//! （CLAP 工作流已经装好两者）。SDK 缺失时**硬失败**，绝不静默跳过 ——
//! "绿在没跑"比不跑更危险。

use hifishifter_clap::abi::*;
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;

/// SDK 与编译器不齐时给出可照做的说明，而不是一句 "not found"。
const ORACLE_ENV_HELP: &str = "\
the CLAP ABI oracle needs the pinned CLAP SDK and a native C compiler:
  HIFISHIFTER_CLAP_SDK_DIR  <-  . ./scripts/env/sdk-env.ps1; Set-SdkEnvironment
  cl.exe (Windows) / cc (Unix) on PATH
CI prepares the pinned SDK before running the CLAP cargo gates.";

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    // tests/ -> crate 根 -> backend/ -> 仓库根
    manifest()
        .ancestors()
        .nth(3)
        .expect("repository root")
        .to_path_buf()
}

fn sdk_include() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIFISHIFTER_CLAP_SDK_DIR") {
        return PathBuf::from(dir).join("include");
    }
    // 便于本地直接 `cargo test`：SDK 按约定就在仓库的 third_party/sdk/clap。
    repo_root().join("third_party/sdk/clap/include")
}

fn compiler() -> (&'static str, Vec<String>) {
    if cfg!(windows) {
        (
            "cl.exe",
            vec!["/nologo".into(), "/std:c11".into(), "/utf-8".into()],
        )
    } else {
        ("cc", vec!["-std=c11".into(), "-O0".into()])
    }
}

/// 编译 `clap_layout.c` 并取回官方头文件算出的 JSON。
fn native_layout() -> serde_json::Value {
    let manifest = manifest();
    let scratch = repo_root().join(".build-tmp/clap-layout");
    std::fs::create_dir_all(&scratch).expect("create CLAP layout scratch dir");
    let include = sdk_include();
    assert!(
        include.join("clap/clap.h").is_file(),
        "pinned CLAP headers missing at {} -- {}",
        include.display(),
        ORACLE_ENV_HELP
    );
    let source = manifest.join("tests/clap_layout.c");
    let exe = scratch.join(if cfg!(windows) {
        "clap-layout.exe"
    } else {
        "clap-layout"
    });
    let (program, base_args) = compiler();
    let mut cmd = Command::new(program);
    cmd.current_dir(&scratch).args(&base_args);
    if cfg!(windows) {
        cmd.arg(format!("/I{}", include.display()))
            .arg(format!("/Fo{}", scratch.join("clap-layout.obj").display()))
            .arg(format!("/Fe{}", exe.display()))
            .arg(&source);
    } else {
        cmd.arg(format!("-I{}", include.display()))
            .arg("-o")
            .arg(&exe)
            .arg(&source);
    }
    let compile = cmd
        .output()
        .unwrap_or_else(|e| panic!("{} -- spawning {} failed: {}", ORACLE_ENV_HELP, program, e));
    assert!(
        compile.status.success(),
        "CLAP ABI oracle compile failed: {}\nstdout: {}\nstderr: {}\n{}",
        compile.status,
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr),
        ORACLE_ENV_HELP
    );
    let native = Command::new(&exe).output().expect("run CLAP ABI oracle");
    assert!(
        native.status.success(),
        "CLAP ABI oracle failed: {native:?}"
    );
    serde_json::from_slice(&native.stdout).expect("CLAP ABI oracle must emit valid JSON")
}

/// 把 `Option<fn>` 的字段偏移与原生函数指针字段一一核对。
///
/// 【为什么函数指针可以只比偏移】CLAP 的函数指针都是普通数据指针宽度，
/// `Option<unsafe extern "C" fn>` 在 Rust 里保证与 `*const c_void` 同尺寸同对齐
/// （空指针 niche）；oracle 的 `FIELD_LAYOUT` 用 `offsetof` 取原生字段地址，
/// 因此只需验证"槽位在哪里、总尺寸/对齐是多少"，不需要逐个类型名映射。
macro_rules! check {
    ($layout:expr, $name:expr, $value:expr) => {
        assert_eq!(
            $layout[$name]
                .as_u64()
                .unwrap_or_else(|| panic!("missing oracle key {}", $name)),
            $value as u64,
            "CLAP ABI {}",
            $name
        );
    };
}
macro_rules! kind {
    ($layout:expr, $ty:ty) => {
        check!($layout, concat!(stringify!($ty), ".size"), size_of::<$ty>());
        check!(
            $layout,
            concat!(stringify!($ty), ".align"),
            align_of::<$ty>()
        );
    };
}
macro_rules! field {
    ($layout:expr, $ty:ty, $name:ident) => {
        check!(
            $layout,
            concat!(stringify!($ty), ".", stringify!($name)),
            offset_of!($ty, $name)
        );
    };
}

#[test]
fn rust_clap_layout_matches_the_pinned_native_sdk() {
    let layout = native_layout();

    kind!(layout, clap_version_t);
    kind!(layout, clap_event_header_t);
    field!(layout, clap_event_header_t, time);
    field!(layout, clap_event_header_t, space_id);
    field!(layout, clap_event_header_t, flags);
    // C 侧的 `size` 字段与 TYPE_LAYOUT 的 `.size` 键重名，oracle 用 `size_field` 输出。
    check!(
        layout,
        "clap_event_header_t.size_field",
        offset_of!(clap_event_header_t, size)
    );

    kind!(layout, clap_event_transport_t);
    field!(layout, clap_event_transport_t, flags);
    field!(layout, clap_event_transport_t, song_pos_beats);
    field!(layout, clap_event_transport_t, song_pos_seconds);
    field!(layout, clap_event_transport_t, tempo);
    field!(layout, clap_event_transport_t, tempo_inc);
    field!(layout, clap_event_transport_t, loop_start_beats);
    field!(layout, clap_event_transport_t, loop_end_beats);
    field!(layout, clap_event_transport_t, loop_start_seconds);
    field!(layout, clap_event_transport_t, loop_end_seconds);
    field!(layout, clap_event_transport_t, bar_start);
    field!(layout, clap_event_transport_t, bar_number);
    field!(layout, clap_event_transport_t, tsig_num);
    field!(layout, clap_event_transport_t, tsig_denom);

    kind!(layout, clap_audio_buffer_t);
    field!(layout, clap_audio_buffer_t, data32);
    field!(layout, clap_audio_buffer_t, data64);
    field!(layout, clap_audio_buffer_t, channel_count);
    field!(layout, clap_audio_buffer_t, latency);
    field!(layout, clap_audio_buffer_t, constant_mask);

    kind!(layout, clap_process_t);
    field!(layout, clap_process_t, steady_time);
    field!(layout, clap_process_t, frames_count);
    field!(layout, clap_process_t, transport);
    field!(layout, clap_process_t, audio_inputs);
    field!(layout, clap_process_t, audio_outputs);
    field!(layout, clap_process_t, audio_inputs_count);
    field!(layout, clap_process_t, audio_outputs_count);
    field!(layout, clap_process_t, in_events);
    field!(layout, clap_process_t, out_events);

    kind!(layout, clap_plugin_descriptor_t);
    field!(layout, clap_plugin_descriptor_t, clap_version);
    field!(layout, clap_plugin_descriptor_t, id);
    field!(layout, clap_plugin_descriptor_t, name);
    field!(layout, clap_plugin_descriptor_t, vendor);
    field!(layout, clap_plugin_descriptor_t, url);
    field!(layout, clap_plugin_descriptor_t, manual_url);
    field!(layout, clap_plugin_descriptor_t, support_url);
    field!(layout, clap_plugin_descriptor_t, version);
    field!(layout, clap_plugin_descriptor_t, description);
    field!(layout, clap_plugin_descriptor_t, features);

    kind!(layout, clap_host_t);
    field!(layout, clap_host_t, clap_version);
    field!(layout, clap_host_t, host_data);
    field!(layout, clap_host_t, name);
    field!(layout, clap_host_t, vendor);
    field!(layout, clap_host_t, url);
    field!(layout, clap_host_t, version);
    field!(layout, clap_host_t, get_extension);
    field!(layout, clap_host_t, request_restart);
    field!(layout, clap_host_t, request_process);
    field!(layout, clap_host_t, request_callback);

    kind!(layout, clap_plugin_t);
    field!(layout, clap_plugin_t, desc);
    field!(layout, clap_plugin_t, plugin_data);
    field!(layout, clap_plugin_t, init);
    field!(layout, clap_plugin_t, destroy);
    field!(layout, clap_plugin_t, activate);
    field!(layout, clap_plugin_t, deactivate);
    field!(layout, clap_plugin_t, start_processing);
    field!(layout, clap_plugin_t, stop_processing);
    field!(layout, clap_plugin_t, reset);
    field!(layout, clap_plugin_t, process);
    field!(layout, clap_plugin_t, get_extension);
    field!(layout, clap_plugin_t, on_main_thread);

    kind!(layout, clap_plugin_factory_t);
    field!(layout, clap_plugin_factory_t, get_plugin_count);
    field!(layout, clap_plugin_factory_t, get_plugin_descriptor);
    field!(layout, clap_plugin_factory_t, create_plugin);

    kind!(layout, clap_plugin_entry_t);
    field!(layout, clap_plugin_entry_t, clap_version);
    field!(layout, clap_plugin_entry_t, init);
    field!(layout, clap_plugin_entry_t, deinit);
    field!(layout, clap_plugin_entry_t, get_factory);

    // 流、端口、state/render/timer 扩展表：这些结构体分别定义在 `clap/stream.h`
    // 与各 `clap/ext/*.h`，Rust 侧每个扩展模块各有一份手写镜像。
    type Istream = hifishifter_clap::state::clap_istream_t;
    type Ostream = hifishifter_clap::state::clap_ostream_t;
    type StateExt = hifishifter_clap::state::clap_plugin_state_t;
    type PortInfo = hifishifter_clap::ports::AudioPortInfo;
    type RenderExt = hifishifter_clap::render_ext::RenderExt;
    type TimerExt = hifishifter_clap::timer::TimerExt;

    check!(layout, "clap_istream_t.size", size_of::<Istream>());
    check!(layout, "clap_istream_t.align", align_of::<Istream>());
    check!(layout, "clap_istream_t.ctx", offset_of!(Istream, ctx));
    check!(layout, "clap_istream_t.read", offset_of!(Istream, read));
    check!(layout, "clap_ostream_t.size", size_of::<Ostream>());
    check!(layout, "clap_ostream_t.align", align_of::<Ostream>());
    check!(layout, "clap_ostream_t.ctx", offset_of!(Ostream, ctx));
    check!(layout, "clap_ostream_t.write", offset_of!(Ostream, write));
    check!(layout, "clap_plugin_state_t.size", size_of::<StateExt>());
    check!(
        layout,
        "clap_plugin_state_t.save",
        offset_of!(StateExt, save)
    );
    check!(
        layout,
        "clap_plugin_state_t.load",
        offset_of!(StateExt, load)
    );
    check!(layout, "clap_audio_port_info_t.size", size_of::<PortInfo>());
    check!(
        layout,
        "clap_audio_port_info_t.id",
        offset_of!(PortInfo, id)
    );
    check!(
        layout,
        "clap_audio_port_info_t.name",
        offset_of!(PortInfo, name)
    );
    check!(
        layout,
        "clap_audio_port_info_t.flags",
        offset_of!(PortInfo, flags)
    );
    check!(
        layout,
        "clap_audio_port_info_t.channel_count",
        offset_of!(PortInfo, channel_count)
    );
    check!(
        layout,
        "clap_audio_port_info_t.port_type",
        offset_of!(PortInfo, port_type)
    );
    check!(
        layout,
        "clap_audio_port_info_t.in_place_pair",
        offset_of!(PortInfo, in_place_pair)
    );
    check!(layout, "clap_plugin_render_t.size", size_of::<RenderExt>());
    check!(
        layout,
        "clap_plugin_render_t.has_hard_realtime_requirement",
        offset_of!(RenderExt, has_hard_realtime_requirement)
    );
    check!(
        layout,
        "clap_plugin_render_t.set",
        offset_of!(RenderExt, set)
    );
    check!(
        layout,
        "clap_plugin_timer_support_t.size",
        size_of::<TimerExt>()
    );
    check!(
        layout,
        "clap_plugin_timer_support_t.on_timer",
        offset_of!(TimerExt, on_timer)
    );

    // GUI 表当前不返回给宿主（阶段 1 是 Capture-only），但一旦有人打开
    // `get_extension("clap.gui")`，错位的函数指针会让宿主跳进无关代码。
    // 提前把布局钉住，避免"未接线"变成"未验证"。
    type Window = hifishifter_clap::gui::clap_window_t;
    type GuiExt = hifishifter_clap::gui::clap_plugin_gui_t;
    type HostGui = hifishifter_clap::gui::clap_host_gui_t;
    check!(layout, "clap_window_t.size", size_of::<Window>());
    check!(layout, "clap_window_t.api", offset_of!(Window, api));
    check!(layout, "clap_window_t.ptr", offset_of!(Window, handle));
    check!(layout, "clap_plugin_gui_t.size", size_of::<GuiExt>());
    check!(
        layout,
        "clap_plugin_gui_t.is_api_supported",
        offset_of!(GuiExt, is_api_supported)
    );
    check!(
        layout,
        "clap_plugin_gui_t.get_preferred_api",
        offset_of!(GuiExt, get_preferred_api)
    );
    check!(
        layout,
        "clap_plugin_gui_t.create",
        offset_of!(GuiExt, create)
    );
    check!(
        layout,
        "clap_plugin_gui_t.destroy",
        offset_of!(GuiExt, destroy)
    );
    check!(
        layout,
        "clap_plugin_gui_t.set_scale",
        offset_of!(GuiExt, set_scale)
    );
    check!(
        layout,
        "clap_plugin_gui_t.get_size",
        offset_of!(GuiExt, get_size)
    );
    check!(
        layout,
        "clap_plugin_gui_t.can_resize",
        offset_of!(GuiExt, can_resize)
    );
    check!(
        layout,
        "clap_plugin_gui_t.get_resize_hints",
        offset_of!(GuiExt, get_resize_hints)
    );
    check!(
        layout,
        "clap_plugin_gui_t.adjust_size",
        offset_of!(GuiExt, adjust_size)
    );
    check!(
        layout,
        "clap_plugin_gui_t.set_size",
        offset_of!(GuiExt, set_size)
    );
    check!(
        layout,
        "clap_plugin_gui_t.set_parent",
        offset_of!(GuiExt, set_parent)
    );
    check!(
        layout,
        "clap_plugin_gui_t.set_transient",
        offset_of!(GuiExt, set_transient)
    );
    check!(
        layout,
        "clap_plugin_gui_t.suggest_title",
        offset_of!(GuiExt, suggest_title)
    );
    check!(layout, "clap_plugin_gui_t.show", offset_of!(GuiExt, show));
    check!(layout, "clap_plugin_gui_t.hide", offset_of!(GuiExt, hide));
    check!(layout, "clap_host_gui_t.size", size_of::<HostGui>());
    check!(
        layout,
        "clap_host_gui_t.resize_hints_changed",
        offset_of!(HostGui, resize_hints_changed)
    );
    check!(
        layout,
        "clap_host_gui_t.request_resize",
        offset_of!(HostGui, request_resize)
    );
    check!(
        layout,
        "clap_host_gui_t.request_show",
        offset_of!(HostGui, request_show)
    );
    check!(
        layout,
        "clap_host_gui_t.request_hide",
        offset_of!(HostGui, request_hide)
    );
    check!(
        layout,
        "clap_host_gui_t.closed",
        offset_of!(HostGui, closed)
    );

    // `clap_event_header_t.type_` 的 C 字段名是 `type`，oracle 用同一个键输出。
    check!(
        layout,
        "clap_event_header_t.type",
        offset_of!(clap_event_header_t, type_)
    );
    // `Option<fn>` 与原生函数指针同宽：整张表能被逐槽比对的前提。
    assert_eq!(
        size_of::<Option<unsafe extern "C" fn()>>(),
        size_of::<*const c_void>()
    );
}
