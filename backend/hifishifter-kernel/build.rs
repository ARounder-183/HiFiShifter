//! 内核的原生依赖构建。
//!
//! 【为什么内核要有 build.rs】内核里有三个模块直接声明了原生 FFI：
//! - `time_stretch` → `sstretch`（Signalsmith Stretch，**静态链接**）
//! - `time_stretch` → `soundtouch`（`#[link(name = "SoundTouchDLL")]`，**链接期依赖**）
//!
//! 这两段构建原先住在 `backend/src-tauri/build.rs` 里。只要内核还只是 app 的一个模块，
//! 那样就够用 —— 因为链接参数来自 app 的 build 脚本，最终二进制照样能链接上。
//!
//! **但插件（`hifishifter-plugin`）是另一个构建目标**：它单独链接内核时，app 的 build
//! 脚本根本不参与。那时这两个符号无处解析，会变成**链接错误** —— 而症状只在插件构建时
//! 出现，`git mv` 之后在 app 里跑测试是**完全正常的**。这是"看起来成功、到 DAW 里才炸"
//! 的一类陷阱，所以构建归属必须跟着模块一起搬。
//!
//! 设计依据：docs/superpowers/specs/2026-10-04-ara-plugin-v1-design.md §4.9
//!
//! 【为什么 `third_party/` 不一起搬】它被 `tauri.conf.json` 的三份配置以相对路径引用
//! 做资源打包。把它移出 `src-tauri/` 会让 Tauri 的资源路径指向 crate 之外，
//! 风险大于收益。所以这里用相对路径引用它，并在下面写明这个耦合。
//!
//! 【为什么构建身份（commit / 脏标志 / 仓库链接）也在这里注入】`build_info` 住在
//! **内核**里，而 `rustc-env` 是**按 crate** 注入的。此前只有 app 与插件的 build
//! script 调 `git_info::emit`，内核没有 —— 于是 `build_info` 在自己那个编译单元里
//! 读到的四个变量**全部未定义**，独立 App 与插件的「关于」对话框、日志会话头、
//! 诊断包里的 commit 一律为空（而键名测试照样全绿）。注入必须发生在**读取者所在的
//! crate**，这是 `option_env!` 的语义，不是可以绕过的约定。

#[path = "../build-support/git_info.rs"]
mod git_info;
#[path = "../build-support/product_version.rs"]
mod product_version;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// 第三方原生源码目录。
///
/// 【为什么是 `../src-tauri/...`】`third_party/` 目前仍在 app crate 里（见文件头说明）。
/// 构建脚本的当前工作目录是**包根**（`backend/hifishifter-kernel`），所以从这个相对
/// 路径可以稳定到达它。这是**构建期**耦合，不影响运行期依赖方向。
const THIRD_PARTY: &str = "../src-tauri/third_party";

/// 所有原生依赖共用的一处构建根（`<target>/hfs-native`）。
///
/// 【为什么不能放在 OUT_DIR】OUT_DIR 的目录名来自 cargo 的**单元指纹**：`cargo check`、
/// `cargo test`、`cargo clippy`、`-p crate` 与 `--workspace` 各有一份，于是同一份 C++
/// 会被完整编译很多次。实测这台机器上留下过 **19 份** `target/debug/build/hifishifter-kernel-*`
/// （每份 7.8–9.6 MB，合计约 170 MB），而它们的源码、编译器、旗标一字不差。
///
/// 放在 target 根的固定路径之后，"切 cargo 子命令"不再重编 C++：`cc` 的产物在那里按
/// mtime 判新鲜，cmake 树在那里按 cache + 输入 mtime 判新鲜。两个 profile 共享同一份
/// 产物是安全的 —— 这些库一律按 `/O2` / `-O2` 构建，与 cargo 的 profile 无关。
///
/// 【为什么从 OUT_DIR 反推 target 根】`OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out`，
/// 上跳四级正是 `<target>`。`CARGO_TARGET_DIR` 指向别处时该推导同样成立。
fn native_build_root() -> PathBuf {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    Path::new(&out_dir)
        .ancestors()
        .nth(4)
        .expect("[kernel-native] unexpected OUT_DIR depth")
        .join("hfs-native")
}

/// 目标三元组的稳定短名（目录名用）：`x86_64-pc-windows-msvc` 原样保留。
fn target_slug() -> String {
    std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
}

/// **目标**操作系统的编译期事实。
///
/// 【为什么不能用 `cfg!(target_os)`】build script 是给**宿主**编译的，脚本里的
/// `cfg!(target_os = ...)` 回答的是"我在哪台机器上构建"，不是"我在为谁构建"。
/// 交叉编译时两者不同：在 Windows 上构建 Android 包时
/// `cfg!(target_os = "windows")` 为真，于是 `-fPIC` 会被**跳过** ——
/// 而 Android 的 .so 必须有位置无关代码，症状是一串与 PIC 无关的链接错误。
/// 唯一正确的来源是 cargo 注入的 `CARGO_CFG_TARGET_OS`。
fn target_os() -> String {
    std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}

/// **目标**架构，按 CMake 在 Apple 平台上的拼写返回（Rust 说 `aarch64`，CMake 说 `arm64`）。
///
/// 【为什么必须由我们自己给】本模块直接 spawn `cmake`，**不经过 `cmake` crate** —— 那个 crate
/// 会在交叉编译时自动补 `-DCMAKE_OSX_ARCHITECTURES`，但这条代码路径上它根本不存在。少了
/// 这个值，cmake 永远按**宿主**架构产出：在 Apple Silicon 上交叉编 x86_64 时，dylib 仍是
/// arm64，而 Rust 侧一切正常 —— universal 包里于是混进一个单架构依赖，只有 `lipo -archs`
/// 能发现（run 46099076516 的 `Assert universal Mach-O` 就是这样红的）。
fn cmake_apple_arch() -> String {
    match std::env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap_or_default()
        .as_str()
    {
        "aarch64" => "arm64".to_string(),
        other => other.to_string(),
    }
}

/// 目标是否是移动形态（Android / iOS）。
///
/// 【为什么需要它】内核里两处原生依赖是**桌面专属**的：
/// - SoundTouchDLL 是 CMake 构建的桌面动态库（Android/iOS 上根本没有 CMake 工具链，
///   也没有 `.so`/`.dylib` 的打包位置）；
/// - vslib 是 Windows 闭源 DLL。
///
/// 移动端只保留 WORLD（C++ 静态）与 Signalsmith（header-only）—— 这两者用 NDK 的
/// clang 就能编。缺少这道闸门，移动目标会卡在 build script 里，错误信息指向 cmake。
fn is_mobile_target() -> bool {
    matches!(target_os().as_str(), "android" | "ios")
}

fn main() {
    // ── 构建身份（必须**先于**任何早退）────────────────────────────────
    //
    // 【为什么放在跳过开关之前】`HIFISHIFTER_SKIP_NATIVE_BUILD=1` 只跳过**原生库**构建；
    // 构建身份是纯 `git` 查询（毫秒级），与原生库无关。放在早退之后，CI 的类型检查
    // 构建就会得到一份"没有 commit"的 `build_info`，而那个差异只会在 CI 上出现 ——
    // 正是本仓库反复吃过的一类"本地正常、CI 才炸"的陷阱。
    //
    // 监视集只有 `src`：内核进二进制的输入只有自己的源码（模型与前端产物归 app 与
    // 插件各自监视）。脏标志回答"这份二进制对应哪次提交"，多监视无关目录只会让它
    // 变成"总是脏的"，等于没有信号。
    git_info::emit(&["src"]);
    product_version::emit(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        env!("CARGO_PKG_VERSION"),
    );

    // 与 app 的 build.rs 共用同一个跳过开关：CI 只做类型检查时可以整体跳过原生构建。
    //
    // 【为什么必须有 rerun-if-env-changed】cargo 默认只在源码变化时重跑 build script。
    // 少了这一行，"先在 `HIFISHIFTER_SKIP_NATIVE_BUILD=1` 下跑过一次类型检查、再正常
    // 构建"这条最常见的本地流程会**复用那份没有原生库的产物**，随后在链接期报
    // 找不到 sstretch / world 的符号 —— 而真正的根因是 build script 没重跑。
    println!("cargo:rerun-if-env-changed=HIFISHIFTER_SKIP_NATIVE_BUILD");
    println!("cargo:rerun-if-env-changed=HIFISHIFTER_OFFLINE");
    println!("cargo:rerun-if-env-changed=HIFISHIFTER_SDK_MIRROR");
    let skip_native = std::env::var("HIFISHIFTER_SKIP_NATIVE_BUILD").unwrap_or_default();
    if skip_native == "1" {
        println!(
            "cargo:warning=[kernel-native] Skipping native library builds (HIFISHIFTER_SKIP_NATIVE_BUILD=1)"
        );
        return;
    }

    build_signalsmith_stretch();
    build_soundtouch();
    build_vslib();
    build_world_static();
}

/// 把 WORLD 声码器编译成静态库。
///
/// 从 `backend/src-tauri/build.rs` 原样搬来，只改了 `third_party` 的根路径。
///
/// 【为什么跟着内核走】`world_vocoder` / `streaming_world` / `renderer::world` 现在
/// 都在内核里，而它们调用的是 WORLD 的 C++ 符号（`CheapTrick` / `D4C` / `Synthesis` 等）。
/// 这些符号由本函数编译出的静态库提供；留在 app 的 build.rs 里，
/// 内核的**测试二进制**与插件都会报 `LNK2019: unresolved external symbol`（实测）。
fn build_world_static() {
    let world_src_dir = format!("{}/world-static/World/src", THIRD_PARTY);
    let world_src_path = Path::new(&world_src_dir);

    if !world_src_path.exists() {
        eprintln!("\n========================================");
        eprintln!("ERROR: WORLD source code not found!");
        eprintln!("========================================");
        eprintln!("\nExpected location: {}", world_src_path.display());
        eprintln!("\nTo fix this, run:");
        eprintln!("  cd backend/src-tauri/third_party/world-static");
        eprintln!("  git clone https://github.com/mmorise/World.git");
        eprintln!("========================================\n");
        panic!("WORLD sources missing. See error message above for instructions.");
    }

    let required_files = [
        "cheaptrick.cpp",
        "codec.cpp",
        "common.cpp",
        "d4c.cpp",
        "dio.cpp",
        "fft.cpp",
        "harvest.cpp",
        "matlabfunctions.cpp",
        "stonemask.cpp",
        "synthesis.cpp",
        "synthesisrealtime.cpp",
    ];

    for file in &required_files {
        let file_path = world_src_path.join(file);
        if !file_path.exists() {
            panic!(
                "Required WORLD source file not found: {}",
                file_path.display()
            );
        }
    }

    println!("cargo:rerun-if-changed={}", world_src_dir);

    let mut world = cc::Build::new();
    // 【共享产物】见 `native_build_root`：这条 C++ 不该为每个 cargo 单元重编一遍。
    let shared = native_build_root().join("world");
    std::fs::create_dir_all(&shared).expect("[world] failed to create the shared build root");
    world.out_dir(&shared);
    world.cpp(true).include(&world_src_dir).files(
        required_files
            .iter()
            .map(|f| format!("{}/{}", world_src_dir, f)),
    );

    // C++ 标准旗标按编译器家族分发（与 sstretch 构建同一模式）：
    // MSVC 的 cl 不认识 GCC 风格的 `-std:c++11`，传入只会得到 D9002
    // "ignoring unknown option" 警告并被忽略 —— cl 默认即 ≥C++14，
    // 显式给 /std:c++14 行为不变、警告消失。
    if world.get_compiler().is_like_msvc() {
        world.flag("/std:c++14");
        world.flag("/utf-8");
    } else {
        world.flag("-std=c++11");
    }

    world.compile("world");

    println!("cargo:rustc-link-lib=static=world");
}

/// 链接 vslib（闭源、仅 Windows）的导入库，并把 DLL 放到目标目录。
///
/// 从 `backend/src-tauri/build.rs` 原样搬来，只改了 `third_party` 的根路径。
///
/// 【为什么跟着内核走】`vslib.rs` 现在在内核里（`renderer/vslib_processor.rs` 要用它），
/// 而 `#[link(name = "vslib_x64")]` 是**链接期**依赖 —— 与 SoundTouch 同理：
/// 留在 app 的 build.rs 里，插件单独链接内核时会报 `LNK1181: cannot open input file
/// 'vslib_x64.lib'`（实测）。
fn build_vslib() {
    if !cfg!(feature = "vslib") {
        return;
    }

    // Only link/copy for x86_64 Windows targets. Non-target platforms should
    // not require third_party/vslib assets to exist.
    let target = std::env::var("TARGET").unwrap_or_default();
    let target_lc = target.to_lowercase();
    if !(target_lc.contains("windows") && target_lc.contains("x86_64")) {
        println!(
            "cargo:warning=[vslib] target '{}' not an x86_64 Windows target; skipping link/copy of vslib_x64",
            target
        );
        return;
    }

    let lib_dir = Path::new(THIRD_PARTY).join("vslib");

    if !lib_dir.exists() {
        panic!(
            "[vslib] third_party/vslib/ not found. \
             Place vslib_x64.dll and vslib_x64.lib there."
        );
    }

    // Resolve to an absolute path so rustc can find the import lib
    let abs = lib_dir
        .canonicalize()
        .expect("[vslib] failed to canonicalize third_party/vslib path");

    println!("cargo:rerun-if-changed={}/vslib_x64.lib", lib_dir.display());
    println!("cargo:rerun-if-changed={}/vslib_x64.dll", lib_dir.display());

    println!("cargo:rustc-link-search=native={}", abs.display());
    println!("cargo:rustc-link-lib=dylib=vslib_x64");

    // OUT_DIR = .../target/<profile>/build/<pkg>/out  →  4 levels up = target/<profile>/
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let dll_src = lib_dir.join("vslib_x64.dll");
        let target_dir = Path::new(&out_dir)
            .ancestors()
            .nth(3)
            .expect("[vslib] unexpected OUT_DIR depth");
        let dll_dst = target_dir.join("vslib_x64.dll");
        if let Err(e) = std::fs::copy(&dll_src, &dll_dst) {
            println!(
                "cargo:warning=[vslib] could not copy DLL to {}: {}",
                dll_dst.display(),
                e
            );
        }
        // Test executables live in target/<profile>/deps/, where the loader
        // looks for DLLs; copy there as well or cargo test cannot start.
        let deps_dir = target_dir.join("deps");
        let _ = std::fs::create_dir_all(&deps_dir);
        let dll_dst_deps = deps_dir.join("vslib_x64.dll");
        if dll_dst_deps != dll_dst {
            let _ = std::fs::copy(&dll_src, &dll_dst_deps);
        }
    } else {
        println!("cargo:warning=[vslib] OUT_DIR not set; skipping DLL copy")
    }
}

/// 编译 Signalsmith Stretch 的 C 包装并静态链接。
///
/// 从 `backend/src-tauri/build.rs` 原样搬来，只改了 `third_party` 的根路径。
fn build_signalsmith_stretch() {
    let ss_base = format!("{}/signalsmith-stretch", THIRD_PARTY);
    let ss_lib_dir = format!("{}/signalsmith-stretch", ss_base);
    let ss_wrapper = format!("{}/sstretch-c.cpp", ss_base);
    let ss_lib_path = Path::new(&ss_lib_dir);

    if !ss_lib_path.exists() {
        eprintln!("\n========================================");
        eprintln!("ERROR: Signalsmith Stretch source code not found!");
        eprintln!("========================================");
        eprintln!("\nExpected location: {}", ss_lib_path.display());
        eprintln!("\nTo fix this, run:");
        eprintln!("  cd backend/src-tauri/third_party/signalsmith-stretch");
        eprintln!("  git clone --depth 1 https://github.com/Signalsmith-Audio/signalsmith-stretch.git signalsmith-stretch");
        eprintln!("  git clone --depth 1 https://github.com/Signalsmith-Audio/linear.git signalsmith-stretch/signalsmith-linear");
        eprintln!("========================================\n");
        panic!("Signalsmith Stretch sources missing. See error message above for instructions.");
    }

    let linear_dir = format!("{}/signalsmith-linear", ss_lib_dir);
    if !Path::new(&linear_dir).exists() {
        eprintln!("\n========================================");
        eprintln!("ERROR: Signalsmith Linear (STFT dependency) not found!");
        eprintln!("========================================");
        eprintln!("\nExpected location: {}", linear_dir);
        eprintln!("\nTo fix this, run:");
        eprintln!(
            "  git clone --depth 1 https://github.com/Signalsmith-Audio/linear.git {}",
            linear_dir
        );
        eprintln!("========================================\n");
        panic!("Signalsmith Linear missing. See error message above for instructions.");
    }

    let stretch_h = format!("{}/signalsmith-stretch.h", ss_lib_dir);
    if !Path::new(&stretch_h).exists() {
        panic!("signalsmith-stretch.h not found at {}", stretch_h);
    }

    println!("cargo:rerun-if-changed={}", ss_base);

    let mut build = cc::Build::new();
    // 【共享产物】理由同 world：切 cargo 子命令不该重编这份 C++。
    let shared = native_build_root().join("signalsmith");
    std::fs::create_dir_all(&shared).expect("[sstretch] failed to create the shared build root");
    build.out_dir(&shared);
    build
        .cpp(true)
        .warnings(false)
        .include(&ss_lib_dir)
        .include(&linear_dir)
        .include(&ss_base)
        // 只需编译我们的 C wrapper，stretch 库本身是 header-only
        .file(&ss_wrapper);

    let compiler = build.get_compiler();
    if compiler.is_like_msvc() {
        build.flag("/EHsc");
        build.flag("/std:c++14");
        build.define("NOMINMAX", None);
        // 启用优化以提升 number-crunching 性能（即使在 Debug 模式下）
        build.flag("/O2");
        build.flag("/utf-8");
    } else {
        build.flag("-std=c++14");
        if target_os() != "windows" {
            build.flag("-fPIC");
        }
        build.flag("-O2");
    }

    build.compile("signalsmith_stretch");

    println!("cargo:rustc-link-lib=static=signalsmith_stretch");
}

/// 用 CMake 构建 SoundTouchDLL 并链接。
///
/// 从 `backend/src-tauri/build.rs` 原样搬来，只改了 `third_party` 的根路径。
///
/// 【保留的一处"越界"写入】Step 5 会把构建出的 DLL 回写到源码树里的
/// `third_party/soundtouch-static/soundtouch/` —— 那是**给 Tauri 的资源打包用的**
/// （`tauri.conf.json` 以该相对路径把它打进去），不是给内核用的。
/// 搬过来之后由本脚本继续写，行为与搬家前一致。
/// SoundTouch 供给链钉子：`scripts/env/plugin-sdks.json` 的 `soundtouch` 段是唯一事实
/// 来源（与 ARA/VST3 SDK 的 pin 同一份清单）。
///
/// 【为什么必须校验】clone 按 `--branch 2.3.3` 拉，而 tag 是**可变引用** —— 上游
/// 重打 tag 或 codeberg 供给链事故会直接进入音频时基代码，且两次构建产物可能不同。
/// clone（或复用已有检出）后对 pin 校验 HEAD，不一致即失败。pin 升级走
/// plugin-sdks.json 的评审 diff。
///
/// 【build.rs 不再改上游任何文件】以前这里会就地给 `SoundTouchDLL.rc` 与
/// `CMakeLists.txt` 打补丁；现在补丁全部消失（rc 的 MFC 头用 include shim 顶替，
/// CMake 策略用 `-DCMAKE_POLICY_VERSION_MINIMUM` 顶替），检出因此可以与上游逐字节
/// 一致 —— 这也是"发行物只带 LGPL 全文、不带变更说明"的前提。
const SOUNDTOUCH_PIN_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/env/plugin-sdks.json"
));

fn soundtouch_pinned_commit() -> String {
    // 不为 build.rs 引入 serde_json（build-dependencies 只有 cc）：清单是受控文件，
    // soundtouch 段放在最后，取该标记之后第一个 "commit" 值即可。
    let section = SOUNDTOUCH_PIN_JSON
        .split("\"soundtouch\"")
        .nth(1)
        .expect("scripts/env/plugin-sdks.json must contain a soundtouch section");
    let rest = section
        .split("\"commit\": \"")
        .nth(1)
        .expect("soundtouch section must pin a commit");
    rest.split('"')
        .next()
        .expect("soundtouch commit value")
        .to_string()
}

/// 校验 SoundTouch 检出：HEAD 必须是 pin 的那次提交，且**源码未被改动**。
///
/// 【为什么连"未修改"也要查】发行物里的 `SoundTouchDLL` 声称是"未修改的上游库"，
/// 随包因此只放 LGPL 全文、不放变更说明。这个声称只有靠构建期断言才守得住：旧版本的
/// build.rs 会在原地改 `SoundTouchDLL.rc` 与 `CMakeLists.txt`，一台保留了旧检出的
/// 编译环境会继续用被改过的源码编出 DLL —— 产物与声明不符，而症状只是"许可里少了一页"。
/// 这里只看被跟踪的源码（`--untracked-files=no` + 路径过滤）：Step 5 回写的 DLL 是
/// 产物，不算改动。
///
/// 无 `.git` 的检出版本（离线镜像导入）两项都无法核对，跳过并留一条警告。
fn verify_soundtouch_checkout(st_src_path: &Path) {
    let pinned = soundtouch_pinned_commit();
    if !st_src_path.join(".git").exists() {
        println!("cargo:warning=[soundtouch] source has no .git; skipping pin and pristine checks");
        return;
    }
    // safe.directory=*：检出可能由不同用户/WSL 创建（Windows 的 dubious ownership）。
    // 这是一次只读的 HEAD 元数据查询，放开该限制不引入执行面。
    let output = Command::new("git")
        .args([
            "-c",
            "safe.directory=*",
            "-C",
            st_src_path.to_str().expect("[soundtouch] src path"),
            "rev-parse",
            "HEAD",
        ])
        .output()
        .expect("[soundtouch] failed to run git rev-parse");
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || head != pinned {
        panic!(
            "[soundtouch] checkout HEAD {head} does not match the pinned commit {pinned} \
             (scripts/env/plugin-sdks.json). Delete the checkout and rebuild to re-clone the \
             pinned revision - a retagged upstream must never reach audio code silently."
        );
    }
    println!("cargo:warning=[soundtouch] pin verified: {head}");

    let dirty = Command::new("git")
        .args([
            "-c",
            "safe.directory=*",
            "-C",
            st_src_path.to_str().expect("[soundtouch] src path"),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "source",
            "include",
            "CMakeLists.txt",
        ])
        .output()
        .expect("[soundtouch] failed to run git status");
    let dirty = String::from_utf8_lossy(&dirty.stdout).trim().to_string();
    if !dirty.is_empty() {
        panic!(
            "[soundtouch] checkout is not pristine - these tracked source files were modified:\n{dirty}\n\
             The shipped SoundTouchDLL must be built from unmodified upstream source (that is why \
             only COPYING.LGPL-2.1 ships, with no change notice). Delete the checkout \
             (backend/src-tauri/third_party/soundtouch-static/soundtouch) and rebuild to re-clone \
             the pinned revision."
        );
    }
    println!("cargo:warning=[soundtouch] checkout is pristine (upstream sources unmodified)");
}

/// 本地 SDK 镜像根：`HIFISHIFTER_SDK_MIRROR`，否则仓库根的 `third_party/sdk-mirror`。
///
/// 【为什么在 Rust 里再实现一遍】镜像根的唯一事实来源是 `scripts/env/sdk-env.ps1`
/// (`Get-SdkMirrorRoot`)，但 build.rs 不能点源 PowerShell。两处默认值必须一致：
/// 镜像里放一份 SoundTouch 源（`<mirror>/soundtouch`），离网时这里直接取用，不触网。
fn sdk_mirror_root() -> PathBuf {
    if let Ok(dir) = std::env::var("HIFISHIFTER_SDK_MIRROR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    // 构建脚本的包根是 backend/hifishifter-kernel，仓库根在两级之上。
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("third_party")
        .join("sdk-mirror")
}

/// 是否禁止为取第三方源联网（`-Offline` 会把它写进 `HIFISHIFTER_OFFLINE`）。
fn sdk_offline_requested() -> bool {
    matches!(
        std::env::var("HIFISHIFTER_OFFLINE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// 递归复制目录树（用于从本地镜像导入源码，替代联网克隆）。软链接按目标类型解引用。
fn copy_dir_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        // `Path::is_dir` 跟随软链接，因此软链接目录会被展开复制。
        if entry.path().is_dir() {
            copy_dir_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// 资源编译器用的 `afxres.h` 替身（构建期 include shim）。
///
/// 上游 `source/SoundTouchDLL/SoundTouchDLL.rc` 第一段就 `#include "afxres.h"` ——
/// 那是 MFC 的头，编译环境没装 MFC 就编不过。以前的做法是**就地改上游文件**，代价是
/// 发行物里的 SoundTouchDLL 成了"修改过的库"：LGPL-2.1 §2(b) 要求每个被改的文件都
/// 带变更声明，于是每个发行包都得塞一份 SOUNDTOUCH-CHANGES.md。
///
/// 现在只改**编译器的搜索路径**：在 OUT_DIR 里生成同名头文件，再用
/// `CMAKE_RC_FLAGS=/I<dir>` 把它排进 rc.exe 的 include 路径。上游源码树因此与
/// 2.3.3 逐字节一致，分发时只需要 LGPL 全文，不需要任何变更披露。
///
/// 【为什么不能改 CMakeLists 加 include_directories】那同样是在改上游文件；
/// `CMAKE_RC_FLAGS` 是**调用方**的配置，源码不动。
///
/// 【为什么是 `-1`】`afxres.h` 里 `IDC_STATIC` 就是 `-1`（标准控件的空 ID）。
const RC_SHIM_AFXRES_H: &str = r#"/* Generated at build time by backend/hifishifter-kernel/build.rs - do not edit.
 *
 * Stand-in for the MFC header that SoundTouch's SoundTouchDLL.rc includes.
 * The SoundTouch source tree stays byte-for-byte identical to upstream 2.3.3;
 * this directory is only added to the resource compiler's include path.
 */
#ifndef HFS_RC_SHIM_AFXRES_H
#define HFS_RC_SHIM_AFXRES_H

#include <windows.h>

#ifndef IDC_STATIC
#define IDC_STATIC -1
#endif

#endif /* HFS_RC_SHIM_AFXRES_H */
"#;

fn build_soundtouch() {
    // ── 三道早退（顺序有意义）────────────────────────────────────────────
    //
    // 1. **移动目标**：SoundTouchDLL 是桌面动态库，移动端没有它的打包位置，也没有
    //    CMake 工具链。移动端只保留 WORLD + Signalsmith 两条链。
    // 2. **feature 关闭**：`soundtouch` 是**可选的运行期依赖**（模块用 libloading
    //    按绝对路径加载，见 soundtouch.rs 的模块头）。关掉它时不该再跑一遍 cmake ——
    //    此前这里无条件执行，于是"不要 SoundTouch 的构建"仍然会去联网克隆上游源码。
    // 3. **跳过开关**：由 main() 处理。
    if is_mobile_target() {
        println!(
            "cargo:warning=[soundtouch] target {} is mobile; skipping SoundTouchDLL (desktop-only)",
            target_os()
        );
        return;
    }
    if std::env::var_os("CARGO_FEATURE_SOUNDTOUCH").is_none() {
        println!(
            "cargo:warning=[soundtouch] feature 'soundtouch' is off; skipping SoundTouchDLL build"
        );
        return;
    }

    println!("cargo:warning=[kernel-native] starting build_soundtouch...");

    let st_src = format!("{}/soundtouch-static/soundtouch", THIRD_PARTY);

    // 【只 watch 真正的输入，不 watch 我们自己写的那棵树】
    //
    // 这棵树在构建过程中**仍会被写入一处**：Step 5 把构建产物 `SoundTouchDLL.dll`
    // 回写到树根（给 tauri_build 做资源校验）；`.git/` 也在里面（pin 校验与"未修改"
    // 校验都要读它）。**源码本身一个字节都不改** —— 资源脚本缺的那个 MFC 头由
    // `soundtouch_rc_shim` 补上，见下文。
    // 早先这里 watch 的是整棵树，于是每次 cargo 都判定"源变了"：cmake configure +
    // cmake build 每次都重跑（实测 ~1.5-2 s/次），产物一变还要连带重编内核、重链
    // 90 MB 的插件测试二进制（~16 s/次）。下面只列真正的输入（C++ 源与头文件）；
    // 产物回写不在 watch 集合里，循环从结构上消失。
    for input in ["source", "include", "CMakeLists.txt"] {
        println!("cargo:rerun-if-changed={}/{}", st_src, input);
    }

    let st_src_path = Path::new(&st_src);
    if !st_src_path.join("CMakeLists.txt").exists() {
        let mirror_entry = sdk_mirror_root().join("soundtouch");
        if mirror_entry.join("CMakeLists.txt").exists() {
            // 本地镜像优先：离网机器靠它取源，全程不触网（见 scripts/setup/export-sdk-mirror.ps1）。
            println!(
                "cargo:warning=[soundtouch] importing source from the local mirror: {}",
                mirror_entry.display()
            );
            if st_src_path.exists() {
                let _ = std::fs::remove_dir_all(st_src_path);
            }
            let parent = st_src_path
                .parent()
                .expect("[soundtouch] invalid source path");
            let _ = std::fs::create_dir_all(parent);
            copy_dir_tree(&mirror_entry, st_src_path)
                .expect("[soundtouch] failed to import source from the local mirror");
            println!("cargo:warning=[soundtouch] SoundTouch source imported from the mirror");
            verify_soundtouch_checkout(st_src_path);
        } else if sdk_offline_requested() {
            eprintln!("\n========================================");
            eprintln!("ERROR: SoundTouch source is missing and offline mode is on!");
            eprintln!("========================================");
            eprintln!("\nHIFISHIFTER_OFFLINE forbids cloning from the network.");
            eprintln!("Provide it via a local SDK mirror instead:");
            eprintln!("  set HIFISHIFTER_SDK_MIRROR=<dir containing a soundtouch/ entry>");
            eprintln!(
                "(create the mirror with scripts/setup/export-sdk-mirror.ps1 on a networked machine)."
            );
            eprintln!("========================================\n");
            panic!("SoundTouch source unavailable offline. See the message above.");
        } else {
            println!("cargo:warning=[soundtouch] SoundTouch source not found, auto-cloning...");
            if st_src_path.exists() {
                let _ = std::fs::remove_dir_all(st_src_path);
            }
            let parent = st_src_path
                .parent()
                .expect("[soundtouch] invalid source path");
            let _ = std::fs::create_dir_all(parent);

            let mut clone = Command::new("git");
            clone.args([
                "clone",
                "--depth",
                "1",
                "--branch",
                "2.3.3",
                "https://codeberg.org/soundtouch/soundtouch.git",
                "soundtouch",
            ]);
            clone.current_dir(parent);

            let status = clone
                .status()
                .expect("[soundtouch] failed to run git clone");
            if !status.success() {
                eprintln!("\n========================================");
                eprintln!("ERROR: Failed to auto-clone SoundTouch source!");
                eprintln!("========================================");
                eprintln!("\nPlease clone manually:");
                eprintln!("  cd backend/src-tauri/third_party/soundtouch-static");
                eprintln!("  git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch");
                eprintln!("========================================\n");
                panic!("SoundTouch source clone failed. See error message above for instructions.");
            }
            println!("cargo:warning=[soundtouch] SoundTouch source cloned successfully");
            verify_soundtouch_checkout(st_src_path);
        }
    } else {
        // 已有检出同样校验：本地残留的旧 clone 不该静默参与构建。
        verify_soundtouch_checkout(st_src_path);
    }

    // Only re-run if build.rs itself changes - the SoundTouch source tree is written to
    // during the build (cmake outputs, the built DLL copied back to its root) which would
    // cause an infinite rebuild loop.
    println!("cargo:rerun-if-changed=build.rs");

    let target = std::env::var("TARGET").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .unwrap_or_else(|_| target.split('-').nth(2).unwrap_or_default().to_string());
    println!(
        "cargo:warning=[soundtouch] TARGET={} TARGET_OS={}",
        target, target_os
    );

    let is_windows = target_os == "windows";
    let is_apple = target_os == "macos";

    println!(
        "cargo:warning=[soundtouch] is_windows={} is_apple={}",
        is_windows, is_apple
    );

    // 【生成器必须显式钉住】CMake 一旦生成过 CMakeCache.txt，就只会复用缓存里的
    // CMAKE_GENERATOR:INTERNAL —— 此后 CMAKE_GENERATOR 环境变量改了也不会生效。本机
    // 实测 8 份旧缓存都写着 Visual Studio 17 2022，而 msvc-env.ps1 给的是
    // NMake Makefiles；旧缓存让 clippy 门禁死在 build script 里，看起来像源码坏了。
    // 因此：声明环境变量触发重跑、把生成器显式交给 -G、并在缓存不一致时清理重建。
    println!("cargo:rerun-if-env-changed=CMAKE_GENERATOR");
    let requested_generator = std::env::var("CMAKE_GENERATOR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        // 【没有环境变量时钉一个确定的答案（Windows）】共享构建目录是**跨进程复用**的，
        // 而"没钉生成器"意味着 cmake 会按 PATH 自己挑 —— vcvars 会把 Visual Studio 自带的
        // Ninja 放上 PATH，于是同一台机器上 A 进程记 Ninja、B 进程记 NMake，共享目录每次
        // 都要擦掉重建（实测一次 = 18 个 C++ 文件重编）。Windows 上 nmake 随 MSVC 一起装，
        // 永远可用，所以它是"没人发话"时的答案；其它平台保持 cmake 默认（Unix Makefiles /
        // Xcode），因为那边没有这个二义性。
        .or_else(|| is_windows.then(|| "NMake Makefiles".to_string()));

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    // 【为什么不再放 OUT_DIR】见 `native_build_root`：`cargo check` / `test` / `clippy`
    // 各有一份 OUT_DIR，cmake 树因此被完整重建过 19 次。放到 target 根的共享位置之后，
    // 只有"源码/旗标/生成器真的变了"才重建。
    // 【目录名带上生成器】否则一次环境变化（Ninja ↔ NMake）会把这棵树整个擦掉重来；
    // 带上之后两种答案各自保留一份，切换只是换目录。
    let generator_key = requested_generator
        .as_deref()
        .map(|value| value.replace(' ', "-").to_lowercase())
        .unwrap_or_else(|| "default".to_string());
    let build_dir =
        native_build_root().join(format!("soundtouch-{}-{}", target_slug(), generator_key));
    println!(
        "cargo:warning=[soundtouch] build_dir={}",
        build_dir.display()
    );

    // 资源脚本的 include shim：只有 Windows 会编 .rc，占位目录也只在那里生成。
    //
    // 【内容必须逐字节稳定】build.rs 每次编译都会跑；内容一样就不重写，免得 rc.exe
    // 的依赖时间戳被无谓刷新（会连带触发资源重新编译）。
    let rc_shim_dir = Path::new(&out_dir).join("soundtouch_rc_shim");
    if is_windows {
        let shim_header = rc_shim_dir.join("afxres.h");
        if std::fs::read_to_string(&shim_header).ok().as_deref() != Some(RC_SHIM_AFXRES_H) {
            std::fs::create_dir_all(&rc_shim_dir)
                .expect("[soundtouch] failed to create the RC include shim directory");
            std::fs::write(&shim_header, RC_SHIM_AFXRES_H)
                .expect("[soundtouch] failed to write the RC include shim");
            println!(
                "cargo:warning=[soundtouch] wrote RC include shim: {}",
                shim_header.display()
            );
        }
    }

    let cache_path = build_dir.join("CMakeCache.txt");
    // 【一个没有 CMakeCache.txt 的非空目录，不是可用的 CMake 树】
    //
    // 现场（2026-11-26 20:07，build.ps1 的插件阶段）：
    //
    //   NMAKE : fatal error U1073: don't know how to make 'CMakeFiles\soundstretch.dir\build.make'
    //
    // 目录里躺着上一次 configure 留下的 `Makefile` / `Makefile2`，而 `CMakeFiles/*/build.make`
    // 与 `CMakeCache.txt` 已经不在 —— 那是一次**被中断、或被部分删除**的 configure 的残骸
    // （`Clear-StaleCmakeCaches` 在文件被占用时只删掉一半，留下的正是这个形状）。cmake 见到
    // `Makefile` 就不再生成，nmake 却按 `Makefile2` 去找已经不存在的 `build.make`，于是
    // **这台机器再也构建不了**，而错误信息里没有一个字提到"缓存坏了"。
    //
    // 判据只用一条：**没有 cache 的目录不能复用**。有 cache 的目录由 cmake 自己保证一致性。
    if !cache_path.is_file() && has_entries(&build_dir) {
        println!(
            "cargo:warning=[soundtouch] {} is a build tree without CMakeCache.txt (an interrupted configure); wiping it",
            build_dir.display()
        );
        remove_tree(&build_dir);
    }

    if let Some(generator) = requested_generator.as_deref() {
        let cached = std::fs::read_to_string(&cache_path).ok();
        let cached_generator = cached.as_deref().and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("CMAKE_GENERATOR:INTERNAL="))
        });
        if cached_generator.is_some_and(|value| value != generator) {
            println!(
                "cargo:warning=[soundtouch] generator changed from {} to {}; reconfiguring",
                cached_generator.unwrap_or("unknown"),
                generator
            );
            remove_tree(&build_dir);
        }
    }

    // 【已经最新就不必再问 cmake】构建脚本在**每次 `git commit`、每次改 `src/`** 时都会
    // 重跑；而这段代码是它里面唯一一段"跑子进程"的工作。三条 stat 就能回答"要不要跑"，
    // 于是那些必然发生的重跑从 0.5–1.5 s 变成 0.1 ms。
    let apple_arch = is_apple.then(cmake_apple_arch);
    if soundtouch_is_fresh(
        &build_dir,
        st_src_path,
        requested_generator.as_deref(),
        is_windows,
        is_apple,
        apple_arch.as_deref(),
    ) {
        println!(
            "cargo:warning=[soundtouch] reusing {} (up to date)",
            build_dir.display()
        );
    } else {
        // 【为什么要重试一次】上面那条判据能挡住"残骸目录"，但挡不住"configure 本身被中断在
        // 半路"。一次擦除重来把任何这类残骸都变成一次成功构建 —— 这正是本仓库在别处已经
        // 用过的纪律（`Clear-StaleCmakeCaches` 的"陈旧缓存自愈"）。重试只做一次：第二次仍
        // 失败说明问题在源码或工具链，而不是缓存。
        let mut attempt = 0;
        loop {
            attempt += 1;
            match configure_and_build_soundtouch(
                &build_dir,
                st_src_path,
                requested_generator.as_deref(),
                &rc_shim_dir,
                is_windows,
                is_apple,
                apple_arch.as_deref(),
            ) {
                Ok(()) => break,
                Err(message) => {
                    if attempt >= 2 {
                        panic!("{message}");
                    }
                    println!(
                        "cargo:warning=[soundtouch] {message}; wiping {} and retrying once",
                        build_dir.display()
                    );
                    remove_tree(&build_dir);
                }
            }
        }
        println!("cargo:warning=[soundtouch] cmake build succeeded");
    }

    // Step 3: Find the built SoundTouchDLL shared library
    let lib_name = "SoundTouchDLL";
    let lib_filename = if is_windows {
        format!("{}.dll", lib_name)
    } else if is_apple {
        format!("lib{}.dylib", lib_name)
    } else {
        format!("lib{}.so", lib_name)
    };

    let lib_src = find_file(&build_dir, &lib_filename).unwrap_or_else(|| {
        panic!(
            "[soundtouch] Could not find {} in build directory {}",
            lib_filename,
            build_dir.display()
        )
    });
    println!(
        "cargo:warning=[soundtouch] found shared lib: {}",
        lib_src.display()
    );

    // Force a stable, relocatable Mach-O install name.
    if is_apple {
        let install_name = format!("@rpath/{}", lib_filename);
        let status = Command::new("install_name_tool")
            .arg("-id")
            .arg(&install_name)
            .arg(&lib_src)
            .status()
            .expect("[soundtouch] failed to run install_name_tool");
        if !status.success() {
            panic!(
                "[soundtouch] install_name_tool failed to set {} on {}",
                install_name,
                lib_src.display()
            );
        }
    }

    // Step 4: Link against the shared library
    let lib_search = lib_src.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", lib_search.display());
    println!("cargo:rustc-link-lib=dylib={}", lib_name);

    if is_apple {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources/macos");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    } else if !is_windows {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/HiFiShifter");
    }

    // Step 5: Copy shared library to the target dir (for runtime linking) AND to
    // the source tree (for Tauri resource bundling / tauri_build validation).
    let target_dir = Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .expect("[soundtouch] unexpected OUT_DIR depth");
    let lib_dst_target = target_dir.join(&lib_filename);

    if let Err(e) = std::fs::copy(&lib_src, &lib_dst_target) {
        println!(
            "cargo:warning=[soundtouch] could not copy {} to {}: {}",
            lib_src.display(),
            lib_dst_target.display(),
            e
        );
    }

    // Test executables live in target/<profile>/deps/.
    let deps_dir = target_dir.join("deps");
    let _ = std::fs::create_dir_all(&deps_dir);
    let lib_dst_deps = deps_dir.join(&lib_filename);
    if lib_dst_deps != lib_dst_target {
        let _ = std::fs::copy(&lib_src, &lib_dst_deps);
    }

    // Also copy to source tree path for tauri_build resource validation.
    // IMPORTANT: only write if bytes differ - writing unconditionally updates the
    // file timestamp every build, which triggers Tauri's dev watcher and causes
    // an infinite rebuild loop.
    let lib_dst_resource = st_src_path.join(&lib_filename);
    let src_bytes = std::fs::read(&lib_src).unwrap_or_default();
    let dst_bytes = std::fs::read(&lib_dst_resource).unwrap_or_default();
    if src_bytes != dst_bytes {
        // 【为什么先写临时文件再改名】这棵树**同时被另一个 target 目录的构建读**：app 的
        // build.rs 与 tauri 的资源校验都会打开它，而 debug 与 release 共享同一份源码树
        // （两个 cargo 进程不会互斥，它们的 target 目录不同）。`fs::write` 先截断再写
        // 57 KB，读到中间态的构建会拿到半个 DLL —— 那是一个只在"两边同时构建"时出现、
        // 且报错完全指向别处的故障。同目录改名是原子的：读方要么看到旧文件，要么看到新文件。
        let temp =
            lib_dst_resource.with_file_name(format!("{}.{}.tmp", lib_filename, std::process::id()));
        let written = std::fs::write(&temp, &src_bytes)
            .and_then(|()| std::fs::rename(&temp, &lib_dst_resource));
        if let Err(e) = written {
            let _ = std::fs::remove_file(&temp);
            println!(
                "cargo:warning=[soundtouch] could not write {}: {}",
                lib_dst_resource.display(),
                e
            );
        }
    }

    // 回写完成的事实以 metadata 形式交给依赖方 build script（DEP_SOUNDTOUCH_*）。
    // links 键保证本脚本先于它们的 build script 结束 —— mac 打包的资源校验
    // （tauri.conf.json 的 frameworks）因此不再与本回写竞态。
    println!("cargo:soundtouch_dll={}", lib_dst_resource.display());
}

/// 共享的 SoundTouch 构建树是否已经是最新的。
///
/// 判据只有三条，都是**廉价 stat**：
/// 1. `CMakeCache.txt` 在，且记录的生成器与本次要求一致；
/// 2. 目标库在构建树里；
/// 3. 它的 mtime 不早于源码树里的**每一个**输入（`source/`、`include/`、`CMakeLists.txt`）。
///
/// 三条同时成立时，configure + build 只会把同一份字节再确认一遍（实测 0.5–1.5 s）。
/// 不确定时一律返回 false —— "多跑一次 cmake"只是慢，"少跑一次"会拿到旧库。
fn soundtouch_is_fresh(
    build_dir: &Path,
    src: &Path,
    generator: Option<&str>,
    is_windows: bool,
    is_apple: bool,
    apple_arch: Option<&str>,
) -> bool {
    let Ok(text) = std::fs::read_to_string(build_dir.join("CMakeCache.txt")) else {
        return false;
    };
    if let Some(wanted) = generator {
        let recorded = text
            .lines()
            .find_map(|line| line.strip_prefix("CMAKE_GENERATOR:INTERNAL="));
        if recorded.is_some_and(|value| value != wanted) {
            return false;
        }
    }
    // 【为什么架构也算"新鲜"的一部分】缓存里没有这一项 = 这棵树是在"还没把架构传给 cmake"
    // 的时候 configure 的，产物是宿主架构。判为不新鲜，让它重配一次 —— 否则修好之后，凡是
    // 本地已经跑过一次旧代码的人都会继续拿到那份错的 dylib。
    if let Some(wanted) = apple_arch {
        let recorded = text
            .lines()
            .find_map(|line| line.strip_prefix("CMAKE_OSX_ARCHITECTURES:"))
            .and_then(|rest| rest.split_once('=').map(|(_kind, value)| value));
        if recorded != Some(wanted) {
            return false;
        }
    }
    let lib_filename = if is_windows {
        "SoundTouchDLL.dll"
    } else if is_apple {
        "libSoundTouchDLL.dylib"
    } else {
        "libSoundTouchDLL.so"
    };
    let Some(lib) = find_file(build_dir, lib_filename) else {
        return false;
    };
    let Ok(lib_time) = std::fs::metadata(&lib).and_then(|meta| meta.modified()) else {
        return false;
    };
    // 【只比真正的输入】源码树根上还有 Step 5 回写的 `SoundTouchDLL.dll` —— 它是**产物**，
    // 拿它当输入会让判据永远为假（刚写的 DLL 永远比构建树里的新），于是每次都重跑 cmake。
    // 所以比较集合与 `rerun-if-changed` 完全一致：`source/`、`include/`、`CMakeLists.txt`。
    let inputs = [
        src.join("source"),
        src.join("include"),
        src.join("CMakeLists.txt"),
    ];
    let Some(newest_input) = newest_mtime(&inputs) else {
        return false;
    };
    lib_time >= newest_input
}

/// 一组路径里最新的 mtime（目录会递归；取不到任何时间戳时返回 None）。
fn newest_mtime(roots: &[PathBuf]) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;
    let mut stack: Vec<PathBuf> = Vec::new();
    for root in roots {
        if let Ok(modified) = std::fs::metadata(root).and_then(|meta| meta.modified()) {
            newest = Some(newest.map_or(modified, |current| current.max(modified)));
        }
        stack.push(root.clone());
    }
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                stack.push(entry.path());
            }
            if let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) {
                newest = Some(newest.map_or(modified, |current| current.max(modified)));
            }
        }
    }
    newest
}

/// 目录存在且非空。
fn has_entries(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// 删掉一棵构建树（不存在也算成功）。
///
/// 【为什么不 expect】这里唯一的调用时机是"已知这棵树坏了"。删不掉（文件被另一个进程占用）
/// 不该让构建以一个与本问题无关的 panic 结束 —— 让后面的 cmake 去报它自己的错，读日志的人
/// 才能看到真正的现场。
fn remove_tree(dir: &Path) {
    if let Err(error) = std::fs::remove_dir_all(dir) {
        if error.kind() != std::io::ErrorKind::NotFound {
            println!(
                "cargo:warning=[soundtouch] could not remove {}: {error}",
                dir.display()
            );
        }
    }
    let _ = std::fs::create_dir_all(dir);
}

/// 配置并构建 SoundTouchDLL；失败时返回**可读的原因**，而不是直接 panic。
///
/// 【为什么返回 Result 而不是 panic】调用方要"失败就擦掉重来一次"（见 [`build_soundtouch`]）。
/// 在这里 panic 会把一次可以自愈的缓存残骸，变成一台再也构建不了的机器。
/// CMake 生成器真正依赖的那个可执行文件（`NMake Makefiles` → `nmake`，`Ninja` → `ninja`）。
///
/// 【为什么要在 build script 里知道这件事】生成器是**环境层**钉住的（见
/// `scripts/env/native-toolchain.ps1`），而环境层只在"导入过 MSVC 环境"的进程里才生效。
/// 一旦某个入口没有导入（2026-11-27 的 windows-arm64 CI 腿就是如此），cmake 会先报
/// `Running 'nmake' '-?' failed` 再报 `CMAKE_CXX_COMPILER not set`，最后由这里 panic ——
/// 整条链路没有一个字提到"工具链没进来"，排查要翻四十分钟日志。
fn generator_driver(generator: &str) -> Option<&'static str> {
    match generator {
        "NMake Makefiles" => Some("nmake"),
        "Ninja" => Some("ninja"),
        _ => None,
    }
}

/// PATH 上是否存在可执行的 `name`（Windows 上同时试 `.exe`）。
fn command_on_path(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    let mut names = vec![name.to_string()];
    if cfg!(windows) {
        names.push(format!("{name}.exe"));
    }
    for dir in std::env::split_paths(&path) {
        for candidate in &names {
            let full = dir.join(candidate);
            if full.is_file() {
                return true;
            }
        }
    }
    false
}
#[allow(clippy::too_many_arguments)]
fn configure_and_build_soundtouch(
    build_dir: &Path,
    st_src_path: &Path,
    requested_generator: Option<&str>,
    rc_shim_dir: &Path,
    is_windows: bool,
    is_apple: bool,
    apple_arch: Option<&str>,
) -> Result<(), String> {
    // 【缺工具时立刻说人话】生成器由环境层钉住；这里只负责在"它要的那个可执行文件不在
    // PATH 上"时，把可执行的下一步直接写进错误信息，而不是让 cmake 抛两条互不相关的报错。
    if let Some(generator) = requested_generator {
        if let Some(driver) = generator_driver(generator) {
            if !command_on_path(driver) {
                panic!(
                    "[soundtouch] CMake generator `{}` needs `{}` on PATH, but it is not there.\n\
                     On Windows the MSVC environment must be imported into the SAME process first:\n    \
                     . .\\scripts\\env\\msvc-env.ps1          # local / build.ps1\n    \
                     scripts\\setup\\export-msvc-env.ps1     # CI (writes $GITHUB_ENV)\n\
                     TARGET={}",
                    generator,
                    driver,
                    std::env::var("TARGET").unwrap_or_default()
                );
            }
        }
    }

    // Step 1: CMake configure - build SoundTouchDLL as a shared library.
    let mut cfg = Command::new("cmake");
    cfg.arg("-S").arg(st_src_path);
    cfg.arg("-B").arg(build_dir);
    cfg.arg("-DCMAKE_POLICY_VERSION_MINIMUM=3.5");
    // 上游 CMakeLists.txt 的 cmake_minimum_required(VERSION 3.1) 在现代 CMake 上会发
    // deprecation 警告。上游源码必须保持 pristine（见 verify_soundtouch_checkout），
    // 所以从调用方关掉这条与构建结果无关的噪音。
    cfg.arg("-Wno-deprecated");
    cfg.arg("-DCMAKE_BUILD_TYPE=Release");
    cfg.arg("-DSOUNDTOUCH_DLL=ON");
    // 【为什么不构建 soundstretch】上游的 `SOUNDSTRETCH` 默认 ON：它多编三个 CLI 源文件、
    // 再链一个我们从不加载的控制台程序。本仓库只 load `SoundTouchDLL`（见 soundtouch.rs），
    // 多出来的目标既是纯粹的时间，也是 2026-11-26 那次
    // `U1073: ... soundstretch.dir\build.make` 的落点 —— 不让它进构建图，那条失败路径
    // 就不存在了。
    cfg.arg("-DSOUNDSTRETCH=OFF");
    if let Some(generator) = requested_generator {
        cfg.arg("-G").arg(generator);
        println!(
            "cargo:warning=[soundtouch] using pinned CMAKE_GENERATOR={}",
            generator
        );
    }
    if is_windows {
        // 引号是给 rc.exe 的：OUT_DIR 可能落在带空格的路径下（用户名里带空格很常见）。
        cfg.arg(format!(
            "-DCMAKE_RC_FLAGS:STRING=/I\"{}\"",
            rc_shim_dir.display()
        ));
    }

    if is_apple {
        cfg.arg("-DCMAKE_INSTALL_NAME_DIR=@rpath");
        cfg.arg("-DCMAKE_MACOSX_RPATH=ON");
        // 见 `cmake_apple_arch`：不给这一项，cmake 就按宿主架构编。
        if let Some(arch) = apple_arch {
            cfg.arg(format!("-DCMAKE_OSX_ARCHITECTURES={arch}"));
        }
    }

    println!("cargo:warning=[soundtouch] spawning cmake configure...");
    let output = cfg
        .output()
        .map_err(|error| format!("[soundtouch] failed to run cmake configure: {error}"))?;
    if !output.status.success() {
        println!(
            "cargo:warning=[soundtouch] cmake configure stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        println!(
            "cargo:warning=[soundtouch] cmake configure stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        return Err(format!(
            "[soundtouch] CMake configure failed with exit code {:?}",
            output.status.code()
        ));
    }
    println!("cargo:warning=[soundtouch] cmake configure succeeded");

    // Step 2: CMake build - only the DLL target we actually load.
    let mut bld = Command::new("cmake");
    bld.arg("--build").arg(build_dir);
    bld.arg("--target").arg("SoundTouchDLL");
    bld.arg("--config").arg("Release");

    println!("cargo:warning=[soundtouch] spawning cmake build...");
    let output = bld
        .output()
        .map_err(|error| format!("[soundtouch] failed to run cmake build: {error}"))?;
    if !output.status.success() {
        println!(
            "cargo:warning=[soundtouch] cmake build stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        println!(
            "cargo:warning=[soundtouch] cmake build stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        return Err(format!(
            "[soundtouch] CMake build failed with exit code {:?}",
            output.status.code()
        ));
    }
    Ok(())
}

/// Recursively search for a file by name under `dir`.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }

    let mut dirs_to_visit = vec![dir.to_path_buf()];

    while let Some(current) = dirs_to_visit.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs_to_visit.push(path);
            } else if path.is_file() {
                if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                    if fname == name {
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}
