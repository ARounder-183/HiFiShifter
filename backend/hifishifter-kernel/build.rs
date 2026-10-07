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

use std::path::{Path, PathBuf};
use std::process::Command;

/// 第三方原生源码目录。
///
/// 【为什么是 `../src-tauri/...`】`third_party/` 目前仍在 app crate 里（见文件头说明）。
/// 构建脚本的当前工作目录是**包根**（`backend/hifishifter-kernel`），所以从这个相对
/// 路径可以稳定到达它。这是**构建期**耦合，不影响运行期依赖方向。
const THIRD_PARTY: &str = "../src-tauri/third_party";

fn main() {
    // 与 app 的 build.rs 共用同一个跳过开关：CI 只做类型检查时可以整体跳过原生构建。
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
    world
        .cpp(true)
        .include(&world_src_dir)
        .files(required_files.iter().map(|f| format!("{}/{}", world_src_dir, f)));

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
        if !cfg!(target_os = "windows") {
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
fn build_soundtouch() {
    println!("cargo:warning=[kernel-native] starting build_soundtouch...");

    let st_src = format!("{}/soundtouch-static/soundtouch", THIRD_PARTY);

    // Re-run this script only when the SoundTouch source tree changes.
    println!("cargo:rerun-if-changed={}", st_src);

    let st_src_path = Path::new(&st_src);
    if !st_src_path.join("CMakeLists.txt").exists() {
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
    }

    // Only re-run if build.rs itself changes - the SoundTouch source tree is modified
    // during the build (cmake outputs, .rc patching) which would cause an infinite rebuild loop.
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

    // Patch SoundTouchDLL.rc to use windows.h instead of afxres.h (MFC header not always available)
    if is_windows {
        let rc_file = st_src_path
            .join("source")
            .join("SoundTouchDLL")
            .join("SoundTouchDLL.rc");
        if rc_file.exists() {
            let content = std::fs::read_to_string(&rc_file)
                .expect("[soundtouch] failed to read SoundTouchDLL.rc");
            // Only write if the file actually needs patching to avoid triggering Tauri's file watcher.
            if content.contains("afxres.h") && !content.contains("#include <windows.h>") {
                let patched = content.replace("#include \"afxres.h\"", "#include <windows.h>");
                // IDC_STATIC is normally defined in afxres.h as -1
                let patched = if !patched.contains("IDC_STATIC") {
                    patched.replace(
                        "#include <windows.h>",
                        "#include <windows.h>\n#ifndef IDC_STATIC\n#define IDC_STATIC -1\n#endif",
                    )
                } else {
                    patched
                };
                if patched != content {
                    std::fs::write(&rc_file, &patched)
                        .expect("[soundtouch] failed to write patched SoundTouchDLL.rc");
                    println!("cargo:warning=[soundtouch] patched SoundTouchDLL.rc to use windows.h");
                }
            }
        }
    }

    // Patch SoundTouch CMakeLists.txt - cmake_minimum_required(VERSION 3.1) is
    // deprecated in CMake ≥3.27 and a hard error in CMake ≥4.0.  Bump to 3.5.
    {
        let cmake_file = st_src_path.join("CMakeLists.txt");
        if cmake_file.exists() {
            let content = std::fs::read_to_string(&cmake_file)
                .expect("[soundtouch] failed to read CMakeLists.txt");
            let patched = content.replace(
                "cmake_minimum_required(VERSION 3.1)",
                "cmake_minimum_required(VERSION 3.5)",
            );
            if patched != content {
                std::fs::write(&cmake_file, &patched)
                    .expect("[soundtouch] failed to write patched CMakeLists.txt");
                println!("cargo:warning=[soundtouch] patched CMakeLists.txt: cmake_minimum_required 3.1 → 3.5");
            }
        }
    }

    println!(
        "cargo:warning=[soundtouch] is_windows={} is_apple={}",
        is_windows, is_apple
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let build_dir = Path::new(&out_dir).join("soundtouch_build");
    println!(
        "cargo:warning=[soundtouch] build_dir={}",
        build_dir.display()
    );

    // Step 1: CMake configure - build SoundTouchDLL as a shared library.
    let mut cfg = Command::new("cmake");
    cfg.arg("-S").arg(st_src_path);
    cfg.arg("-B").arg(&build_dir);
    cfg.arg("-DCMAKE_POLICY_VERSION_MINIMUM=3.5");
    cfg.arg("-DCMAKE_BUILD_TYPE=Release");
    cfg.arg("-DSOUNDTOUCH_DLL=ON");

    if is_apple {
        cfg.arg("-DCMAKE_INSTALL_NAME_DIR=@rpath");
        cfg.arg("-DCMAKE_MACOSX_RPATH=ON");
    }

    println!("cargo:warning=[soundtouch] spawning cmake configure...");
    let status = cfg
        .status()
        .expect("[soundtouch] failed to run cmake configure");
    if !status.success() {
        panic!(
            "[soundtouch] CMake configure failed with exit code {:?}",
            status.code()
        );
    }
    println!("cargo:warning=[soundtouch] cmake configure succeeded");

    // Step 2: CMake build - build SoundTouchDLL target
    let mut bld = Command::new("cmake");
    bld.arg("--build").arg(&build_dir);
    bld.arg("--config").arg("Release");

    println!("cargo:warning=[soundtouch] spawning cmake build...");
    let output = bld
        .output()
        .expect("[soundtouch] failed to run cmake build");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("cargo:warning=[soundtouch] cmake build stderr:\n{}", stderr);
        println!("cargo:warning=[soundtouch] cmake build stdout:\n{}", stdout);
        panic!(
            "[soundtouch] CMake build failed with exit code {:?}",
            output.status.code()
        );
    }
    println!("cargo:warning=[soundtouch] cmake build succeeded");

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
        if let Err(e) = std::fs::write(&lib_dst_resource, &src_bytes) {
            println!(
                "cargo:warning=[soundtouch] could not write {}: {}",
                lib_dst_resource.display(),
                e
            );
        }
    }
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
