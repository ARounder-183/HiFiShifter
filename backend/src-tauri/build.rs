fn main() {
    build_frontend();

    // Allow skipping expensive native builds in CI checks via env var
    // Set HIFISHIFTER_SKIP_NATIVE_BUILD=1 to skip WORLD/Signalsmith/VSLIB builds
    let skip_native = std::env::var("HIFISHIFTER_SKIP_NATIVE_BUILD").unwrap_or_default();
    if skip_native != "1" {
        build_world_static();
        build_signalsmith_stretch();
        build_vslib();
        build_soundtouch();
    } else {
        println!("cargo:warning=[build.rs] Skipping native library builds (HIFISHIFTER_SKIP_NATIVE_BUILD=1)");
        // Create placeholder files so tauri_build resource validation passes
        for placeholder in &[
            "third_party/soundtouch-static/soundtouch/SoundTouchDLL.dll",
            "third_party/soundtouch-static/soundtouch/libSoundTouchDLL.so",
            "third_party/soundtouch-static/soundtouch/libSoundTouchDLL.dylib",
        ] {
            let p = std::path::Path::new(placeholder);
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(p, b"");
        }
    }

    // Create placeholder for vslib_x64.dll on non-x86_64 Windows targets
    // so tauri_build resource validation passes.
    create_vslib_placeholder();

    // Copy DirectML.dll from the ORT build output to the resources directory
    // so Tauri can bundle it into the NSIS installer. DirectML.dll is loaded
    // dynamically by ONNX Runtime (not linked at compile time), so Tauri's
    // dependency scanner does not detect it. Explicit resource listing in
    // tauri.windows.conf.json handles the bundling.
    copy_directml_dll();

    // Stage the ONNX Runtime dylibs (WebGPU/Dawn, CoreML providers, ...) into
    // resources/macos so tauri.macos.conf.json can bundle them into
    // Contents/Resources/macos. Without this, the app links
    // @rpath/libwebgpu_dawn.dylib but the .app does not contain it, and dyld
    // fails at launch.
    stage_ort_macos_dylibs();

    // Generate a CoreML-compatible NSF-HiFiGAN model variant (macOS ARM64
    // only): the stock model derives Pad pads at runtime, which the CoreML EP
    // cannot compile.  The generated file is gitignored and produced on
    // demand during the build.
    generate_coreml_model_variant();

    // tauri_build validates resources listed in tauri.conf.json and its
    // platform-specific merges (tauri.windows.conf.json, tauri.linux.conf.json),
    // and copies bundle.macOS.frameworks for darwin targets. All referenced
    // files must exist before this call.
    tauri_build::build();

    // ── Windows：为【测试二进制】嵌入 ComCtl32 v6 应用清单 ─────────────────
    // tauri_build 只通过 `cargo:rustc-link-arg-bins` 给 bin 目标嵌清单；
    // cargo test 的测试 harness exe 不在其中。而依赖树（winit/tauri dialog）
    // 静态导入了 comctl32.dll!TaskDialogIndirect —— 该函数只存在于 v6
    // side-by-side 程序集，无清单时 loader 绑定到 System32 的 v5 副本，
    // 进程初始化阶段直接 STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139) 失败，
    // 所有 cargo test 在 Windows 上都无法启动。
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        let manifest_out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default())
            .join("hifishifter_tests.manifest");
        let manifest_xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="*"
        publicKeyToken="6595b64144ccf1df"
        language="*"
      />
    </dependentAssembly>
  </dependency>
</assembly>
"#;
        let _ = std::fs::write(&manifest_out, manifest_xml);
        // 注意：只能用 `-tests`（作用于 tests/ 下的集成测试目标）。
        // 通用的 `rustc-link-arg` 会同时打到主 bin —— 主程序已由 tauri_build
        // 通过 resource.lib 内嵌了清单，再来一份 /MANIFEST:EMBED 会触发
        // CVT1100 duplicate resource。而 lib 单元测试 harness 没有对应的
        // link-arg 通道（cargo 限制），其在 Windows 上无法启动的问题见
        // tests/smoke.rs 的说明。
        println!("cargo:rustc-link-arg-tests=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-tests=/MANIFESTINPUT:{}",
            manifest_out.display()
        );
        println!("cargo:rustc-link-arg-tests=/MANIFESTUAC:NO");
    }
}

/// 在编译时自动构建前端静态资源。
///
/// 当 `frontend/dist` 目录不存在时，自动执行 `npm run build` 生成前端产物，
/// 确保 Tauri 能找到 `frontendDist`。
/// 若 dist 已存在则跳过（开发者可手动删除 dist 目录强制重建）。
fn build_frontend() {
    use std::path::Path;
    use std::process::Command;

    // build.rs 的工作目录是 src-tauri/，前端目录在上两级
    let frontend_dir = Path::new("../../frontend");
    let dist_dir = frontend_dir.join("dist");

    if !frontend_dir.exists() {
        println!("cargo:warning=[Frontend] frontend 目录不存在，跳过前端构建");
        return;
    }

    // 当关键文件变更时重新触发 build.rs
    println!("cargo:rerun-if-changed=../../frontend/src");
    println!("cargo:rerun-if-changed=../../frontend/package.json");
    println!("cargo:rerun-if-changed=../../frontend/vite.config.ts");
    println!("cargo:rerun-if-changed=../../frontend/vite.config.js");

    // Allow CI to skip frontend build if artifact is provided.
    // Set HIFISHIFTER_SKIP_FRONTEND_BUILD=1 to skip building frontend here.
    let skip_frontend = std::env::var("HIFISHIFTER_SKIP_FRONTEND_BUILD").unwrap_or_default();
    if skip_frontend == "1" {
        println!(
            "cargo:warning=[Frontend] HIFISHIFTER_SKIP_FRONTEND_BUILD=1 -> skipping frontend build"
        );
        return;
    }

    // dist 已存在则跳过，避免每次编译都重新构建前端
    if dist_dir.exists() {
        println!("cargo:warning=[Frontend] dist 已存在，跳过构建（删除 frontend/dist 可强制重建）");
        return;
    }

    println!("cargo:warning=[Frontend] 正在构建前端，请稍候...");

    let npm_cmd = if cfg!(target_os = "windows") {
        "npm.cmd"
    } else {
        "npm"
    };

    let status = Command::new(npm_cmd)
        .arg("run")
        .arg("build")
        .current_dir(frontend_dir)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=[Frontend] 前端构建成功");
        }
        Ok(s) => {
            panic!("[Frontend] 前端构建失败，退出码: {:?}", s.code());
        }
        Err(e) => {
            panic!(
                "[Frontend] 无法执行 npm run build: {}。请确保已安装 Node.js 和 npm。",
                e
            );
        }
    }
}

/// Build WORLD vocoder as a static library using cc crate.
///
/// Since v2026.03, WORLD is statically linked at compile time instead of
/// dynamically loaded via DLL. This approach provides:
/// - Single self-contained binary (no external DLL dependencies)
/// - Improved reliability (no runtime loading failures)
/// - Simplified cross-platform builds
/// - Faster startup (no DLL search overhead)
///
/// Source location: third_party/world-static/World/
/// Build time: ~60-90s on first build, ~5-10s incremental
///
/// The WORLD library (https://github.com/mmorise/World) provides:
/// - Dio/Harvest: F0 (pitch) analysis algorithms
/// - CheapTrick: Spectral envelope estimation
/// - D4C: Aperiodicity estimation
/// - Synthesis: High-quality vocoder reconstruction
fn build_world_static() {
    use std::path::Path;

    let world_src_dir = "third_party/world-static/World/src";
    let world_src_path = Path::new(world_src_dir);

    // Check if WORLD sources exist
    if !world_src_path.exists() {
        eprintln!("\n========================================");
        eprintln!("ERROR: WORLD source code not found!");
        eprintln!("========================================");
        eprintln!("\nExpected location: {}", world_src_path.display());
        eprintln!("\nTo fix this, run:");
        eprintln!("  cd backend/src-tauri/third_party/world-static");
        eprintln!("  git clone https://github.com/mmorise/World.git");
        eprintln!("\nOr from project root:");
        eprintln!("  git clone https://github.com/mmorise/World.git backend/src-tauri/third_party/world-static/World");
        eprintln!("========================================\n");
        panic!("WORLD sources missing. See error message above for instructions.");
    }

    // Verify all required source files exist
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

    // Compile WORLD as static library
    cc::Build::new()
        .cpp(true)
        .std("c++11")
        .include(world_src_dir)
        .file(format!("{}/cheaptrick.cpp", world_src_dir))
        .file(format!("{}/codec.cpp", world_src_dir))
        .file(format!("{}/common.cpp", world_src_dir))
        .file(format!("{}/d4c.cpp", world_src_dir))
        .file(format!("{}/dio.cpp", world_src_dir))
        .file(format!("{}/fft.cpp", world_src_dir))
        .file(format!("{}/harvest.cpp", world_src_dir))
        .file(format!("{}/matlabfunctions.cpp", world_src_dir))
        .file(format!("{}/stonemask.cpp", world_src_dir))
        .file(format!("{}/synthesis.cpp", world_src_dir))
        .file(format!("{}/synthesisrealtime.cpp", world_src_dir))
        .compile("world");

    println!("cargo:rustc-link-lib=static=world");
}

/// Build Signalsmith Stretch as a static library using cc crate.
///
/// Signalsmith Stretch (https://github.com/Signalsmith-Audio/signalsmith-stretch)
/// is a header-only C++ library for pitch and time stretching.
/// We compile a thin C wrapper (sstretch-c.cpp) that exposes a C API for Rust FFI.
///
/// License: MIT (no GPL restrictions)
/// Build time: ~10-30s (much faster than Rubber Band)
///
/// Dependencies:
///   - signalsmith-linear (STFT library): git submodule in signalsmith-stretch/
///
/// Source location: third_party/signalsmith-stretch/
fn build_signalsmith_stretch() {
    use std::path::Path;

    let ss_base = "third_party/signalsmith-stretch";
    let ss_lib_dir = format!("{}/signalsmith-stretch", ss_base);
    let ss_wrapper = format!("{}/sstretch-c.cpp", ss_base);
    let ss_lib_path = Path::new(&ss_lib_dir);

    // Check if Signalsmith Stretch sources exist
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

    // Verify signalsmith-linear dependency exists
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

    // Verify critical files
    let stretch_h = format!("{}/signalsmith-stretch.h", ss_lib_dir);
    if !Path::new(&stretch_h).exists() {
        panic!("signalsmith-stretch.h not found at {}", stretch_h);
    }

    println!("cargo:rerun-if-changed={}", ss_base);

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .warnings(false)
        // Include paths:
        // - signalsmith-stretch/ 目录（signalsmith-stretch.h 所在）
        // - signalsmith-stretch/signalsmith-linear/ 目录（stft.h 等依赖）
        // - sstretch-c.h 所在的 wrapper 目录
        .include(&ss_lib_dir)
        .include(&linear_dir)
        .include(ss_base)
        // 只需编译我们的 C wrapper，stretch 库本身是 header-only
        .file(&ss_wrapper);

    // Platform-specific flags
    let compiler = build.get_compiler();
    if compiler.is_like_msvc() {
        build.flag("/EHsc");
        build.flag("/std:c++14");
        build.define("NOMINMAX", None);
        // 启用优化以提升 number-crunching 性能（即使在 Debug 模式下）
        build.flag("/O2");
    } else {
        build.flag("-std=c++14");
        if !cfg!(target_os = "windows") {
            build.flag("-fPIC");
        }
        // 启用优化（Signalsmith 文档建议即使 Debug 也开启优化）
        build.flag("-O2");
    }

    build.compile("signalsmith_stretch");

    println!("cargo:rustc-link-lib=static=signalsmith_stretch");
}

/// Link against vslib_x64.dll via its import library.
///
/// The DLL and import lib live in third_party/vslib/:
///   vslib_x64.dll  - needs to sit next to the final binary at runtime
///   vslib_x64.lib  - import library linked at compile time
///
/// Enabled only when the `vslib` cargo feature is active.
fn build_vslib() {
    if !cfg!(feature = "vslib") {
        return;
    }

    // Only link/copy for x86_64 Windows targets. Non-target platforms should
    // not require third_party/vslib assets to exist.
    let target = std::env::var("TARGET").unwrap_or_default();
    let target_lc = target.to_lowercase();
    if !(target_lc.contains("windows") && target_lc.contains("x86_64")) {
        println!("cargo:warning=[vslib] target '{}' not an x86_64 Windows target; skipping link/copy of vslib_x64", target);
        return;
    }

    let lib_dir = std::path::Path::new("third_party/vslib");

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

    println!("cargo:rerun-if-changed=third_party/vslib/vslib_x64.lib");
    println!("cargo:rerun-if-changed=third_party/vslib/vslib_x64.dll");

    println!("cargo:rustc-link-search=native={}", abs.display());
    println!("cargo:rustc-link-lib=dylib=vslib_x64");

    // OUT_DIR = .../target/<profile>/build/<pkg>/out  →  4 levels up = target/<profile>/
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let dll_src = lib_dir.join("vslib_x64.dll");
        let target_dir = std::path::Path::new(&out_dir)
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
        } else {
            println!(
                "cargo:warning=[vslib] copied vslib_x64.dll to {}",
                dll_dst.display()
            );
        }
    } else {
        println!("cargo:warning=[vslib] OUT_DIR not set; skipping DLL copy")
    }
}

/// Build SoundTouch as a shared library via CMake for all platforms.
///
/// Compiles SoundTouch from source located at third_party/soundtouch-static/soundtouch/
/// and links against the resulting shared library (dynamic linking for LGPL compliance).
///
/// Strategy:
///   1. CMake builds the core SoundTouch C++ library as a static lib
///   2. We manually compile SoundTouchDLL.cpp (the C API wrapper) as a shared lib,
///      linking it against the static SoundTouch lib
///
/// Supported targets:
///   - Windows x86_64 / ARM64  → SoundTouchDLL.dll
///   - macOS   x86_64 / ARM64  → libSoundTouchDLL.dylib
///   - Linux   x86_64 / ARM64  → libSoundTouchDLL.so
fn build_soundtouch() {
    use std::path::Path;
    use std::process::Command;

    println!("cargo:warning=[soundtouch] starting build_soundtouch...");

    let st_src = "third_party/soundtouch-static/soundtouch";

    // Verify SoundTouch source exists; auto-clone if missing
    let st_src_path = Path::new(st_src);
    if !st_src_path.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=[soundtouch] SoundTouch source not found, auto-cloning..."
        );
        if st_src_path.exists() {
            let _ = std::fs::remove_dir_all(st_src_path);
        }
        let parent = st_src_path.parent().expect("[soundtouch] invalid source path");
        let _ = std::fs::create_dir_all(parent);

        let mut clone = Command::new("git");
        clone.args([
            "clone",
            "--depth", "1",
            "--branch", "2.3.3",
            "https://codeberg.org/soundtouch/soundtouch.git",
            "soundtouch",
        ]);
        clone.current_dir(parent);

        let status = clone.status().expect("[soundtouch] failed to run git clone");
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
        println!(
            "cargo:warning=[soundtouch] SoundTouch source cloned successfully"
        );
    }

    // Only re-run if build.rs itself changes - the SoundTouch source tree is modified
    // during the build (cmake outputs, .rc patching) which would cause an infinite rebuild loop.
    println!("cargo:rerun-if-changed=build.rs");

    let target = std::env::var("TARGET").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| {
        target
            .split('-')
            .nth(2)
            .unwrap_or_default()
            .to_string()
    });
    println!(
        "cargo:warning=[soundtouch] TARGET={} TARGET_OS={}",
        target, target_os
    );

    let is_windows = target_os == "windows";
    let is_apple = target_os == "macos";

    // Patch SoundTouchDLL.rc to use windows.h instead of afxres.h (MFC header not always available)
    if is_windows {
        let rc_file = st_src_path.join("source").join("SoundTouchDLL").join("SoundTouchDLL.rc");
        if rc_file.exists() {
            let content = std::fs::read_to_string(&rc_file).expect("[soundtouch] failed to read SoundTouchDLL.rc");
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
                    std::fs::write(&rc_file, &patched).expect("[soundtouch] failed to write patched SoundTouchDLL.rc");
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

    println!("cargo:warning=[soundtouch] is_windows={} is_apple={}", is_windows, is_apple);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let build_dir = Path::new(&out_dir).join("soundtouch_build");
    println!("cargo:warning=[soundtouch] build_dir={}", build_dir.display());

    // Step 1: CMake configure - build SoundTouchDLL as a shared library.
    // Use the path as-is (cmake handles relative paths fine, and canonicalize
    // produces \\?\ extended paths on Windows which break CMake/MSBuild).
    println!("cargo:warning=[soundtouch] running cmake configure...");
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
    let status = cfg.status().expect("[soundtouch] failed to run cmake configure");
    println!("cargo:warning=[soundtouch] cmake configure exit status: {}", status);
    if !status.success() {
        panic!("[soundtouch] CMake configure failed with exit code {:?}", status.code());
    }
    println!("cargo:warning=[soundtouch] cmake configure succeeded");

    // Step 2: CMake build - build SoundTouchDLL target
    let mut bld = Command::new("cmake");
    bld.arg("--build").arg(&build_dir);
    bld.arg("--config").arg("Release");

    println!("cargo:warning=[soundtouch] spawning cmake build...");
    let output = bld.output().expect("[soundtouch] failed to run cmake build");
    println!("cargo:warning=[soundtouch] cmake build exit status: {}", output.status);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("cargo:warning=[soundtouch] cmake build stderr:\n{}", stderr);
        println!("cargo:warning=[soundtouch] cmake build stdout:\n{}", stdout);
        panic!("[soundtouch] CMake build failed with exit code {:?}", output.status.code());
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

    let lib_src = find_file(&build_dir, &lib_filename)
        .unwrap_or_else(|| {
            panic!(
                "[soundtouch] Could not find {} in build directory {}",
                lib_filename,
                build_dir.display()
            )
        });
    println!("cargo:warning=[soundtouch] found shared lib: {}", lib_src.display());

    // Force a stable, relocatable Mach-O install name.  The library is bundled
    // into HiFiShifter.app/Contents/Frameworks and loaded through @rpath, so it
    // must record `@rpath/libSoundTouchDLL.dylib` instead of an absolute path
    // or an executable-relative path (which would point at Contents/MacOS).
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
        println!(
            "cargo:warning=[soundtouch] set dylib install name to {}",
            install_name
        );
    }

    // Step 4: Link against the shared library
    let lib_search = lib_src.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", lib_search.display());
    println!("cargo:rustc-link-lib=dylib={}", lib_name);

    // Set rpath so the binary finds the shared library at runtime.
    // macOS:
    //   1. @executable_path/../Frameworks — primary location: Tauri
    //      bundle.macOS.frameworks copies the dylib into Contents/Frameworks
    //   2. @executable_path/../Resources  — fallback location: Tauri
    //      bundle.macOS.files copies the dylib into Contents/Resources
    //   3. @executable_path/../Resources/macos — ONNX Runtime dylibs staged
    //      by stage_ort_macos_dylibs() (libwebgpu_dawn.dylib, providers, ...)
    //   4. @executable_path                — finds the copy next to the binary
    //      during plain cargo runs (target/<triple>/release/)
    // Linux:
    //   1. $ORIGIN — finds libSoundTouchDLL.so next to the binary
    //   2. $ORIGIN/../lib/HiFiShifter — finds it in the AppImage AppDir layout
    if is_apple {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Resources/macos");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    } else if !is_windows {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/HiFiShifter");
    }

    // Step 5: Copy shared library to target dir (for runtime linking) AND to
    // source tree (for Tauri macOS framework bundling / tauri_build validation).
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
    } else {
        println!(
            "cargo:warning=[soundtouch] copied {} to {}",
            lib_src.display(),
            lib_dst_target.display()
        );
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
                "cargo:warning=[soundtouch] could not copy {} to framework source path {}: {}",
                lib_src.display(),
                lib_dst_resource.display(),
                e
            );
        } else {
            println!(
                "cargo:warning=[soundtouch] updated macOS framework dylib at {}",
                lib_dst_resource.display()
            );
        }
    } else {
        println!("cargo:warning=[soundtouch] macOS framework dylib unchanged, skipping write");
    }

}

/// Recursively search for a file by name under `dir`.
fn find_file(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
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
                // Skip symlink loops by only pushing actual dirs
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

/// Create a placeholder for vslib_x64.dll on non-x86_64 Windows targets
/// so tauri_build resource validation passes.  On x86_64 with the vslib
/// feature active, the real DLL is linked by build_vslib().
fn create_vslib_placeholder() {
    if !cfg!(target_os = "windows") {
        return;
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    // Only needed when vslib is not available (ARM64 or vslib feature disabled).
    if target.contains("x86_64") && cfg!(feature = "vslib") {
        return;
    }
    let p = std::path::Path::new("third_party/vslib/vslib_x64.dll");
    if !p.exists() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, b"");
    }
}

/// Copy DirectML.dll from the ORT build output (placed by ort-sys copy-dylibs)
/// to the resources directory so Tauri can validate and bundle it into the
/// NSIS installer. DirectML.dll is loaded dynamically by ONNX Runtime, so
/// Tauri's dependency scanner does not detect it; explicit resource listing
/// in tauri.windows.conf.json compensates for this.
///
/// Only runs on Windows x86_64 (DirectML is Windows-only, and aarch64 uses
/// a different build path).
fn copy_directml_dll() {
    use std::path::Path;

    if !cfg!(target_os = "windows") {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    // OUT_DIR = .../target/<triple>/<profile>/build/<crate>-<hash>/out
    // DLL is at   .../target/<triple>/<profile>/DirectML.dll
    let target_dir = Path::new(&out_dir).ancestors().nth(3).unwrap();
    let dll_src = target_dir.join("DirectML.dll");

    if !dll_src.exists() {
        println!(
            "cargo:warning=[ort] DirectML.dll not found at {}",
            dll_src.display()
        );
        return;
    }

    let resource_dir = Path::new("resources");
    let _ = std::fs::create_dir_all(resource_dir);
    let dll_dst = resource_dir.join("DirectML.dll");

    // Only write if bytes differ to avoid infinite dev rebuild loops.
    let src_bytes = std::fs::read(&dll_src).unwrap_or_default();
    let dst_bytes = std::fs::read(&dll_dst).unwrap_or_default();
    if src_bytes != dst_bytes {
        if let Err(e) = std::fs::write(&dll_dst, &src_bytes) {
            println!(
                "cargo:warning=[ort] could not copy DirectML.dll to {}: {}",
                dll_dst.display(),
                e
            );
        } else {
            println!(
                "cargo:warning=[ort] copied DirectML.dll ({} bytes) to {}",
                src_bytes.len(),
                dll_dst.display()
            );
        }
    } else {
        println!("cargo:warning=[ort] DirectML.dll resource unchanged, skipping write");
    }
}

/// Stage the ONNX Runtime shared libraries into `resources/macos/` so Tauri
/// can bundle them into `Contents/Resources/macos`.
///
/// ort-sys (`copy-dylibs` feature) places every `.dylib` from the downloaded
/// ORT package next to the final binary (target/<triple>/release). The app
/// links against some of them (e.g. `libwebgpu_dawn.dylib` for the WebGPU
/// execution provider) with the install name `@rpath/<name>`, so they must be
/// present inside the .app or dyld fails before main() runs.
fn stage_ort_macos_dylibs() {
    use std::path::Path;

    if !cfg!(target_os = "macos") {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    if out_dir.is_empty() {
        return;
    }

    // OUT_DIR = .../target/<triple>/<profile>/build/<pkg>-<hash>/out
    // 4 levels up = target/<triple>/<profile>/ (same dir ort-sys copies into).
    let target_dir = Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .expect("[ort] unexpected OUT_DIR depth");
    let staging_dir = Path::new("resources/macos");
    // This directory is a build output. Without an explicit rerun guard, Cargo's
    // default filesystem tracking sees newly staged dylibs as inputs and starts
    // another build, creating a watch loop in `cargo tauri dev`.
    println!("cargo:rerun-if-changed=build.rs");

    if let Err(e) = std::fs::create_dir_all(staging_dir) {
        println!(
            "cargo:warning=[ort] could not create {}: {}",
            staging_dir.display(),
            e
        );
        return;
    }

    let mut staged = 0usize;
    let entries = match std::fs::read_dir(target_dir) {
        Ok(entries) => entries,
        Err(e) => {
            println!(
                "cargo:warning=[ort] could not read {}: {}",
                target_dir.display(),
                e
            );
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        // SoundTouch is staged and bundled separately.
        if !name.ends_with(".dylib") || name == "libSoundTouchDLL.dylib" {
            continue;
        }

        let dst = staging_dir.join(&name);
        let src_bytes = std::fs::read(&path).unwrap_or_default();
        let dst_bytes = std::fs::read(&dst).unwrap_or_default();
        if src_bytes == dst_bytes {
            println!("cargo:warning=[ort] {} unchanged, skipping write", dst.display());
            continue;
        }

        let temp_dst = staging_dir.join(format!("{}.tmp", name));
        if let Err(e) = std::fs::write(&temp_dst, &src_bytes) {
            println!(
                "cargo:warning=[ort] could not stage {} to {}: {}",
                path.display(),
                temp_dst.display(),
                e
            );
            continue;
        }
        if let Err(e) = std::fs::rename(&temp_dst, &dst) {
            println!(
                "cargo:warning=[ort] could not replace {} with {}: {}",
                temp_dst.display(),
                dst.display(),
                e
            );
            let _ = std::fs::remove_file(&temp_dst);
            continue;
        }
        println!("cargo:warning=[ort] staged {} to {}", path.display(), dst.display());

        // Make the staged dylib relocatable: set @rpath/<name> as its install
        // name and rewrite absolute dependency paths to @rpath/<basename>.
        normalize_macos_dylib(&dst, staging_dir);
        staged += 1;
    }

    if staged == 0 {
        println!(
            "cargo:warning=[ort] no ORT dylibs found in {} (expected on macOS ARM64)",
            target_dir.display()
        );
    } else {
        println!("cargo:warning=[ort] staged {} ORT dylib(s) into {}", staged, staging_dir.display());
    }
}

/// Make a dylib relocatable inside the app bundle: force its install name to
/// `@rpath/<name>` and rewrite absolute LC_LOAD_DYLIB entries to
/// `@rpath/<basename>` — but only for dependencies that are bundled next to it
/// (system libraries under /usr/lib and /System/Library are left untouched).
/// Failures are warnings only; the prebuilt ORT dylibs are unsigned so
/// install_name_tool can always modify them.
fn normalize_macos_dylib(path: &std::path::Path, staging_dir: &std::path::Path) {
    use std::process::Command;

    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return,
    };

    let install_name = format!("@rpath/{}", name);
    let id_status = Command::new("install_name_tool")
        .arg("-id")
        .arg(&install_name)
        .arg(path)
        .status();
    if let Ok(status) = id_status {
        if status.success() {
            println!("cargo:warning=[ort] set install name {} on {}", install_name, path.display());
        }
    }

    let output = match Command::new("otool").arg("-L").arg(path).output() {
        Ok(output) => output,
        Err(e) => {
            println!(
                "cargo:warning=[ort] otool failed for {}: {}",
                path.display(),
                e
            );
            return;
        }
    };

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let line = line.trim();
        // Only rewrite absolute dependency paths.
        let Some(rest) = line.strip_prefix('/') else {
            continue;
        };
        let dep_path = rest.split_whitespace().next().unwrap_or_default();
        if dep_path.is_empty() {
            continue;
        }
        let dep_name = match std::path::Path::new(dep_path)
            .file_name()
            .and_then(|n| n.to_str())
        {
            Some(dep_name) if dep_name.ends_with(".dylib") => dep_name.to_string(),
            _ => continue,
        };
        // Only rewrite ORT-internal dependencies that we bundle; never touch
        // system libraries.
        if !staging_dir.join(&dep_name).exists() {
            continue;
        }
        let old = format!("/{}", dep_path);
        let new = format!("@rpath/{}", dep_name);
        let change_status = Command::new("install_name_tool")
            .arg("-change")
            .arg(&old)
            .arg(&new)
            .arg(path)
            .status();
        if let Ok(status) = change_status {
            if status.success() {
                println!(
                    "cargo:warning=[ort] rewrote dependency {} -> {} in {}",
                    old,
                    new,
                    path.display()
                );
            }
        }
    }
}


/// Minimal protobuf reader/writer (no external crates) used to rewrite the
/// NSF-HiFiGAN ONNX model so the Pad node's runtime-derived `pads` become a
/// constant initializer.  CoreML EP cannot compile the stock model's dynamic
/// pads ("output_features has no value for 'Sub_output_0'").
#[cfg(target_os = "macos")]
mod coreml_pb {
    pub fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        buf.push(v as u8);
    }
    pub fn write_tag(buf: &mut Vec<u8>, num: u32, wire: u8) {
        write_varint(buf, ((num as u64) << 3) | wire as u64);
    }
    pub fn write_bytes_field(buf: &mut Vec<u8>, num: u32, payload: &[u8]) {
        write_tag(buf, num, 2);
        write_varint(buf, payload.len() as u64);
        buf.extend_from_slice(payload);
    }
    pub fn write_varint_field(buf: &mut Vec<u8>, num: u32, v: u64) {
        write_tag(buf, num, 0);
        write_varint(buf, v);
    }

    pub struct Field {
        pub num: u32,
        pub wire: u8,
        pub payload: Vec<u8>,
    }

    pub fn parse(data: &[u8]) -> Result<Vec<Field>, String> {
        let mut fields = Vec::new();
        let mut pos = 0usize;
        while pos < data.len() {
            let tag = read_varint(data, &mut pos)?;
            let num = (tag >> 3) as u32;
            let wire = (tag & 7) as u8;
            match wire {
                0 => {
                    let start = pos;
                    read_varint(data, &mut pos)?;
                    fields.push(Field { num, wire, payload: data[start..pos].to_vec() });
                }
                2 => {
                    let len = read_varint(data, &mut pos)? as usize;
                    if pos + len > data.len() {
                        return Err("protobuf length overflow".to_string());
                    }
                    fields.push(Field { num, wire, payload: data[pos..pos + len].to_vec() });
                    pos += len;
                }
                5 => {
                    if pos + 4 > data.len() {
                        return Err("protobuf fixed32 overflow".to_string());
                    }
                    fields.push(Field { num, wire, payload: data[pos..pos + 4].to_vec() });
                    pos += 4;
                }
                1 => {
                    if pos + 8 > data.len() {
                        return Err("protobuf fixed64 overflow".to_string());
                    }
                    fields.push(Field { num, wire, payload: data[pos..pos + 8].to_vec() });
                    pos += 8;
                }
                w => return Err(format!("unsupported protobuf wire type {w}")),
            }
        }
        Ok(fields)
    }

    pub fn encode(fields: &[Field]) -> Vec<u8> {
        let mut buf = Vec::new();
        for f in fields {
            write_tag(&mut buf, f.num, f.wire);
            match f.wire {
                0 => buf.extend_from_slice(&f.payload),
                2 => {
                    write_varint(&mut buf, f.payload.len() as u64);
                    buf.extend_from_slice(&f.payload);
                }
                5 | 1 => buf.extend_from_slice(&f.payload),
                _ => {}
            }
        }
        buf
    }

    fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
        let mut shift = 0;
        let mut val = 0u64;
        while *pos < data.len() && shift < 64 {
            let b = data[*pos];
            *pos += 1;
            val |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(val);
            }
            shift += 7;
        }
        Err("truncated protobuf varint".to_string())
    }
}

/// Rewrite CoreML-incompatible parts of the stock model:
/// - Pad: dynamic `pads` inputs become constant initializers.
/// - ConvTranspose: explicit `kernel_shape` attributes are removed so CoreML EP
///   accepts the upsampling layers.
/// Only the node/initializer protobuf fields are touched; all other bytes
/// (including the ~54 MB weight raw_data) are preserved verbatim.
#[cfg(target_os = "macos")]
fn rewrite_coreml_model(src: &std::path::Path, dst: &std::path::Path) -> Result<(), String> {
    use coreml_pb::{Field, encode, parse};

    let patch = load_pads_patch()?;
    let data = std::fs::read(src).map_err(|e| e.to_string())?;
    let model_fields = parse(&data)?;

    let mut out_model: Vec<Field> = Vec::new();
    let mut changed = false;
    for f in &model_fields {
        if f.num == 7 && f.wire == 2 {
            out_model.push(Field { num: 7, wire: 2, payload: rewrite_graph(&f.payload, &patch)? });
            changed = true;
        } else {
            out_model.push(Field { num: f.num, wire: f.wire, payload: f.payload.clone() });
        }
    }
    if !changed {
        return Err("ONNX graph field not found".to_string());
    }
    std::fs::write(dst, encode(&out_model)).map_err(|e| e.to_string())?;
    Ok(())
}

/// Load the Pad-pads patch (name -> constant int64 values).  The stock
/// model derives Pad pads at runtime; CoreML EP cannot compile dynamic pads,
/// so this patch records the (constant) values and build.rs rewrites every
/// affected Pad node to use a constant initializer.
#[cfg(target_os = "macos")]
fn load_pads_patch() -> Result<std::collections::HashMap<String, Vec<i64>>, String> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("resources/models/nsf_hifigan/coreml_pads_patch.txt");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut map = std::collections::HashMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, vals) = line
            .split_once('=')
            .ok_or_else(|| format!("bad pads patch line: {line}"))?;
        let vals = vals
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<i64>()
                    .map_err(|e| format!("bad pads patch value '{v}': {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        map.insert(name.trim().to_string(), vals);
    }
    Ok(map)
}

/// Encode a TensorProto with dims=[len], data_type=INT64, int64_data=vals,
/// name=name.  Values are expected to be non-negative (pads are 0/1).
#[cfg(target_os = "macos")]
fn build_int64_tensor(name: &str, vals: &[i64]) -> Vec<u8> {
    let mut t = Vec::new();
    let mut dims = Vec::new();
    coreml_pb::write_varint(&mut dims, vals.len() as u64);
    coreml_pb::write_bytes_field(&mut t, 1, &dims);
    coreml_pb::write_varint_field(&mut t, 2, 7);
    let mut data = Vec::new();
    for v in vals {
        coreml_pb::write_varint(&mut data, *v as u64);
    }
    coreml_pb::write_bytes_field(&mut t, 7, &data);
    coreml_pb::write_bytes_field(&mut t, 8, name.as_bytes());
    t
}

#[cfg(target_os = "macos")]
fn rewrite_graph(
    graph: &[u8],
    patch: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<Vec<u8>, String> {
    use coreml_pb::{Field, encode, parse};

    let gfields = parse(graph)?;
    let mut nodes: Vec<Field> = Vec::new();
    let mut inits: Vec<Field> = Vec::new();
    let mut others: Vec<Field> = Vec::new();
    let mut next_id = 0usize;
    let mut new_inits: Vec<Vec<u8>> = Vec::new();
    for f in &gfields {
        match (f.num, f.wire) {
            (1, 2) => nodes.push(Field {
                num: 1,
                wire: 2,
                payload: rewrite_node(&f.payload, patch, &mut next_id, &mut new_inits)?,
            }),
            (5, 2) => inits.push(Field { num: 5, wire: 2, payload: f.payload.clone() }),
            _ => others.push(Field { num: f.num, wire: f.wire, payload: f.payload.clone() }),
        }
    }

    let mut out = Vec::new();
    for f in others {
        out.extend(encode(&[f]));
    }
    for f in nodes {
        out.extend(encode(&[f]));
    }
    for f in inits {
        out.extend(encode(&[f]));
    }
    for t in new_inits {
        out.extend(encode(&[Field { num: 5, wire: 2, payload: t }]));
    }
    Ok(out)
}

/// Rewrite CoreML-incompatible nodes:
/// - Pad: replace a dynamic `pads` input listed in the patch with a fresh
///   constant initializer.
/// - ConvTranspose: remove the explicit `kernel_shape` attribute.  CoreML EP
///   only accepts ConvTranspose nodes when `kernel_shape` is not present
///   (inferred from the constant weight), while the stock model exports it
///   explicitly for every upsampling layer.
#[cfg(target_os = "macos")]
fn rewrite_node(
    node: &[u8],
    patch: &std::collections::HashMap<String, Vec<i64>>,
    next_id: &mut usize,
    new_inits: &mut Vec<Vec<u8>>,
) -> Result<Vec<u8>, String> {
    use coreml_pb::{Field, encode, parse};

    let nf = parse(node)?;
    let mut op_type = String::new();
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    for f in &nf {
        if f.num == 4 && f.wire == 2 {
            op_type = String::from_utf8_lossy(&f.payload).into_owned();
        }
        if f.num == 1 && f.wire == 2 {
            inputs.push(f.payload.clone());
        }
    }

    if op_type == "ConvTranspose" {
        let mut out = Vec::new();
        for f in &nf {
            if f.num == 5 && f.wire == 2 {
                let attr = parse(&f.payload)?;
                let mut name = String::new();
                for af in &attr {
                    if af.num == 1 && af.wire == 2 {
                        name = String::from_utf8_lossy(&af.payload).into_owned();
                    }
                }
                if name == "kernel_shape" {
                    continue;
                }
            }
            out.extend(encode(&[Field { num: f.num, wire: f.wire, payload: f.payload.clone() }]));
        }
        return Ok(out);
    }

    if op_type == "Pad" && inputs.len() >= 2 {
        let pads_name = String::from_utf8_lossy(&inputs[1]).into_owned();
        if let Some(vals) = patch.get(&pads_name) {
            let const_name = format!("/coreml_pads_const_{}", *next_id);
            *next_id += 1;
            new_inits.push(build_int64_tensor(&const_name, vals));

            let mut out = Vec::new();
            for f in &nf {
                if f.num == 1 && f.wire == 2 {
                    continue; // rebuilt below
                }
                out.extend(encode(&[Field { num: f.num, wire: f.wire, payload: f.payload.clone() }]));
            }
            for (i, inp) in inputs.iter().enumerate() {
                let name: &[u8] = if i == 1 { const_name.as_bytes() } else { inp };
                coreml_pb::write_bytes_field(&mut out, 1, name);
            }
            return Ok(out);
        }
    }
    Ok(node.to_vec())
}

/// Generate the CoreML-compatible model variant during macOS ARM64 builds.
/// The output file lives in the resources tree (so Tauri bundles it) but is
/// gitignored; it is regenerated whenever the stock model, the pads patch or
/// this build script changes.
#[cfg(target_os = "macos")]
fn generate_coreml_model_variant() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("resources/models/nsf_hifigan/pc_nsf_hifigan.onnx");
    let dst = manifest.join("resources/models/nsf_hifigan/pc_nsf_hifigan_coreml.onnx");
    if !src.is_file() {
        println!("cargo:warning=[build.rs] pc_nsf_hifigan.onnx not found; skipping CoreML variant");
        return;
    }
    let patch_path = manifest.join("resources/models/nsf_hifigan/coreml_pads_patch.txt");
    let build_script = std::path::Path::new(file!());
    let src_m = std::fs::metadata(&src).and_then(|m| m.modified()).ok();
    let patch_m = std::fs::metadata(&patch_path).and_then(|m| m.modified()).ok();
    let build_m = std::fs::metadata(build_script).and_then(|m| m.modified()).ok();
    let dst_m = std::fs::metadata(&dst).and_then(|m| m.modified()).ok();
    if dst.is_file()
        && dst_m >= src_m
        && (patch_m.is_none() || dst_m >= patch_m)
        && (build_m.is_none() || dst_m >= build_m)
    {
        return;
    }
    match rewrite_coreml_model(&src, &dst) {
        Ok(()) => println!("cargo:warning=[build.rs] generated CoreML model variant: {}", dst.display()),
        Err(e) => println!("cargo:warning=[build.rs] failed to generate CoreML model variant: {e}"),
    }
}

#[cfg(not(target_os = "macos"))]
fn generate_coreml_model_variant() {}
