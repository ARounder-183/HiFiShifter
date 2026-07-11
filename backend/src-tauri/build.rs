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

    // tauri_build validates resources, so soundtouch must run first to populate
    // the shared library at the resource path.
    tauri_build::build();
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
///   vslib_x64.dll  — needs to sit next to the final binary at runtime
///   vslib_x64.lib  — import library linked at compile time
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

    println!("cargo:rerun-if-changed={}", st_src);

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
            if content.contains("afxres.h") {
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
                std::fs::write(&rc_file, &patched).expect("[soundtouch] failed to write patched SoundTouchDLL.rc");
                println!("cargo:warning=[soundtouch] patched SoundTouchDLL.rc to use windows.h");
            }
        }
    }
    println!("cargo:warning=[soundtouch] is_windows={} is_apple={}", is_windows, is_apple);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let build_dir = Path::new(&out_dir).join("soundtouch_build");
    println!("cargo:warning=[soundtouch] build_dir={}", build_dir.display());

    // Step 1: CMake configure — build SoundTouchDLL as a shared library.
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
        cfg.arg("-DCMAKE_INSTALL_NAME_DIR=@executable_path");
        cfg.arg("-DCMAKE_MACOSX_RPATH=ON");
    }

    println!("cargo:warning=[soundtouch] spawning cmake configure...");
    let status = cfg.status().expect("[soundtouch] failed to run cmake configure");
    println!("cargo:warning=[soundtouch] cmake configure exit status: {}", status);
    if !status.success() {
        panic!("[soundtouch] CMake configure failed with exit code {:?}", status.code());
    }
    println!("cargo:warning=[soundtouch] cmake configure succeeded");

    // Step 2: CMake build — build SoundTouchDLL target
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

    // Step 4: Link against the shared library
    let lib_search = lib_src.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", lib_search.display());
    println!("cargo:rustc-link-lib=dylib={}", lib_name);

    // Set rpath so the binary finds the shared library in its own directory at runtime.
    // Linux/ELF uses `$ORIGIN`, while macOS uses dyld-specific loader paths.
    if is_apple {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    } else if !is_windows {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }

    // Step 5: Copy shared library to target dir (for runtime linking) AND to
    // source tree (for Tauri resource bundling on all platforms)
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

    // Also copy to a stable path under third_party/ so tauri.conf.json can reference it
    let lib_dst_resource = st_src_path.join(&lib_filename);
    if lib_dst_resource != lib_dst_target {
        if let Err(e) = std::fs::copy(&lib_src, &lib_dst_resource) {
            println!(
                "cargo:warning=[soundtouch] could not copy {} to resource path {}: {}",
                lib_src.display(),
                lib_dst_resource.display(),
                e
            );
        }
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
