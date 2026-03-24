fn main() {
    build_frontend();
    tauri_build::build();

    // Allow skipping expensive native builds in CI checks via env var
    // Set HIFISHIFTER_SKIP_NATIVE_BUILD=1 to skip WORLD/Signalsmith/VSLIB builds
    let skip_native = std::env::var("HIFISHIFTER_SKIP_NATIVE_BUILD").unwrap_or_default();
    if skip_native == "1" {
        println!("cargo:warning=[build.rs] Skipping native library builds (HIFISHIFTER_SKIP_NATIVE_BUILD=1)");
    } else {
        build_world_static();
        build_signalsmith_stretch();
        build_vslib();
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
        println!("cargo:warning=[Frontend] HIFISHIFTER_SKIP_FRONTEND_BUILD=1 -> skipping frontend build");
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
