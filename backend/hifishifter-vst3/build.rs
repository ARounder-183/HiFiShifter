//! VST3 外壳的构建脚本。
//!
//! 三件事：把构建身份（commit / 脏标志 / GitHub 仓库链接）烘进 DLL；把产品版本的
//! **单一来源**（`src-tauri/tauri.conf.json`）注入为 `HFS_PRODUCT_VERSION`；给 ELF /
//! Mach-O 声明 `$ORIGIN` 运行时搜索路径。
//!
//! 【为什么与 `hifishifter-plugin` / `hifishifter-clap` 的 build.rs 各有一份】真正
//! 产出 cdylib 的是外壳 crate：cargo 对非 cdylib/bin 目标的 `cargo:rustc-link-arg`
//! 是**丢弃**的（`hifishifter-plugin` 的 build.rs 注释记载了这段历史）。三份保持
//! 各自最小、职责一致。

#[path = "../build-support/git_info.rs"]
mod git_info;
#[path = "../build-support/product_version.rs"]
mod product_version;

fn main() {
    git_info::emit(&["src"]);
    product_version::emit(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        env!("CARGO_PKG_VERSION"),
    );
    link_runtime_search_paths();
}

/// 让本 cdylib 只依赖**同 bundle 内**的共享库，不依赖编译环境的任何绝对路径。
///
/// 【为什么是 `$ORIGIN` 而不是改 `LD_LIBRARY_PATH`】插件跑在宿主进程里，那是用户的
/// 环境；能改的只有自己的 ELF 动态段。macOS 对应的是 `@loader_path`（Mach-O 没有
/// 进程级搜索路径可言）。
fn link_runtime_search_paths() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        // 同目录：`Contents/x86_64-linux/` 里同时放模块与它的两个 .so。
        "linux" | "freebsd" | "android" => {
            println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        // `Contents/MacOS/HiFiShifter` + `Contents/Frameworks/*.dylib`。
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
        }
        _ => {}
    }
}
