//! 插件引擎的构建脚本。
//!
//! 两件事：把构建身份（commit / 脏标志 / GitHub 仓库链接）烘进 DLL；把产品版本的
//! **单一来源**（`src-tauri/tauri.conf.json`）注入为 `HFS_PRODUCT_VERSION`。
//!
//! 【为什么现在才有】在此之前插件完全没有 build script，于是它的「关于」对话框
//! 只能报一个 crate 版本号 —— 没有 commit、没有仓库链接，用户报 bug 时无法定位到
//! 具体构建。实现与 App 共用 `backend/build-support/git_info.rs`，避免两份逻辑分叉。
//!
//! 监视集只有 `src` 与 `native`：它们是本 crate 会进入二进制的输入。模型与前端
//! 产物由打包阶段决定、不编译进 DLL，因此不参与脏标志判定。

#[path = "../build-support/git_info.rs"]
mod git_info;
#[path = "../build-support/product_version.rs"]
mod product_version;

fn main() {
    git_info::emit(&["src", "native"]);
    product_version::emit(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        env!("CARGO_PKG_VERSION"),
    );
    link_runtime_search_paths();
}

/// 让本 cdylib 只依赖**同 bundle 内**的共享库，不依赖构建机的任何绝对路径。
///
/// 【为什么必须写在这里】kernel 的 build script 里确实有 `-Wl,-rpath,$ORIGIN`，
/// 但 kernel 只是 rlib：cargo 对非 cdylib/bin 目标的 `cargo:rustc-link-arg` 是**丢弃**的，
/// 于是产物的 `readelf -d` 里一个 RPATH 都没有，`ldd` 直接报 `libwebgpu_dawn.so`、
/// `libSoundTouchDLL.so` "not found"。真正产出 cdylib 的是本 crate，所以只能写在这里。
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
