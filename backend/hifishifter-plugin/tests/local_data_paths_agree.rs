//! 本地数据目录守卫：CLAP 外壳与内核必须算出**同一个**用户数据根。
//!
//! 【为什么需要这条测试】CLAP 外壳刻意不依赖推理内核（依赖图守卫，见
//! `no_tauri_in_dependency_tree.rs`），所以它在
//! `hifishifter_plugin_core::local_data` 里带了一份同规则的路径解析。两份实现
//! 一旦漂移，症状是**两个产品目录**：WebView2 profile、日志与插件记事本散到两处，
//! 用户删掉一处就等于只清了一半 —— 而且没有任何报错。
//!
//! 判据：同一台机器上两个函数必须逐字相等（含环境变量缺席时的临时目录兜底）。

use std::path::PathBuf;

fn kernel_root() -> PathBuf {
    hifishifter_kernel::config_location::local_data_subdir("webview")
}

fn shell_root() -> PathBuf {
    hifishifter_plugin_core::local_data::local_data_subdir("webview")
}

#[test]
fn the_plugin_core_and_the_kernel_agree_on_the_local_data_root() {
    assert_eq!(
        shell_root(),
        kernel_root(),
        "the CLAP shell and the kernel resolve the local data root differently"
    );
}

#[test]
fn the_local_data_root_is_absolute_and_per_user() {
    let root = shell_root();
    assert!(root.is_absolute(), "{}", root.display());
    // 不能落在程序目录下：那正是 CLAP 的 profile 打不开窗口的原因。
    assert!(
        !root.to_string_lossy().contains("Program Files"),
        "{} must not point into the install tree",
        root.display()
    );
}
