//! VST3 模块入口的守卫：三个导出符号必须存在且拼写正确。
//!
//! 【为什么值得一条测试】VST3 在 Windows 上就是一个导出 `GetPluginFactory` /
//! `InitDll` / `ExitDll` 的 DLL。符号名拼错时 REAPER 的表现是"这个文件 0 个类"
//! 而**不报错** —— 探针期实测过（IID 布局写错时就是这个症状）。那是这类问题里
//! 最难定位的一种，所以用测试把它钉住。
//!
//! 前置：先 `cargo build -p hifishifter-plugin`（cdylib 由它产出）。

#[test]
fn the_vst3_module_entry_points_are_exported() {
    let dll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/debug/hifishifter_plugin.dll");
    assert!(
        dll.is_file(),
        "找不到 {} —— 先跑 `cargo build -p hifishifter-plugin` 再跑测试（cdylib 不由 cargo test 产出）",
        dll.display()
    );

    let output = std::process::Command::new("dumpbin")
        .args(["/exports", dll.to_str().unwrap()])
        .output()
        .expect("需要 MSVC 的 dumpbin（先点源 tools/msvc-env.ps1）");
    assert!(output.status.success(), "dumpbin 失败: {output:?}");

    let text = String::from_utf8_lossy(&output.stdout);
    for name in ["GetPluginFactory", "InitDll", "ExitDll"] {
        assert!(text.contains(name), "缺少 VST3 导出符号 `{name}`：\n{text}");
    }
}
