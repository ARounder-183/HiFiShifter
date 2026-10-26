//! 安全配置守卫：把"改坏了也不会有症状"的安全判据钉成测试。
//!
//! 【为什么需要这条测试】2026-10-26 的安全排查把 CSP 从"能跑就行"收紧到一组明确的
//! 指令（`object-src` / `base-uri` / `frame-src` / `form-action`），并确认
//! `script-src` 里**没有** `data:` / `blob:` —— 那正是 Tauri "CSP nonce 被 scheme
//! 源旁路"（CVE-2026-95626）的前提条件。这类判据的共同点是：删掉一条指令，应用照常
//! 启动、测试照常全绿，直到某天有人用一条注入链把它利用掉。所以写成测试。
//!
//! 判据分三层：
//! 1. 发布配置 `tauri.conf.json`：CSP 非空、脚本源只有 `'self'`、四个收口指令在、
//!    危险开关（`dangerousDisableAssetCspModification` /
//!    `dangerousRemoteDomainIpcAccess`）没有被打开；
//! 2. 插件页面 `frontend/plugin.html`（WebView 里没有 Tauri 注入，靠 meta 自带）：
//!    `frame-src` / `object-src` 必须继续是 `'none'`；
//! 3. 开发配置 `tauri.conf.dist-dev.json`：必须带 CSP，且**不许**被任何构建脚本或
//!    工作流拿去出包 —— 它把 `devUrl`（localhost:5173）写死在配置里，一旦出包就是
//!    一个指向空页面的产物。

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // tests/ -> crate 根 -> backend/ -> 仓库根
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repository root")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// 取一条 CSP 指令的取值部分（`script-src a b` -> `a b`）；不存在时为 `None`。
///
/// 只做"按分号切、按首个空格切"这一层解析：CSP 的指令语法就这么简单，引一个解析库
/// 反而会让"到底比了什么"变得不可见。
fn directive<'a>(csp: &'a str, name: &str) -> Option<&'a str> {
    csp.split(';').find_map(|part| {
        let part = part.trim();
        let (head, tail) = match part.split_once(' ') {
            Some((head, tail)) => (head, tail.trim()),
            None => (part, ""),
        };
        (head == name).then_some(tail)
    })
}

fn csp_of(config: &serde_json::Value) -> String {
    config["app"]["security"]["csp"]
        .as_str()
        .unwrap_or_else(|| panic!("app.security.csp must be a string, got {config}"))
        .to_string()
}

/// **不变量：发布配置的 CSP 必须保持 2026-10-26 收紧后的形状。**
#[test]
fn the_release_csp_keeps_the_hardening_directives() {
    let root = repo_root();
    let text = read(&root.join("backend/src-tauri/tauri.conf.json"));
    let config: serde_json::Value = serde_json::from_str(&text).expect("tauri.conf.json parses");
    let csp = csp_of(&config);

    // 1) 脚本源只有 `'self'`：`data:` / `blob:` 一旦出现在这里，CSP nonce 就形同虚设。
    let script_src = directive(&csp, "script-src").expect("script-src must be declared");
    let tokens: Vec<&str> = script_src.split_whitespace().collect();
    assert_eq!(
        tokens,
        vec!["'self'"],
        "script-src must stay 'self' only (data:/blob: defeat the nonce), got {script_src}"
    );

    // 2) 四个"注入即被放大"的收口必须都在。
    for name in ["object-src", "base-uri", "frame-src", "form-action"] {
        assert_eq!(
            directive(&csp, name),
            Some("'none'"),
            "{name} must be 'none' in the release CSP: {csp}"
        );
    }

    // 3) 危险开关不许被打开（它们会让上面两条同时失效）。
    for forbidden in [
        "dangerousDisableAssetCspModification",
        "dangerousRemoteDomainIpcAccess",
    ] {
        assert!(
            !text.contains(forbidden),
            "{forbidden} must not appear in tauri.conf.json"
        );
    }
}

/// **不变量：插件页面的自带 CSP 继续挡住 frame 与 object。**
///
/// 插件窗口在 DAW 进程里，页面由自定义 scheme 提供，**没有** Tauri 注入的 CSP，
/// 唯一的策略就是这份 meta。
#[test]
fn the_plugin_page_csp_still_blocks_frames_and_objects() {
    let root = repo_root();
    let html = read(&root.join("frontend/plugin.html"));
    for needle in ["frame-src 'none'", "object-src 'none'", "script-src 'self'"] {
        assert!(html.contains(needle), "plugin.html must keep {needle}");
    }
}

/// **不变量：开发配置带 CSP，且没有任何构建入口引用它。**
///
/// 它把 `devUrl` 写死成 `http://localhost:5173`：拿它出包会得到一个指向空页面的
/// 产物。引用面只允许 `scripts/release/set-version.ps1`（改版本号）与插件侧的
/// `version_parity` 测试（比对版本）—— 新增任何引用都必须是有意识的决定。
#[test]
fn the_dev_config_never_reaches_a_build() {
    let root = repo_root();
    let dev_config = read(&root.join("backend/src-tauri/tauri.conf.dist-dev.json"));
    let parsed: serde_json::Value = serde_json::from_str(&dev_config).expect("dev config parses");
    assert!(
        parsed["app"]["security"]["csp"].is_string(),
        "the dev config must carry a CSP too: a build with it must not ship unprotected"
    );

    let allowed = [
        "scripts/release/set-version.ps1",
        "backend/hifishifter-plugin/tests/version_parity.rs",
        "backend/src-tauri/tests/security_config.rs",
    ];
    let mut offenders = Vec::new();
    for dir in ["scripts", ".github/workflows"] {
        collect_references(&root.join(dir), &root, &mut offenders);
    }
    offenders.retain(|path| !allowed.contains(&path.as_str()));
    assert!(
        offenders.is_empty(),
        "tauri.conf.dist-dev.json is referenced by {offenders:?}; it hardcodes devUrl and must not build a release"
    );
}

/// 递归收集 `dir` 下所有"提到了开发配置"的文件（相对仓库根的路径）。
///
/// 只看脚本、工作流与 Rust 源码：文档里提到它是正常的（本仓的审查记录就写着它的来历）。
fn collect_references(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_references(&path, root, out);
            continue;
        }
        let relevant = matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("ps1" | "sh" | "mjs" | "js" | "yml" | "yaml" | "rs")
        );
        if !relevant {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !text.contains("tauri.conf.dist-dev.json") {
            continue;
        }
        if let Ok(relative) = path.strip_prefix(root) {
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}
