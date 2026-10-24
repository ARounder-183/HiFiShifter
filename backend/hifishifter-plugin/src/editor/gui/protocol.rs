//! 页面 ↔ 原生之间的**纯函数**契约：消息形状、长度、资产路径、MIME、导航白名单。
//!
//! 【为什么单独成文件】这些判据必须能被单测，而它们的调用点全都在
//! 「要 COM / 要 NSView / 要 GTK」的函数里 —— 一旦内联，就只能靠"跑一遍宿主"来验证。
//! 这里没有任何 `use` 平台类型，也没有 `unsafe`：三个平台共用同一份判据。
//!
//! 【为什么桥接与形状放在一起】`4.2` 的桥接脚本（`bridge.js`）与这里的校验是同一份
//! 契约的两端：脚本负责"把字符串原样送到原生"，这里负责"只接受一个形状"。改一端必须
//! 改另一端，放在同一个目录里才能被同一次评审看到。

//
// 【为什么 Windows 上允许 §dead_code§】本文件的消费者是 macOS/Linux 的两个原语
// （自定义 scheme + 导航白名单）。Windows 用 WebView2 的虚拟主机映射，走不到它们，
// 但判据必须继续编译 —— 否则等到那个平台时才腐烂。
#![cfg_attr(windows, allow(dead_code))]

use std::path::{Path, PathBuf};

/// 一条页面消息的字节上限。
///
/// 【为什么是 1 MiB】与 Windows 现值的同一个数字（`webview.rs` 的 `1024*1024`）：它护的
/// 不是内存，而是"宿主 UI 线程不被一条消息拖住" —— 解析 1 MiB JSON 已经在毫秒级，
/// 再大就该走文件通道（导入音频、导出诊断包都是这么做的）。
pub(crate) const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

/// 前端 `id` 的上限：JS 的 `Number` 只能精确表示到 2^53-1。
///
/// 【为什么要判】超过它的 id 在 `JSON.stringify` 之后已经丢精度，回执永远对不上，
/// 请求会挂到超时。拒绝比"静默挂 30 秒"好。
pub(crate) const MAX_ID: u64 = 9_007_199_254_740_991;

/// 自定义 scheme 与页面地址（macOS / Linux 共用）。
///
/// 【为什么不用 `file://`】三条硬理由：ES module 的跨源加载需要非公开偏好；
/// `file://` 的 origin 是 opaque，`localStorage` 直接抛（`uiStorage` 的同步读取会炸）；
/// CSP 里的 `'self'` 在 `file://` 下语义含混。见方案 `5.3`。
pub(crate) const SCHEME: &str = "hfs-plugin";
pub(crate) const ORIGIN: &str = "hfs-plugin://local/";
pub(crate) const PAGE_URL: &str = "hfs-plugin://local/plugin.html";

/// 前端唯一的新增消息：页面脚本装好监听之后的第一句。
///
/// 【为什么不进命令表】它不需要回执、不产生任何状态，只是把"白屏"变成一条能搜到的
/// 日志（`[gui] page ready after 143ms`）。让它走命令表就要给它编一个 id，然后前端
/// 得为它维护一个永不 resolve 的 promise。
pub(crate) const PAGE_READY: &str = "plugin_ready";

/// 校验一条页面消息，返回可用的请求对象。
///
/// 【为什么返回 `Option` 而不是 `Result`】调用方（三个平台的回调）对"不合法"的处理
/// **完全一致**：丢掉、什么都不做。给它一个错误类型只会让每个平台各写一遍日志。
pub(crate) fn parse_request(text: &str, view_id: &str) -> Option<serde_json::Value> {
    if text.len() > MAX_MESSAGE_BYTES {
        return None;
    }
    let request = serde_json::from_str::<serde_json::Value>(text).ok()?;
    if request["version"] != 1 || request["viewId"] != view_id {
        return None;
    }
    let id = request["id"].as_u64()?;
    if id == 0 || id > MAX_ID {
        return None;
    }
    Some(request)
}

/// 资源扩展名 → MIME。
///
/// 【为什么手写这张表而不是查 `mime_guess`】只有七种扩展名，而**写错一个就是白屏**：
/// `.js` 给成 `text/plain` 会被 module 加载拒绝，`.wasm` 给错会拒绝实例化。
/// 手写让"支持哪些类型"在代码里一眼可见，也让这条判据能被单测。
pub(crate) fn mime_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("html") | Some("htm") => "text/html",
        Some("js") | Some("mjs") => "text/javascript",
        Some("css") => "text/css",
        Some("json") => "application/json",
        Some("wasm") => "application/wasm",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("map") => "application/json",
        _ => "application/octet-stream",
    }
}

/// 把自定义 scheme 的请求路径解析成 `Resources/frontend` 下的真实文件。
///
/// 三道判据，缺一不可：
/// 1. **拒绝绝对路径与 `..` 段** —— 请求路径来自页面，页面可以被注入的脚本改写。
/// 2. `canonicalize` 之后再比前缀 —— 否则 `frontend-evil/` 这类兄弟目录能靠前缀匹配混进来。
/// 3. 必须是**文件** —— 目录会被 `fs::read` 报错，但错误信息会泄漏目录内容。
pub(crate) fn resolve_asset(root: &Path, request_path: &str) -> Option<PathBuf> {
    let relative = request_path.trim_start_matches('/');
    if relative.is_empty() || relative.contains('\\') || relative.contains('\0') {
        return None;
    }
    let mut path = root.to_path_buf();
    for segment in relative.split('/') {
        // 空段（`a//b`）、`.` 与 `..` 一律拒绝 —— 页面不该用它们，
        // 而放宽任何一段都会让判据多出一个特例。
        if segment.is_empty() || segment == "." || segment == ".." {
            return None;
        }
        path.push(segment);
    }
    let canonical_root = root.canonicalize().ok()?;
    let canonical = path.canonicalize().ok()?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return None;
    }
    Some(canonical)
}

/// 把一段 JSON 文本包成一条**安全的 JS 表达式**。
///
/// 【为什么不是直接拼接】回执里带着用户数据（片段名、路径、错误文本）。直接拼进
/// `window.__hfsDeliver(<json>)` 会踩两个坑：JSON 是 JS 对象字面量语法的**近似**子集
/// （`__proto__`、U+2028/2029 都能改变语义），而包成字符串再 `JSON.parse` 就只有
/// 一种解释。
///
/// 产出形状：`window.__hfsDeliver(JSON.parse('…'))`。
pub(crate) fn js_expression(json: &str) -> String {
    let mut escaped = String::with_capacity(json.len() + 32);
    for character in json.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\'' => escaped.push_str("\\'"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            // 【为什么连这两个也转义】ES2019 之前 U+2028/2029 是字符串字面量的终止符；
            // 转义它们零成本，却免掉一整类"某个旧 WebKit 上白屏"的可能。
            '\u{2028}' => escaped.push_str("\\u2028"),
            '\u{2029}' => escaped.push_str("\\u2029"),
            _ => escaped.push(character),
        }
    }
    format!("window.__hfsDeliver(JSON.parse('{escaped}'))")
}

/// 导航白名单：只允许本 bundle 的页面。
///
/// 【为什么用前缀而不是全等】`plugin.html` 之后可能跟查询串或 `#` 片段；
/// 真正的安全边界是 scheme + host（`hfs-plugin://local/`），不是路径。
pub(crate) fn is_allowed_navigation(uri: &str) -> bool {
    uri.starts_with(ORIGIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> serde_json::Value {
        serde_json::json!({"version":1,"viewId":"view-1","id":7,"command":"ping"})
    }

    /// **不变量：版本/视图/编号三者同时正确才算一条请求。**
    ///
    /// 【为什么钉死】三个平台各自实现回调，任何一处少判一个字段，都会让"另一个视图的
    /// 回执"或"上一个实例的迟到请求"落到本视图上 —— 那是串消息，不是崩溃，所以只能
    /// 靠这条测试守住。
    #[test]
    fn a_request_needs_version_view_and_a_bounded_id() {
        let text = view().to_string();
        assert!(parse_request(&text, "view-1").is_some());

        for broken in [
            serde_json::json!({"version":2,"viewId":"view-1","id":7}),
            serde_json::json!({"version":1,"viewId":"view-2","id":7}),
            serde_json::json!({"version":1,"viewId":"view-1","id":0}),
            serde_json::json!({"version":1,"viewId":"view-1","id":MAX_ID + 1}),
            serde_json::json!({"version":1,"viewId":"view-1"}),
            serde_json::json!({"version":1,"viewId":"view-1","id":"7"}),
        ] {
            assert!(
                parse_request(&broken.to_string(), "view-1").is_none(),
                "{broken}"
            );
        }
        assert!(parse_request("not json", "view-1").is_none());
        // 上限之上的消息在解析前就被拒：不做 JSON 解析就等于不给宿主 UI 线程机会。
        let oversized = format!("{{\"pad\":\"{}\"}}", "x".repeat(MAX_MESSAGE_BYTES));
        assert!(parse_request(&oversized, "view-1").is_none());
    }

    /// **不变量：`.js` 必须是 `text/javascript`，`.wasm` 必须是 `application/wasm`。**
    ///
    /// 【为什么钉死】这两条错了，浏览器**不会报错**，只会拒绝执行 —— 用户看到的是白屏，
    /// 日志里什么都没有。这是本方案里最贵的一类缺陷，所以给它一条测试。
    #[test]
    fn module_and_wasm_mime_types_are_exact() {
        assert_eq!(mime_for(Path::new("a/b/plugin.html")), "text/html");
        assert_eq!(
            mime_for(Path::new("a/b/index-abc123.js")),
            "text/javascript"
        );
        assert_eq!(mime_for(Path::new("a/b/mod.mjs")), "text/javascript");
        assert_eq!(mime_for(Path::new("a/b/style.css")), "text/css");
        assert_eq!(
            mime_for(Path::new("a/b/kernel_bg.wasm")),
            "application/wasm"
        );
        assert_eq!(mime_for(Path::new("a/b/font.woff2")), "font/woff2");
        assert_eq!(mime_for(Path::new("a/b/noext")), "application/octet-stream");
        // 大写扩展名来自用户文件系统，不是我们的产物；仍然要给出正确类型。
        assert_eq!(mime_for(Path::new("A/B/INDEX.JS")), "text/javascript");
    }

    /// **不变量：自定义 scheme 只服务 `Resources/frontend` 之下的真实文件。**
    #[test]
    fn asset_requests_cannot_escape_the_frontend_folder() {
        let root = std::env::temp_dir().join(format!("hfs-asset-{}", std::process::id()));
        std::fs::create_dir_all(root.join("assets")).unwrap();
        std::fs::write(root.join("plugin.html"), "<html></html>").unwrap();
        std::fs::write(root.join("assets/app.js"), "1").unwrap();
        // 兄弟目录：前缀匹配会把它当成"在 root 之下"。
        let sibling = root.with_extension("evil");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("secret.js"), "2").unwrap();

        assert!(resolve_asset(&root, "/plugin.html").is_some());
        assert!(resolve_asset(&root, "/assets/app.js").is_some());
        assert!(resolve_asset(&root, "assets/app.js").is_some());

        for bad in [
            "/../secret.js",
            "/assets/../../secret.js",
            "/./plugin.html",
            "/",
            "",
            "/assets",
            "/nope.js",
            "/assets//app.js",
        ] {
            assert!(resolve_asset(&root, bad).is_none(), "{bad}");
        }
        // 绝对路径的另一种写法：直接给一个存在的系统文件。
        assert!(resolve_asset(&root, "/etc/passwd").is_none());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&sibling).ok();
    }

    /// **不变量：回执里的任意字符都不会改变 JS 表达式的结构。**
    ///
    /// 【为什么钉死】这条链路上带着用户数据（片段名、路径、错误文本）。一个没转义的
    /// 引号就能把"投递一条回执"变成"执行一段脚本"。
    #[test]
    fn delivery_expressions_survive_hostile_payloads() {
        let payload = serde_json::json!({
            "name": "it's a \\ test",
            "path": "C:\\Users\\z\\a.wav",
            "note": "line1\nline2\u{2028}end",
        })
        .to_string();
        let expression = js_expression(&payload);
        assert!(expression.starts_with("window.__hfsDeliver(JSON.parse('"));
        assert!(expression.ends_with("'))"));
        // 表达式里不能出现未转义的引号（除了最外层那两个）。
        let inner = &expression["window.__hfsDeliver(JSON.parse('".len()..expression.len() - 3];
        let mut escaped = false;
        for character in inner.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            match character {
                '\\' => escaped = true,
                '\'' => panic!("an unescaped quote escaped into the expression"),
                _ => {}
            }
        }
        assert!(!escaped, "the payload must not end with a lone backslash");
        // 反解析回原 JSON 文本：这是"表达式语义等于一条回执"的最直接证据。
        // 【为什么单遍扫描而不是一串 `replace`】顺序敏感：先把 `\'` 折成 `'` 再折
        // `\\\\` 会把 `\\\'` 折坏。单遍扫描没有顺序问题。
        let mut unescaped = String::new();
        let mut chars = inner.chars();
        while let Some(character) = chars.next() {
            if character != '\\' {
                unescaped.push(character);
                continue;
            }
            match chars.next() {
                Some('\'') => unescaped.push('\''),
                Some('\\') => unescaped.push('\\'),
                Some('n') => unescaped.push('\n'),
                Some('r') => unescaped.push('\r'),
                Some('u') => {
                    let mut digits = String::new();
                    for _ in 0..4 {
                        digits.extend(chars.next());
                    }
                    let code = u32::from_str_radix(&digits, 16).expect("a four-digit escape");
                    unescaped.push(char::from_u32(code).expect("a valid scalar value"));
                }
                other => panic!("unexpected escape: {other:?}"),
            }
        }
        assert_eq!(unescaped, payload);
    }

    /// **不变量：只有本 bundle 的 origin 可以导航。**
    #[test]
    fn only_the_bundle_origin_may_navigate() {
        assert!(is_allowed_navigation(PAGE_URL));
        assert!(is_allowed_navigation("hfs-plugin://local/plugin.html?v=1"));
        for bad in [
            "https://hifishifter.invalid/plugin.html",
            "hfs-plugin://evil/plugin.html",
            "file:///etc/passwd",
            "about:blank",
            "hfs-plugin:/local/plugin.html",
        ] {
            assert!(!is_allowed_navigation(bad), "{bad}");
        }
    }
}
