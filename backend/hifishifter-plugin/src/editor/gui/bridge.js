// 平台垫片：让 WebKit 假装成 WebView2 的那个 `chrome.webview`。
//
// 【为什么只有这一个文件是 JS】前端（`services/pluginHost.ts`）判"是不是插件窗口"
// 只看两件事：`window.__HFS_PLUGIN_BOOTSTRAP__` 与 `window.chrome.webview`。
// WebView2 原生提供后者，WebKit 没有 —— 于是这里补一个**只有六个方法**的替身，
// 前端一行都不用改。这条约束是整套方案的第一原则：一旦为了平台方便去改前端，
// 就等于承认"同一份 UI 两套协议"。
//
// 【为什么只传字符串】WebKit 的结构化 `postMessage` 会走 `NSDictionary` /
// `JSCValue` 转换，`u64` 的 id 在那里会丢精度（`9007199254740993` 变成 `…992`），
// 回执就对不上请求了。字符串是三条链路上唯一逐字节一致的形式。
//
// 【为什么不实现 `postMessageWithAdditionalObjects`】它承载的是"WebView2 拿到的真实
// 磁盘 File 对象"。WebKit 的 `File` 没有路径（沙箱），这条路走不通 —— 两个 WebKit
// 平台改走原生拖放事件（`native_drop`）。前端早就判了这件事
// （`pluginHost.ts` 在没有该方法时明确提示"用 File 菜单导入"）。

(function () {
    "use strict";
    if (typeof window === "undefined") return;
    // 幂等：万一被注入两次，不要覆盖 WebView2 原生提供的那个。
    if (window.chrome && window.chrome.webview) return;

    var listeners = [];

    /**
     * 原生侧的唯一入口：`evaluateJavaScript("window.__hfsDeliver(JSON.parse('…'))")`。
     *
     * 【为什么容忍字符串与对象两种入参】原生侧把回执序列化成 JSON 文本再包成
     * JS 字符串字面量（见 `protocol::js_expression`）；这里统一解析一次，调用方
     * 就不用关心它来自哪条平台链路。
     */
    window.__hfsDeliver = function (value) {
        var data;
        if (typeof value === "string") {
            try {
                data = JSON.parse(value);
            } catch (error) {
                return;
            }
        } else {
            data = value;
        }
        var current = listeners.slice();
        for (var index = 0; index < current.length; index += 1) {
            try {
                current[index]({ data: data });
            } catch (error) {
                // 页面自己的监听器抛错不该打死整条投递链：后面的监听器仍应收到。
            }
        }
    };

    window.chrome = window.chrome || {};
    window.chrome.webview = {
        postMessage: function (message) {
            window.webkit.messageHandlers.hfs.postMessage(JSON.stringify(message));
        },
        addEventListener: function (name, listener) {
            if (name === "message") listeners.push(listener);
        },
        removeEventListener: function (name, listener) {
            if (name !== "message") return;
            var index = listeners.indexOf(listener);
            if (index >= 0) listeners.splice(index, 1);
        },
    };
})();
