package com.arounder.hifishifter

import android.os.Bundle
import android.webkit.WebView
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge

/**
 * 应用唯一的 Activity。
 *
 * 【为什么它需要**手写**而不是留着模板生成的那 10 行】模板只做了一件事：
 * `enableEdgeToEdge()`（Tauri 2.4 起 Android 15+ 强制边到边，不这么做也会被迫边到边）。
 * 但边到边之后有两件事必须由应用自己补上，而模板都没有：
 *
 * 1. **返回键**。`TauriActivity` 把 `handleBackNavigation` 关掉了（它只在
 *    多页 WebView 里有用），于是返回键直接走系统默认 —— `finish()`，
 *    应用**立刻退出**，哪怕用户正开着一个对话框。移动端的约定是"返回键先关
 *    最上层的东西"，这只有应用自己知道（见下面的回调）。
 * 2. **安全区与系统栏**。见 `NativeBridgePlugin`（内边距监听 + 图标明暗）。
 */
class MainActivity : TauriActivity() {
    companion object {
        /**
         * 交给网页的返回键判定脚本。
         *
         * 【为什么把判定放进网页】"现在最上层开着什么"只有 DOM 知道（Radix 的
         * 对话框与菜单、应用自己的抽屉、快速搜索…）。原生侧若也维护一份栈，
         * 就会与真实界面分叉 —— 那是本仓反复禁止的"两处决定同一件事"。
         *
         * 脚本返回 `"exit"` 表示网页没有可以关闭的东西，此时才真的退出。
         * `evaluateJavascript` 会把返回值 JSON 编码后回调，所以这里比对的是
         * 结果字符串里有没有 `exit`。
         */
        private const val BACK_SCRIPT =
            "window.__HFS_HANDLE_BACK__ ? window.__HFS_HANDLE_BACK__() : 'exit'"
    }

    /** 由 `onWebViewCreate` 填入；返回键要拿它求值。 */
    private var webView: WebView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        // 边到边必须在 super.onCreate 之前：它要改的是窗口的 decor 属性。
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                val view = webView
                if (view == null) {
                    finish()
                    return
                }
                view.evaluateJavascript(BACK_SCRIPT) { result ->
                    // 结果形如 ""handled"" / ""exit""（JSON 字符串）。
                    if (result == null || result.contains("exit")) {
                        finish()
                    }
                }
            }
        })
    }

    override fun onWebViewCreate(webView: WebView) {
        super.onWebViewCreate(webView)
        this.webView = webView
    }
}
