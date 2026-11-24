package com.arounder.hifishifter

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.WebView
import androidx.activity.result.ActivityResult
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import app.tauri.Logger
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSArray
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * 应用自己的原生桥（Kotlin 侧）。
 *
 * 【为什么需要它 —— 三件 WebView 做不到的事】
 *
 * 1. **系统文件选择器**。`rfd` 在移动目标上编译不过（只有 gtk/macos/win32 后端），
 *    而 wry 的 `RustWebChromeClient` **没有实现 `onShowFileChooser`**（wry 0.57.0
 *    的 kotlin 目录里可以核实），所以 `<input type="file">` 在 Android 上点了
 *    毫无反应。系统选择器是 SAF（`ACTION_OPEN_DOCUMENT` / `ACTION_CREATE_DOCUMENT`），
 *    只有原生侧能发起。
 * 2. **安全区内边距**。`enableEdgeToEdge()` 让 WebView 铺到状态栏与手势条底下，
 *    而 Android WebView 的 `env(safe-area-inset-*)` 只反映**刘海**（displayCutout），
 *    不反映系统栏高度 —— 于是 CSS 里那套 `--qt-safe-*` 在真机上恒为 0，
 *    汉堡菜单与关闭按钮落在状态栏正下方，既看不见也点不到。
 * 3. **系统栏图标明暗**。应用自己的主题（浅色/深色）与系统夜间模式无关，状态栏
 *    图标必须按**应用**主题切换，否则浅色主题下会是白底白字。
 *
 * 【为什么是一个 Tauri 插件而不是 MainActivity 里的普通对象】Tauri 的
 * `PluginApi::register_android_plugin` 只认 `@TauriPlugin` + `@Command` 的类，
 * 而这条路带来了两样现成的东西：`startActivityForResult` 的**异步回调**
 * （Rust 侧 `run_mobile_plugin` 直接等到结果，不需要前端参与）与
 * `Invoke` 的 JSON 编解码。手写 JNI 或 JS 接口都要把这两样重新发明一遍。
 *
 * 【为什么命令是同步返回值 + 活动回调两种形状】`insets` / `exportDir` 是**瞬时**
 * 查询，同步返回最简单；`pickFiles` / `saveToLocation` 必须等用户在系统选择器里
 * 操作，只能走 `startActivityForResult`。
 */
@InvokeArg
class PickFilesArgs {
    /** 允许的 MIME 类型；空数组表示不限。 */
    var mimeTypes: Array<String> = emptyArray()

    /** 是否允许多选。 */
    var multiple: Boolean = false
}

@InvokeArg
class SaveToLocationArgs {
    /** 已经写好、要交给用户的文件（应用沙箱内的绝对路径）。 */
    lateinit var sourcePath: String

    /** 建议文件名（SAF 用它预填标题）。 */
    lateinit var fileName: String

    /** 目标文档的 MIME 类型；决定 SAF 的后缀提示。 */
    var mimeType: String = "application/octet-stream"
}

@InvokeArg
class SetSystemBarsArgs {
    /** true = 深色背景、需要**浅色**图标。 */
    var dark: Boolean = true
}

@TauriPlugin
class NativeBridgePlugin(private val activity: Activity) : Plugin(activity) {
    companion object {
        private const val TAG = "HiFiShifterBridge"

        /** 选择器结果回调的方法名（与下面的 `@ActivityCallback` 方法同名）。 */
        private const val PICK_CALLBACK = "pickFilesResult"
        private const val SAVE_CALLBACK = "saveToLocationResult"
    }

    /**
     * 最近一次计算出的安全区（**dp**，即 CSS px）。
     *
     * 【为什么缓存而不是每次现算】`insets` 命令由 Rust 在启动期同步调用，
     * 那时窗口可能还没完成第一次布局；缓存让"布局完成前"的调用拿到 0 而不是抛异常，
     * 而布局一完成监听器就会把真实值填进来。
     */
    private var insetTop = 0
    private var insetRight = 0
    private var insetBottom = 0
    private var insetLeft = 0

    /** 正在等待 SAF 写入的源文件（`saveToLocation` 跨活动结果期间保存）。 */
    private var pendingSourcePath: String? = null

    /**
     * WebView 创建完成：挂上窗口内边距监听。
     *
     * 【为什么在这里而不是 MainActivity】`Plugin.load(webView)` 是 Tauri 给插件准备的
     * "WebView 就绪"钩子，由 `PluginManager.onWebViewCreated` 调用 —— 多窗口（本应用
     * 目前单窗口，但插件形态会有多个）时每个 WebView 都能各自挂一份，而
     * MainActivity.onWebViewCreate 只有一个。
     */
    override fun load(webView: WebView) {
        ViewCompat.setOnApplyWindowInsetsListener(webView) { view, insets ->
            applyInsets(view as WebView, insets)
            // 返回未消费的 insets：WebView 内部还要用它们处理自己的滚动与命中。
            insets
        }
        ViewCompat.requestApplyInsets(webView)
    }

    private fun applyInsets(webView: WebView, insets: WindowInsetsCompat) {
        val density = webView.resources.displayMetrics.density.takeIf { it > 0f } ?: 1f
        val bars = insets.getInsets(
            WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
        )
        val ime = insets.getInsets(WindowInsetsCompat.Type.ime())

        insetTop = Math.round(bars.top / density)
        insetRight = Math.round(bars.right / density)
        insetLeft = Math.round(bars.left / density)
        /*
         * 键盘弹起时底部安全区报 0：此时导航栏被键盘挡住，而 WebView 已经被下面的
         * 内边距抬到了键盘上方。两个都算会让底部工具栏凭空多出 48dp 的空白。
         */
        insetBottom = if (insets.isVisible(WindowInsetsCompat.Type.ime())) {
            0
        } else {
            Math.round(bars.bottom / density)
        }

        /*
         * 键盘用**给 WebView 加内边距**的方式让位，而不是把 IME 高度也交给 CSS。
         *
         * 【为什么不用 CSS】`enableEdgeToEdge()` 之后窗口不再为键盘缩小
         * （decorFitsSystemWindows = false），页面自己没有任何"键盘弹出了"的信号；
         * 而给 WebView 加内边距会让视口真正变矮 —— `100dvh`、flex 布局、对话框的
         * 底部 sheet 全都自动正确，无需在几十个组件里各判断一次。
         *
         * 【为什么只加底部】IME 只从底部进入。左右上三边由 CSS 变量处理（见
         * `index.css` 的 `--qt-safe-*`），这样状态栏区域仍然由页面自己的底色填充。
         */
        val imeBottom = Math.round(ime.bottom / density)
        val targetBottom = if (imeBottom > 0) imeBottom else 0
        if (webView.paddingBottom != targetBottom) {
            webView.setPadding(
                webView.paddingLeft,
                webView.paddingTop,
                webView.paddingRight,
                targetBottom
            )
        }
    }

    // ── 瞬时查询 ────────────────────────────────────────────────────────────

    /** 当前安全区（dp）。Rust 在 `get_surface_profile` 里读它。 */
    @Command
    fun insets(invoke: Invoke) {
        val payload = JSObject()
        payload.put("top", insetTop)
        payload.put("right", insetRight)
        payload.put("bottom", insetBottom)
        payload.put("left", insetLeft)
        invoke.resolve(payload)
    }

    /**
     * 用户能真正找到的导出目录。
     *
     * 【为什么不是 `getExternalFilesDir`】那是 `Android/data/<pkg>/files/...`，
     * 从 Android 11 起**在系统"文件"应用里不可见**（DocumentsUI 明确隐藏
     * `Android/data`），导出的音频等于扔进黑洞 —— 这正是"导出音频不工作"的真实成因。
     * `getExternalMediaDirs()` 给的是 `Android/media/<pkg>`，同一个应用、同样零权限，
     * 但**不隐藏**：文件管理器与 MTP（USB 连电脑）都能看到。
     *
     * 取不到外部媒体目录（极少数无外部存储的设备）时退回 `getExternalFilesDir`：
     * 那时至少还能通过应用自己的"打开/分享"把文件交出去。
     */
    @Command
    fun exportDir(invoke: Invoke) {
        val base = activity.getExternalMediaDirs()?.firstOrNull()
            ?: activity.getExternalFilesDir(null)
        if (base == null) {
            invoke.reject("no external directory is mounted")
            return
        }
        val dir = File(base, "HiFiShifter")
        if (!dir.exists() && !dir.mkdirs()) {
            invoke.reject("cannot create ${dir.absolutePath}")
            return
        }
        val payload = JSObject()
        payload.put("path", dir.absolutePath)
        invoke.resolve(payload)
    }

    /**
     * 状态栏 / 导航栏图标的明暗。
     *
     * 【为什么必须由应用决定】`enableEdgeToEdge()` 装的是 `SystemBarStyle.auto`，
     * 它跟随**系统**夜间模式；而本应用的主题是用户自己选的。两者不一致时
     * 浅色主题 + 深色系统 = 白底白图标，状态栏内容直接消失。
     */
    @Command
    fun setSystemBars(invoke: Invoke) {
        val args = invoke.parseArgs(SetSystemBarsArgs::class.java)
        activity.runOnUiThread {
            val controller = WindowCompat.getInsetsController(activity.window, activity.window.decorView)
            // 深色背景需要浅色图标：isAppearanceLightStatusBars = !dark。
            controller.isAppearanceLightStatusBars = !args.dark
            controller.isAppearanceLightNavigationBars = !args.dark
        }
        invoke.resolve()
    }

    // ── 系统文件选择器：读 ──────────────────────────────────────────────────

    /**
     * 打开系统文件选择器，把选中的文件**拷贝到应用缓存**后返回路径。
     *
     * 【为什么必须拷贝】SAF 给出的是 `content://` URI，不是路径：它没有
     * `File` 语义、不能被 `std::fs` 打开、也不保证长期有效（`ACTION_OPEN_DOCUMENT`
     * 的授权在进程重启后需要 `takePersistableUriPermission` 才延续）。应用其余
     * 部分全部按**路径**工作（导入、工程、波形缓存），所以边界处一次性物化。
     *
     * 【为什么落在 cacheDir 而不是工作区】工作区的布局（`media/` 与 `imports/`）
     * 由 Rust 的 `platform::paths` 决定，Kotlin 不该有第二份意见。这里只负责
     * "把 URI 变成一个真实文件"，随后的归类由 `import_copy` 做 —— 两者在同一个
     * 数据分区内，`rename` 即可，没有第二次拷贝。
     */
    @Command
    fun pickFiles(invoke: Invoke) {
        val args = invoke.parseArgs(PickFilesArgs::class.java)
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            if (args.mimeTypes.isNotEmpty()) {
                putExtra(Intent.EXTRA_MIME_TYPES, args.mimeTypes)
            }
            putExtra(Intent.EXTRA_ALLOW_MULTIPLE, args.multiple)
        }
        try {
            startActivityForResult(invoke, intent, PICK_CALLBACK)
        } catch (error: Exception) {
            Logger.error(TAG, "no activity handles ACTION_OPEN_DOCUMENT", error)
            invoke.reject("no system file picker is available on this device")
        }
    }

    @ActivityCallback
    fun pickFilesResult(invoke: Invoke, result: ActivityResult) {
        val uris = if (result.resultCode == Activity.RESULT_OK) collectUris(result.data) else emptyList()
        if (uris.isEmpty()) {
            // 用户按返回 = 取消。**resolve 而不是 reject**：Rust 侧要能把它与
            // "读文件失败"区分开，否则一次取消会变成界面上的一条错误。
            resolveCanceled(invoke)
            return
        }
        /*
         * 【为什么必须在后台线程拷贝】`@Command` 与活动回调默认跑在**主线程**，
         * 而这里要搬运的是用户刚选的媒体文件（几百 MB 的 WAV 很常见）。
         * 在主线程上做这件事就是一次 ANR：系统选择器一关，界面就冻住。
         */
        Thread {
            try {
                val staging = File(activity.cacheDir, "pick").apply { mkdirs() }
                val paths = JSArray()
                val names = JSArray()
                for (uri in uris) {
                    val name = displayName(uri) ?: "import"
                    val destination = uniqueFile(staging, name)
                    copyUriToFile(uri, destination)
                    paths.put(destination.absolutePath)
                    names.put(name)
                }
                val payload = JSObject()
                payload.put("paths", paths)
                payload.put("names", names)
                invoke.resolve(payload)
            } catch (error: Exception) {
                Logger.error(TAG, "failed to materialise the picked files", error)
                invoke.reject(error.message ?: "failed to read the picked files")
            }
        }.start()
    }

    // ── 系统文件选择器：写 ──────────────────────────────────────────────────

    /**
     * 把已经写好的文件交给用户选定的位置（SAF `ACTION_CREATE_DOCUMENT`）。
     *
     * 【为什么"先写后拷"而不是"先选后写"】本应用的导出/保存全部按**路径**写
     * （`std::fs`、编码器、zip），无法直接写 `content://`。倒过来做只需一次
     * 文件到文件的拷贝，而且拷贝发生在导出**成功之后** —— 不会出现"用户选好了
     * 位置、导出却失败，目标处留下一个空文件"。
     */
    @Command
    fun saveToLocation(invoke: Invoke) {
        val args = invoke.parseArgs(SaveToLocationArgs::class.java)
        if (!File(args.sourcePath).isFile) {
            invoke.reject("source file does not exist: ${args.sourcePath}")
            return
        }
        pendingSourcePath = args.sourcePath
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = args.mimeType
            putExtra(Intent.EXTRA_TITLE, args.fileName)
        }
        try {
            startActivityForResult(invoke, intent, SAVE_CALLBACK)
        } catch (error: Exception) {
            pendingSourcePath = null
            Logger.error(TAG, "no activity handles ACTION_CREATE_DOCUMENT", error)
            invoke.reject("no system save dialog is available on this device")
        }
    }

    @ActivityCallback
    fun saveToLocationResult(invoke: Invoke, result: ActivityResult) {
        val source = pendingSourcePath
        pendingSourcePath = null
        val uri = if (result.resultCode == Activity.RESULT_OK) result.data?.data else null
        if (uri == null || source == null) {
            resolveCanceled(invoke)
            return
        }
        // 同 `pickFilesResult`：拷贝可能几百 MB，不能在主线程做。
        Thread {
            try {
                activity.contentResolver.openOutputStream(uri, "wt")?.use { output ->
                    File(source).inputStream().use { input -> input.copyTo(output) }
                } ?: throw IOException("the chosen location is not writable")
                val payload = JSObject()
                payload.put("name", displayName(uri) ?: File(source).name)
                invoke.resolve(payload)
            } catch (error: Exception) {
                Logger.error(TAG, "failed to write into the chosen location", error)
                invoke.reject(error.message ?: "failed to write the file")
            }
        }.start()
    }

    /** 统一的"用户取消"应答；Rust 侧据此返回 `None` 而不是一个错误。 */
    private fun resolveCanceled(invoke: Invoke) {
        val payload = JSObject()
        payload.put("canceled", true)
        invoke.resolve(payload)
    }

    // ── 小工具 ──────────────────────────────────────────────────────────────

    private fun collectUris(data: Intent?): List<Uri> {
        if (data == null) return emptyList()
        val clip = data.clipData
        if (clip != null) {
            return (0 until clip.itemCount).mapNotNull { index -> clip.getItemAt(index).uri }
        }
        return listOfNotNull(data.data)
    }

    private fun displayName(uri: Uri): String? {
        return try {
            activity.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
        } catch (error: Exception) {
            Logger.error(TAG, "failed to resolve the display name of $uri", error)
            null
        }
    }

    private fun copyUriToFile(uri: Uri, destination: File) {
        activity.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(destination).use { output -> input.copyTo(output) }
        } ?: throw IOException("cannot open $uri")
    }

    /**
     * 同名冲突时加序号（`name (2).ext`）。
     *
     * 【为什么不用哈希】用户是按文件名认素材的。这一条与 Rust 侧
     * `platform::paths::import_copy` 的取舍相同 —— 那里的注释解释了原因。
     */
    private fun uniqueFile(directory: File, name: String): File {
        var candidate = File(directory, name)
        if (!candidate.exists()) return candidate
        val dot = name.lastIndexOf('.')
        val stem = if (dot > 0) name.substring(0, dot) else name
        val extension = if (dot > 0) name.substring(dot) else ""
        var index = 2
        while (candidate.exists()) {
            candidate = File(directory, "$stem ($index)$extension")
            index += 1
        }
        return candidate
    }
}
