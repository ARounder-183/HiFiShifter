/**
 * 应用挂载点。
 *
 * 【为什么从 `main.tsx` 拆出来】偏好必须在**任何模块读取它们之前**就位：
 * `keybindingsSlice` 在模块加载期就读快捷键覆盖项、`themeStorage` 读外观、
 * `fileBrowserSlice` 读上次目录。这些读取是同步的（调用点太多，改成异步会迫使
 * 它们重排时序），所以启动顺序必须是「先灌数据 → 再加载这些模块」。
 * `main.tsx` 因此先 `await hydrateUiStorage()`，再动态 import 本文件。
 *
 * 本文件只在动态 import 之后被求值，因此它内部所有的静态 import 都发生在
 * 偏好就位之后 —— 这正是需要的保证。
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
/*
 * Radix 的样式表**不在这里** import：它必须落在 `@layer radix` 里才能被 Tailwind
 * 工具类覆盖，而 CSS 的 `@import ... layer()` 只能写在 CSS 内（见 `index.css`
 * 顶部的层顺序说明）。在这里再 import 一次会得到一份**未分层**的副本，
 * `.rt-reset { all: unset }` 会重新压过所有工具类。
 */
import "./index.css";
import App from "./App.tsx";
import { store } from "./app/store";
import { getDockDragState } from "./features/dock/dockDragStore";
import { isFileBrowserDragActive } from "./features/fileBrowser/fileBrowserDragStore";
import { AppTooltipProvider } from "./components/AppTooltip";
import { AppRootErrorBoundary } from "./components/AppRootErrorBoundary";
import { fadeToolTipSuppress } from "./components/layout/timeline/FadeContextMenu";
import { I18nProvider } from "./i18n/I18nProvider";
import { AppThemeProvider } from "./theme/AppThemeProvider";
import { SessionUnavailable } from "./features/ara/SessionUnavailable";
import { probeSession } from "./services/sessionProbe";
import { appForm, isPluginMode } from "./services/hostDetect";
import { surfaceProfile } from "./services/surfaceProfileStore";
import { installSurfaceInsets } from "./services/surfaceInsets";
import { installBackGesture } from "./services/backGesture";
import { invoke } from "./services/invoke";
import { installTouchContextMenu } from "./utils/useLongPress";
import { setGlRasterMaxDpr } from "./components/layout/renderKernel/gl/glRaster";

/**
 * 把当前形态写进 `<html>` 的 `data-*` 属性。
 *
 * | 属性 | 取值 | 用途 |
 * |---|---|---|
 * | `data-form` | desktop / plugin / mobile | 外壳与布局分支 |
 * | `data-input` | pointer / touch | 令牌层（控件高度、滚动条、字号） |
 * | `data-os` | windows / macos / linux / android / ios | 平台微调（例如 iOS 的橡皮筋） |
 *
 * 【为什么 `data-input` 不直接等于形态】桌面可以有触摸屏（Windows 平板），
 * 移动端也可以接鼠标。CSS 该跟着**输入通道**走，而不是跟着"是不是手机"走 ——
 * 这条区分在 `inputProfile.ts` 里已经是既有做法（`profileFor(event)`）。
 */
function applySurfaceMarkers(): void {
    const root = document.documentElement;
    const profile = surfaceProfile();
    root.dataset.form = appForm();
    root.dataset.input = profile.input.primary;
    root.dataset.os = profile.os;
}

export async function mountApp(): Promise<void> {
    // dev-only 性能工程脚手架：动态 import 保证生产构建完全不打包该模块。
    // 用法：`?perf=400`（clip 总数，按 10 轨均分）或 `?perf=10x40`（轨数 × 每轨
    // clip 数）冷启动即全览；运行时也可用控制台 `window.__hsPerf({...})` 重生成。
    if (import.meta.env.DEV) {
        void import("./dev/perfProject").then((module) => module.installPerfProjectDevtools());
        // dev-only 调试出口：把 store 挂到 window，便于在浏览器里读取**真实**运行期
        // 状态（选区来源标记 `multiSelectionIntentional` 这类"必须有 Redux 上下文
        // 才能验证"的事实，从内核句柄读不到）。生产构建不挂载。
        (window as unknown as { __hfsStore?: typeof store }).__hfsStore = store;
    }

    // 探活必须在扇出之前完成（见 `probeSession` 的说明）。
    const session = await probeSession();

    // ── 形态标记写进 <html> ───────────────────────────────────────────────
    //
    // 【为什么用 `data-*` 而不是给根组件传 className】CSS 令牌层要按形态覆写
    // （`html[data-input="touch"]` 抬升控件高度、放大滚动条、加安全区内边距），
    // 而这些规则必须能作用到 `html`/`body` 自身（`100dvh`、`padding: env(...)`）——
    // 组件的 className 够不到那里。属性写在 `<html>` 上，两处都能命中。
    //
    // 另外这两个属性是**调试入口**：`?form=mobile` 之后 DevTools 里一眼能看出
    // 当前跑的是哪个形态，不必去翻 store。
    applySurfaceMarkers();

    // ── 安全区内边距写进 CSS 变量 ──────────────────────────────────────────
    //
    // 【为什么在这里而不是某个组件里】`--qt-safe-*` 的消费点遍布全应用（外壳、
    // 抽屉、对话框、菜单），它们分属不同的 React 子树与 portal 容器，只有
    // `<html>` 是共同祖先。而"首帧就要对"是硬要求：晚一帧，顶栏会先画在状态栏
    // 底下再跳下来，用户看到的是闪一下。
    //
    // 传剖面里的值而不是只订阅事件：`main.tsx` 已经在挂载前取到了剖面，
    // 这里用同一份数据，第一次绘制就是对的。
    void installSurfaceInsets(surfaceProfile().viewport.insets);

    // ── 返回键 / 返回手势 ──────────────────────────────────────────────────
    //
    // 【为什么只在移动形态装】原生侧只有 Android 会执行 `window.__HFS_HANDLE_BACK__`。
    // 桌面上装一个没人调用的全局函数既无意义，也让"桌面路径不变"这条纪律多一个
    // 需要解释的例外。
    if (appForm() === "mobile") {
        installBackGesture();
    }

    // ── 触摸的长按 → 上下文菜单 ────────────────────────────────────────────
    //
    // 【为什么装在这里、只装一次】全仓有 136 处 `onContextMenu`（52 个文件）。逐个
    // 改写成"触摸也支持"意味着 136 次漏掉的机会，而漏掉的症状是"这个对象在手机上
    // 不能右键"，没有任何报错。装在 document 上之后，**今天存在的和明天新增的**
    // 处理器都会自动生效 —— 这是唯一不会随功能增加而退化的做法。
    //
    // 桌面形态下 `installTouchContextMenu()` 直接返回空卸载函数（能力表里
    // `input.secondaryButton` 为真），因此桌面路径一行都不变。
    installTouchContextMenu();

    // ── 光栅化 DPR 上限 ────────────────────────────────────────────────────
    //
    // 【为什么是 2.0】移动端旗舰机的 DPR 是 3：390×844 的画布在 DPR 3 下约 9.0 Mpx，
    // 单个 RGBA 帧缓冲 36 MB，而全应用有 6 个 WebGL2 上下文。iOS 给单进程的内存上限
    // 在 1~1.5 GB —— 几张这样的帧缓冲加上 ORT 会话就会触发 jetsam（无异常、无日志、
    // 进程直接消失）。降到 2.0 像素量减半，而在 6 英寸屏上看不出差别（波形与曲线都是
    // 低频内容）。
    //
    // 桌面传 Infinity：行为与引入这个上限之前**逐字节相同**。
    setGlRasterMaxDpr(surfaceProfile().form === "mobile" ? 2 : Number.POSITIVE_INFINITY);

    createRoot(document.getElementById("root")!).render(
        <StrictMode>
            <Provider store={store}>
                <I18nProvider>
                    <AppThemeProvider>
                        <AppTooltipProvider
                            isSuppressedExternal={() =>
                                // 停靠拖拽期间必须抑制悬停提示：它不再是原生 tooltip，
                                // 不会自己消失，会正好盖住拖拽时给用户看的落点提示。
                                //
                                // 文件浏览器拖拽同理：行既带 `data-tooltip` 又是拖拽源，
                                // 不抑制的话气泡会被钉住并一路跟着鼠标飘到时间轴上方。
                                fadeToolTipSuppress.isSuppressed ||
                                getDockDragState()?.started === true ||
                                isFileBrowserDragActive()
                            }
                        >
                            {/*
                              根级错误边界：兜住面板子树之外抛出的渲染异常。
                              没有它时，任何非面板位置抛错都会卸载整棵树、留下空白窗口。
                              放在 Provider 内侧以便用上主题与语言，包住 App 整体。
                            */}
                            <AppRootErrorBoundary>
                                {session === "unavailable" ? <SessionUnavailable /> : <App />}
                            </AppRootErrorBoundary>
                        </AppTooltipProvider>
                    </AppThemeProvider>
                </I18nProvider>
            </Provider>
        </StrictMode>,
    );
    if (isPluginMode()) {
        requestAnimationFrame(() =>
            requestAnimationFrame(() => {
                void invoke("plugin_ui_ready").catch(() => undefined);
            }),
        );
    }
}
