/**
 * 安全区内边距：把**真实**的系统栏高度写进 CSS 自定义属性。
 *
 * 【为什么 CSS 的 `env(safe-area-inset-*)` 不够】那条规范只承诺两件事：刘海
 * （display cutout）与 iOS 的安全区。Android WebView 把它实现成"只反映刘海" ——
 * 状态栏与导航栏的高度**不在其中**。而 Tauri 的 Android 模板从 2.4 起就调
 * `enableEdgeToEdge()`（Android 15+ 更是强制边到边），WebView 铺满整屏、
 * 内容直接画到状态栏底下。
 *
 * 两者相加的结果是：`env(safe-area-inset-top)` 在绝大多数 Android 设备上等于 0，
 * 而汉堡菜单、抽屉的关闭按钮、对话框标题正好落在状态栏那一条上 —— 既被系统栏
 * 盖住、又因为系统栏吃掉触摸而**点不到**。这不是"样式不好看"，是入口不可达。
 *
 * 因此真实值由原生侧（`WindowInsets`）给出，经后端剖面 `viewport.insets` 送到
 * 这里，写成 `--qt-inset-*`；`index.css` 再把它与 `env()` 取较大者得到
 * `--qt-safe-*`。两端各管一件事：原生给系统栏，CSS 给刘海。
 */
import { invoke } from "./invoke";

/** 四边内边距，单位是 CSS px。 */
export interface SurfaceInsets {
    top: number;
    right: number;
    bottom: number;
    left: number;
}

const ZERO: SurfaceInsets = { top: 0, right: 0, bottom: 0, left: 0 };

/** CSS 自定义属性名。与 `index.css` 的 `--qt-inset-*` 逐字一致。 */
const PROPERTIES: Array<[keyof SurfaceInsets, string]> = [
    ["top", "--qt-inset-top"],
    ["right", "--qt-inset-right"],
    ["bottom", "--qt-inset-bottom"],
    ["left", "--qt-inset-left"],
];

function sanitize(value: unknown): SurfaceInsets {
    if (!value || typeof value !== "object") return ZERO;
    const raw = value as Partial<SurfaceInsets>;
    const pick = (candidate: unknown): number => {
        const number = typeof candidate === "number" ? candidate : Number(candidate);
        // 负值没有意义（内边距不能为负），非有限值同样丢弃 —— 一个 NaN 写进
        // CSS 会让整条 `padding` 声明失效，表现为"安全区完全没生效"。
        return Number.isFinite(number) && number > 0 ? number : 0;
    };
    return {
        top: pick(raw.top),
        right: pick(raw.right),
        bottom: pick(raw.bottom),
        left: pick(raw.left),
    };
}

/**
 * 把内边距写进 `<html>` 的样式。
 *
 * 【为什么写在 `documentElement` 上】`--qt-safe-*` 的消费点遍布全应用（外壳、
 * 抽屉、对话框、菜单、Toast），而它们分属不同的 React 子树与 portal 容器。
 * 只有 `html` 是它们共同的祖先。
 */
export function applySurfaceInsets(insets: unknown): void {
    const value = sanitize(insets);
    const style = document.documentElement.style;
    for (const [key, property] of PROPERTIES) {
        style.setProperty(property, `${value[key]}px`);
    }
}

/**
 * 应用一次并保持更新；返回卸载函数。
 *
 * 【为什么还要订阅 resize】原生侧的内边距会随**旋转**、**折叠屏展开**、**分屏
 * 尺寸变化**而改变，而这些都不会重新加载页面。桌面形态下这个订阅是空操作
 * （内边距恒为 0，命令也不存在），因此不需要额外的形态判断。
 *
 * 【为什么是防抖 120ms 而不是每次事件都问】旋转时 `resize` 会连续触发十几帧，
 * 每次一次 IPC 纯属浪费；而 120ms 远小于用户能察觉的延迟。
 */
export function installSurfaceInsets(initial?: unknown): () => void {
    applySurfaceInsets(initial);
    if (typeof window === "undefined") return () => undefined;

    let timer: number | null = null;
    let disposed = false;

    const refresh = () => {
        if (disposed) return;
        void invoke("get_surface_insets")
            .then((value) => {
                if (!disposed) applySurfaceInsets(value);
            })
            .catch(() => {
                // 命令不存在（旧后端 / 浏览器预览）或移动端没有原生桥：保留上一次
                // 的值。宁可沿用旧内边距，也不要把它清成 0 —— 那会让界面在旋转后
                // 突然压到状态栏底下。
            });
    };

    const schedule = () => {
        if (timer !== null) window.clearTimeout(timer);
        timer = window.setTimeout(() => {
            timer = null;
            refresh();
        }, 120);
    };

    window.addEventListener("resize", schedule);
    window.addEventListener("orientationchange", schedule);

    return () => {
        disposed = true;
        if (timer !== null) window.clearTimeout(timer);
        window.removeEventListener("resize", schedule);
        window.removeEventListener("orientationchange", schedule);
    };
}
