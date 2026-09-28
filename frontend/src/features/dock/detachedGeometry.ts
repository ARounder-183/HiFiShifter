/**
 * 独立窗口与"界面内浮窗"之间的几何换算。
 *
 * 【为什么需要】独立窗口不是"另一份位置"，而是同一个浮窗的**另一种呈现**：把浮窗
 * 拆出去时窗口落在原处（原地转换），关掉时浮窗继承窗口的位置与大小。两者坐标系不同：
 *
 * - 界面内浮窗：相对主窗口**客户区**（webview）的 CSS 像素；
 * - 独立窗口：屏幕坐标（逻辑像素），且创建参数 `x/y` 是**外框**左上角，而
 *   `width/height` 是**客户区**尺寸（tauri 侧 `inner_size(config.width, config.height)`）。
 *
 * 因此换算必须显式带上"客户区原点"与"外框厚度"：少了前者窗口会跑到屏幕角落，
 * 少了后者每拆一次偏一个标题栏、每关一次缩一圈（棘轮效应）。
 *
 * 纯函数、无 Tauri 依赖 —— 见同名测试。
 */

/** 逻辑像素矩形（浮窗坐标或屏幕坐标，取决于上下文）。 */
export interface GeometryRect {
    x: number;
    y: number;
    w: number;
    h: number;
}

/**
 * 主窗口的几何（全部为**逻辑像素**，屏幕坐标）。
 *
 * @param clientOriginX 客户区左上角的屏幕 x。
 * @param clientOriginY 客户区左上角的屏幕 y。
 * @param clientWidth 客户区宽度。
 * @param clientHeight 客户区高度。
 * @param frameInsetX 客户区相对**外框**左上角的横向偏移（边框宽度）。
 * @param frameInsetY 客户区相对外框左上角的纵向偏移（标题栏 + 边框高度）。
 */
export interface MainWindowFrame {
    clientOriginX: number;
    clientOriginY: number;
    clientWidth: number;
    clientHeight: number;
    frameInsetX: number;
    frameInsetY: number;
}

/**
 * 界面内浮窗矩形 → 独立窗口的创建参数（外框位置 + 客户区尺寸）。
 *
 * 独立窗口的客户区左上角 = 主窗口客户区原点 + 浮窗坐标；创建参数要的是**外框**
 * 位置，故再减去外框厚度。
 */
export function floatRectToWindowRect(float: GeometryRect, frame: MainWindowFrame): GeometryRect {
    return {
        x: Math.round(frame.clientOriginX + float.x - frame.frameInsetX),
        y: Math.round(frame.clientOriginY + float.y - frame.frameInsetY),
        w: Math.round(float.w),
        h: Math.round(float.h),
    };
}

/**
 * 独立窗口的客户区屏幕矩形 → 界面内浮窗矩形。
 *
 * @param clientRect 独立窗口**客户区**的屏幕矩形（`innerPosition` + `innerSize`）。
 */
export function windowRectToFloatRect(
    clientRect: GeometryRect,
    frame: MainWindowFrame,
): GeometryRect {
    return {
        x: Math.round(clientRect.x - frame.clientOriginX),
        y: Math.round(clientRect.y - frame.clientOriginY),
        w: Math.round(clientRect.w),
        h: Math.round(clientRect.h),
    };
}

/**
 * 把独立窗口的屏幕矩形夹进显示器范围。
 *
 * 【为什么必须夹】窗口位置来自主窗口客户区，主窗口贴近屏幕边缘（或跨显示器）时
 * 换算结果可能整块落在显示器之外 —— 用户会得到一个看不见、也点不到的窗口，且
 * 它的位置会被记下来，下次启动依旧不可达。夹紧保证：装得下就整窗可见，装不下也
 * 至少让左上角留在显示器内。
 *
 * @param rect 待夹紧的窗口矩形（外框，屏幕逻辑像素）。
 * @param monitor 显示器矩形（屏幕逻辑像素）。
 * @param marginPx 与显示器边缘保留的间距。
 */
export function clampScreenRectToMonitor(
    rect: GeometryRect,
    monitor: GeometryRect,
    marginPx = 8,
): GeometryRect {
    const minX = monitor.x + marginPx;
    const minY = monitor.y + marginPx;
    // 窗口比显示器还大时上界会小于下界：此时取上界（左上角贴边），至少保证可达。
    const maxX = Math.max(minX, monitor.x + monitor.w - rect.w - marginPx);
    const maxY = Math.max(minY, monitor.y + monitor.h - rect.h - marginPx);
    return {
        x: Math.round(Math.min(maxX, Math.max(minX, rect.x))),
        y: Math.round(Math.min(maxY, Math.max(minY, rect.y))),
        w: rect.w,
        h: rect.h,
    };
}
