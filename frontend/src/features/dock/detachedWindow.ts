/**
 * 独立窗口的生命周期管理（把一个窗体拆到主窗口之外）。
 *
 * 【为什么可行】创建第二个 webview 的能力在本项目里已经存在并且在用 —— 外观设置
 * 窗口就是前端 `new WebviewWindow(...)` 建出来的独立窗口（独立 Vite 入口 + 独立
 * JS 上下文），`capabilities/default.json` 里也已放行
 * `core:webview:allow-create-webview-window`。因此这里不需要任何 Rust 侧改动。
 *
 * 【代价与边界】独立窗口是**另一个 JS 上下文**：面板必须在那边重新挂载，且状态
 * 需要跨窗口同步（见 `detachBridge`）。因此只有声明了 `detachable` 的面板才会被
 * 拆出去（时间轴 / 参数编辑器带着 WebGL 上下文与波形缓存，不在此列）。
 *
 * 【失败即降级】任何一步失败（权限、平台限制、label 冲突）都返回失败原因，调用方
 * 回退为进程内浮窗 —— 用户不会因为"窗口没开出来"而丢掉面板。
 */

/** 独立窗口的 label 前缀（与主窗口的 `main` 区分）。 */
const DETACHED_LABEL_PREFIX = "hs-detached-";

/** 由窗体 id 得到窗口 label。 */
export function detachedWindowLabel(formId: string): string {
    return `${DETACHED_LABEL_PREFIX}${formId}`;
}

/** 由窗口 label 反查窗体 id（不是独立窗口时返回 null）。 */
export function formIdFromWindowLabel(label: string): string | null {
    return label.startsWith(DETACHED_LABEL_PREFIX)
        ? label.slice(DETACHED_LABEL_PREFIX.length)
        : null;
}

/** 卫星窗口的 URL（相对应用根，与 `appearance.html` 同一约定）。 */
export function detachedWindowUrl(formId: string): string {
    return `detached.html?hsDetachedForm=${encodeURIComponent(formId)}`;
}

export interface DetachedWindowRequest {
    formId: string;
    title: string;
    width: number;
    height: number;
    /** 屏幕坐标（缺省居中）。 */
    screen?: { x: number; y: number } | null;
}

export type DetachedWindowResult = { ok: true } | { ok: false; reason: string };

/** 动态载入 Tauri 窗口 API（非 Tauri 环境返回 null）。 */
async function loadWebviewWindowApi(): Promise<{
    WebviewWindow: typeof import("@tauri-apps/api/webviewWindow").WebviewWindow;
} | null> {
    try {
        const mod = await import("@tauri-apps/api/webviewWindow");
        return { WebviewWindow: mod.WebviewWindow };
    } catch {
        return null;
    }
}

/**
 * 打开（或聚焦）一个承载指定窗体的独立窗口。
 *
 * 已存在同 label 的窗口时只聚焦，不重建 —— 避免用户重复点击开出多个副本。
 */
export async function openDetachedWindow(
    request: DetachedWindowRequest,
): Promise<DetachedWindowResult> {
    const api = await loadWebviewWindowApi();
    if (api === null) return { ok: false, reason: "window-api-unavailable" };
    const label = detachedWindowLabel(request.formId);
    try {
        const existing = await api.WebviewWindow.getByLabel(label);
        if (existing) {
            await existing.setFocus();
            return { ok: true };
        }
    } catch {
        // 查不到（或平台不支持查询）时继续创建：创建本身会报真正的错误。
    }
    try {
        const win = new api.WebviewWindow(label, {
            url: detachedWindowUrl(request.formId),
            title: request.title,
            width: Math.max(240, Math.round(request.width)),
            height: Math.max(160, Math.round(request.height)),
            x: request.screen ? Math.round(request.screen.x) : undefined,
            y: request.screen ? Math.round(request.screen.y) : undefined,
            // 用系统标题栏：标题与关闭键由操作系统提供，面板内不再重复画一遍。
            decorations: true,
            resizable: true,
            focus: true,
        });
        return await new Promise<DetachedWindowResult>((resolve) => {
            let settled = false;
            const finish = (result: DetachedWindowResult) => {
                if (settled) return;
                settled = true;
                resolve(result);
            };
            void win.once("tauri://created", () => finish({ ok: true }));
            void win.once("tauri://error", (event) => {
                finish({
                    ok: false,
                    reason: `create-failed:${String((event as { payload?: unknown }).payload ?? "")}`,
                });
            });
            // 兜底：某些平台上 created 事件可能早于监听注册。
            setTimeout(() => finish({ ok: true }), 1500);
        });
    } catch (error) {
        return {
            ok: false,
            reason: error instanceof Error ? error.message : String(error),
        };
    }
}

/** 关闭指定窗体的独立窗口（不存在时静默）。 */
export async function closeDetachedWindow(formId: string): Promise<void> {
    const api = await loadWebviewWindowApi();
    if (api === null) return;
    try {
        const win = await api.WebviewWindow.getByLabel(detachedWindowLabel(formId));
        await win?.close();
    } catch {
        // 窗口已不在：无需处理。
    }
}

/** 关闭所有独立窗口（主窗口退出前调用）。 */
export async function closeAllDetachedWindows(): Promise<void> {
    const api = await loadWebviewWindowApi();
    if (api === null) return;
    try {
        const all = await api.WebviewWindow.getAll();
        await Promise.all(
            all
                .filter((win) => formIdFromWindowLabel(win.label) !== null)
                .map((win) => win.close().catch(() => undefined)),
        );
    } catch {
        // 查询失败时无从清理：主窗口退出会带走子窗口（见 lib.rs 的退出处理）。
    }
}

/**
 * 订阅某个独立窗口的关闭事件。
 *
 * @returns 取消订阅函数。
 */
export async function onDetachedWindowClosed(
    formId: string,
    handler: () => void,
): Promise<() => void> {
    const api = await loadWebviewWindowApi();
    if (api === null) return () => {};
    const label = detachedWindowLabel(formId);
    let unlisten: (() => void) | null = null;
    try {
        const win = await api.WebviewWindow.getByLabel(label);
        if (!win) return () => {};
        const off = await win.onCloseRequested(() => {
            handler();
        });
        unlisten = off;
    } catch {
        return () => {};
    }
    return () => unlisten?.();
}

/** 读取独立窗口当前的屏幕坐标（用于持久化）。 */
export async function readDetachedWindowPosition(
    formId: string,
): Promise<{ x: number; y: number } | null> {
    const api = await loadWebviewWindowApi();
    if (api === null) return null;
    try {
        const win = await api.WebviewWindow.getByLabel(detachedWindowLabel(formId));
        if (!win) return null;
        const position = await win.outerPosition();
        const scale = await win.scaleFactor().catch(() => 1);
        // `outerPosition` 返回物理像素；持久化用逻辑像素（与创建参数同一坐标系）。
        return { x: position.x / (scale || 1), y: position.y / (scale || 1) };
    } catch {
        return null;
    }
}
