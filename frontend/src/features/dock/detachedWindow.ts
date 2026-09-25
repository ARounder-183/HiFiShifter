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

/**
 * 主窗口的 **Tauri 窗口 label**。
 *
 * `tauri.conf.json` 的窗口没有显式声明 `label`，Tauri 的默认值就是 `main`。它与
 * `resolveWindowLabel()` 的逻辑角色同名但含义不同：这里是真实窗口 label，用于给
 * 独立窗口建立 owner 关系。
 */
const MAIN_WINDOW_LABEL = "main";

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
    let createError: string | null = null;
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
            // 【独立窗口不是"第二个应用"，而是主窗口的从属窗口】把主窗口设为 owner：
            // - 任务栏只留主窗口一个按钮（shell 会跳过有 owner 的窗口）；
            // - 从属窗口恒在 owner **之上** ⇒ 永远压在主界面前面，又不会盖住别的程序；
            // - 多个从属窗口之间按**激活顺序**排列 ⇒ 焦点在谁身上谁就在最前；
            // - owner 最小化时它们随之隐藏、owner 销毁时被系统一并销毁（正是我们要的
            //   "主窗口退出即带走全部独立窗口"）。
            // 语义在各平台一致（Linux 为 transient-for、macOS 为 child window），
            // 因此不做平台分支。`skipTaskbar` 只作兜底：即便某些平台的 owner 关系
            // 不参与任务栏判定，也不会多占一个按钮。
            parent: MAIN_WINDOW_LABEL,
            skipTaskbar: true,
        });
        // 错误事件尽力监听（`once` 自身是异步的，可能错过已发出的错误 —— 因此下面
        // 还有一次基于"窗口是否真的出现"的确认，两者互补）。
        void win
            .once("tauri://error", (event) => {
                createError = String((event as { payload?: unknown }).payload ?? "unknown");
            })
            .catch(() => undefined);
    } catch (error) {
        // 构造函数本身通常不抛（它只是发起 invoke），但平台差异下可能抛。
        return {
            ok: false,
            reason: error instanceof Error ? error.message : String(error),
        };
    }

    // 【为什么轮询确认而不是等事件】`tauri://created` 可能在 `once` 注册完成之前就已
    // 发出（注册本身要走一次 IPC），只靠事件会**漏判**；而"超时即成功"又会把真正的
    // 失败（label 冲突、权限不足）报告成成功。曾经两个方向都踩过：先是无条件成功
    // （失败被吞、面板两边都不见），随后改成"超时查一次"又产生假失败（窗口 1.5s 内
    // 还没登记 ⇒ 判失败 ⇒ 回退进程内浮层 ⇒ 用户看到窗口没出来）。
    //
    // 唯一可靠的判据是**观察窗口是否存在**：轮询到它出现即成功，到时限仍未出现即
    // 失败，并把原因上报（绝不静默回退）。
    const deadline = Date.now() + DETACHED_CREATE_TIMEOUT_MS;
    for (;;) {
        if (createError !== null) return { ok: false, reason: `create-failed:${createError}` };
        try {
            const created = await api.WebviewWindow.getByLabel(label);
            if (created) return { ok: true };
        } catch {
            // 查询失败按"还没登记"处理，继续轮询。
        }
        if (Date.now() >= deadline) {
            return { ok: false, reason: "create-not-registered" };
        }
        await new Promise((resolve) => setTimeout(resolve, DETACHED_CREATE_POLL_MS));
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

/** 创建窗口后确认其存在的轮询间隔与时限。 */
const DETACHED_CREATE_POLL_MS = 200;
const DETACHED_CREATE_TIMEOUT_MS = 6000;

/** 监视独立窗口是否已消失的轮询间隔（与首次检查的宽限期）。 */
const DETACHED_WATCH_INTERVAL_MS = 1000;
const DETACHED_WATCH_GRACE_MS = 2000;

/**
 * 独立窗口的屏幕几何（**逻辑像素**，与创建参数同一坐标系）。
 *
 * 位置用于下次重新开启时恢复（`form.floatScreen`），尺寸用于恢复浮窗大小
 * （`form.float.w/h`）—— 两者都是"用户上次摆成什么样"的一部分。
 */
export interface DetachedWindowGeometry {
    x: number;
    y: number;
    w: number;
    h: number;
}

/** 读取一个窗口实例的屏幕几何（逻辑像素）；读取失败返回 null。 */
async function readGeometryOf(win: {
    outerPosition: () => Promise<{ x: number; y: number }>;
    outerSize: () => Promise<{ width: number; height: number }>;
    scaleFactor: () => Promise<number>;
}): Promise<DetachedWindowGeometry | null> {
    try {
        const scale = (await win.scaleFactor().catch(() => 1)) || 1;
        const position = await win.outerPosition();
        const size = await win.outerSize();
        // Tauri 的坐标与尺寸都是**物理像素**；持久化用逻辑像素（与创建参数一致），
        // 否则在系统缩放率 ≠ 100% 时窗口每次重开都会按比例漂移。
        return {
            x: position.x / scale,
            y: position.y / scale,
            w: size.width / scale,
            h: size.height / scale,
        };
    } catch {
        return null;
    }
}

/**
 * 监视某个独立窗口是否已被关闭，并**顺带上报它的屏幕几何**。
 *
 * 【为什么用轮询，而不是 `onCloseRequested`】`WebviewWindow.listen` 会把监听
 * **限定到该窗口的 label**（`target: { kind: "Webview", label: this.label }`），
 * 而事件只投递给目标 webview。于是从**主窗口**给另一个窗口注册 `onCloseRequested`
 * 时，监听器装在主窗口的 JS 上下文里却永远收不到事件 —— 而 Tauri 的 JS 封装正是
 * 靠那个回调在"未 preventDefault"时调用 `destroy()`：回调不触发，**窗口就永远关
 * 不掉**（实测：向独立窗口发 WM_CLOSE 与 SC_CLOSE 都无效，窗口一直留在屏幕上）。
 *
 * 轮询窗口是否存在则完全不依赖跨窗口事件语义，而且能覆盖"进程被杀 / 崩溃"这类
 * 事件根本不会到达的情况。
 *
 * 【几何为什么也在这里上报】用户拖动/缩放独立窗口时，主窗口这一侧**收不到任何
 * 事件**（跨窗口事件语义不可靠，见上）；轮询是唯一稳定时机。每秒两次 IPC 调用，
 * 代价可忽略，换来"关窗/退出/崩溃后下次仍能恢复到上次的位置与大小"。
 *
 * @param formId 窗体 id。
 * @param handlers.onGone 窗口消失时的回调（主窗口据此把窗体收回进程内浮层）。
 * @param handlers.onGeometry 每次轮询到几何时的回调（调用方自行做变化判定）。
 * @returns 停止监视（收回窗体、或主窗口卸载时调用）。
 */
export function watchDetachedWindow(
    formId: string,
    handlers: {
        onGone: () => void;
        onGeometry?: (geometry: DetachedWindowGeometry) => void;
    },
): () => void {
    const { onGone, onGeometry } = handlers;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
        if (stopped) return;
        const api = await loadWebviewWindowApi();
        if (api === null || stopped) return;
        try {
            const win = await api.WebviewWindow.getByLabel(detachedWindowLabel(formId));
            if (!win) {
                if (!stopped) onGone();
                return;
            }
            if (onGeometry) {
                const geometry = await readGeometryOf(win);
                if (geometry && !stopped) onGeometry(geometry);
            }
        } catch {
            // 查询失败按"仍在"处理：下一个周期再确认，避免误回收。
        }
        if (!stopped) timer = setTimeout(tick, DETACHED_WATCH_INTERVAL_MS);
    };

    // 宽限期：窗口刚创建时可能还没登记进窗口管理器，立即查询会误判为"已消失"。
    timer = setTimeout(tick, DETACHED_WATCH_GRACE_MS);
    return () => {
        stopped = true;
        if (timer !== null) clearTimeout(timer);
    };
}

/** 读取独立窗口当前的屏幕几何（用于持久化）；窗口不在时返回 null。 */
export async function readDetachedWindowGeometry(
    formId: string,
): Promise<DetachedWindowGeometry | null> {
    const api = await loadWebviewWindowApi();
    if (api === null) return null;
    try {
        const win = await api.WebviewWindow.getByLabel(detachedWindowLabel(formId));
        if (!win) return null;
        return await readGeometryOf(win);
    } catch {
        return null;
    }
}
