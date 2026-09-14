/**
 * 渲染内核 · WebGL 可用性诊断
 *
 * 【主要内容】
 * 在**已经失败**的环境里收集可展示的排障信息：WebGL2 / WebGL1 是否可用、
 * userAgent、devicePixelRatio。
 *
 * 【作用】
 * 时间轴内核是唯一渲染路径，WebGL2 不可用时没有回退，用户看到的失败界面就是唯一
 * 出口——它必须给出能自助排障的信息。
 *
 * 【为什么拿不到显卡型号（重要）】
 * renderer / vendor 需要创建 `webgl2` 上下文并读 `WEBGL_debug_renderer_info`，而在
 * 本函数运行的环境里这次探测**同样返回 null**。因此本模块**不提供**该字段，而不是
 * 给一个 undefined 让人误以为拿到了。
 *
 * 【与其他模块的关系】
 * - 上游：`KernelUnavailableNotice` 在渲染时调用。
 * - 下游：纯探测，不做任何写入或渲染。
 * - 独立性：只依赖 `document`（可缺失），不依赖 React；`document` 缺失或
 *   `getContext` 抛异常时按"不可用"处理，绝不向上抛错——它本就运行在有问题的机器上。
 */

/** 诊断结果。 */
export interface GlDiagnostics {
    /** 能否创建 WebGL2 上下文。 */
    readonly webgl2: boolean;
    /** 能否创建 WebGL1 上下文（用于判断"是否完全无 GL"）。 */
    readonly webgl1: boolean;
    /** 浏览器标识（Tauri 下为 WebView 的 UA）。 */
    readonly userAgent: string;
    /** 设备像素比（影响渲染成本，排障常需）。 */
    readonly devicePixelRatio: number;
}

/**
 * 探测某个 WebGL 上下文类型是否可用。
 *
 * 流程：取 `document` → `createElement("canvas")` → `getContext(type)` → 非 null 即可用。
 *
 * 特殊说明：`createElement` / `getContext` 都可能不存在或抛错（非浏览器环境、老驱动
 * 的异常路径），一律按不可用处理，不向上抛。
 *
 * @param type 上下文类型（`"webgl2"` / `"webgl"`）。
 * @returns 可创建时为 true。
 */
function canCreateContext(type: string): boolean {
    try {
        const doc = globalThis.document;
        if (doc == null || typeof doc.createElement !== "function") return false;
        const canvas = doc.createElement("canvas") as HTMLCanvasElement;
        if (canvas == null || typeof canvas.getContext !== "function") return false;
        return canvas.getContext(type) != null;
    } catch {
        return false;
    }
}

/**
 * 收集 GL 诊断信息。
 *
 * 流程：分别探测 webgl2 / webgl1 → 读取 userAgent 与 devicePixelRatio（缺失时给安全
 * 默认值）→ 组装成可直接展示 / 复制的对象。
 *
 * @returns 见 `GlDiagnostics`。
 */
export function collectGlDiagnostics(): GlDiagnostics {
    return {
        webgl2: canCreateContext("webgl2"),
        webgl1: canCreateContext("webgl"),
        userAgent:
            typeof globalThis.navigator?.userAgent === "string"
                ? globalThis.navigator.userAgent
                : "未知",
        devicePixelRatio:
            typeof globalThis.devicePixelRatio === "number" ? globalThis.devicePixelRatio : 1,
    };
}
