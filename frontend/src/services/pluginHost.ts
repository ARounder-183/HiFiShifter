// 原GUI的原生插件通信：不伪造Tauri，不让命令/事件串到其它FX实例。
export type HostEvent<T> = { event: string; id: number; payload: T };
export type PluginBootstrap = {
    version: 1;
    viewId: string;
    /**
     * 宿主提供 ARA 播放请求（`ARAPlaybackControllerInterface`）。
     *
     * 【为什么**不**放进 `hostCapabilities`】它描述的是**插件**能否向宿主发播放请求，
     * 不是宿主能否提供 transport 读数（后者是 `hostCapabilities.transport`）。
     */
    transportControl?: boolean;
    /**
     * 插件私有的参数分组是否可用。守卫的是纯插件的 `move_private_track`，**不需要任何
     * 宿主写能力**，因此恒为 `true`（见后端 `BootstrapFlags::to_json`）。
     */
    trackGrouping?: boolean;
    /**
     * 宿主用哪一套淡化轴。
     *
     * - `"legacy"`：REAPER ≤7.80，`C_FADE*SHAPE` 决定形状 → 预设号直接写；
     * - `"continuous"`：REAPER ≥7.81，curvature/S 两轴决定形状 → 预设按钮照摆，
     *   由 Rust 侧按实测表翻成 `(curvature, S)` 一对分量（见 `timeline/hostFadeAxes.ts`）；
     * - 缺省/`null`：版本读不出来 → 整块淡变编辑保持只读。
     */
    fadeAxes?: "legacy" | "continuous" | null;
    /**
     * 撤销历史由谁承载：`"host"`（宿主权威栈）或 `"local"`（插件自管栈）。
     *
     * 【为什么需要它】两条栈共用同一个「操作记录」面板，但记录里 `label` 的含义不同：
     * 本地栈是 op key（按 `history_op_<key>` 查 catalog），宿主栈是宿主自己的可读字符串
     * （原样显示）。缺了这个判别式，前端只能无差别加前缀 —— 那正是
     * `history_op_Move items` 这类原始键名被摊给用户的原因。
     *
     * 缺省按 `"local"`（独立 App / 旧后端）。
     */
    historyBackend?: "host" | "local";
    /**
     * 当前编辑后端身份：`"reaper"`（宿主给了原生扩展）或 `"ara_only"`（只有 ARA 2.0）。
     *
     * 【为什么前端需要它】ARA-only 后端上片段几何**只读**（ARA 2.0 不允许插件改 region
     * 几何），前端据此显示只读横幅，而不是让用户点了写操作再撞一个错误。
     */
    /**
     * 当前编辑后端身份。
     *
     * `"capture"` = 宿主没给 ARA，素材从 VST3 输入总线录进来（见 Capture 方案）。
     * 它与 `"ara_only"` 的区别是**实质性的**：Capture 的时间线由捕获存储构建、
     * 完全属于插件自己，而 ara_only 的时间线是宿主投影。
     */
    hostKind?: "reaper" | "ara_only" | "capture";
    /**
     * 生效的宿主能力位快照：**写片段 / 分割 / 剪贴板 / 导入 / 建轨 / 撤销栈的唯一来源**。
     *
     * 【为什么不再有并存的顶层布尔】此前 `clipEditing` / `audioImport` 等与这里并存，
     * 由后端两处各自计算 —— 必然漂移。现在只读这一份（后端
     * `ExtensionOwner::effective_host_capabilities` 统一算出）。
     */
    hostCapabilities?: {
        transport: boolean;
        clipGeometry: boolean;
        clipWrite: boolean;
        split: boolean;
        undoStack: boolean;
        mediaImport: boolean;
        trackCreate: boolean;
        itemClipboard: boolean;
        fadeAxes: boolean;
        /**
         * 插件能否把处理结果导出到文件。
         *
         * 【为什么不是恒真】导出只读授权 PCM + 写用户选定的文件，不碰宿主写接口，
         * 因此两条后端都为 `true`；但它是 `project.export` 判据的唯一来源 ——
         * 见 `hostCapabilities.ts` 的 `canExportAudio`。
         */
        exportAudio: boolean;
        /** 插件能否导出纯数据文件（布局 / 主题 / 颤音预设）。 */
        exportData: boolean;
    };
    /**
     * Capture 通路的**观察位**（`hostKind === "capture"` 时才有值，否则 `null`）。
     *
     * 【为什么是"观察到过"而不是能力声明】它们描述宿主**实际**做了什么 —— 给过播放
     * 位置吗、把音频送到输入总线了吗、走过离线（bounce）吗。宿主不会声明这些，
     * 所以只能观察。前端据此选引导语：见过位置就能录；没见过位置就该明说
     * "这个宿主没有把播放位置交给插件"，而不是让用户对着按钮猜。
     *
     * 【为什么不放进 `hostCapabilities`】那里回答"宿主能做什么"，这里回答
     * "我们观察到了什么"。混在一起会让"宿主不支持"与"还没播放过"变成同一个值。
     */
    captureCaps?: {
        sawContext: boolean;
        sawInput: boolean;
        sawOffline: boolean;
        /** 现在能不能录（等于 `sawContext`；单独给出让前端读一个字段就够）。 */
        canCapture: boolean;
    } | null;
};
export interface WebViewMessagePort {
    postMessage(message: unknown): void;
    postMessageWithAdditionalObjects?(message: unknown, objects: File[]): void;
    addEventListener(name: "message", listener: (event: { data: unknown }) => void): void;
    removeEventListener(name: "message", listener: (event: { data: unknown }) => void): void;
}
export interface PluginHostBridge {
    readonly kind: "plugin";
    invoke<T>(command: string, args?: Record<string, unknown>, files?: File[]): Promise<T>;
    listen<T>(event: string, listener: (event: HostEvent<T>) => void): Promise<() => void>;
    dispose(): void;
}
declare global {
    interface Window {
        __HFS_PLUGIN_BOOTSTRAP__?: PluginBootstrap;
        chrome?: { webview?: WebViewMessagePort };
    }
}
type Pending = {
    resolve(value: unknown): void;
    reject(error: Error): void;
    timer: ReturnType<typeof setTimeout>;
};

/**
 * 每条插件命令的等待上限（毫秒）。
 *
 * 【为什么不能只有一个 30 s】`run_vocoder_benchmark` 会依次建 CPU 会话、跑 5 轮、
 * 再建 DirectML 会话、再跑 5 轮 —— 实测在 RTX 3050 上**134 秒**。此前所有命令共用
 * 30 s，于是前端在第 30 秒就 reject，用户看到 `Backend invoke failed:
 * plugin:run_vocoder_benchmark`，而后端其实正常跑完、回执因为待办项已删而被丢弃。
 * 日志里只有一份正常结果，与"报错"完全对不上——这正是该缺陷难定位的原因。
 *
 * 【为什么是白名单而不是统一调大】默认值承担的是"插件卡死时别把界面挂住"的职责。
 * 把它一起调大，一个真正卡死的命令会让窗口僵 15 分钟。只有明确知道会跑很久的命令
 * 才给长预算，且它们都有用户可见的进度（见基准测试对话框）。
 *
 * 【这些预算是**客户端**的】后端对慢车道上的请求另有自己的绊线（30 分钟，见
 * `hifishifter-plugin` 的 `SLOW_PENDING_TTL`），它只用来不让一个真正卡住的条目永久
 * 占位。两者**故意**不等：用户可见的那个预算才该说话，所以永远是客户端先放弃。
 */
const COMMAND_TIMEOUT_MS: Readonly<Record<string, number>> = {
    run_vocoder_benchmark: 15 * 60_000,
    // 导出诊断包要读模型摘要 + 打包日志，大工程上远超 30 s。
    export_diagnostics: 5 * 60_000,
};

/** 默认等待上限（毫秒）。 */
export const DEFAULT_COMMAND_TIMEOUT_MS = 30_000;

/** 某条命令的等待上限。导出供回归测试锁定"基准测试必须拿到长预算"。 */
export function commandTimeoutMs(command: string): number {
    return COMMAND_TIMEOUT_MS[command] ?? DEFAULT_COMMAND_TIMEOUT_MS;
}

/**
 * 命令失败的**机器可读分类**，由后端随 IPC 回执一起给出。
 *
 * - `terminal`：这个编辑会话已经**永久**结束（文档/组件已关闭），再问一万次也一样。
 *   调用方应当**停止轮询**并如实显示，而不是无限重试 —— 用户报障的 Cubase 场景
 *   正是后端把同一个终态错误答了 19,056 次、把日志刷满。
 * - `transient`：暂时失败，值得稍后重试。
 */
export type PluginErrorKind = "terminal" | "transient";

/** 带分类的插件命令错误。调用方按 `kind` 决策，**不解析错误文本**。 */
export class PluginInvokeError extends Error {
    readonly kind: PluginErrorKind;
    /**
     * 后端协议码（回执的 `errorCode`，见 `services/sessionGate.ts` 的 `ERROR_CODE`）。
     *
     * 【为什么需要它】`kind` 只说"是不是终态"，说不出是**哪一种**终态。而
     * "编辑文档已关闭"（关窗重开有用）与"宿主从未绑定 ARA"（关窗重开无用）需要给
     * 用户完全不同的指引。码是协议承诺（只增不改），文本只做展示 —— 与 `kind`
     * 同一纪律：**按码分支，不解析文本**。旧后端不带该字段时为 `null`。
     */
    readonly code: number | null;
    constructor(message: string, kind: PluginErrorKind, code: number | null = null) {
        super(message);
        this.name = "PluginInvokeError";
        this.kind = kind;
        this.code = code;
    }
}

/** 宿主IPlugView兜底按键继续经过原GUI监听器，保留输入框、模态及自定义键位的规则。 */
function forwardPluginKeyboard(payload: unknown): void {
    if (!payload || typeof payload !== "object" || typeof document === "undefined") return;
    const key = payload as Record<string, unknown>;
    if (
        (key.type !== "keydown" && key.type !== "keyup") ||
        typeof key.key !== "string" ||
        !["c", "x", "v", "z", "y"].includes(key.key) ||
        key.ctrlKey !== true
    )
        return;
    const target = document.activeElement ?? document.body;
    target.dispatchEvent(
        new KeyboardEvent(key.type, {
            key: key.key,
            code: `Key${key.key.toUpperCase()}`,
            ctrlKey: true,
            shiftKey: key.shiftKey === true,
            altKey: key.altKey === true,
            metaKey: key.metaKey === true,
            repeat: key.repeat === true,
            bubbles: true,
            cancelable: true,
        }),
    );
}

/** 用宿主注入的view身份关联有界请求；只能绑定原生WebView消息口。 */
export function createPluginHost(
    port: WebViewMessagePort,
    boot: PluginBootstrap,
): PluginHostBridge {
    if (boot.version !== 1 || !boot.viewId || boot.viewId.length > 128) {
        throw new Error("Invalid plugin host bootstrap");
    }
    let closed = false;
    let nextId = 1;
    let eventId = 1;
    const pending = new Map<number, Pending>();
    const listeners = new Map<string, Set<(event: HostEvent<unknown>) => void>>();
    function receive(event: { data: unknown }) {
        if (closed || !event.data || typeof event.data !== "object") return;
        const data = event.data as Record<string, unknown>;
        if (data.version !== 1 || data.viewId !== boot.viewId) return;
        if (data.event === "plugin_keyboard") {
            forwardPluginKeyboard(data.payload);
            return;
        }
        if (typeof data.event === "string") {
            const notification = { event: data.event, id: eventId++, payload: data.payload };
            for (const handler of Array.from(listeners.get(data.event) ?? [])) {
                try {
                    handler(notification);
                } catch (error) {
                    console.error("Plugin event handler failed", error);
                }
            }
            return;
        }
        if (typeof data.id !== "number" || !Number.isSafeInteger(data.id)) return;
        const slot = pending.get(data.id);
        if (!slot) return;
        pending.delete(data.id);
        clearTimeout(slot.timer);
        if (data.ok === true) slot.resolve(data.value);
        else
            slot.reject(
                new PluginInvokeError(
                    typeof data.error === "string" ? data.error : "Invalid plugin response",
                    data.errorKind === "terminal" ? "terminal" : "transient",
                    typeof data.errorCode === "number" ? data.errorCode : null,
                ),
            );
    }
    port.addEventListener("message", receive);
    return {
        kind: "plugin",
        invoke<T>(command: string, args?: Record<string, unknown>, files?: File[]): Promise<T> {
            if (closed) return Promise.reject(new Error("Plugin editor closed"));
            if (pending.size >= 128)
                return Promise.reject(new Error("Too many pending plugin requests"));
            if (!command || command.length > 128 || !Number.isSafeInteger(nextId)) {
                return Promise.reject(
                    new Error("Invalid plugin command or exhausted request identity"),
                );
            }
            if (
                files &&
                (!port.postMessageWithAdditionalObjects ||
                    command !== "import_native_audio_file" ||
                    files.length !== 1)
            ) {
                return Promise.reject(
                    new Error("Native File import unavailable; use File menu import"),
                );
            }
            const id = nextId++;
            return new Promise<T>((resolve, reject) => {
                const timer = setTimeout(() => {
                    pending.delete(id);
                    reject(new Error("Plugin command timed out: " + command));
                }, commandTimeoutMs(command));
                pending.set(id, { resolve: (value) => resolve(value as T), reject, timer });
                try {
                    const message = { version: 1, viewId: boot.viewId, id, command, args };
                    if (files) port.postMessageWithAdditionalObjects!(message, files);
                    else port.postMessage(message);
                } catch (error) {
                    clearTimeout(timer);
                    pending.delete(id);
                    reject(error instanceof Error ? error : new Error(String(error)));
                }
            });
        },
        async listen<T>(event: string, listener: (event: HostEvent<T>) => void) {
            if (closed) throw new Error("Plugin editor closed");
            const handlers = listeners.get(event) ?? new Set();
            const handler = listener as (event: HostEvent<unknown>) => void;
            handlers.add(handler);
            listeners.set(event, handlers);
            return () => {
                handlers.delete(handler);
                if (!handlers.size) listeners.delete(event);
            };
        },
        dispose() {
            if (closed) return;
            closed = true;
            port.removeEventListener("message", receive);
            for (const slot of pending.values()) {
                clearTimeout(slot.timer);
                slot.reject(new Error("Plugin editor closed"));
            }
            pending.clear();
            listeners.clear();
        },
    };
}

let active: PluginHostBridge | null = null;

/** 独立app没有插件bootstrap，原调用路径不会被WebView2对象本身误判为插件。 */
export function getPluginHost(): PluginHostBridge | null {
    if (typeof window === "undefined" || !window.__HFS_PLUGIN_BOOTSTRAP__) return null;
    if (active) return active;
    const port = window.chrome?.webview;
    if (!port) throw new Error("Plugin bootstrap present but native message port unavailable");
    active = createPluginHost(port, window.__HFS_PLUGIN_BOOTSTRAP__);
    // 【为什么监听 bootstrap 变化】窗口可能先于 ARA 绑定打开；绑定后后端会补发一条
    // `plugin_bootstrap_changed`。这里更新全局快照，并广播一个 DOM 事件让关心形态的
    // 界面在下一轮渲染时重新读取 `hostCapabilities` / `hostKind`。
    void active.listen("plugin_bootstrap_changed", (event) => {
        const payload = event.payload;
        if (!payload || typeof payload !== "object") return;
        const current = window.__HFS_PLUGIN_BOOTSTRAP__;
        if (!current) return;
        window.__HFS_PLUGIN_BOOTSTRAP__ = {
            ...current,
            ...(payload as Partial<PluginBootstrap>),
            // 视图身份属于本窗口，不能被事件载荷覆盖。
            viewId: current.viewId,
        };
        window.dispatchEvent(new Event("hfs-plugin-bootstrap-changed"));
    });
    window.addEventListener("pagehide", () => active?.dispose(), { once: true });
    return active;
}
