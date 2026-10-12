/**
 * 会话闸门 —— 后端编辑会话的**终态**判定与全局停止。
 *
 * 【为什么需要它，而不是让每个轮询点各自判断】
 * 终态（"编辑文档已关闭"、"宿主从未绑定 ARA"…）是**会话级**事实，不是每条命令各自的
 * 属性。此前后端已经随回执给出 `errorKind: "terminal"`，但只有 `pluginApplyStore`
 * 一个轮询点采纳了它 —— App 的常驻播放轮询完全不看，于是实测在 Cubase 上同一个终态
 * 错误被问 750 次（`get_playback_state` 每 400ms 一次，持续 5 分钟以上），每次都要跨
 * 一次 IPC 并把整条错误对象（含最长 4000 字符的栈）序列化过去。
 *
 * 只要还有**任何一个**轮询点不看 `kind`，缺陷就以另一种形式回来。因此这里做一次判定、
 * 全局生效：闸门一落，后续所有插件调用**本地失败**，不再跨 IPC。
 *
 * 【为什么不可逆】终态不会自行恢复 —— 恢复路径是"用户关窗重开"，那会重建 bootstrap
 * 与全新的 `PluginHostBridge`（见 `pluginHost.ts` 的 `getPluginHost`）。让闸门可逆只会
 * 制造一个"看起来还能用、实际每次都在失败"的界面。
 */

/**
 * 后端协议码，与 `backend/hifishifter-ara-ipc/src/lib.rs` 的 `error_code` 一一对应。
 *
 * 【为什么在前端也写一份】它是**协议承诺**（只增不改）。前端按码分支（例如区分
 * "会话已结束"与"宿主从未绑定 ARA"），不解析错误文本 —— 与 `pluginHost.ts` 里
 * "调用方按 kind 决策，不解析错误文本"是同一条纪律的延伸。
 */
export const ERROR_CODE = {
    /** 编辑文档已关闭。 */
    DOCUMENT_CLOSED: 1,
    /** FX 处理器（组件）已销毁。 */
    FX_PROCESSOR_CLOSED: 2,
    /** 编辑器视图不存在或身份未知。 */
    VIEW_CLOSED_OR_UNKNOWN: 3,
    /** 视图身份与会话不匹配。 */
    VIEW_IDENTITY_MISMATCH: 4,
    /** 请求排队期间编辑路由已改变。 */
    ROUTE_CHANGED_WHILE_QUEUED: 5,
    /** 编辑路由属于另一份文档。 */
    ROUTE_OTHER_DOCUMENT: 6,
    /** 编辑租约已被撤销。 */
    LEASE_REVOKED: 7,
    /** 插件实例已关闭。 */
    INSTANCE_CLOSED: 8,
    /** 宿主从未把本实例绑定到 ARA 文档。 */
    ARA_NOT_BOUND: 9,
} as const;

export type SessionTerminal = {
    /** 协议码；旧后端不带时为 `null`（此时只能按文本展示，不能按码分支）。 */
    code: number | null;
    /** 后端原始错误串（仅用于展示与日志）。 */
    message: string;
};

let terminal: SessionTerminal | null = null;
const listeners = new Set<() => void>();

function notify(): void {
    for (const listener of Array.from(listeners)) {
        try {
            listener();
        } catch {
            // 订阅者自身出错不得影响闸门状态（与 `pluginHost` 的事件派发同一纪律）。
        }
    }
}

/**
 * 标记会话进入终态。幂等：第一次之后不再改变（终态不会"更终态"）。
 *
 * 【为什么要保留第一条】第一条终态携带的**协议码**决定界面给出哪一句指引
 * （"关窗重开" vs "宿主尚未接入 ARA"）。后到的终态只会是同一个会话的连带失败，
 * 覆盖它会让指引漂移。
 */
export function markBackendTerminal(message: string, code?: number | null): void {
    if (terminal) return;
    terminal = { code: typeof code === "number" ? code : null, message };
    notify();
}

/** 会话是否已进入终态。 */
export function isBackendTerminal(): boolean {
    return terminal !== null;
}

/** 终态详情；未进入终态时为 `null`。 */
export function sessionTerminal(): SessionTerminal | null {
    return terminal;
}

/** 订阅闸门变化（`useSyncExternalStore` 的 `subscribe`）。 */
export function subscribeSessionGate(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/**
 * `useSyncExternalStore` 的 `getSnapshot`。
 *
 * 【为什么返回同一个对象】`terminal` 一旦置位就不再替换，因此快照引用天然稳定；
 * 每次新建对象会让 React 认为"一直在变"并无限重渲染。
 */
export function getSessionGateSnapshot(): SessionTerminal | null {
    return terminal;
}

/** 仅供测试重置模块级状态。 */
export function resetSessionGateForTests(): void {
    terminal = null;
    listeners.clear();
}
