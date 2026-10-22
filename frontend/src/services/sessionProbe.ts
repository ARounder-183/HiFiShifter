/**
 * 挂载前的会话探活：后端编辑会话是否**已经**处于终态。
 *
 * 【为什么必须在挂载 App 之前问一次】会话在启动前就已经终结时，App 的初始化扇出会
 * **并发**发出十几条命令，每条都失败、每条都进日志 —— 实测 Cubase 上那一秒有 15 条
 * `Invoke failed`。闸门拦不住它们（IPC 往返约 200ms，大部分命令在第一个响应回来前就
 * 发出去了），只有在扇出**之前**问一次才拦得住。
 *
 * 【为什么不用 `ping`】`ping` 只回 `link.owner().is_ok()`，不检查 owner 是否已关闭、
 * 也不报告形态；它无法区分"会话正常"与"组件已销毁但路由还在"。
 *
 * 【为什么不再用 `get_runtime_info`】它会走 `editor_session()` + actor 队列。重开
 * ARA 工程时，会话创建（模型加载 / actor 创建）可能远超 30 s IPC 超时，于是探活本身
 * 变成用户看到的 `Backend invoke failed: plugin:get_runtime_info`。
 * `plugin_health` 是内联命令：只读 owner 的原子/弱引用，不创建会话、不排队。
 */
import { hostKind, isPluginMode } from "./hostCapabilities";
import { invoke, invokeErrorKind } from "./invoke";

/**
 * 这个错误是否意味着"整页替换成会话不可用"。
 *
 * 【为什么瞬时失败不算】"编辑器尚未连上处理器"这类是**暂时**的：组件创建与绑定之间必然
 * 经历它，正常挂载后 App 自己会重试（见 `editor/mod.rs` 的分类测试）。
 *
 * 【为什么 Capture 形态不看终态】`ARA_NOT_BOUND` 对 ARA 形态是终态（宿主从不走 ARA
 * 路径，关窗重开一万次也一样），但 Capture 形态**本来就不需要 ARA 文档**：它的编辑会话
 * 由捕获存储供时间线（`EditorSession::SessionBacking`）。把这条码当终态会让所有不支持
 * ARA 的宿主（Cubase 也在内）永远停在"会话不可用"整页 —— 而那正是 Capture 存在的理由。
 * 其它终态（组件已销毁、实例已关闭）在 Capture 下仍由闸门正常反映。
 */
export function sessionUnavailableFrom(error: unknown): boolean {
    if (invokeErrorKind(error) !== "terminal") return false;
    return hostKind() !== "capture";
}

/** 探活结论。`unavailable` ⇒ 渲染整页提示而不是 App。 */
export async function probeSession(): Promise<"ready" | "unavailable"> {
    if (!isPluginMode()) return "ready";
    try {
        await invoke("plugin_health");
        return "ready";
    } catch (error) {
        return sessionUnavailableFrom(error) ? "unavailable" : "ready";
    }
}
