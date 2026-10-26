/**
 * 编辑器路由（视图 ↔ 处理器）的状态与通知。
 *
 * 【为什么单独一个模块】这条状态的判据只有一个来源（后端 bootstrap 的 `editorRoute`），
 * 而读它的地方有两处：状态栏提示（这里的事件订阅）与 Capture 轮询的降频
 * （`features/capture/captureStore.ts` 直接读 bootstrap）。把判据留在
 * `hostDetect` 的 bootstrap 里、把**措辞**留在这里，两边就不会各自发明一套。
 *
 * 【为什么不是一个 Redux 切片】它描述的是"这一次会话的宿主做到了哪一步"，
 * 与界面选择无关；放进 store 只会让它在换工程时被重置，而"这个宿主不转发 VST3 消息"
 * 不会因为换工程而改变。
 */
import { pluginBootstrap } from "./hostDetect";
import { getPluginHost } from "./pluginHost";

/** 后端 `editor_route_notice` 事件的载荷。 */
export interface EditorRouteNotice {
    /** `"unbound"`（当前只有这一种会发通知）。 */
    state: string;
    /** 后端给出的可读原因（无活实例 / 多个活实例 …）。 */
    reason: string;
}

/**
 * 编辑器是否**确定**连不上处理器。
 *
 * 【为什么只认 `unbound`】`waiting-host` 是正常的启动瞬间（宿主的握手还在路上），
 * 把它当成故障会让每次开窗都闪一条警告。
 */
export function editorRouteIsUnbound(): boolean {
    return pluginBootstrap()?.editorRoute?.state === "unbound";
}

/**
 * 订阅"路由确定失败"的通知；返回退订函数。
 *
 * 【为什么与让路通知同一个形状】两者都是"后端发现了一件用户需要知道、但不由任何一次
 * `invoke` 触发的事"。共用同一套排队/退订语义，卸载时不会漏掉一次迟到的订阅。
 */
export function subscribeEditorRouteNotice(
    listener: (notice: EditorRouteNotice) => void,
): () => void {
    let unsubscribe: (() => void) | null = null;
    let disposed = false;
    void (async () => {
        try {
            const host = getPluginHost();
            if (!host) return;
            const off = await host.listen<EditorRouteNotice>("editor_route_notice", (event) => {
                const payload = event.payload;
                if (!payload || typeof payload.state !== "string") return;
                if (payload.state !== "unbound") return;
                listener({
                    state: payload.state,
                    reason: typeof payload.reason === "string" ? payload.reason : "",
                });
            });
            if (disposed) off();
            else unsubscribe = off;
        } catch {
            // 独立 App / 消息口不可用：没有通知可订阅，也不需要提示。
        }
    })();
    return () => {
        disposed = true;
        unsubscribe?.();
        unsubscribe = null;
    };
}
