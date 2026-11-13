/**
 * 「另存为」失败的提示通道。
 *
 * 【为什么需要一条总线】三种格式的菜单子项、三个可绑定快捷键、`project.saveAs`
 * 与「未保存 → 另存为」共用同一个 thunk。失败必须变成一句可读的提示，而唯一拥有
 * 状态栏提示位的是 `App`；让每个调用点各自 `catch` 会把同一段映射抄四份，还会
 * 漏掉键盘通道（那里没有组件可以显示任何东西）。
 *
 * 因此 thunk 只发布**错误码**，措辞留在 `App`（那里有 `tf` 与状态栏）。
 *
 * 【为什么不是 Redux 切片】它不描述应用状态，只是一次性事件。写进 store 反而要
 * 逼消费方在渲染后清空，多一层时序；订阅集合与同窗口轻量事件总线同范式。
 */

/** 失败码（后端错误码原样透传，例如 `format_extension_mismatch`）。 */
export type SaveProjectAsFailureCode = string;

type Listener = (code: SaveProjectAsFailureCode) => void;

const listeners = new Set<Listener>();

/**
 * 发布一次失败。
 *
 * 没有订阅者时静默丢弃：单元测试里的纯 thunk 调用就属于这种情况，不该因为它
 * 没有挂状态栏而抛错。
 */
export function publishSaveProjectAsFailure(code: SaveProjectAsFailureCode): void {
    for (const listener of listeners) {
        try {
            listener(code);
        } catch {
            // 单个订阅者异常不影响其余订阅者（与其它总线同策略）。
        }
    }
}

/** 订阅失败；返回退订函数。 */
export function subscribeSaveProjectAsFailure(listener: Listener): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 仅测试用：清空订阅者，避免用例之间互相污染。 */
export function resetSaveProjectAsFailureForTests(): void {
    listeners.clear();
}
