/**
 * 参数编辑器「数据加载中」状态总线（轻量订阅，供状态栏读取）。
 *
 * # 为什么不用 React Context
 * 这个状态**只有状态栏一处消费**，却曾经放在包裹整个 `AppInner` 的 Context 里。
 * `AppInner` 的第一行就 `useContext` 它，于是这个 boolean 每翻转一次，整棵应用树
 * （时间轴、参数编辑器、全部面板）就重渲染一次。
 *
 * 而参数取数极其频繁 —— 水平缩放、滚动、每一笔编辑提交都会触发（见
 * `usePianoRollData` 的取数 effect）。结果是编辑过程中持续的全树重渲染，
 * 参数编辑器的笔画因此被打断、跳变。
 *
 * 换成"外部 store + `useSyncExternalStore`"后，只有真正订阅它的那个小组件
 * （`ParamDataLoadingChip`）重渲染，其余部分完全不受影响。这与仓库里
 * `pianoRollSelectionBus` 是同一范式。
 *
 * # 为什么按实例记账
 * 参数编辑器**可以多开**（停靠系统支持同一面板的多个窗体）。若共用一个 boolean，
 * 一个实例卸载时上报 `false` 会把另一个仍在取数的实例的状态抹掉。因此按发布者 id
 * 记账，对外只暴露"有没有任何一个在取数"。
 */

/** 发布者 id → 是否在取数。id 用窗体 id（见 `dockFormId`）。 */
const loadingById = new Map<string, boolean>();
/** 对外快照：`loadingById` 中是否有任一为真。 */
let anyLoading = false;
const listeners = new Set<() => void>();

function recompute(): void {
    let next = false;
    for (const value of loadingById.values()) {
        if (value) {
            next = true;
            break;
        }
    }
    // **幂等**：对外状态未变化时不通知订阅者。取数成功/失败/取消各条路径都会走到
    // 这里，多实例也会各自上报；不去重就会让订阅者收到大量无意义的通知。
    if (next === anyLoading) return;
    anyLoading = next;
    for (const listener of listeners) {
        try {
            listener();
        } catch {
            // 忽略订阅者异常（与 pianoRollSelectionBus 同策略）
        }
    }
}

/**
 * 发布某个面板实例的取数状态。
 *
 * `publisherId` 必须跨渲染稳定（用窗体 id）；同一 id 重复发布同值会被短路。
 */
export function setPianoRollLoading(publisherId: string, loading: boolean): void {
    if (loadingById.get(publisherId) === loading) return;
    loadingById.set(publisherId, loading);
    recompute();
}

/**
 * 注销某个面板实例（卸载时调用）。
 *
 * 必须调用：否则该实例的最后一次状态会永远粘在记账表里，状态栏再也不会收起。
 */
export function clearPianoRollLoading(publisherId: string): void {
    if (!loadingById.delete(publisherId)) return;
    recompute();
}

/** 读取当前状态（`useSyncExternalStore` 的 snapshot）。 */
export function getPianoRollLoading(): boolean {
    return anyLoading;
}

/** 订阅状态变化；返回取消订阅函数。 */
export function subscribePianoRollLoading(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 仅测试用：复位内部状态与订阅者。 */
export function resetPianoRollLoadingForTests(): void {
    loadingById.clear();
    anyLoading = false;
    listeners.clear();
}
