/**
 * selectionEditInFlight.ts — 参数编辑器「选区编辑」在途标记（模块级单例）。
 *
 * 参数线上移/下移（选择范围）的执行体在 PianoRollPanel（hifi:editOp 消费
 * 端），而长按重复的节拍器在 App（beginHoldRepeat）。两端靠本模块共享
 * 「上一拍是否仍在途」：App 的 fire 在在途期间跳过本拍（不派发事件），
 * 消费端兜底再次检查 —— 与 App.tsx 音频块范围平移的 paramShiftBusyRef
 * 同构，避免 50ms 节奏下 IPC 请求与历史检查点堆积。
 */

let inFlight = false;

/**
 * 标记一次选区编辑开始。已在途时返回 false（调用方跳过本拍）。
 * 消费端在进入异步工作前同步调用，保证同一事件循环内的后续拍可见。
 */
export function beginSelectionParamEdit(): boolean {
    if (inFlight) return false;
    inFlight = true;
    return true;
}

/** 标记选区编辑结束（消费端 finally 中调用）。 */
export function endSelectionParamEdit(): void {
    inFlight = false;
}

/** 是否有选区编辑仍在途（App 长按 fire 的跳拍依据）。 */
export function isSelectionParamEditInFlight(): boolean {
    return inFlight;
}
