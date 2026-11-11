// 参数刷新显示策略：同一作用域内刷新时保留已提交曲线，只有切作用域才清窗。
/**
 * 强制重新取数不等于必须先清空画面。
 *
 * 时间线刷新（Clip 编辑 / 撤销 / 分析回填）会让 `paramsEpoch` 前进，但参数窗口的
 * 作用域（root + param）没有变化。此时清空 `paramView` 会让曲线消失到下一次
 * IPC 回包；参数块也会因此失去采样值。保留旧曲线，等新数据原子替换即可。
 */
export function shouldClearParamRefresh(scopeChanged: boolean): boolean {
    return scopeChanged;
}

/**
 * Same-window empty responses are usually a transient refresh; keep the last
 * visible curve. A different window's empty response is authoritative and must
 * replace the old window, otherwise scrolling to an empty region leaves the
 * previous curve painted on the new coordinates.
 */
export function shouldAcceptParamPayload(args: {
    readonly hasValues: boolean;
    readonly requestedKey: string;
    readonly previousKey: string | null;
    readonly previousHasValues: boolean;
}): boolean {
    if (args.hasValues) return true;
    if (!args.previousHasValues || args.previousKey === null) return true;
    return args.previousKey !== args.requestedKey;
}
