/**
 * 时间轴右键框选的**纯合并规则**。
 *
 * 框选手势本体由内核宿主实现（`kernel/interaction/boxSelection.ts`，DOM 版
 * hook 已随内核唯一路径删除）；这里只保留"框选结果如何并入现有多选"的
 * 规则函数，供时间轴面板在内核回调里复用。
 *
 * 注意：右键框选**刻意不做吸附、也不显示吸附高亮** —— 它是 Clip 框选
 * 手势，不与时间轴网格/候选直接交互（吸附仅服务于移动/编辑类拖拽）。
 */

export function computeTimelineRectSelection(params: {
    selectionBeforeDrag: string[];
    selectedInRect: string[];
    primaryModifierPressedAtStart: boolean;
}): string[] {
    const { selectionBeforeDrag, selectedInRect, primaryModifierPressedAtStart } = params;
    if (!primaryModifierPressedAtStart) {
        return selectedInRect;
    }
    const beforeSet = new Set(selectionBeforeDrag);
    const inRectSet = new Set(selectedInRect);
    const kept = selectionBeforeDrag.filter((id) => !inRectSet.has(id));
    const appended = selectedInRect.filter((id) => !beforeSet.has(id));
    return [...kept, ...appended];
}
