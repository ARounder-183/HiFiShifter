/**
 * 内核宿主的 DOM 写入去重工具。
 *
 * 【为什么单独成文件】timelineKernelHost 已在超大文件预算内（见
 * tests/fileSizeBudget.test.ts），这两块与宿主装配无关的纯工具先搬出来；
 * 与 `createPlayheadElementWriter` 同一契约，这里按值类型泛化。
 */

/**
 * 判断 DOM 写入值是否需要提交（值去重）。
 *
 * 【为什么用取反写法】去重变量以 `NaN` 表示「从未写入」，而 `NaN > eps` 恒为
 * **false**：直接写 `Math.abs(next - prev) > eps` 会把**首次**写入判定为
 * 「无需写入」而跳过，表现为标尺 / 播放头 / 细节层在第一次交互前完全不动
 * （后续也永远追不上，因为 `last` 一直是 NaN）。取反写法让 NaN 走「需要写入」分支。
 *
 * @param next 本次要写入的值。
 * @param previous 上一次写入的值（NaN = 从未写入）。
 * @param epsilon 视为「无变化」的最大差值。
 * @returns 需要写入时为 true。
 */
export function shouldWrite(next: number, previous: number, epsilon = 0.01): boolean {
    return !(Math.abs(next - previous) <= epsilon);
}

/**
 * 「元素身份 + 值」双键去重的 DOM 写入器。
 *
 * 【为什么不能只按值去重】被同步的图层 / 滚动容器可能在宿主创建之后被 React
 * 重建（面板停靠重排会搬动 DOM、标尺随视图出现）。若元素被重建而值未变，纯值
 * 去重会判定「不必写」—— 新元素就停在静态位置（例如吸附高亮整层没有 transform、
 * 轨道头容器 scrollTop 仍是 0），直到值下一次变化为止。键加入元素身份后，元素一换
 * 就无条件写一次。
 *
 * @param equals 值的相等判定（缺省 `Object.is`）。
 * @returns `(slot, element, value, apply)`；element 为 null 时丢弃该槽位，
 *          元素回来时无条件写一次。
 */
export function createElementValueWriter<T>(
    equals: (a: T, b: T) => boolean = Object.is,
): (
    slot: string,
    element: HTMLElement | null,
    value: T,
    apply: (target: HTMLElement, value: T) => void,
) => void {
    const slots = new Map<string, { element: HTMLElement; value: T }>();
    return (slot, element, value, apply) => {
        if (element === null) {
            slots.delete(slot);
            return;
        }
        const previous = slots.get(slot);
        if (
            previous !== undefined &&
            previous.element === element &&
            equals(previous.value, value)
        ) {
            return;
        }
        slots.set(slot, { element, value });
        apply(element, value);
    };
}
