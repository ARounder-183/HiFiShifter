/**
 * Select 组件滚轮切换辅助。
 *
 * 将鼠标滚轮映射为上一个/下一个选项，供 Radix Themes Select.Trigger 使用。
 */

/** 滚轮事件里本模块真正依赖的那部分；原生与合成事件都满足。 */
export type WheelLikeEvent = Pick<
    globalThis.WheelEvent,
    "deltaY" | "preventDefault" | "stopPropagation"
>;

export function applySelectWheelChange<T extends string>(args: {
    /**
     * 滚轮事件。
     *
     * 【为什么是结构化类型而不是 `React.WheelEvent`】本函数只用到 `deltaY` 与两个
     * 阻止方法，而它们**原生事件与 React 合成事件都有** —— 调用方两种也都有：
     * 工具栏里的下拉走 React `onWheel`，对话框里的下拉走 `useNonPassiveWheel` 的原生
     * 监听。声明成 `React.WheelEvent` 会把原生调用方挡在门外，逼它们写
     * `as unknown as`，而那层强转正是"按合成事件取值"这类错误的藏身处。
     */
    event: WheelLikeEvent;
    currentValue: T;
    options: readonly T[];
    onChange: (next: T) => void;
}) {
    const { event, currentValue, options, onChange } = args;
    if (!Array.isArray(options) || options.length <= 1) return;
    if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;

    event.preventDefault();
    event.stopPropagation();

    const currentIndex = options.findIndex((opt) => opt === currentValue);
    if (currentIndex < 0) return;

    const direction = event.deltaY < 0 ? -1 : 1;
    const nextIndex = currentIndex + direction;
    if (nextIndex < 0 || nextIndex >= options.length) return;

    const nextValue = options[nextIndex];
    if (nextValue !== currentValue) {
        onChange(nextValue);
    }
}
