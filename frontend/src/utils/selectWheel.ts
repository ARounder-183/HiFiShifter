/**
 * Select 组件滚轮切换辅助。
 *
 * 将鼠标滚轮映射为上一个/下一个选项，供 Radix Themes Select.Trigger 使用。
 */

import type { WheelEvent as ReactWheelEvent } from "react";

export function applySelectWheelChange<T extends string>(args: {
    event: ReactWheelEvent<HTMLElement>;
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
