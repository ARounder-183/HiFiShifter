/**
 * 下拉选择原语 —— ComboBox 的**唯一**实现。
 *
 * 【它内建了什么】审查发现 65 个 Radix `Select.Root` 里只有 36 个接了滚轮
 * （55%），另外 29 个（录音设置 7 个、记事本设置 10 个、渲染缓存 4 个、
 * 吸附设置 3 个…）**完全没接**。原因是滚轮不是 `Select` 自带能力，而要每个
 * 调用点自己写 `onWheel` 并自己找 options 列表 —— 忘掉是常态。
 *
 * 本组件把滚轮内建：调用方给出 `options`，滚轮换项自动生效，且用
 * `useNonPassiveWheel` 保证不会连带滚动祖先容器。
 *
 * 【为什么 options 用数组而不是 children】滚轮换项需要知道"下一个选项是什么"，
 * 而 children 是 React 节点树，拿不到有序的值列表。`options` 同时承担渲染与
 * 换项两件事，因此二者不可能不一致。
 */
import { Select } from "@radix-ui/themes";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { useNonPassiveWheel } from "../utils/useNonPassiveWheel";
import { applySelectWheelChange } from "../utils/selectWheel";

export interface AppSelectItem {
    value: string;
    label: ReactNode;
    disabled?: boolean;
}

/** 选项条目：普通项或分组分隔线。 */
export type AppSelectEntry = AppSelectItem | { separator: true };

export interface AppSelectProps {
    value: string;
    onValueChange: (next: string) => void;
    /** 条目列表（可含 `{ separator: true }` 作分组）。 */
    options: readonly AppSelectEntry[];
    disabled?: boolean;
    /** 触发器宽度铺满容器（表单里默认如此）。 */
    fullWidth?: boolean;
    ariaLabel?: string;
    className?: string;
}

function isSeparator(entry: AppSelectEntry): entry is { separator: true } {
    return "separator" in entry;
}

/**
 * 下拉选择。
 *
 * @example
 * <AppSelect
 *     value={session.grid}
 *     options={GRID_SIZES.map((grid) => ({ value: grid, label: grid }))}
 *     onValueChange={setGrid}
 * />
 */
export function AppSelect({
    value,
    onValueChange,
    options,
    disabled = false,
    fullWidth = true,
    ariaLabel,
    className,
}: AppSelectProps) {
    /*
     * 滚轮换项只关心"值"的有序列表，分隔线要滤掉 —— 否则滚轮会停在分隔线上。
     */
    const wheelOptions = options.flatMap((entry) => (isSeparator(entry) ? [] : [entry.value]));

    const setWheelTarget = useNonPassiveWheel<HTMLButtonElement>((event) => {
        if (disabled) return;
        applySelectWheelChange({
            event,
            currentValue: value,
            options: wheelOptions,
            onChange: onValueChange,
        });
    });

    return (
        <Select.Root value={value} onValueChange={onValueChange} disabled={disabled}>
            <Select.Trigger
                ref={setWheelTarget}
                aria-label={ariaLabel}
                className={cx(fullWidth && "w-full", className)}
            />
            <Select.Content>
                {options.map((entry, index) =>
                    isSeparator(entry) ? (
                        // 分隔线用索引做 key：它没有稳定标识，且位置固定
                        <Select.Separator key={`sep-${index}`} />
                    ) : (
                        <Select.Item key={entry.value} value={entry.value} disabled={entry.disabled}>
                            {entry.label}
                        </Select.Item>
                    ),
                )}
            </Select.Content>
        </Select.Root>
    );
}
