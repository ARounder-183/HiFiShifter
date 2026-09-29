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
import { radixSizeFor, useDensity, type AppDensity } from "./density";
import { useNonPassiveWheel } from "../utils/useNonPassiveWheel";
import { useFrameCommitter, useWheelStepAccumulator } from "./useFrameCommit";

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
    /**
     * 密度覆盖。默认继承容器（对话框 = `form`，工具条 = `compact`）。
     * 只给**不是工具条**的紧凑表面用（快速搜索排序行、轨道头算法下拉）。
     */
    density?: AppDensity;
    /**
     * 触发器最小宽度（px）。
     *
     * 【为什么需要】有些下拉的内容是**不定长文本**（设备名、应用名、路径），
     * 不设下限时触发器会塌缩成"当前选项那么宽"，切换选项时宽度还会跳动。
     * 旧写法用内联 `style={{minWidth: 260}}`；迁移到原语时原语没有这一维，
     * 于是 `RecordingSettingsDialog` 的设备/回环/应用三个下拉失去了下限。
     */
    minWidth?: number;
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
    density,
    minWidth,
    ariaLabel,
    className,
}: AppSelectProps) {
    const size = radixSizeFor(useDensity(density));
    /*
     * 滚轮换项只关心"值"的有序列表，分隔线要滤掉 —— 否则滚轮会停在分隔线上。
     * 同一份列表也用来判定"这个值是不是本组件的选项"，见下方的守卫。
     */
    const optionValues = options.flatMap((entry) => (isSeparator(entry) ? [] : [entry.value]));

    /*
     * 滚轮提交走帧合并 + 手势累积（见 `useFrameCommit.ts` 的长注释）：
     * 一次滚轮手势最多每帧提交一次，且值一次走到位。这是必须的 —— 有些下拉的
     * `onValueChange` 会 dispatch **同步 Tauri 命令**，逐格提交会阻塞 UI 线程
     * 直到窗口未响应。
     */
    const committer = useFrameCommitter(onValueChange);
    const accumulator = useWheelStepAccumulator<string>();

    const setWheelTarget = useNonPassiveWheel<HTMLButtonElement>((event) => {
        if (disabled) return;
        if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
        if (optionValues.length <= 1) return;
        const at = optionValues.indexOf(value);
        if (at < 0) return;
        // 先接管这次滚轮（阻止祖先滚动），再算值 —— 算不出下一格时也不该滚动容器。
        event.preventDefault();
        event.stopPropagation();
        const next = accumulator.advance(value, (base) => {
            const from = optionValues.indexOf(base);
            if (from < 0) return base;
            const direction = event.deltaY < 0 ? -1 : 1;
            return optionValues[from + direction] ?? base;
        });
        if (next !== value) committer.schedule(next);
    });

    return (
        <Select.Root
            value={value}
            /*
             * 手动选择（点菜单项）不经过滚轮，直接提交；无需节流。
             *
             * 【必须守卫：只接受本组件自己的选项值】Radix 会为表单兼容渲染一个隐藏的
             * 原生 `<select>`，并在受控值变化时把它镜像进去（`SelectBubbleInput`：
             * `setValue.call(select, value)` 后派发一个冒泡的 `change`）。受控值一旦
             * **不在 `<option>` 里**，浏览器会把 `select.value` 归成 `""`，Radix 就把这个
             * `""` 原样转发给 `onValueChange` —— 于是调用方会收到一次它从未请求过的
             * 空字符串变更。
             *
             * 真实后果：`RenderCacheDialog` 的两个数字框旁边曾有预设下拉，用 "custom"
             * 表示"当前值不是预设"。滚轮把 4096 调成 4097 时，下拉的受控值变成 "custom"
             * ⇒ 触发一次 `onValueChange("")` ⇒ 调用方 `Number("") === 0` ⇒ 占用上限被
             * 静默改成"不限"。用户看到的是"滚轮直接跳到 0"。
             *
             * 受控下拉**只可能**报告它自己的选项，因此这里把不属于选项集的值一律丢掉。
             */
            onValueChange={(next) => {
                if (!optionValues.includes(next)) return;
                onValueChange(next);
            }}
            disabled={disabled}
            size={size}
        >
            <Select.Trigger
                ref={setWheelTarget}
                aria-label={ariaLabel}
                className={cx(fullWidth && "w-full", className)}
                style={minWidth === undefined ? undefined : { minWidth }}
            />
            <Select.Content>
                {options.map((entry, index) =>
                    isSeparator(entry) ? (
                        // 分隔线用索引做 key：它没有稳定标识，且位置固定
                        <Select.Separator key={`sep-${index}`} />
                    ) : (
                        <Select.Item
                            key={entry.value}
                            value={entry.value}
                            disabled={entry.disabled}
                        >
                            {entry.label}
                        </Select.Item>
                    ),
                )}
            </Select.Content>
        </Select.Root>
    );
}
