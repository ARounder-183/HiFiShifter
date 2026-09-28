/**
 * 数字输入原语 —— 数值型字段的**唯一**实现。
 *
 * 【它内建了什么】审查发现 34 个数字输入里只有 6 个支持滚轮、其中 12 个支持
 * 精细调整修饰键；另有 2 个各写一份的本地 `NumberField` 完全没有这两样。
 * 逐个补 `onWheel` 治标不治本 —— 下一处新字段还是会忘。
 *
 * 本组件把三件事内建，调用方无法遗漏：
 *   1. **滚轮步进**（`useNonPassiveWheel`，因此不会连带滚动祖先容器）；
 *   2. **精细调整修饰键**（走 `stepPolicy` 的粗/精二元组）；
 *   3. **步长按单位语义**（`unit="bpm"` 而不是 `step={1}`）。
 *
 * 【提交语义】打字过程中只更新本地草稿，**blur 或 Enter 才提交**；
 * 滚轮每格是离散动作，立即提交。这与设置类表单的心智一致：改到一半不会
 * 每敲一个字符就往 Redux / 后端写一次。
 */
import { TextField } from "@radix-ui/themes";
import { useState } from "react";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { stepFor, stepValue, type StepUnit } from "./stepPolicy";
import { useFineAdjustModifier } from "./useFineAdjustModifier";
import { useNonPassiveWheel } from "../utils/useNonPassiveWheel";

export interface AppNumberFieldProps {
    value: number;
    /** 提交回调（blur / Enter / 滚轮每格）。 */
    onCommit: (next: number) => void;
    /**
     * 单位语义，决定步长与小数位。
     * 例：`bpm` → 粗调 1 / 精调 0.1；`percent` → 5 / 1。
     */
    unit: StepUnit;
    min: number;
    max: number;
    disabled?: boolean;
    /** 宽度（px）。默认 72，与既有数字字段一致。 */
    width?: number;
    /** 字段后缀（如 `px`、`ms`）。 */
    suffix?: ReactNode;
    /** 无障碍名称；缺省时用 `aria-label` 传入。 */
    ariaLabel?: string;
    className?: string;
}

/**
 * 数字字段。
 *
 * @example
 * <AppNumberField
 *     value={snap.gridMinSpacingPx}
 *     unit="pixels"
 *     min={2}
 *     max={200}
 *     suffix="px"
 *     ariaLabel={t("snap_grid_min_spacing_px")}
 *     onCommit={(v) => patch({ gridMinSpacingPx: v })}
 * />
 */
export function AppNumberField({
    value,
    onCommit,
    unit,
    min,
    max,
    disabled = false,
    width = 72,
    suffix,
    ariaLabel,
    className,
}: AppNumberFieldProps) {
    const spec = stepFor(unit);
    const isFine = useFineAdjustModifier();

    /*
     * 草稿状态。
     *
     * 【为什么记一个 source，而不是用 useEffect 同步】React 的官方做法是
     * "渲染期调整状态"：外部值与我们播种时的值不同，就地重新播种。用 effect
     * 同步会触发级联渲染（React Compiler 的 set-state-in-effect 规则直接报错），
     * 而且用户正在打字时上游刷新会与草稿打架。
     *
     * `source` 表示"这份 text 是从哪个外部值派生的"：打字时它不变（外部值也
     * 没变），提交后我们乐观地把它推进到新值，于是紧随其后的 props 更新不会
     * 覆盖草稿。
     */
    const [draft, setDraft] = useState(() => ({
        source: value,
        text: format(value, spec.decimals),
    }));
    if (draft.source !== value) {
        setDraft({ source: value, text: format(value, spec.decimals) });
    }
    const text = draft.text;

    const commitText = () => {
        const parsed = Number(text);
        if (!Number.isFinite(parsed)) {
            // 输入非法时回退显示，不提交 —— 静默写入 0 会悄悄改掉用户的数据。
            setDraft({ source: value, text: format(value, spec.decimals) });
            return;
        }
        const clamped = Math.min(max, Math.max(min, parsed));
        setDraft({ source: clamped, text: format(clamped, spec.decimals) });
        if (clamped !== value) onCommit(clamped);
    };

    /*
     * 滚轮：非被动监听，因此 `preventDefault` 真正生效（React 的合成 `onWheel`
     * 是 passive，阻止不了祖先滚动）。每格立即提交。
     */
    const setWheelTarget = useNonPassiveWheel<HTMLDivElement>((event) => {
        if (disabled) return;
        if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
        event.preventDefault();
        const direction: 1 | -1 = event.deltaY < 0 ? 1 : -1;
        const next = stepValue({ value, direction, unit, fine: isFine(event), min, max });
        if (next !== value) {
            setDraft({ source: next, text: format(next, spec.decimals) });
            onCommit(next);
        }
    });

    return (
        <div
            ref={setWheelTarget}
            className={cx("flex items-center gap-1", className)}
            style={{ width: suffix ? undefined : width }}
        >
            <TextField.Root
                type="number"
                size="2"
                value={text}
                min={min}
                max={max}
                step={spec.coarse}
                disabled={disabled}
                aria-label={ariaLabel}
                onChange={(event) => setDraft({ source: draft.source, text: event.target.value })}
                onBlur={commitText}
                onKeyDown={(event) => {
                    if (event.key !== "Enter") return;
                    // Enter 提交本字段；不 `preventDefault` 的话会继续冒泡触发
                    // 对话框表单的默认动作（AppDialog 已在 submitter 层拦一道，
                    // 这里再拦一道保证"改完按 Enter 不会顺手关窗"）。
                    event.preventDefault();
                    commitText();
                }}
                style={suffix ? { width } : { width: "100%" }}
            />
            {suffix ? <span className="hs-type-caption shrink-0">{suffix}</span> : null}
        </div>
    );
}

/** 按小数位格式化，避免 `0.30000000000000004` 出现在输入框里。 */
function format(value: number, decimals: number): string {
    const factor = 10 ** decimals;
    return String(Math.round(value * factor) / factor);
}
