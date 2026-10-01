/**
 * 滑块原语 —— 滑块的**唯一**实现。
 *
 * 【它内建了什么】
 *   1. **滚轮步进**：20 个滑块里有 4 个（全是 Radix `Slider`）完全不能滚轮调值；
 *   2. **精细调整修饰键**：5 个 Radix `Slider` 全部不支持；
 *   3. **统一外观**：15 个裸 `range` 里有 14 个没带 `.qt-range`，渲染成浏览器
 *      默认样式 —— 与应用其余部分格格不入。
 *
 * 统一走 Radix `Slider`（自带主题适配与键盘支持），因此不再需要 `.qt-range`
 * 这个为裸 `range` 准备的类。已有的裸 `range` 迁过来即可，不必逐个补 class。
 *
 * 【为什么不用裸 `range`】Radix `Slider` 已支持方向键/Home/End 与正确的
 * 无障碍角色；裸 `range` 的样式与键盘行为都要自己补，正是当初 14 处漏掉
 * `.qt-range` 的原因。
 */
import { Slider } from "@radix-ui/themes";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { stepValue, type StepUnit } from "./stepPolicy";
import { useFineAdjustModifier } from "./useFineAdjustModifier";
import { useNonPassiveWheel } from "../utils/useNonPassiveWheel";
import { useFrameCommitter, useWheelStepAccumulator } from "./useFrameCommit";
import { radixSizeFor, useDensity, type AppDensity } from "./density";

/**
 * 滑块头的**装饰盒高**（CSS px）——滑块头本体加上它的焦点环。
 *
 * 由 Radix 的两条公式推出（`@radix-ui/themes/src/components/slider.css`）：
 *   `--slider-track-size`：size 1 = `space-2 × 0.75` = 6px，size 2 = `space-2` = 8px
 *   `--slider-thumb-size` = track + `space-1`  → size 1 = 10px，size 2 = 12px
 *   可见滑块头（`::after`）= thumb + 2 × (0.25 × track) → size 1 = 13px，size 2 = 16px
 * 即 `thumb + 0.5 × track`。盒高必须 ≥ 可见滑块头，否则滑块头上下会被裁平。
 *
 * 盒高同时也是**行高**（同一行里的读数文字也是 16px），因此不能再往上加：
 * 横向的余量走 `padding-inline`，纵向的焦点环走内嵌（见 `index.css` 的
 * `.hs-slider-box`）。
 */
const DECORATION_BOX_PX: Record<"1" | "2", number> = { "1": 13, "2": 16 };

export interface AppSliderProps {
    value: number;
    /** 拖动/滚轮过程中的实时回调（轻量，用于预览）。 */
    onChange: (next: number) => void;
    /**
     * 松手 / 滚轮一格后的提交回调（用于落盘、checkpoint、IPC）。
     * 省略时只在 `onChange` 上报。
     */
    onCommit?: (next: number) => void;
    /**
     * 单位语义，决定滚轮步长。
     * 注意：拖动步长固定为 1（与 `Slider` 的 `step` 一致），滚轮才用粗/精步长
     * —— 原实现里两者不一致（`step={1}` 但滚轮滚 5），手感割裂。
     */
    unit: StepUnit;
    min: number;
    max: number;
    disabled?: boolean;
    ariaLabel?: string;
    /** 密度覆盖，默认继承容器。 */
    density?: AppDensity;
    className?: string;
}

/**
 * 滑块。
 *
 * @example
 * <AppSlider
 *     value={snap.swingPercent}
 *     unit="percent"
 *     min={0}
 *     max={100}
 *     ariaLabel={t("snap_grid_swing_strength")}
 *     onChange={handleSwingPreview}
 *     onCommit={handleSwingCommit}
 * />
 */
export function AppSlider({
    value,
    onChange,
    onCommit,
    unit,
    min,
    max,
    disabled = false,
    density,
    ariaLabel,
    className,
}: AppSliderProps) {
    const size = radixSizeFor(useDensity(density));
    const isFine = useFineAdjustModifier();

    /*
     * 滚轮走帧合并 + 手势累积（见 useFrameCommit.ts）。拖动本身由 Radix 以帧率
     * 回调，不需要节流；滚轮是每格一个事件，必须合并。
     */
    const committer = useFrameCommitter<number>((next) => {
        onChange(next);
        onCommit?.(next);
    });
    const accumulator = useWheelStepAccumulator<number>();

    const setWheelTarget = useNonPassiveWheel<HTMLSpanElement>((event) => {
        if (disabled) return;
        if (!Number.isFinite(event.deltaY) || event.deltaY === 0) return;
        event.preventDefault();
        const direction: 1 | -1 = event.deltaY < 0 ? 1 : -1;
        const fine = isFine(event);
        const next = accumulator.advance(value, (base) =>
            stepValue({ value: base, direction, unit, fine, min, max }),
        );
        if (next !== value) committer.schedule(next);
    });

    return (
        // Radix Slider 的根是 span；滚轮监听挂在它上面，指针落在滑块任意位置都生效。
        //
        // 【为什么要自带盒高并收住溢出】Radix 的滑块头是**绝对定位的装饰**：可见
        // 部分比轨道高（size 2 是 16px / 8px），另有放大命中区（`::before`，滑块头 × 3）。
        // 这份溢出**不会**被任何内在尺寸计算算进去（CSS 的 intrinsic sizing 从不含
        // scrollable overflow），却会被祖先的滚动容器算成"可滚动" —— 于是每一个含滑块的
        // 对话框都恒定挂着一条滚不动的竖直滚动条（实测正文 20px 高、scrollHeight 26px；
        // 参数编辑器右键菜单里的 8 个对话框有 7 个如此，唯一没有滑块的「添加颤音」没有）。
        //
        // 修法只能是把这份装饰溢出**收在自己的盒子里**：给足可见装饰需要的高度
        // （见 `DECORATION_BOX_PX`），再由 `index.css` 的 `.hs-slider-box` 裁掉多余
        // 部分（纵向的命中区余量）并留出横向余量（可见滑块头在 0% / 100% 时会越过
        // 轨道两端 2px）。被裁掉的只有那个 36px 命中区在纵向的余量 —— 它在 16px 高的
        // 行里本来就与相邻行重叠。
        <span
            ref={setWheelTarget}
            // 行高不变：盒高 ≤ 同一行里读数文字的 16px（紧凑表面里更是如此）。
            style={{ minHeight: DECORATION_BOX_PX[size] }}
            className={cx("hs-slider-box inline-flex min-w-0 flex-1 items-center", className)}
        >
            <Slider
                value={[value]}
                size={size}
                min={min}
                max={max}
                step={1}
                disabled={disabled}
                aria-label={ariaLabel}
                onValueChange={(values) => onChange(values[0] ?? value)}
                onValueCommit={(values) => onCommit?.(values[0] ?? value)}
                style={{ width: "100%" }}
            />
        </span>
    );
}

/**
 * 滑块的数值读数（右侧百分比/数值）。
 *
 * 与滑块成对使用，保证读数与滑块的排版一致（等宽数字、右对齐、固定宽度），
 * 而不是每处自己写 `style={{ width: 36, textAlign: "right" }}`。
 */
export function AppSliderReadout({
    children,
    minWidth = 40,
}: {
    children: ReactNode;
    /** 读数最小宽度；旧的各处用 40/52/56 不等，统一由调用方按需给出。 */
    minWidth?: number;
}) {
    return (
        <span className="hs-type-mono shrink-0 text-right" style={{ minWidth }}>
            {children}
        </span>
    );
}
