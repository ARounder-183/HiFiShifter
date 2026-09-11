import React from "react";

/**
 * 内核态 clip 行内编辑浮层（重命名 / 增益 / 速率）。
 *
 * 【为什么需要它】
 * 旧实现的行内输入框在 `ClipHeader`（DOM）内。内核模式下 clip 是自绘的、
 * `ClipHeader` 不挂载，因此必须有一个独立的输入浮层，否则双击名称、单击增益 /
 * 速率标签这些交互都没有落点。
 *
 * 【定位策略】
 * 浮层是内核容器内的**绝对定位**元素，位置由外部在 rAF 内命令式写入
 * （内容坐标 − 滚动量，见 `TimelineKernelView` 的视口图层注册）。刻意**不用**
 * React state 每帧更新位置：那会让输入框在滚动时抖动，甚至因重渲染丢失焦点与
 * 已输入内容。
 *
 * 【职责边界】
 * 只负责「输入 + 提交 / 取消」与最小可用性（自动聚焦全选、Enter 提交、Esc 取消、
 * 失焦提交）。**值的格式化与解析由调用方负责**——那是领域知识（增益是 dB、
 * 速率有 `x` / `%` 前缀），不属于输入框。
 */

/** 行内编辑浮层的 props。 */
export interface KernelClipInlineEditorProps {
    /** 初始文本（由调用方格式化）。 */
    readonly initialValue: string;
    /**
     * 输入框宽度（CSS px）。
     *
     * 由外部在 rAF 内**命令式**写入（与 `left` / `top` 同一处），这里只给首帧的
     * 兜底值。刻意不用 React state 驱动宽度：那会让「挂载」与「拿到宽度」分成
     * 两次渲染，切换编辑目标时会出现"状态已更新但浮层没出现"的时序空档。
     */
    readonly widthPx?: number;
    /** 数值键盘提示：`gain` / `rate` 用 `decimal`，名称用 `text`。 */
    readonly inputMode?: "text" | "decimal";
    readonly placeholder?: string;
    /**
     * 提交回调（Enter 或失焦）。
     *
     * 传入的是**原始文本**（未 trim 以外的加工）：调用方决定如何解析与校验。
     */
    readonly onCommit: (value: string) => void;
    /** 取消回调（Esc，或失焦时值未变化）。 */
    readonly onCancel: () => void;
}

/**
 * 内核态 clip 行内编辑浮层。
 *
 * @param props 见 `KernelClipInlineEditorProps`。
 * @param ref 浮层根元素的 ref（外部据此在 rAF 内写 `left` / `top`）。
 * @returns 输入框元素。
 */
export const KernelClipInlineEditor = React.forwardRef<
    HTMLDivElement,
    KernelClipInlineEditorProps
>(function KernelClipInlineEditor(props, ref) {
    const { initialValue, widthPx = 0, inputMode = "text", placeholder, onCommit, onCancel } =
        props;
    const [value, setValue] = React.useState(initialValue);
    const inputRef = React.useRef<HTMLInputElement | null>(null);
    /**
     * 是否已收尾。
     *
     * 失焦与 Enter / Esc 都可能触发收尾，且失焦会在元素被移除时再次触发——
     * 用 ref 做幂等守卫，避免同一次编辑提交两遍（表现为后端收到两笔相同写入，
     * 撤销栈里多出一个空步）。
     */
    const settledRef = React.useRef(false);

    React.useLayoutEffect(() => {
        const input = inputRef.current;
        if (input === null) return;
        input.focus();
        input.select();
    }, []);

    /** 收尾（幂等）：提交或取消。 */
    const settle = React.useCallback(
        (commit: boolean) => {
            if (settledRef.current) return;
            settledRef.current = true;
            if (commit) onCommit(value);
            else onCancel();
        },
        [onCancel, onCommit, value],
    );

    return (
        <div
            ref={ref}
            data-hs-kernel-inline-editor="1"
            className="absolute z-30"
            style={{ left: 0, top: 0, width: widthPx > 0 ? widthPx : 160 }}
        >
            <input
                ref={inputRef}
                value={value}
                inputMode={inputMode === "decimal" ? "decimal" : "text"}
                placeholder={placeholder}
                className="w-full rounded-sm border border-qt-highlight bg-qt-window px-1.5 py-0.5 text-[11px] text-qt-text outline-none"
                onChange={(event) => setValue(event.target.value)}
                onPointerDown={(event) => {
                    // 输入框内的按下不得触发时间轴手势（否则会开始一次拖拽 / seek）。
                    event.stopPropagation();
                }}
                onKeyDown={(event) => {
                    // 快捷键不得外泄到时间轴的键盘处理（Delete / 空格等会误触发）。
                    event.stopPropagation();
                    if (event.key === "Enter") {
                        event.preventDefault();
                        settle(true);
                    } else if (event.key === "Escape") {
                        event.preventDefault();
                        settle(false);
                    }
                }}
                onBlur={() => {
                    // 值未变化时视为取消：避免一次「点开又点走」产生无意义的远端写入。
                    settle(value !== initialValue);
                }}
            />
        </div>
    );
});
