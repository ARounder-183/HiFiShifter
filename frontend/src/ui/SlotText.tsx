/**
 * 固定槽位读数 —— 会让数字逐帧变化的文本必须用它，而不是自然宽度。
 *
 * 【它解决什么】时间、捕获计时、录音倒计时这类读数每多一位数字，整段文本就变宽；
 * 放在 shrink-0 的工具栏组里时，组宽随内容变化，后续所有控件一起移动。人眼把这读成
 * "整个菜单在抖"，而不是"数字变长了"。
 *
 * 【做法】把连续数字拆成等宽槽位，并用一段**隐藏的预留文本**按真实字体量出最长宽度；
 * 容器始终按最长文本占位。数字只在槽位里变化，非数字字符（小数点、冒号、负号、单位）
 * 保持自然宽度，因此 "12.3s" 与 "9.9s" 的总宽完全相同。
 *
 * 【为什么不用 tabular-nums 就够了】等宽数字只让每个数字等宽，不会为"未来可能出现的
 * 位数"预留空间；1 位变 2 位仍然会让容器变宽。槽位 + reserveText 才真正把宽度钉死。
 */
import { useCallback, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import { cx } from "./cx";

/** 把文本拆为连续数字段 / 非数字段，顺序与原文一致。 */
export function splitDigitRuns(text: string): Array<{ text: string; digits: boolean }> {
    const parts: Array<{ text: string; digits: boolean }> = [];
    let current = "";
    let currentDigits = false;
    const flush = () => {
        if (current.length > 0) {
            parts.push({ text: current, digits: currentDigits });
            current = "";
        }
    };
    for (const ch of text) {
        const isDigit = ch >= "0" && ch <= "9";
        if (isDigit !== currentDigits) {
            flush();
            currentDigits = isDigit;
        }
        current += ch;
    }
    flush();
    return parts;
}

export interface SlotTextProps {
    text: string;
    /** 用于预留宽度的最长文本；缺省 = text。 */
    reserveText?: string;
    className?: string;
    style?: CSSProperties;
    /** 允许用户选择这段文本（显式覆盖全局 DAW 禁选策略）。 */
    selectable?: boolean;
    /** 自定义气泡；空串不写 data-tooltip。 */
    tooltip?: string;
    /** 测试/诊断出口。 */
    "data-testid"?: string;
}

function useSlotMetrics(
    reserveText: string,
    text: string,
): {
    reserveRef: (node: HTMLSpanElement | null) => void;
    textProbeRef: (node: HTMLSpanElement | null) => void;
    probeRef: (node: HTMLSpanElement | null) => void;
    digitWidth: number | null;
    minWidth: number | null;
} {
    const reserveRef = useRef<HTMLSpanElement | null>(null);
    const textProbeRef = useRef<HTMLSpanElement | null>(null);
    const probeRef = useRef<HTMLSpanElement | null>(null);
    const [digitWidth, setDigitWidth] = useState<number | null>(null);
    const [minWidth, setMinWidth] = useState<number | null>(null);

    const measureDigitWidth = useCallback(() => {
        const probe = probeRef.current;
        if (!probe) return;
        const style = window.getComputedStyle(probe);
        const probeWidth = probe.getBoundingClientRect().width;
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            if (probeWidth > 0) setDigitWidth((prev) => (prev === probeWidth ? prev : probeWidth));
            return;
        }
        ctx.font = style.font;
        let max = 0;
        for (const digit of "0123456789") {
            max = Math.max(max, ctx.measureText(digit).width);
        }
        const slotWidth = Math.max(max, probeWidth);
        if (slotWidth > 0) {
            setDigitWidth((prev) =>
                prev != null && Math.abs(prev - slotWidth) < 0.01 ? prev : slotWidth,
            );
        }
    }, []);

    const measureWidth = useCallback(() => {
        const reserve = reserveRef.current;
        const current = textProbeRef.current;
        if (!reserve || !current) return;
        const reserveWidth = reserve.getBoundingClientRect().width;
        const currentWidth = current.getBoundingClientRect().width;
        const next = Math.ceil(Math.max(reserveWidth, currentWidth)) + 2;
        if (next > 0) {
            setMinWidth((prev) => (prev != null && Math.abs(prev - next) < 0.01 ? prev : next));
        }
    }, []);

    useLayoutEffect(() => {
        measureDigitWidth();
        measureWidth();
        const observer =
            typeof ResizeObserver === "undefined"
                ? null
                : new ResizeObserver(() => {
                      measureDigitWidth();
                      measureWidth();
                  });
        if (probeRef.current) observer?.observe(probeRef.current);
        if (reserveRef.current) observer?.observe(reserveRef.current);
        if (textProbeRef.current) observer?.observe(textProbeRef.current);
        // WebView2 的自定义字体可能在首帧之后才完成加载；加载完成补测一次。
        if (typeof document !== "undefined" && document.fonts?.ready) {
            void document.fonts.ready.then(() => {
                measureDigitWidth();
                measureWidth();
            });
        }
        return () => observer?.disconnect();
    }, [measureDigitWidth, measureWidth, reserveText, text]);

    const setTextProbe = useCallback((node: HTMLSpanElement | null) => {
        textProbeRef.current = node;
    }, []);
    const setReserve = useCallback((node: HTMLSpanElement | null) => {
        reserveRef.current = node;
    }, []);
    const setProbe = useCallback((node: HTMLSpanElement | null) => {
        probeRef.current = node;
    }, []);

    return {
        reserveRef: setReserve,
        textProbeRef: setTextProbe,
        probeRef: setProbe,
        digitWidth,
        minWidth,
    };
}

/**
 * 固定槽位文本。
 *
 * 真实文本放在独立节点里，隐藏的测量节点是它的兄弟 —— 因此 [data-slot-text]
 * 的 textContent 仍是原始字符串，复制、查找与既有测试都不会看到预留文本。
 */
export function SlotText({
    text,
    reserveText,
    className,
    style,
    selectable = false,
    tooltip,
    "data-testid": testId,
}: SlotTextProps) {
    const reserve = reserveText ?? text;
    const parts = useMemo(() => splitDigitRuns(text), [text]);
    const { reserveRef, textProbeRef, probeRef, digitWidth, minWidth } = useSlotMetrics(
        reserve,
        text,
    );

    const onDoubleClick = selectable
        ? (event: React.MouseEvent<HTMLSpanElement>) => {
              event.preventDefault();
              event.stopPropagation();
              const selection = window.getSelection();
              if (!selection) return;
              const range = document.createRange();
              range.selectNodeContents(event.currentTarget);
              selection.removeAllRanges();
              selection.addRange(range);
          }
        : undefined;

    return (
        <span
            className={cx(
                "relative inline-flex max-w-full align-middle",
                selectable && "cursor-text",
                className,
            )}
            style={{
                ...style,
                minWidth: minWidth ?? undefined,
                userSelect: selectable ? "text" : undefined,
                WebkitUserSelect: selectable ? "text" : undefined,
            }}
            data-tooltip={tooltip || undefined}
            data-slot-text-container=""
            data-testid={testId}
        >
            <span
                className="inline-flex min-w-0 items-baseline whitespace-nowrap"
                style={{ maxWidth: "100%" }}
                onDoubleClick={onDoubleClick}
                data-slot-text=""
                data-reserve-text={reserve}
            >
                {parts.map((part, index) =>
                    part.digits && digitWidth != null && digitWidth > 0 ? (
                        <span key={index} className="inline-flex">
                            {part.text.split("").map((ch, chIndex) => (
                                <span
                                    key={chIndex}
                                    style={{
                                        display: "inline-block",
                                        width: digitWidth,
                                        textAlign: "center",
                                    }}
                                >
                                    {ch}
                                </span>
                            ))}
                        </span>
                    ) : (
                        <span key={index}>{part.text}</span>
                    ),
                )}
            </span>
            <span
                ref={reserveRef}
                aria-hidden="true"
                className={cx(
                    "pointer-events-none invisible absolute whitespace-nowrap",
                    className,
                )}
            >
                {reserve}
            </span>
            <span
                aria-hidden="true"
                className={cx(
                    "pointer-events-none invisible absolute whitespace-nowrap tabular-nums",
                    className,
                )}
                ref={textProbeRef}
            >
                {text}
            </span>
            <span
                ref={probeRef}
                aria-hidden="true"
                className={cx(
                    "pointer-events-none invisible absolute whitespace-nowrap tabular-nums",
                    className,
                )}
            >
                0
            </span>
        </span>
    );
}
