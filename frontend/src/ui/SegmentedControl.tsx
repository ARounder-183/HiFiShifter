/*
 * 段控原语 —— 「一组互斥选项里选一个」的紧凑控件。
 *
 * 【为什么必须有它】此前同一控件有**两种视觉语言**：ExportAudioDialog /
 * QuickClipExportDialog 用 Radix `SegmentedControl`（灰色激活块），外观设置
 * 面板手写了一个 accent 激活段控 —— 同一个 app 里"选中"长两种样子。根因是
 * 没有原语，作者只能各找各的路。
 *
 * 【激活态用 --qt-accent】与 `AppButton` 的 primary 同源（`var(--accent-9)`）：
 * "激活"与"主操作"都是"当前生效"的语义，颜色系统不应给出第二种答案。
 *
 * 【键盘】`role="radiogroup"` + roving tabindex：整组只占一个 Tab 停留点
 * （激活项），方向键在组内循环移动并直接选中 —— 与 DockTabBar / 菜单同一套
 * 约定，也让全局快捷键分发器自动让出方向键。
 *
 * 【宽度】容器默认 `inline-flex`（跟随内容，页签场景）；传 `w-full` 让它铺满
 * 表单行（导出格式场景）。按钮本身始终 `flex-1` 均分容器。
 */
import { useRef, type KeyboardEvent, type ReactNode } from "react";

import { cx } from "./cx";

export interface AppSegmentedOption<T extends string> {
    value: T;
    label: ReactNode;
}

export interface AppSegmentedControlProps<T extends string> {
    /** 当前生效的选项值（受控）。 */
    value: T;
    options: ReadonlyArray<AppSegmentedOption<T>>;
    onChange: (value: T) => void;
    /** `sm`（表单字段内）/ `md`（页签行）。默认 `sm`。 */
    size?: "sm" | "md";
    /** 无障碍名称：radiogroup 需要可读名称。 */
    ariaLabel?: string;
    className?: string;
}

export function AppSegmentedControl<T extends string>({
    value,
    options,
    onChange,
    size = "sm",
    ariaLabel,
    className,
}: AppSegmentedControlProps<T>) {
    const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

    const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        const index = options.findIndex((option) => option.value === value);
        let next = -1;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
            next = (index + 1) % options.length;
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
            next = (index - 1 + options.length) % options.length;
        } else if (event.key === "Home") {
            next = 0;
        } else if (event.key === "End") {
            next = options.length - 1;
        }
        if (next < 0) return;
        event.preventDefault();
        const target = options[next];
        if (target.value !== value) onChange(target.value);
        buttonRefs.current[next]?.focus();
    };

    return (
        <div
            role="radiogroup"
            aria-label={ariaLabel}
            className={cx(
                "inline-flex gap-1 rounded border border-qt-border bg-qt-panel p-1",
                className,
            )}
            onKeyDown={onKeyDown}
        >
            {options.map((option, index) => {
                const active = option.value === value;
                return (
                    <button
                        key={option.value}
                        ref={(el) => {
                            buttonRefs.current[index] = el;
                        }}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        tabIndex={active ? 0 : -1}
                        onClick={() => onChange(option.value)}
                        className={cx(
                            "flex-1 rounded px-2.5 text-qt-xs font-semibold cursor-pointer select-none whitespace-nowrap transition-colors",
                            size === "md" ? "py-1.5" : "py-1",
                            active
                                ? "bg-qt-accent text-white"
                                : "text-qt-text-muted hover:bg-qt-hover hover:text-qt-text",
                        )}
                    >
                        {option.label}
                    </button>
                );
            })}
        </div>
    );
}
