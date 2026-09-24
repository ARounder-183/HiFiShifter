/*
 * 记事本里的小型上下文菜单。
 *
 * 与时间轴 `ClipContextMenu` 同构（手搓的 fixed 定位 div，不引入 Radix 的
 * portal 菜单）：菜单项由调用方给，`shortcut` 只作展示。
 */

import { useEffect, useRef } from "react";

export interface NotebookMenuItem {
    key: string;
    label: string;
    onSelect: () => void;
    /** 展示用快捷键文本（不参与绑定）。 */
    shortcut?: string;
    danger?: boolean;
    disabled?: boolean;
    /** 上方加一条分隔线（分组）。 */
    separatorBefore?: boolean;
}

export interface NotebookContextMenuProps {
    x: number;
    y: number;
    items: NotebookMenuItem[];
    onClose: () => void;
}

export function NotebookContextMenu({ x, y, items, onClose }: NotebookContextMenuProps) {
    const ref = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) onClose();
        }
        function onKeyDown(event: KeyboardEvent) {
            if (event.key === "Escape") onClose();
        }
        // 捕获阶段：编辑器自身的 pointerdown 会先选中节点，不能让菜单先关掉
        // 再被下层重新打开，因此统一在 document 捕获里判定。
        document.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("keydown", onKeyDown, true);
        return () => {
            document.removeEventListener("pointerdown", onPointerDown, true);
            document.removeEventListener("keydown", onKeyDown, true);
        };
    }, [onClose]);

    // 贴近窗口边缘时向内收，避免菜单跑出可视区。
    const width = 210;
    const height = items.length * 26 + 12;
    const left = Math.max(4, Math.min(x, window.innerWidth - width - 4));
    const top = Math.max(4, Math.min(y, window.innerHeight - height - 4));

    return (
        <div
            ref={ref}
            role="menu"
            data-hs-context-menu="1"
            className="fixed z-[999] min-w-[190px] rounded border border-qt-border bg-qt-window py-1 text-qt-text shadow-lg"
            style={{ left, top, width }}
            onPointerDown={(event) => event.stopPropagation()}
        >
            {items.map((item) => (
                <button
                    key={item.key}
                    type="button"
                    role="menuitem"
                    disabled={item.disabled}
                    className={[
                        "flex w-full items-center justify-between gap-3 px-3 py-1 text-left text-xs",
                        item.disabled
                            ? "cursor-default text-qt-text-muted"
                            : item.danger
                              ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
                              : "hover:bg-qt-hover",
                    ].join(" ")}
                    style={
                        item.separatorBefore
                            ? {
                                  borderTop: "1px solid var(--qt-border)",
                                  marginTop: 4,
                                  paddingTop: 6,
                              }
                            : undefined
                    }
                    onClick={() => {
                        if (item.disabled) return;
                        item.onSelect();
                        onClose();
                    }}
                >
                    <span className="truncate">{item.label}</span>
                    {item.shortcut ? (
                        <span className="shrink-0 text-qt-text-muted">{item.shortcut}</span>
                    ) : null}
                </button>
            ))}
        </div>
    );
}
