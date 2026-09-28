/*
 * 标签右键菜单。
 *
 * 【为什么它没有迁到 `AppContextMenu`】菜单第一行是**重命名输入框**，带自己的
 * 模式机（Esc 退回菜单、Enter/blur 提交）。`AppContextMenu` 是"扁平项列表 +
 * 选中即关闭"，没有容纳输入框的槽位；强行迁会在这个文件里留下第二套手写外壳，
 * 反而更差。等原语支持内联编辑器后再统一。
 *
 * 【它是贡献点的第一个生产用例】第三方 / 内置面板可以往这里加自己的标签菜单项
 * （见 `features/dock/contributions.ts` 的 `registerPanelTabMenuItem`）——
 * 此前面板注册了却只能在 Window 菜单里被找到。
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { EDGE_GAP, clampAxisPosition } from "../appTooltipPosition";

import { useMenuKeyboard } from "../../ui/useMenuKeyboard";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { renameForm } from "../../features/dock/dockSlice";
import { usePanelTabMenuItems } from "../../features/dock/contributions";
import { useI18n } from "../../i18n/I18nProvider";

export interface DockTabMenuProps {
    formId: string;
    x: number;
    y: number;
    onClose: () => void;
    onFloat: () => void;
    onCloseForm: () => void;
    /**
     * 拆到独立窗口 / 从独立窗口收回。
     *
     * `null` 表示当前窗格不支持（面板未声明 `detachable`，或已经是独立窗口且
     * 收回入口由那个窗口自己的标题栏提供）—— 此时不渲染该项，而不是给一个点了
     * 没反应的按钮。
     */
    detachAction?: { labelKey: string; run: () => void } | null;
}

/** 菜单宽度：与 `AppContextMenu` 的默认 `minWidth` 保持一致。 */
const MENU_WIDTH = 190;

export function DockTabMenu({
    formId,
    x,
    y,
    onClose,
    onFloat,
    onCloseForm,
    detachAction,
}: DockTabMenuProps) {
    const dispatch = useAppDispatch();
    const { t, tf } = useI18n();
    /** 本窗体所属面板：贡献项按面板作用域筛选（全局项对所有面板可见）。 */
    const panelId = useAppSelector((state) => state.dock.layout.forms[formId]?.panelId);
    const contributedItems = usePanelTabMenuItems({ panelId });
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [renaming, setRenaming] = useState(false);
    const [draft, setDraft] = useState("");
    // 重命名模式下容器里只有输入框，没有菜单项可导航 —— 交给输入框自己。
    useMenuKeyboard(menuRef, !renaming);

    /*
     * 关闭时归还焦点（与 `AppContextMenu` 同一约定）。
     *
     * 菜单是弹出表面：卸载后若不归还焦点，键盘用户会被丢到 `<body>`，下一个 Tab
     * 从文档头重新开始。触发者可能已随菜单一起消失（例如"关闭标签"删掉了它），
     * 故归还前先查 `isConnected`。
     */
    const openerRef = useRef<HTMLElement | null>(null);
    useEffect(() => {
        openerRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null;
        return () => {
            const opener = openerRef.current;
            if (opener?.isConnected) opener.focus();
        };
    }, []);

    /*
     * 视口夹紧：右键点在屏幕右下角时菜单不能跑出可视区。
     *
     * 与 `AppContextMenu` 统一为**按实测尺寸**夹紧。原先这里是估算（固定 24px
     * 行高 + 18px chrome），但本菜单有几种高度不确定的内容：重命名输入框、
     * 贡献项（第三方可以加任意多项，标签长度也不受控）。估算一旦偏低，菜单底部
     * 就会跑出视口 —— 用户看不到"关闭标签"这一项。测量在绘制前完成，无闪动。
     */
    const [position, setPosition] = useState<{ x: number; y: number; ready: boolean }>({
        x,
        y,
        ready: false,
    });
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        setPosition({
            x: clampAxisPosition(x, rect.width, window.innerWidth, 0, EDGE_GAP),
            y: clampAxisPosition(y, rect.height, window.innerHeight, 0, EDGE_GAP),
            ready: true,
        });
    }, [x, y, renaming, contributedItems.length, detachAction]);

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                event.stopPropagation();
                onClose();
            }
        };
        const onPointerDown = (event: PointerEvent) => {
            if (menuRef.current?.contains(event.target as Node)) return;
            onClose();
        };
        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("pointerdown", onPointerDown, true);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("pointerdown", onPointerDown, true);
        };
    }, [onClose]);

    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            data-hs-context-menu="1"
            className="fixed z-qt-menu min-w-[190px] rounded border border-qt-border bg-qt-window py-1 text-qt-text shadow-lg"
            style={{
                left: position.x,
                top: position.y,
                minWidth: MENU_WIDTH,
                visibility: position.ready ? undefined : "hidden",
            }}
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            {renaming ? (
                <div className="px-2 py-1">
                    <input
                        autoFocus
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === "Enter") {
                                dispatch(renameForm({ formId, title: draft }));
                                onClose();
                            }
                            if (event.key === "Escape") setRenaming(false);
                        }}
                        onBlur={() => {
                            dispatch(renameForm({ formId, title: draft }));
                            onClose();
                        }}
                        className="w-full rounded border border-qt-border bg-qt-base px-1 py-0.5 text-qt-xs text-qt-text outline-none"
                    />
                </div>
            ) : (
                <MenuItem
                    label={tf("dock_rename_tab")}
                    onClick={() => {
                        setDraft("");
                        setRenaming(true);
                    }}
                />
            )}
            <MenuItem label={tf("dock_float")} onClick={onFloat} />
            {detachAction ? (
                <MenuItem
                    label={tf(detachAction.labelKey)}
                    onClick={() => {
                        detachAction.run();
                        onClose();
                    }}
                />
            ) : null}
            {contributedItems.length > 0 ? (
                <>
                    <div className="my-1 border-t border-qt-border" />
                    {contributedItems.map((item) => (
                        <MenuItem
                            key={item.id}
                            label={item.label}
                            danger={item.danger}
                            disabled={item.enabled ? !item.enabled() : false}
                            onClick={() => {
                                item.onSelect();
                                onClose();
                            }}
                        />
                    ))}
                </>
            ) : null}
            <div className="my-1 border-t border-qt-border" />
            <MenuItem
                label={t("close")}
                danger
                onClick={() => {
                    onCloseForm();
                    onClose();
                }}
            />
        </div>,
        document.body,
    );
}

function MenuItem({
    label,
    onClick,
    danger,
    disabled = false,
}: {
    label: string;
    onClick: () => void;
    danger?: boolean;
    disabled?: boolean;
}) {
    return (
        <button
            type="button"
            role="menuitem"
            disabled={disabled}
            className={`block w-full px-3 py-1.5 text-left text-qt-xs ${
                disabled
                    ? "cursor-default text-qt-text-muted"
                    : danger
                      ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
                      : "hover:bg-qt-hover"
            }`}
            onClick={onClick}
        >
            {label}
        </button>
    );
}
