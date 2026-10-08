/*
 * 标签右键菜单。
 *
 * 【收编自手写壳】本菜单曾是 `AppContextMenu` 之外的第二套手写实现，卡点是
 * 菜单第一行的**重命名输入框**（自带 Enter/blur 提交、Esc 退回菜单的模式机）。
 * `AppContextMenu` 增加 `header` 槽并把 keydown 监听挪到冒泡阶段后，输入框可以
 * 优先于菜单的全局处理吃到按键，卡点消除 —— 条目、贡献项、危险项全部走壳。
 *
 * 【它是贡献点的第一个生产用例】第三方 / 内置面板可以往这里加自己的标签菜单项
 * （见 `features/dock/contributions.ts` 的 `registerPanelTabMenuItem`）——
 * 此前面板注册了却只能在 Window 菜单里被找到。
 *
 * 【重命名模式的行为】点「重命名」后整份菜单变成输入框（items 清空、只留
 * header）：Enter 或失焦提交、Esc 退回条目列表、点外面提交 —— 与收编前一致。
 */

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { AppContextMenu, type AppMenuItemSpec } from "../../ui/Menu";
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
     *
     * 【为什么还要 `disabled` / `tooltip`】有些"不支持"是**模式级**的：插件里没有
     * Tauri 窗口 API，拆出去必然失败。那时保留入口并写明原因比直接隐藏更有用 ——
     * 用户至少知道"这里本来有这个功能，是当前宿主环境不允许"。
     */
    detachAction?: {
        labelKey: string;
        run: () => void;
        disabled?: boolean;
        tooltip?: string;
    } | null;
}

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
    const [renaming, setRenaming] = useState(false);

    const items: AppMenuItemSpec[] = renaming
        ? []
        : [
              {
                  key: "rename",
                  label: tf("dock_rename_tab"),
                  onSelect: () => setRenaming(true),
              },
              { key: "float", label: tf("dock_float"), onSelect: onFloat },
              ...(detachAction
                  ? [
                        {
                            key: "detach",
                            label: tf(detachAction.labelKey),
                            disabled: detachAction.disabled === true,
                            tooltip: detachAction.tooltip,
                            onSelect: () => {
                                detachAction.run();
                                onClose();
                            },
                        },
                    ]
                  : []),
              ...(contributedItems.length > 0
                  ? contributedItems.map((item, index) => ({
                        key: item.id,
                        label: item.label,
                        danger: item.danger,
                        disabled: item.enabled ? !item.enabled() : false,
                        separatorBefore: index === 0,
                        onSelect: () => {
                            item.onSelect();
                            onClose();
                        },
                    }))
                  : []),
              {
                  key: "close",
                  label: t("close"),
                  danger: true,
                  separatorBefore: true,
                  onSelect: () => {
                      onCloseForm();
                      onClose();
                  },
              },
          ];

    return createPortal(
        <AppContextMenu
            x={x}
            y={y}
            ariaLabel={tf("dock_rename_tab")}
            items={items}
            onClose={onClose}
            header={
                renaming ? (
                    <RenameInput
                        onCommit={(title) => {
                            dispatch(renameForm({ formId, title }));
                            onClose();
                        }}
                        onCancel={() => setRenaming(false)}
                    />
                ) : undefined
            }
        />,
        document.body,
    );
}

/**
 * 重命名输入框（header 槽内容）。
 *
 * 【按键时序】Enter/Escape 在输入框自己的 onKeyDown 里 `stopPropagation()` ——
 * `AppContextMenu` 的 keydown 监听在冒泡阶段，输入框（目标）先跑，退回菜单
 * 不会被菜单的全局 Esc 处理抢先把整个菜单关掉。
 *
 * 【点外面 = 提交】与收编前一致：捕获阶段监听 pointerdown，点在输入框之外
 * 视为确认（失焦提交在 portal 卸载时不会触发，所以这里显式监听）。
 */
function RenameInput({
    onCommit,
    onCancel,
}: {
    onCommit: (title: string) => void;
    onCancel: () => void;
}) {
    const [draft, setDraft] = useState("");
    const ref = useRef<HTMLInputElement | null>(null);
    /** 外部点击提交时读最新草稿：在事件处理器里同步（渲染期写 ref 会被
     *  React Compiler 的引用规则拒绝）。 */
    const draftRef = useRef("");

    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current?.contains(event.target as Node)) return;
            onCommit(draftRef.current);
        }
        window.addEventListener("pointerdown", onPointerDown, true);
        return () => window.removeEventListener("pointerdown", onPointerDown, true);
    }, [onCommit]);

    return (
        <input
            ref={ref}
            autoFocus
            value={draft}
            onChange={(event) => {
                setDraft(event.target.value);
                draftRef.current = event.target.value;
            }}
            onKeyDown={(event) => {
                if (event.key === "Enter") {
                    event.stopPropagation();
                    onCommit(draft);
                }
                if (event.key === "Escape") {
                    event.stopPropagation();
                    onCancel();
                }
            }}
            className="w-full rounded border border-qt-border bg-qt-base px-1 py-0.5 text-qt-xs text-qt-text outline-none"
        />
    );
}
