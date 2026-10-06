/*
 * 记事本的右键菜单**表面**。
 *
 * 壳层（定位夹紧 / 外部点击与 Esc 关闭 / 方向键与 Home/End 导航 / 层级与
 * role）统一委托给共享原语 `AppContextMenu`（见 src/ui/Menu.tsx）；菜单**内容**
 * 由 `notebookMenu.ts` 决定。本文件只负责一件事：把菜单挂到 `document.body`。
 *
 * 【为什么必须 portal】菜单留在触发它的布局盒里，会被沿途任何一层
 * `overflow: hidden` 裁掉。记事本正文外面套着 `.hs-scroll-gutter`（`overflow:
 * auto`），卡片菜单还要再穿过几层 —— 挂到 body 之后这些都不再影响它。
 * `designSystemGates.test.ts` 也把这条写成了门禁（渲染 `hs-menu` 壳的文件必须
 * 出现 `createPortal(`）。
 *
 * 【不设 `floating`】那是"挂在时间轴/剪辑上的浮动菜单"契约，有三个独立读取方
 * （`AppTooltip` 的悬停抑制、`TimelinePanel` 的内联编辑失焦提交、`ClipHeader`
 * 的角标编辑守卫）。记事本是普通面板，套上它只会误伤。
 *
 * 【项类型直接用 `AppMenuItemSpec`】此前这里有一个更窄的本地 `NotebookMenuItem`
 * （少了 `icon` / `checked` / `tooltip` / `heading` / `ariaLabel`），而右键菜单
 * 恰恰要用到后四个 —— 再包一层只会让"能表达什么"取决于包在哪。直接沿用原语的
 * 类型，调用方与 `AppContextMenu` 的文档示例完全一致。
 */

import { createPortal } from "react-dom";

import { AppContextMenu, type AppMenuItemSpec } from "../../../ui";

export type { AppMenuItemSpec };

export interface NotebookContextMenuProps {
    x: number;
    y: number;
    items: AppMenuItemSpec[];
    onClose: () => void;
    /** 无障碍名称：菜单是弹出表面，需要有可读名称。 */
    ariaLabel?: string;
}

/**
 * 记事本的右键菜单。
 *
 * 【为什么总是 `autoFocus`】触发面（富文本正文 / 源码 textarea）**本身就是可编辑
 * 元素**，焦点天然在它身上。而菜单原语的方向键守卫是"焦点元素吞方向键就让路"
 * （为菜单内的内联输入框而设），于是不收焦点时方向键全被判给编辑器 —— 键盘用户
 * 用 `ContextMenu` 键打开菜单后一格都动不了（实测：`menuActive` 始终为 null，
 * 光标却在动）。收焦点之后方向键归菜单，关闭时原语再把焦点还给编辑器。
 *
 * 代价是编辑器会收到一次 `blur`；`useNotebookEditor` 的失焦处理已按
 * `relatedTarget` 识别"焦点是进了菜单"并跳过收尾（见那里的注释）。
 */
export function NotebookContextMenu({ x, y, items, onClose, ariaLabel }: NotebookContextMenuProps) {
    return createPortal(
        <AppContextMenu
            x={x}
            y={y}
            items={items}
            onClose={onClose}
            ariaLabel={ariaLabel}
            autoFocus
        />,
        document.body,
    );
}
