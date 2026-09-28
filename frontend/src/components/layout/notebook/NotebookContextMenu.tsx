/*
 * 记事本里的小型上下文菜单。
 *
 * 壳层（定位夹紧 / 外部点击与 Esc 关闭 / 方向键与 Home/End 导航 / 层级与
 * role）统一委托给共享原语 `AppContextMenu`（见 src/ui/Menu.tsx）：菜单项由
 * 调用方给，`shortcut` 只作展示。
 */

import { AppContextMenu } from "../../../ui/Menu";

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
    return <AppContextMenu x={x} y={y} items={items} onClose={onClose} />;
}
