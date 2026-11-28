/**
 * 桌面形态的顶栏栈。
 *
 * 【为什么这也要单独一个组件】它只有三个槽位，看起来"不值得一个文件"。但它承载的是
 * 一条**布局契约**：三层（菜单栏 / 动作栏 / 宿主提示条）必须都是 `shrink-0`，
 * 且顺序固定 —— 提示条在动作栏**之下**、工作区之上，因为它在描述"你看到的这片
 * 区域为什么没有声音"，必须紧挨着那片区域。
 *
 * 把这条契约写在组件里，`App.tsx` 就只需要传三个节点；写回 `App.tsx` 的 render 里，
 * 它会和 40 个对话框的 JSX 混在一起，下一个改动的人只能靠肉眼确认顺序没被打乱。
 */
import type { ReactNode } from "react";

export interface DesktopChromeProps {
    /** 菜单栏（`MenuBar`）。 */
    menuBar: ReactNode;
    /** 动作栏（`ActionBar`）。 */
    actionBar: ReactNode;
    /** 宿主音频提示条（仅插件形态；独立 App 为 `null`）。 */
    notice?: ReactNode;
}

export function DesktopChrome({ menuBar, actionBar, notice }: DesktopChromeProps) {
    return (
        <>
            <div className="shrink-0">{menuBar}</div>
            <div className="shrink-0">{actionBar}</div>
            {notice ? <div className="shrink-0">{notice}</div> : null}
        </>
    );
}
