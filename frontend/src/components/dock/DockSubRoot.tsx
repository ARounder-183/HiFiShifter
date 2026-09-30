/*
 * 一棵布局树的挂载点：主界面与面板窗体共用。
 *
 * 【为什么要抽出来】主界面就是"主布局根"，面板就是"另一些布局根"——两者的
 * 渲染结构完全相同（`data-dock-root` 容器 + 递归的 `DockNodeView`），差别只在
 * 根 id 与嵌套样式。抽成这一个组件，"主界面是一种特殊的面板"就不再是一句
 * 概念话，而是同一份代码的两个调用点。
 *
 * 【空根 = 占位井】`roots[rootId]` 缺失即空面板。占位井自己带上
 * `data-dock-root`，于是它立刻获得根级边缘带与"整块就是落点"的语义 ——
 * 空面板从建好那一刻起就能接住拖进来的窗体，不需要先有内容。
 */

import { useAppSelector } from "../../app/hooks";
import { readRoot } from "../../features/dock/dockTree";
import { useI18n } from "../../i18n/I18nProvider";
import { DockNodeView } from "./DockNodeView";

export interface DockSubRootProps {
    rootId: string;
    /** `main` = 主布局根（无嵌套视觉），`panel` = 面板内部（嵌套描边）。 */
    kind: "main" | "panel";
}

export function DockSubRoot({ rootId, kind }: DockSubRootProps) {
    const layout = useAppSelector((state) => state.dock.layout);
    const { tf } = useI18n();
    const tree = readRoot(layout, rootId);

    if (!tree) {
        return (
            <div
                className="hs-dock-root hs-dock-empty-root"
                data-dock-root={rootId}
                data-dock-root-kind={kind}
            >
                <span className="hs-dock-empty-hint">{tf("dock_panel_empty_hint")}</span>
            </div>
        );
    }

    return (
        <div className="hs-dock-root" data-dock-root={rootId} data-dock-root-kind={kind}>
            <DockNodeView node={tree} />
        </div>
    );
}
