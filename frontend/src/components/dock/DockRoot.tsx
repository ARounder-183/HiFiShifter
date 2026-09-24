/*
 * 停靠根：主窗口工作区的布局宿主。
 *
 * 这一层取代了原先写死在 `App.tsx` 里的"时间轴 / 分隔条 / 参数编辑器 + 右侧
 * 固定宽度栏"结构。它做四件事：
 * 1. 渲染布局树（`DockNodeView`）；
 * 2. 挂载面板宿主层（`DockPanelHosts`，只挂载一次，见 `panelHostRegistry`）；
 * 3. 渲染浮动层与拖拽覆盖层（两者都是 portal/fixed，不占布局）；
 * 4. 负责面板注册完成后的布局同步，以及布局持久化。
 */

import { useEffect } from "react";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { isFormVisible } from "../../features/dock/dockTree";
import { markFormsMounted, syncRegisteredPanels } from "../../features/dock/dockSlice";
import { DockDropOverlay } from "./DockDropOverlay";
import { DockFloatingLayer } from "./DockFloatingLayer";
import { DockNodeView } from "./DockNodeView";
import { DockPanelHosts } from "./DockPanelHosts";
import "./dock.css";

export function DockRoot() {
    const dispatch = useAppDispatch();
    const layout = useAppSelector((s) => s.dock.layout);
    const mountedFormIds = useAppSelector((s) => s.dock.mountedFormIds);

    // 内置面板在 App 模块加载期注册，早于首次渲染；但注册表也可能在运行期
    // 变化（热更新重放注册、将来插件加载）。这里同步一次，把新注册的面板
    // 补成"已关闭"的窗体记录，让"显示窗体"菜单立刻能看到它们。
    useEffect(() => {
        dispatch(syncRegisteredPanels());
    }, [dispatch]);

    // 可见窗体（每次布局变化重算）→ 登记为"需要挂载"。宿主层只挂载这个并集，
    // 因此关闭再打开是零成本、状态全在；从未打开过的面板不挂载，避免启动时
    // 为记事本构建富文本编辑器、为未打开的面板创建 WebGL 上下文。
    const visibleIds = layout.order.filter((formId) => isFormVisible(layout, formId));
    const visibleKey = visibleIds.join("|");

    useEffect(() => {
        dispatch(markFormsMounted(visibleKey ? visibleKey.split("|") : []));
    }, [dispatch, visibleKey]);

    const mountedForms = mountedFormIds
        .map((formId) => layout.forms[formId])
        .filter((form): form is NonNullable<typeof form> => Boolean(form));

    return (
        <>
            <DockPanelHosts forms={mountedForms} />
            <div className="hs-dock-root" data-dock-root="1">
                <DockNodeView node={layout.tree} />
            </div>
            <DockFloatingLayer />
            <DockDropOverlay />
        </>
    );
}
