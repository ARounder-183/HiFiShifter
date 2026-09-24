/*
 * 把窗体的 DOM 宿主搬进一个槽位的共享逻辑。
 *
 * 停靠标签组与浮动窗都需要同一件事：槽位渲染出来是空的，真实面板 DOM 由
 * 本 hook 从 `panelHostRegistry` 搬进来；切走时把旧宿主送回停泊区（而不是
 * 让它留在槽位里继续占位）。两处若各写一份，"停靠态正常、浮动态白屏"这类
 * 分叉几乎必然出现，所以收敛到这一个 hook。
 */

import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from "react";

import { store } from "../../app/store";
import { useAppSelector } from "../../app/hooks";
import { getPanel } from "../../features/dock/panelRegistry";
import {
    acquirePanelHost,
    attachPanelHost,
    createSlotOwner,
    getPanelHostVersion,
    parkPanelHost,
    subscribePanelHosts,
} from "./panelHostRegistry";

/** 面板定义里的尺寸兜底（停泊时用，避免停泊区里 0 尺寸）。 */
export function panelFallbackSize(panelId: string): { w: number; h: number } {
    const definition = getPanel(panelId);
    return { w: definition?.defaultWidth ?? 420, h: definition?.defaultHeight ?? 320 };
}

/**
 * 返回应当挂载到内容区的槽位 ref。
 *
 * `formId` 为 null 表示当前没有窗体（空槽位）—— 旧宿主会被送回停泊区。
 */
export function useDockSlot(formId: string | null): React.RefObject<HTMLDivElement | null> {
    const slotRef = useRef<HTMLDivElement | null>(null);
    const forms = useAppSelector((s) => s.dock.layout.forms);
    // 宿主集合变化（新窗体首次出现）时必须重试搬家，因此版本号要进 effect 依赖：
    // 少了它，"面板挂载晚于槽位渲染"的那一次就永远不会把宿主搬进来。
    const hostVersion = useSyncExternalStore(
        subscribePanelHosts,
        getPanelHostVersion,
        getPanelHostVersion,
    );
    const previousRef = useRef<string | null>(null);
    // 本槽位的归属令牌：停泊时必须证明"宿主仍归我"，否则会把刚被新槽位接手的
    // 宿主搬走（见 `panelHostRegistry.owners`）。
    const ownerRef = useRef<symbol | null>(null);
    ownerRef.current ??= createSlotOwner();

    useLayoutEffect(() => {
        const slot = slotRef.current;
        const previous = previousRef.current;

        const owner = ownerRef.current;
        if (!owner) return;

        if (previous && previous !== formId) {
            const panelId = forms[previous]?.panelId;
            parkPanelHost(previous, panelFallbackSize(panelId ?? previous), owner);
            previousRef.current = null;
        }

        if (!slot || !formId) return;
        // 宿主可能尚未创建（`PanelMount` 的 effect 还没跑）——`acquirePanelHost`
        // 会按需创建，随后的 `subscribePanelHosts` 通知会让本 effect 重跑。
        acquirePanelHost(formId);
        attachPanelHost(formId, slot, owner);
        previousRef.current = formId;
    }, [formId, forms, hostVersion]);

    // 槽位卸载（面板关闭 / 树结构变化）时把宿主送回停泊区：否则宿主留在一个
    // 已脱离文档的槽位里，再也不会被复用。
    //
    // 清理函数里现读 store 而不是把 `forms` 存进 ref：在渲染期写 ref 会违反
    // React Compiler 的引用规则（"渲染期不得访问 ref"），而 store 本身就是
    // 权威且随时可读的。
    useEffect(() => {
        return () => {
            const owner = ownerRef.current;
            const previous = previousRef.current;
            previousRef.current = null;
            if (!owner || !previous) return;
            const panelId = store.getState().dock.layout.forms[previous]?.panelId;
            parkPanelHost(previous, panelFallbackSize(panelId ?? previous), owner);
        };
    }, []);

    return slotRef;
}
