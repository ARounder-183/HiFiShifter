/**
 * 播放光标时间各槽位的**挂载事实**。
 *
 * 【为什么需要一份登记表】`auto` 是一条优先级链（见 `AUTO_PLACEMENT_PRIORITY`）：
 * 链上的每一级都以"那个槽位此刻真的挂载着"为条件，链条才会顺延。若各槽位只在
 * 自己的 JSX 里判断"该不该渲染"，就没有任何一处知道**别处**还在不在 —— 解析函数
 * 只能退化成把一个槽位写死，那正是"自动"名不副实的原因。
 *
 * 【为什么是模块级 store 而不是 context】槽位分布在三棵互不相邻的子树里
 * （ActionBar / App 状态栏 / TrackList），挂 context 就得给整个 App 加一层 provider；
 * 而这份状态本来就属于"这三个组件在不在"，与 React 树的结构无关。用外部 store +
 * `useSyncExternalStore` 让每个槽位自己登记自己，三个宿主都不必改。
 *
 * 【首帧为什么是 `null`】登记发生在 effect 里，首帧还没有任何事实。`null` 表示
 * "尚未测量"，解析时按链首处理 —— 读数第一帧就出现在该在的地方，而不是先闪没、
 * 下一帧再出现（那会在工具栏里造成一次可见的横向跳动）。同理，所有槽位都卸载时
 * 回到 `null`：StrictMode 的双次挂载与用例之间的卸载都不会留下"全 false"的残影。
 */
import { useEffect, useSyncExternalStore } from "react";

import type {
    PlayheadTimeSlot,
    PlayheadTimeSlotAvailability,
} from "../../../features/session/playheadTimePlacement";

const mountedCount: Record<PlayheadTimeSlot, number> = {
    toolbar: 0,
    statusBar: 0,
    trackHeader: 0,
};

let snapshot: PlayheadTimeSlotAvailability | null = null;
const listeners = new Set<() => void>();

function publish(): void {
    const total = mountedCount.toolbar + mountedCount.statusBar + mountedCount.trackHeader;
    snapshot =
        total === 0
            ? null
            : {
                  toolbar: mountedCount.toolbar > 0,
                  statusBar: mountedCount.statusBar > 0,
                  trackHeader: mountedCount.trackHeader > 0,
              };
    for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

function getSnapshot(): PlayheadTimeSlotAvailability | null {
    return snapshot;
}

/**
 * 本窗口各槽位的挂载事实；`null` 表示尚未测量（见文件头）。
 *
 * 只读、不登记：标尺右键菜单也要用它把 `auto` 解析成"此刻会落在哪一处"。
 */
export function usePlayheadTimeSlotAvailability(): PlayheadTimeSlotAvailability | null {
    return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/**
 * 把本槽位登记为"已挂载"。
 *
 * 必须**无条件**调用（放在组件的早退之前）：槽位组件在轮不到自己时会返回 `null`，
 * 但它仍然"在那儿"，仍然算这一级可用 —— 否则 `auto` 会因为目标槽位不渲染自己而
 * 判定它不存在，链就永远停在链首。
 */
export function useRegisterPlayheadTimeSlot(slot: PlayheadTimeSlot): void {
    useEffect(() => {
        mountedCount[slot] += 1;
        publish();
        return () => {
            mountedCount[slot] -= 1;
            publish();
        };
    }, [slot]);
}
