/**
 * 自动交叉淡化**写回**的回归测试。
 *
 * 【为什么这些用例存在】旧实现把一次粘贴/复制/导入的自动淡化拆成 N 次
 * `webApi.setClipState`：N 个宿主写回请求、N 个撤销点、N 次完成门。部分成功时
 * 一半 Clip 有淡化、一半没有 —— 用户看到的就是"自动交叉淡化没有执行"。而失败时
 * 本地已经写进去的乐观值从不回滚，界面于是长期显示一份宿主上并不存在的淡化。
 *
 * 这里钉住三件事：
 * 1. **一次批量**：N 条更新只产生一个写回 thunk（不是 N 个）；
 * 2. **永久失败回滚**：本地值回到写前，界面不再撒谎；
 * 3. **瞬时失败不回滚**：宿主还在追几何/媒体时，回滚只会让淡化"闪一下又没了"。
 */
import { describe, expect, test, vi } from "vitest";

import type { AppDispatch } from "../../../../app/store";
import { persistAutoFadeUpdates, autoFadeSnapshot } from "./autoCrossfade";

interface RecordedAction {
    type: string;
    payload?: { clipId?: string; autoFadeInSec?: number; autoFadeOutSec?: number };
}

/** 一个只记录动作的 dispatch；写回 thunk 的成败由 `outcome` 决定。 */
function recordingDispatch(outcome: "ok" | "transient" | "permanent") {
    const actions: RecordedAction[] = [];
    const writes: unknown[] = [];
    const dispatch = ((action: unknown) => {
        if (typeof action === "function") {
            writes.push(action);
            return {
                unwrap: async () => {
                    if (outcome === "ok") return {};
                    throw outcome === "permanent"
                        ? { detail: "invalid host channel mode", kind: "permanent" }
                        : {
                              detail: "clip was edited in the host but its ARA geometry has not caught up",
                          };
                },
            };
        }
        actions.push(action as RecordedAction);
        return { unwrap: async () => ({}) };
    }) as unknown as AppDispatch;
    return { dispatch, actions, writes };
}

const UPDATES = [
    { clipId: "a", autoFadeInSec: 0.1, autoFadeOutSec: 0 },
    { clipId: "b", autoFadeInSec: 0, autoFadeOutSec: 0.2 },
    { clipId: "c", autoFadeInSec: 0.3, autoFadeOutSec: 0.3 },
];

const PREVIOUS = new Map([
    ["a", { in: 0, out: 0 }],
    ["b", { in: 0, out: 0 }],
    ["c", { in: 0.05, out: 0.05 }],
]);

describe("persistAutoFadeUpdates", () => {
    test("三条更新只发一次批量写回", async () => {
        const { dispatch, writes } = recordingDispatch("ok");
        const outcome = await persistAutoFadeUpdates(UPDATES, dispatch, { previous: PREVIOUS });
        expect(outcome).toBe("ok");
        expect(writes).toHaveLength(1);
    });

    test("永久失败把本地值回滚到写前", async () => {
        const { dispatch, actions } = recordingDispatch("permanent");
        const outcome = await persistAutoFadeUpdates(UPDATES, dispatch, { previous: PREVIOUS });
        expect(outcome).toBe("permanent");
        const rollbacks = actions.filter((a) => a.type === "session/setClipAutoFades").slice(-3);
        expect(rollbacks.map((a) => a.payload?.autoFadeInSec)).toEqual([0, 0, 0.05]);
        expect(rollbacks.map((a) => a.payload?.autoFadeOutSec)).toEqual([0, 0, 0.05]);
    });

    test("瞬时失败保留乐观值（等宿主回声对齐）", async () => {
        const { dispatch, actions } = recordingDispatch("transient");
        const outcome = await persistAutoFadeUpdates(UPDATES, dispatch, { previous: PREVIOUS });
        expect(outcome).toBe("transient");
        const applied = actions.filter((a) => a.type === "session/setClipAutoFades");
        expect(applied.map((a) => a.payload?.autoFadeInSec)).toEqual([0.1, 0, 0.3]);
    });

    test("requirePresent 过滤掉本地还不存在的 clip，不让整批失败", async () => {
        const { dispatch, actions, writes } = recordingDispatch("ok");
        const outcome = await persistAutoFadeUpdates(UPDATES, dispatch, {
            requirePresent: new Set(["a"]),
        });
        expect(outcome).toBe("ok");
        expect(writes).toHaveLength(1);
        expect(actions.filter((a) => a.type === "session/setClipAutoFades")).toHaveLength(1);
    });

    test("空更新不产生任何写回", async () => {
        const { dispatch, writes, actions } = recordingDispatch("ok");
        const outcome = await persistAutoFadeUpdates([], dispatch);
        expect(outcome).toBe("ok");
        expect(writes).toHaveLength(0);
        expect(actions).toHaveLength(0);
    });
});

describe("autoFadeSnapshot", () => {
    test("缺省的 clip 记 0，已有值原样取出", () => {
        const session = {
            clips: [{ id: "a", autoFadeInSec: 0.4, autoFadeOutSec: 0.5 }, { id: "b" }],
        } as never;
        const snapshot = autoFadeSnapshot(session, ["a", "b", "missing"]);
        expect(snapshot.get("a")).toEqual({ in: 0.4, out: 0.5 });
        expect(snapshot.get("b")).toEqual({ in: 0, out: 0 });
        expect(snapshot.get("missing")).toEqual({ in: 0, out: 0 });
    });
});
