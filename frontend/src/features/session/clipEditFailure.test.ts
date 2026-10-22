import { test } from "vitest";

import { classifyClipEditFailure } from "./clipEditFailure.ts";

/**
 * **不变量：只有需要用户处置的编辑失败才提示"音频块编辑被拒绝"。**
 *
 * 【为什么这是回归测试】`setClipStateRemote` / `setClipsStateBulkRemote` 的 `.rejected`
 * 此前一律把状态置成 `"Clip edit rejected"`，而**任何**失败模式都走同一条路 —— 包括
 * "宿主还在追上几何/媒体"这种后端会自己重试并收敛的瞬时失败。用户因此每做一次编辑都
 * 被报一次错，即使编辑其实生效了（用户报障："执行任何编辑操作时，总是会闪烁地提示…"
 * 与"一直触发音频块编辑被拒绝"）。
 */
test("features/session/clipEditFailure.test.ts scripted checks", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    // 后端在宿主回调边界产出的**瞬时**文案：插件会自行重试并收敛。
    for (const message of [
        "clip was edited in REAPER but its ARA geometry has not caught up; host Undo remains available",
        "item was split in REAPER but both clips have not caught up; host Undo remains available",
        "item was created in REAPER but its ARA audio is not ready; use host Undo if canceling",
        "Conflict: host changed; local curves preserved, host is not ready",
        "host clip changed before GUI commit; refresh required",
        "native host edit request budget exceeded",
        "host editor lease revoked",
    ]) {
        assertEqual(
            classifyClipEditFailure(message),
            "transient",
            `瞬时失败不得提示用户：${message}`,
        );
    }

    // 需要用户处置的失败仍然必须可见。
    for (const message of [
        "host clip property is not yet writable: color",
        "nonlinear host stretch-marker editing is not supported",
        "unknown editor clip",
        "clip has no unique directly bound REAPER take",
    ]) {
        assertEqual(
            classifyClipEditFailure(message),
            "permanent",
            `需要处置的失败必须提示：${message}`,
        );
    }

    // 未知/空文案默认按"永久"：宁可多报一次，也不能把真正的失败静默掉。
    assertEqual(classifyClipEditFailure("something new went wrong"), "permanent", "未知文案");
    assertEqual(classifyClipEditFailure(""), "permanent", "空文案");
    assertEqual(classifyClipEditFailure(undefined), "permanent", "缺失文案");

    // ── 结构化入参（thunk 的 rejectWithValue 载荷）───────────────────────
    //
    // 【为什么这四条必须一起存在】真缺陷的根因是"调用方传的是通用包装文案"，
    // 于是白名单永远匹配不到；修法是把后端原文与协议分类一起带下来。
    // 第 3 条是**防修过头**的闸：协议的 kind 只说"会话是否还能继续"，
    // 校验失败也是 transient —— 若按 kind 静默，就会把"你的输入不被支持"一起吞掉。

    // 1) 会话已结束 ⇒ 必须提示。
    assertEqual(
        classifyClipEditFailure({ kind: "terminal", detail: "document closed" }),
        "permanent",
        "终态错误必须可见",
    );

    // 2) 后端原文里的瞬时冲突 ⇒ 静默（后端自己会收敛）。
    assertEqual(
        classifyClipEditFailure({
            kind: "transient",
            message: "Backend invoke failed: plugin:set_clip_state",
            detail: "host clip edit failed (Undo block retained): host clip changed before GUI commit; refresh required (host start=1.0 length=2.0, editor start=1.0 length=2.0)",
        }),
        "transient",
        "后端原文的瞬时冲突不得提示",
    );

    // 3) 校验失败虽然也是 transient，但必须提示。
    assertEqual(
        classifyClipEditFailure({ kind: "transient", detail: "invalid host channel mode" }),
        "permanent",
        "校验失败必须仍然可见（不能按 kind 一刀切）",
    );

    // 4) 只有包装文案（旧后端 / 取不到 cause）⇒ 保守判定为永久。
    assertEqual(
        classifyClipEditFailure({ message: "Backend invoke failed: plugin:set_clip_state" }),
        "permanent",
        "取不到原文时按永久处理",
    );
});
