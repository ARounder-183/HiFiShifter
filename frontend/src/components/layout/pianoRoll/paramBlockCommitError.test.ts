/**
 * 失败分类（./paramBlockCommitError）自检。
 *
 * 【本测试守护什么】分类必须按**协议前缀**，不能按 `includes` 猜文案：
 * 猜文案的版本一旦后端改了措辞，冲突就会被当成"未知错误"而不再重试 —— 用户
 * 看到的正是那种"有时能重试、有时直接红条"的不一致。
 */
import { describe, expect, it } from "vitest";

import {
    canRebaseParamBlockCommit,
    classifyParamBlockFailure,
    ParamBlockCommitError,
    paramBlockFailureFeedback,
} from "./paramBlockCommitError";

describe("classifyParamBlockFailure", () => {
    it("maps the three CAS conflicts to their codes", () => {
        const edits = classifyParamBlockFailure(
            new Error("Conflict: parameter block edits changed"),
        );
        expect(edits).toMatchObject({ kind: "conflict", code: "edit_set" });
        const curve = classifyParamBlockFailure(new Error("Conflict: parameter curve changed"));
        expect(curve).toMatchObject({ kind: "conflict", code: "curve" });
        const geometry = classifyParamBlockFailure(
            new Error("Conflict: parameter geometry changed"),
        );
        expect(geometry).toMatchObject({ kind: "conflict", code: "geometry" });
    });

    it("treats any other Conflict: prefix as an unknown conflict (still rebasable)", () => {
        const other = classifyParamBlockFailure(new Error("Conflict: something else"));
        expect(other).toMatchObject({ kind: "conflict", code: "unknown" });
        expect(canRebaseParamBlockCommit(other)).toBe(true);
    });

    /*
     * 【重基判据的回归锁】旧实现只允许 `edit_set` 冲突重试，于是 `curve` /
     * `geometry` 冲突一律变成用户可见的失败。现在写队列把「读基准」搬进了队列槽，
     * 重试等于「重新读一遍世界、按同一个意图重算补丁」—— 三类冲突都可以重基。
     */
    it("★ rebases every conflict but never a transport failure", () => {
        for (const message of [
            "Conflict: parameter block edits changed",
            "Conflict: parameter curve changed",
            "Conflict: parameter geometry changed",
            "Conflict: something else",
        ]) {
            expect(canRebaseParamBlockCommit(classifyParamBlockFailure(new Error(message)))).toBe(
                true,
            );
        }
        expect(canRebaseParamBlockCommit(classifyParamBlockFailure(new Error("ipc down")))).toBe(
            false,
        );
        expect(
            canRebaseParamBlockCommit(
                classifyParamBlockFailure(new Error("Invalid: parameter block overlap")),
            ),
        ).toBe(false);
    });

    it("maps Invalid: to rejected with a stable code", () => {
        const rejected = classifyParamBlockFailure(
            new Error("Invalid: parameter block overlapping members"),
        );
        expect(rejected).toMatchObject({
            kind: "rejected",
            code: "parameter_block_overlapping_members",
        });
        expect(canRebaseParamBlockCommit(rejected)).toBe(false);
        expect(paramBlockFailureFeedback(rejected)).toBe("rejected");
    });

    it("maps cancellations to cancelled and stays silent", () => {
        const cancelled = classifyParamBlockFailure(
            new Error("Block drag commit cancelled: parameter blocks changed during read"),
        );
        expect(cancelled.kind).toBe("cancelled");
        expect(paramBlockFailureFeedback(cancelled)).toBe("silent");
        expect(canRebaseParamBlockCommit(cancelled)).toBe(false);
    });

    it("treats everything else as transport", () => {
        const transport = classifyParamBlockFailure(new Error("ipc disconnected"));
        expect(transport).toMatchObject({ kind: "transport", code: "transport" });
        expect(paramBlockFailureFeedback(transport)).toBe("transport");
    });

    it("does not re-wrap an already classified error", () => {
        const original = new ParamBlockCommitError({ kind: "conflict", code: "curve" });
        expect(classifyParamBlockFailure(original)).toBe(original);
    });

    it("accepts non-Error values without throwing", () => {
        expect(classifyParamBlockFailure("boom").kind).toBe("transport");
        expect(classifyParamBlockFailure(undefined).kind).toBe("transport");
    });
});
