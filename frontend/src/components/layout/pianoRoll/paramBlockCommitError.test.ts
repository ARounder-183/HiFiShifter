/**
 * 失败分类（./paramBlockCommitError）自检。
 *
 * 【本测试守护什么】分类必须按**协议前缀**，不能按 `includes` 猜文案：
 * 猜文案的版本一旦后端改了措辞，冲突就会被当成"未知错误"而不再重试 —— 用户
 * 看到的正是那种"有时能重试、有时直接红条"的不一致。
 */
import { describe, expect, it } from "vitest";

import {
    classifyParamBlockFailure,
    isRetryableParamBlockFailure,
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

    it("treats any other Conflict: prefix as an unknown conflict (still retryable)", () => {
        const other = classifyParamBlockFailure(new Error("Conflict: something else"));
        expect(other).toMatchObject({ kind: "conflict", code: "unknown" });
        expect(isRetryableParamBlockFailure(other)).toBe(true);
    });

    it("maps Invalid: to rejected with a stable code", () => {
        const rejected = classifyParamBlockFailure(
            new Error("Invalid: parameter block overlapping members"),
        );
        expect(rejected).toMatchObject({
            kind: "rejected",
            code: "parameter_block_overlapping_members",
        });
        expect(isRetryableParamBlockFailure(rejected)).toBe(false);
        expect(paramBlockFailureFeedback(rejected)).toBe("rejected");
    });

    it("maps cancellations to cancelled and stays silent", () => {
        const cancelled = classifyParamBlockFailure(
            new Error("Block drag commit cancelled: parameter blocks changed during read"),
        );
        expect(cancelled.kind).toBe("cancelled");
        expect(paramBlockFailureFeedback(cancelled)).toBe("silent");
        expect(isRetryableParamBlockFailure(cancelled)).toBe(false);
    });

    it("treats everything else as transport", () => {
        const transport = classifyParamBlockFailure(new Error("ipc disconnected"));
        expect(transport).toMatchObject({ kind: "transport", code: "transport" });
        expect(isRetryableParamBlockFailure(transport)).toBe(true);
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
