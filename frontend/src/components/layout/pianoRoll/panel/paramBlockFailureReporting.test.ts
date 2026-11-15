import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ reportFrontendError: vi.fn() }));

vi.mock("../../../../services/frontendErrorLog", () => ({
    reportFrontendError: mocks.reportFrontendError,
}));

import { ParamBlockCommitError } from "../paramBlockCommitError";
import { paramBlockFeedbackStore, publishParamBlockFeedback } from "../paramBlockFeedback";
import {
    reportParamBlockLocalRejection,
    reportParamBlockWriteFailure,
} from "./paramBlockFailureReporting";

const tf = (key: string) => key;

describe("paramBlockFailureReporting", () => {
    beforeEach(() => {
        mocks.reportFrontendError.mockReset();
        publishParamBlockFeedback(null);
    });

    /*
     * 【单槽覆盖的回归锁】写入层已经按类别发过反馈。调用点再发一条通用红条
     * 会把分类结果覆盖掉 —— 用户永远只看到「参数块操作失败」，而编辑其实成功了。
     */
    it("★ 写入失败只写诊断，不发布任何用户可见反馈", () => {
        reportParamBlockWriteFailure({
            error: new Error("Conflict: parameter curve changed"),
            gesture: "drag",
            rootTrackId: "root",
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toBeNull();
        expect(mocks.reportFrontendError).toHaveBeenCalledTimes(1);
        expect(mocks.reportFrontendError.mock.calls[0][0]).toBe("param_block_drag:curve");
    });

    it("取消写成一类稳定的诊断码，而不是 danger 提示", () => {
        reportParamBlockWriteFailure({
            error: new ParamBlockCommitError({ kind: "cancelled", code: "blocks_changed" }),
            gesture: "drag",
            rootTrackId: "root",
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toBeNull();
        expect(mocks.reportFrontendError.mock.calls[0][0]).toBe("param_block_drag:write_cancelled");
    });

    it("本地拒绝（没有发起写入）才由调用点发一条 warning", () => {
        reportParamBlockLocalRejection({
            code: "create_block_failed",
            detail: "no clip",
            gesture: "create",
            rootTrackId: "root",
            tf,
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toMatchObject({
            text: "status_param_block_edit_rejected",
            tone: "warning",
            tooltip: "no clip",
        });
        expect(mocks.reportFrontendError).toHaveBeenCalledTimes(1);
    });

    it("本地拒绝也写诊断（同一手势同一码限频 1 秒）", () => {
        for (let index = 0; index < 3; index += 1) {
            reportParamBlockLocalRejection({
                code: "split_too_close",
                gesture: "split",
                rootTrackId: "root",
                tf,
            });
        }
        expect(mocks.reportFrontendError).toHaveBeenCalledTimes(1);
    });
});
