/*
 * 被拒绝的 thunk 显示什么原因。
 *
 * 【为什么必须有】`createAsyncThunk` 配 `rejectWithValue` 时真正的原因在
 * `action.payload`，而 `action.error.message` 恒为字面量 `"Rejected"`。此前状态栏
 * 直接取后者，于是用户看到的是「错误：Rejected」—— 没有任务名、没有原因、没有可
 * 行动的信息（用户报障：点"显示速度映射"就得到这一条）。
 */
import { describe, expect, test } from "vitest";

import { rejectedMessage } from "./sessionSlice";

describe("rejectedMessage", () => {
    test("rejectWithValue 的字符串原因优先于 RTK 的占位符", () => {
        expect(
            rejectedMessage({ payload: "宿主拒绝了这条命令", error: { message: "Rejected" } }),
        ).toBe("宿主拒绝了这条命令");
    });

    test("对象载荷取它的 message 字段", () => {
        expect(
            rejectedMessage({ payload: { message: "tempo_map_commit_failed" }, error: {} }),
        ).toBe("tempo_map_commit_failed");
    });

    test("没有可用载荷时退回真实错误；`Rejected` 不算原因", () => {
        expect(
            rejectedMessage({ error: { message: "Command unavailable in ARA plugin mode" } }),
        ).toBe("Command unavailable in ARA plugin mode");
        // 占位符与空串都等于"什么都没说"。
        expect(rejectedMessage({ error: { message: "Rejected" } })).toBe("Request failed");
        expect(rejectedMessage({})).toBe("Request failed");
    });

    test("空白字符串不被当成原因", () => {
        expect(rejectedMessage({ payload: "   ", error: { message: "Rejected" } })).toBe(
            "Request failed",
        );
    });
});
