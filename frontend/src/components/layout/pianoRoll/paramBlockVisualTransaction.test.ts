import { describe, expect, it } from "vitest";
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import {
    adoptParamBlockVisualStructure,
    isParamBlockVisualEpochCurrent,
    sameParamBlockEpochLayout,
    sameParamBlockExternalEpoch,
    shouldUseParamBlockVisualTransaction,
    type ParamBlockEpoch,
    type ParamBlockVisualTransaction,
} from "./paramBlockVisualTransaction";

const epoch: ParamBlockEpoch = {
    rootTrackId: "root",
    param: "pitch",
    layoutKey: "layout-a",
    structureToken: "structure-1",
    externalHistoryToken: "0:0",
    framePeriodMs: 5,
};

function spec(): PianoRollParamBlockSpec {
    return {
        param: "pitch",
        valueMin: 0,
        valueMax: 127,
        blocks: [],
        borderRgba: [0, 0, 0, 1],
        selectedRgba: [1, 1, 1, 1],
        groupedRgba: [1, 0, 0, 1],
    };
}

function tx(overrides: Partial<ParamBlockVisualTransaction> = {}): ParamBlockVisualTransaction {
    return {
        id: 1,
        epoch,
        kind: "resize",
        spec: spec(),
        startedAt: 0,
        ...overrides,
    };
}

describe("paramBlockVisualTransaction", () => {
    it("rejects a transaction from another layout or parameter", () => {
        expect(
            shouldUseParamBlockVisualTransaction(tx(), { ...epoch, layoutKey: "layout-b" }),
        ).toBe(false);
        expect(shouldUseParamBlockVisualTransaction(tx(), { ...epoch, param: "volume" })).toBe(
            false,
        );
        expect(shouldUseParamBlockVisualTransaction(tx(), epoch)).toBe(true);
    });

    it("keeps drag layout valid across a self commit but not an external jump", () => {
        expect(sameParamBlockEpochLayout(epoch, { ...epoch })).toBe(true);
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, externalHistoryToken: "1:0" })).toBe(
            false,
        );
        expect(sameParamBlockExternalEpoch(epoch, { ...epoch, param: "volume" })).toBe(false);
    });

    /*
     * 【结构令牌取代 revision 的回归锁】
     * - 自己的 value 变更（块中心 / 曲线）不改结构 ⇒ 预览继续有效；
     * - 第三方几何变更（分割 / 伸缩 / 换 Clip）改结构 ⇒ 预览立即失效。
     * 旧实现比 revision 加 ownedEditRevision，而后者永远对不上服务端回包 ——
     * 于是自己提交的回包一落地就撤掉自己的预览（「先回弹」）。
     */
    it("adopts only the structure token after a local create commit", () => {
        const before = tx({ kind: "create" });
        const adopted = adoptParamBlockVisualStructure(before, "structure-2");
        expect(adopted).not.toBeNull();
        if (adopted === null) return;
        expect(adopted.epoch.structureToken).toBe("structure-2");
        expect(adopted.epoch.layoutKey).toBe(before.epoch.layoutKey);
        expect(adopted.epoch.externalHistoryToken).toBe(before.epoch.externalHistoryToken);
        expect(adopted.epoch.param).toBe(before.epoch.param);
        expect(adopted.epoch.framePeriodMs).toBe(before.epoch.framePeriodMs);
        expect(adopted.spec).toBe(before.spec);
        expect(adopted.kind).toBe("create");
        expect(adopted.id).toBe(before.id);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch: adopted.epoch,
                current: { ...epoch, structureToken: "structure-2" },
            }),
        ).toBe(true);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch: adopted.epoch,
                current: { ...epoch, structureToken: "structure-3" },
            }),
        ).toBe(false);
    });

    it("returns null when there is no transaction", () => {
        expect(adoptParamBlockVisualStructure(null, "structure-2")).toBeNull();
    });

    it("★ 自己的值变更不失效，第三方结构变更立即失效", () => {
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch },
            }),
        ).toBe(true);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch, structureToken: "structure-2" },
            }),
        ).toBe(false);
        // 布局 / 历史 / 参数变化仍然立即失效（与结构令牌无关）。
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch, externalHistoryToken: "1:0" },
            }),
        ).toBe(false);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch, layoutKey: "layout-b" },
            }),
        ).toBe(false);
    });
});
