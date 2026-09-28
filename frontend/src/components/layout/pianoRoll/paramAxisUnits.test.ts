/**
 * 纵轴展示单位（倍率 ↔ dB）的换算与解析。
 *
 * 重点覆盖两类容易出错的地方：
 * 1. **dB 读数的边界**：1× 必须恰为 0、2× 为 +6、0× 为 −∞（不可写成有限数，
 *    否则刻度会把"静音"标成某个具体电平）；
 * 2. **设置的宽容度**：手改过的设置文件（未知参数名 / 大小写变体 / 非字符串）
 *    必须被丢弃而不是让界面落进一个"设置了但没生效"的状态。
 */

import { describe, expect, it } from "vitest";

import {
    DEFAULT_PARAM_AXIS_UNIT,
    formatDbLabel,
    formatDbReadout,
    nextParamAxisUnit,
    normalizeParamAxisUnits,
    ratioToDb,
    resolveParamAxisUnit,
    supportsParamAxisUnit,
    type ParamAxisUnits,
} from "./paramAxisUnits";
import { formatAxisMarkLabel } from "./kernel/scene/axisMarkInstances";

describe("supportsParamAxisUnit", () => {
    it("accepts the linear-amplitude params (volume / dyn, incl. legacy ids)", () => {
        expect(supportsParamAxisUnit("volume")).toBe(true);
        expect(supportsParamAxisUnit("dyn")).toBe(true);
        expect(supportsParamAxisUnit("dyn_edit")).toBe(true);
        expect(supportsParamAxisUnit("hifigan_volume")).toBe(true);
    });

    it("rejects params whose unit is fixed by their semantics", () => {
        expect(supportsParamAxisUnit("pitch")).toBe(false);
        expect(supportsParamAxisUnit("tension")).toBe(false);
        expect(supportsParamAxisUnit("formant_shift_cents")).toBe(false);
        expect(supportsParamAxisUnit("")).toBe(false);
        expect(supportsParamAxisUnit(null)).toBe(false);
        expect(supportsParamAxisUnit(undefined)).toBe(false);
    });
});

describe("resolveParamAxisUnit", () => {
    it("defaults to ratio when unset or unknown", () => {
        expect(DEFAULT_PARAM_AXIS_UNIT).toBe("ratio");
        expect(resolveParamAxisUnit(undefined, "dyn")).toBe("ratio");
        expect(resolveParamAxisUnit({}, "dyn")).toBe("ratio");
        // 手改过的设置文件（值不是合法枚举）同样必须落到倍率，绝不静默错位。
        expect(
            resolveParamAxisUnit({ volume: "bogus" } as unknown as ParamAxisUnits, "volume"),
        ).toBe("ratio");
    });

    it("reads the stored value per param", () => {
        expect(resolveParamAxisUnit({ volume: "db", dyn: "ratio" }, "volume")).toBe("db");
        expect(resolveParamAxisUnit({ volume: "db", dyn: "ratio" }, "dyn")).toBe("ratio");
    });
});

describe("nextParamAxisUnit", () => {
    it("round-trips", () => {
        expect(nextParamAxisUnit("ratio")).toBe("db");
        expect(nextParamAxisUnit("db")).toBe("ratio");
        expect(nextParamAxisUnit(nextParamAxisUnit("ratio"))).toBe("ratio");
    });
});

describe("normalizeParamAxisUnits", () => {
    it("keeps only valid params and values", () => {
        expect(
            normalizeParamAxisUnits({
                volume: "db",
                dyn: "ratio",
                pitch: "db", // 不支持切换的参数
                tension: "db", // 同上
                volume_legacy: "db", // 未知参数
            }),
        ).toEqual({ volume: "db", dyn: "ratio" });
    });

    it("drops invalid values and non-object payloads", () => {
        expect(normalizeParamAxisUnits({ volume: "dB" })).toEqual({});
        expect(normalizeParamAxisUnits({ volume: true })).toEqual({});
        expect(normalizeParamAxisUnits(null)).toEqual({});
        expect(normalizeParamAxisUnits("db")).toEqual({});
        expect(normalizeParamAxisUnits(7)).toEqual({});
    });
});

describe("ratioToDb / formatDbLabel", () => {
    it("anchors 1× at 0 dB and 2× at +6 dB", () => {
        expect(ratioToDb(1)).toBe(0);
        expect(ratioToDb(2)).toBeCloseTo(6.0206, 3);
        expect(ratioToDb(0.5)).toBeCloseTo(-6.0206, 3);
    });

    it("maps silence to -Infinity, not a finite level", () => {
        expect(ratioToDb(0)).toBe(Number.NEGATIVE_INFINITY);
        expect(ratioToDb(-1)).toBe(Number.NEGATIVE_INFINITY);
        expect(ratioToDb(Number.NaN)).toBe(Number.NEGATIVE_INFINITY);
        expect(formatDbLabel(0)).toBe("-∞");
    });

    it("formats tick labels compactly (1 decimal, trailing .0 trimmed)", () => {
        expect(formatDbLabel(1)).toBe("0");
        expect(formatDbLabel(0.5)).toBe("-6");
        expect(formatDbLabel(0.25)).toBe("-12");
        expect(formatDbLabel(0.1)).toBe("-20");
        expect(formatDbLabel(2)).toBe("+6");
        expect(formatDbLabel(1.5)).toBe("+3.5");
        expect(formatDbLabel(0.75)).toBe("-2.5");
    });

    it("appends the unit in readouts so dB is not mistaken for a ratio", () => {
        expect(formatDbReadout(0.5)).toBe("-6 dB");
        expect(formatDbReadout(1)).toBe("0 dB");
        expect(formatDbReadout(0)).toBe("-∞ dB");
    });
});

describe("formatAxisMarkLabel with the axis unit", () => {
    it("uses the dB reading for volume / dyn when the unit is db", () => {
        expect(formatAxisMarkLabel(1, "volume", "db")).toBe("0");
        expect(formatAxisMarkLabel(0.5, "dyn", "db")).toBe("-6");
        expect(formatAxisMarkLabel(0, "dyn", "db")).toBe("-∞");
    });

    it("keeps the historical ratio reading when the unit is ratio / unset", () => {
        expect(formatAxisMarkLabel(0.5, "dyn")).toBe("0.5");
        expect(formatAxisMarkLabel(0.5, "dyn", "ratio")).toBe("0.5");
        expect(formatAxisMarkLabel(0, "dyn", "ratio")).toBe("0");
        expect(formatAxisMarkLabel(1.5, "volume", "ratio")).toBe("1.5");
        // 1.0 是满量程：倍率读数写 1，dB 读数写 0 —— 两者都是各自坐标系的原点。
        expect(formatAxisMarkLabel(1, "volume", "ratio")).toBe("1");
    });

    it("never rewrites params whose unit cannot be switched", () => {
        // 张力 / 音分没有"倍率或 dB"两种读法：即便设置里混进了 db，也必须原样。
        expect(formatAxisMarkLabel(-50, "tension", "db")).toBe("-50");
        expect(formatAxisMarkLabel(1200, "formant_shift_cents", "db")).toBe("1200");
    });
});
