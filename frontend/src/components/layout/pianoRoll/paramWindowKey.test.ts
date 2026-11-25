import { describe, expect, it } from "vitest";

import { paramScopeParamId, paramWindowScope, parseParamWindowKey } from "./paramWindowKey";

/**
 * 窗口键解析。
 *
 * 【为什么只剩这些】这个模块此前装着整套"live 覆盖按窗口下标寻址 + 换窗口重锚"的逻辑，
 * 笔画改成绝对帧缓冲后只剩**作用域**这一个用途：它决定"换参数"（该丢弃笔画）与
 * "换窗口"（什么都不该发生）的区别。键的格式由 usePianoRollData 生成，解析口径必须
 * 与之一致 —— 差一个字段就会让"换参数"被当成"换窗口"，正在画的笔画留在新参数上。
 */
describe("parseParamWindowKey", () => {
    it("splits the key into scope, start frame and stride", () => {
        const parsed = parseParamWindowKey("v2|track-1|volume|1200|6400|1");
        expect(parsed.scope).toBe("v2|track-1|volume");
        expect(parsed.startFrame).toBe(1200);
        expect(parsed.stride).toBe(1);
    });

    it("keeps two windows of the same parameter in one scope", () => {
        const a = parseParamWindowKey("v2|track-1|dyn|0|6400|1");
        const b = parseParamWindowKey("v2|track-1|dyn|900|3200|2");
        expect(a.scope).toBe(b.scope);
        expect(a.startFrame).not.toBe(b.startFrame);
        expect(a.stride).not.toBe(b.stride);
    });

    it("separates parameters and tracks", () => {
        const base = parseParamWindowKey("v2|track-1|volume|0|6400|1").scope;
        expect(parseParamWindowKey("v2|track-1|dyn|0|6400|1").scope).not.toBe(base);
        expect(parseParamWindowKey("v2|track-2|volume|0|6400|1").scope).not.toBe(base);
    });

    it("falls back to usable defaults for a malformed key", () => {
        const parsed = parseParamWindowKey("garbage");
        expect(parsed.startFrame).toBe(0);
        expect(parsed.stride).toBe(1);
    });
});

describe("paramWindowScope", () => {
    it("is the identity for keys that do not carry the v2 prefix", () => {
        // 历史 / 手写键没有前缀：拆出来的 "a||" 会让两条不同的键撞成同一个作用域。
        expect(paramWindowScope("garbage")).toBe("garbage");
        expect(paramWindowScope("v2|track-1|volume|0|4|1")).toBe("v2|track-1|volume");
    });
});

describe("paramScopeParamId", () => {
    it("takes the last segment", () => {
        expect(paramScopeParamId("v2|track-1|volume")).toBe("volume");
        expect(paramScopeParamId("v2|track-1|dyn_edit")).toBe("dyn_edit");
        expect(paramScopeParamId("volume")).toBe("volume");
    });
});
