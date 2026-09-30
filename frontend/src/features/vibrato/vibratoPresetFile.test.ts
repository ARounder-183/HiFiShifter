/**
 * 颤音预设文件格式（导入 / 导出）的契约。
 *
 * 【往返是底线】导出的文件必须能原样导回 —— 这是"备份"一词的最低要求。
 * 其余每条失败分类都对应一种真实的用户失误（拿错文件 / 文件比应用新 /
 * 手改坏了），错误分得清，提示才能说得准。
 */
import { describe, expect, it } from "vitest";

import { sanitizeVibratoPreset } from "./vibratoPresets";
import {
    mergeImportedPresets,
    parseVibratoPresets,
    serializeVibratoPresets,
    VIBRATO_PRESET_FILE_KIND,
    VIBRATO_PRESET_FILE_VERSION,
    vibratoPresetFileName,
} from "./vibratoPresetFile";

function samplePreset(name: string, overrides = {}) {
    return sanitizeVibratoPreset({ id: "custom_x", name, depthCents: 42, ...overrides });
}

describe("serializeVibratoPresets / parseVibratoPresets：往返", () => {
    it("导出 → 导入得到参数等价的预设（id 重新生成）", () => {
        const presets = [samplePreset("A"), samplePreset("B", { depthCents: 80 })];
        const text = serializeVibratoPresets(presets);
        const result = parseVibratoPresets(text);
        expect(result.ok).toBe(true);
        if (!result.ok) return;

        expect(result.presets).toHaveLength(2);
        for (let i = 0; i < presets.length; i += 1) {
            const original = presets[i];
            const parsed = result.presets[i];
            // id 必须重生成 —— 文件里的 id 可能与本地撞车（按 id 覆盖会替换掉
            // 用户已有的预设）。
            expect(parsed.id).not.toBe(original.id);
            expect(parsed.id.startsWith("custom_")).toBe(true);
            // 其余参数逐字段等价。
            const strip = (p: typeof original) => {
                const { id: _i, ...rest } = p;
                void _i;
                return rest;
            };
            expect(strip(parsed)).toEqual(strip(original));
        }
    });

    it("头部标记正确（kind + version）", () => {
        const parsed = JSON.parse(serializeVibratoPresets([samplePreset("A")])) as Record<
            string,
            unknown
        >;
        expect(parsed.kind).toBe(VIBRATO_PRESET_FILE_KIND);
        expect(parsed.version).toBe(VIBRATO_PRESET_FILE_VERSION);
    });

    it("二次往返参数不再变化（id 每次都重生成，故剥掉后比较）", () => {
        const once = parseVibratoPresets(serializeVibratoPresets([samplePreset("A")]));
        expect(once.ok).toBe(true);
        if (!once.ok) return;
        const twice = parseVibratoPresets(serializeVibratoPresets(once.presets));
        expect(twice.ok).toBe(true);
        if (!twice.ok) return;
        const strip = (presets: typeof once.presets) =>
            presets.map(({ id: _i, ...rest }) => {
                void _i;
                return rest;
            });
        expect(strip(twice.presets)).toEqual(strip(once.presets));
    });
});

describe("parseVibratoPresets：拒绝", () => {
    it("坏 JSON → badJson", () => {
        expect(parseVibratoPresets("{ not json")).toEqual({ ok: false, error: "badJson" });
    });

    it("kind 不匹配 → wrongKind（拿错文件：布局 / 主题都是 .json）", () => {
        expect(
            parseVibratoPresets(
                JSON.stringify({ kind: "something-else", version: 1, presets: [] }),
            ),
        ).toEqual({
            ok: false,
            error: "wrongKind",
        });
        // 没有 kind 字段的文件同样拒收 —— 主题 / 布局导出就没有这个字段。
        expect(parseVibratoPresets(JSON.stringify({ name: "A", colors: {} }))).toEqual({
            ok: false,
            error: "wrongKind",
        });
    });

    it("version 比应用新 → newerVersion（提示语应当是『升级』，不是『文件不对』）", () => {
        const text = JSON.stringify({
            kind: VIBRATO_PRESET_FILE_KIND,
            version: VIBRATO_PRESET_FILE_VERSION + 1,
            presets: [],
        });
        expect(parseVibratoPresets(text)).toEqual({ ok: false, error: "newerVersion" });
    });

    it("version 缺失或非整数 → wrongKind", () => {
        const base = { kind: VIBRATO_PRESET_FILE_KIND };
        expect(parseVibratoPresets(JSON.stringify({ ...base, presets: [] }))).toEqual({
            ok: false,
            error: "wrongKind",
        });
        expect(parseVibratoPresets(JSON.stringify({ ...base, version: "1", presets: [] }))).toEqual(
            {
                ok: false,
                error: "wrongKind",
            },
        );
    });

    it("presets 缺失或不是数组 → noPresets", () => {
        const base = { kind: VIBRATO_PRESET_FILE_KIND, version: 1 };
        expect(parseVibratoPresets(JSON.stringify(base))).toEqual({
            ok: false,
            error: "noPresets",
        });
        expect(parseVibratoPresets(JSON.stringify({ ...base, presets: "nope" }))).toEqual({
            ok: false,
            error: "noPresets",
        });
    });

    it("空数组是合法文件但 0 条（UI 说『没有预设』，不当成功）", () => {
        const result = parseVibratoPresets(
            JSON.stringify({ kind: VIBRATO_PRESET_FILE_KIND, version: 1, presets: [] }),
        );
        expect(result).toEqual({ ok: true, presets: [] });
    });
});

describe("parseVibratoPresets：净化与 id 重写", () => {
    it("文件里的 builtin. 前缀 id 被重写为新的用户 id（否则导入会静默丢条目）", () => {
        const text = JSON.stringify({
            kind: VIBRATO_PRESET_FILE_KIND,
            version: 1,
            presets: [{ id: "builtin.natural", name: "Stolen", depthCents: 30 }],
        });
        const result = parseVibratoPresets(text);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.presets[0]?.id.startsWith("custom_")).toBe(true);
        // sanitize 把 builtin 前缀标成 builtin:true；重写 id 后必须是 false，
        // 否则 upsert 会拒收（静默丢条目）。
        expect(result.presets[0]?.builtin).toBe(false);
    });

    it("越界数值被钳进合法值域（与设置加载同一条净化路径）", () => {
        const text = JSON.stringify({
            kind: VIBRATO_PRESET_FILE_KIND,
            version: 1,
            presets: [{ id: "custom_a", name: "Crazy", depthCents: 99_999, rateHz: 0 }],
        });
        const result = parseVibratoPresets(text);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.presets[0]?.depthCents).toBeLessThanOrEqual(1200);
        expect(result.presets[0]?.rateHz).toBeGreaterThan(0);
    });

    it("id 每次导入都重新生成：同一文件导两次产生两个独立预设", () => {
        const text = serializeVibratoPresets([samplePreset("A")]);
        const first = parseVibratoPresets(text);
        const second = parseVibratoPresets(text);
        expect(first.ok && second.ok).toBe(true);
        if (!(first.ok && second.ok)) return;
        expect(first.presets[0]?.id).not.toBe(second.presets[0]?.id);
    });
});

describe("mergeImportedPresets", () => {
    it("同一签名跳过：重复导入是幂等的", () => {
        const existing = [samplePreset("A")];
        const incoming = [samplePreset("A"), samplePreset("B")];
        const { imported, skipped } = mergeImportedPresets(incoming, existing, 10);
        expect(imported.map((p) => p.name)).toEqual(["B"]);
        expect(skipped).toBe(1);
    });

    it("名字撞了但参数不同仍导入（删哪个由用户决定）", () => {
        const existing = [samplePreset("A", { depthCents: 10 })];
        const { imported, skipped } = mergeImportedPresets(
            [samplePreset("A", { depthCents: 90 })],
            existing,
            10,
        );
        expect(imported).toHaveLength(1);
        expect(skipped).toBe(0);
    });

    it("一批之内互相重复的也只收一条", () => {
        const { imported, skipped } = mergeImportedPresets(
            [samplePreset("A"), samplePreset("A")],
            [],
            10,
        );
        expect(imported).toHaveLength(1);
        expect(skipped).toBe(1);
    });

    it("容量截断给出准确数字（静默丢弃不如说清楚）", () => {
        const incoming = [samplePreset("A"), samplePreset("B"), samplePreset("C")];
        const { imported, skipped } = mergeImportedPresets(incoming, [], 2);
        expect(imported).toHaveLength(2);
        expect(skipped).toBe(1);
    });

    it("不修改传入的数组", () => {
        const existing = [samplePreset("A")];
        const snapshot = existing.map((p) => p.id);
        mergeImportedPresets([samplePreset("B")], existing, 10);
        expect(existing.map((p) => p.id)).toEqual(snapshot);
    });
});

describe("vibratoPresetFileName", () => {
    it("非法字符被替换，控制字符被去掉", () => {
        // 非法字符被**剔除**（而不是替换成空格再压缩）—— 名字更紧凑。
        expect(vibratoPresetFileName({ name: '我的/颤音:v1?"' })).toBe(
            "hifishifter-vibrato-我的颤音v1.json",
        );
    });

    it("空白压缩成连字符；空名回退 preset", () => {
        expect(vibratoPresetFileName({ name: "  deep   vibrato  " })).toBe(
            "hifishifter-vibrato-deep-vibrato.json",
        );
        expect(vibratoPresetFileName({ name: "???" })).toBe("hifishifter-vibrato-preset.json");
    });
});
