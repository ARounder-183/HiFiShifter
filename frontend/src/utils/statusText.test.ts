import { describe, expect, it } from "vitest";

import { resolveStatusText } from "./statusText";

/**
 * 用真实 i18n 键的形状做替身：模板里带 `{n}` / `{m}` / `{p}`，这样能同时验证
 * "选对了键"与"占位符都回填了"。
 */
const STATUS_KEY = {
    "Fake-stereo scan rejected": "status_fake_stereo_scan_rejected",
    "Waveform cache cleared": "status_waveform_cache_cleared",
    "Export done": "status_export_done",
};

const TEMPLATES: Record<string, string> = {
    status_fake_stereo_scan_rejected: "假立体声扫描被拒绝",
    status_waveform_cache_cleared: "波形缓存已清空（{n} 个文件）",
    status_export_done: "导出完成",
    status_fake_stereo_scan_foldable: "假立体声扫描：{n} 个 Take，{m} 个可折叠",
    status_fake_stereo_scan_folded: "假立体声扫描：{n} 个 Take，{m} 个已折叠为单声道",
    status_fake_stereo_scan_foldable_pending:
        "假立体声扫描：{n} 个 Take，{m} 个可折叠，{p} 个本次读不到",
    status_fake_stereo_scan_folded_pending:
        "假立体声扫描：{n} 个 Take，{m} 个已折叠为单声道，{p} 个本次读不到",
    status_fake_stereo_scan_nearest_hint: "；最接近的一条差异为 {d}，可考虑放宽容差",
};

const t = (key: string) => TEMPLATES[key] ?? `MISSING:${key}`;

const resolve = (status: string) => resolveStatusText(status, STATUS_KEY, t);

describe("resolveStatusText", () => {
    it("translates an exact status", () => {
        expect(resolve("Fake-stereo scan rejected")).toBe("假立体声扫描被拒绝");
    });

    it("fills the single-count template", () => {
        expect(resolve("Waveform cache cleared (3 files)")).toBe(
            "波形缓存已清空（3 个文件）",
        );
    });

    it("keeps a suffix appended to a prefix-matched status", () => {
        expect(resolve("Export done — C:/out.wav")).toBe("导出完成 — C:/out.wav");
    });

    it("returns the raw status when nothing matches", () => {
        expect(resolve("some brand new status")).toBe("some brand new status");
    });

    describe("fake-stereo scan", () => {
        it("renders the dry-run variant without an unreadable suffix", () => {
            expect(resolve("Fake-stereo scan: 5 take(s), 3 foldable")).toBe(
                "假立体声扫描：5 个 Take，3 个可折叠",
            );
        });

        it("renders the applied variant without an unreadable suffix", () => {
            expect(resolve("Fake-stereo scan: 5 take(s), 3 folded to mono")).toBe(
                "假立体声扫描：5 个 Take，3 个已折叠为单声道",
            );
        });

        it("renders the unreadable suffix when sources could not be read", () => {
            // "本次读不到"必须显性出现：那些 Take 会在下次打开时自动重试，
            // 与"单声道、无事可做"是两回事。
            expect(resolve("Fake-stereo scan: 5 take(s), 3 folded to mono, 2 unreadable")).toBe(
                "假立体声扫描：5 个 Take，3 个已折叠为单声道，2 个本次读不到",
            );
            expect(resolve("Fake-stereo scan: 5 take(s), 3 foldable, 2 unreadable")).toBe(
                "假立体声扫描：5 个 Take，3 个可折叠，2 个本次读不到",
            );
        });

        it("leaves no placeholder unfilled", () => {
            for (const status of [
                "Fake-stereo scan: 5 take(s), 3 foldable",
                "Fake-stereo scan: 5 take(s), 3 folded to mono",
                "Fake-stereo scan: 5 take(s), 3 foldable, 2 unreadable",
                "Fake-stereo scan: 5 take(s), 3 folded to mono, 2 unreadable",
            ]) {
                expect(resolve(status)).not.toMatch(/\{[nmp]\}/);
            }
        });

        it("appends the nearest-difference hint as a separate sentence", () => {
            expect(resolve("Fake-stereo scan: 12 take(s), 0 folded to mono — nearest 0.0021")).toBe(
                "假立体声扫描：12 个 Take，0 个已折叠为单声道；最接近的一条差异为 0.0021，可考虑放宽容差",
            );
        });

        it("does not mix a raw English sentence with a translated hint", () => {
            // 前缀无法翻译时，宁可不加提示，也不要拼出中英夹杂的状态行。
            expect(resolve("Totally unknown status — nearest 0.5")).toBe(
                "Totally unknown status — nearest 0.5",
            );
        });
    });
});
