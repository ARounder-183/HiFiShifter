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
    status_fake_stereo_scan_no_clips: "工程里没有音频块",
    status_fake_stereo_scan_range_unmatched: "扫描范围与工程对不上（工程共 {p} 个音频块）",
    status_fake_stereo_scan_no_takes: "范围里的 {c} 个音频块没有 Take 记录",
    status_fake_stereo_scan_no_source: "{n} 个 Take 没有音频源",
    status_fake_stereo_scan_nothing: "没有可判定的对象",
    status_fake_stereo_scan_overridden_suffix: "；其中 {n} 个原本由你设置了声道模式，已覆盖",
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

        describe("zero candidates", () => {
            // 这句话必须**按后端返回的真实计数**成句。过去的实现猜了一个原因，
            // 而计数因字段名前后端不一致恒为 0 —— 于是界面随口断言
            // "选中的音频块在工程里找不到"，一句听起来像用户操作有误、
            // 实际完全虚假的话。
            it("distinguishes an empty project", () => {
                expect(resolve("Fake-stereo scan: the project has no clips")).toBe(
                    "工程里没有音频块",
                );
            });

            it("reports the project clip count when the range matches nothing", () => {
                expect(resolve("Fake-stereo scan: range matches no clip (project has 42)")).toBe(
                    "扫描范围与工程对不上（工程共 42 个音频块）",
                );
            });

            it("reports clips that carry no takes", () => {
                expect(resolve("Fake-stereo scan: 3 clip(s) in range have no takes")).toBe(
                    "范围里的 3 个音频块没有 Take 记录",
                );
            });

            it("reports takes with no audio source", () => {
                expect(resolve("Fake-stereo scan: 4 take(s) have no audio source")).toBe(
                    "4 个 Take 没有音频源",
                );
            });

            it("falls back to a neutral sentence", () => {
                expect(resolve("Fake-stereo scan: nothing to decide")).toBe("没有可判定的对象");
            });

            it("leaves no placeholder unfilled", () => {
                for (const status of [
                    "Fake-stereo scan: range matches no clip (project has 42)",
                    "Fake-stereo scan: 3 clip(s) in range have no takes",
                    "Fake-stereo scan: 4 take(s) have no audio source",
                ]) {
                    expect(resolve(status)).not.toMatch(/\{[npu]\}/);
                }
            });
        });

        it("reports how many user-set channel modes the scan overrode", () => {
            expect(
                resolve("Fake-stereo scan: 5 take(s), 3 folded to mono, 2 setting(s) overridden"),
            ).toBe("假立体声扫描：5 个 Take，3 个已折叠为单声道；其中 2 个原本由你设置了声道模式，已覆盖");
        });

    });
});
