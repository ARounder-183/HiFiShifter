/**
 * 系统剪贴板信封的运行时校验（外来进程可以往同一个原生格式槽位写任意 JSON）。
 */
import { describe, expect, test } from "vitest";

import { parseClipboardObject } from "./systemClipboard";

const template = {
    trackId: "track-1",
    name: "vocal.wav",
    startSec: 1.25,
    lengthSec: 2.5,
};

describe("parseClipboardObject", () => {
    test("接受规范的 clip v1 载荷并丢掉未知信封字段", () => {
        const parsed = parseClipboardObject(
            JSON.stringify({
                version: 1,
                kind: "clip",
                templates: [template],
                groupIds: ["group-1"],
                injected: { prototype: "bad" },
            }),
        );
        expect(parsed).toEqual({
            version: 1,
            kind: "clip",
            templates: [template],
            groupIds: ["group-1"],
        });
    });

    test("残缺模板 / 非有限数值 / 非法 groupIds 不得进入粘贴路径", () => {
        expect(
            parseClipboardObject(
                JSON.stringify({ kind: "clip", templates: [{ ...template, startSec: null }] }),
            ),
        ).toBeNull();
        expect(
            parseClipboardObject(
                JSON.stringify({
                    kind: "clip",
                    templates: [{ ...template, playbackRate: Number.POSITIVE_INFINITY }],
                }),
            ),
        ).toBeNull();
        const parsed = parseClipboardObject(
            JSON.stringify({ kind: "clip", templates: [template], groupIds: [42] }),
        );
        expect((parsed as { groupIds?: string[] } | null)?.groupIds).toBeUndefined();
    });

    test("空 templates 不是可粘贴载荷，外来字段按白名单清洗", () => {
        expect(parseClipboardObject(JSON.stringify({ kind: "clip", templates: [] }))).toBeNull();
        const parsed = parseClipboardObject(
            JSON.stringify({
                kind: "clip",
                templates: [
                    {
                        ...template,
                        lengthSec: 0,
                    },
                ],
            }),
        );
        expect(parsed).toBeNull();

        const cleaned = parseClipboardObject(
            JSON.stringify({
                kind: "clip",
                templates: [
                    {
                        ...template,
                        muted: "yes",
                        gain: 2,
                        color: "not-a-color",
                        waveformPreview: [0.1, 0.2],
                        injected: { prototype: "bad" },
                    },
                ],
            }),
        ) as { templates: Record<string, unknown>[] } | null;
        expect(cleaned).not.toBeNull();
        expect(cleaned!.templates[0].gain).toBe(2);
        expect(cleaned!.templates[0].waveformPreview).toEqual([0.1, 0.2]);
        expect(cleaned!.templates[0].muted).toBeUndefined();
        expect(cleaned!.templates[0].color).toBeUndefined();
        expect(cleaned!.templates[0].injected).toBeUndefined();
        expect(
            (
                parseClipboardObject(
                    JSON.stringify({
                        kind: "clip",
                        templates: [{ ...template, loopPeriodSec: null }],
                    }),
                ) as { templates: Record<string, unknown>[] } | null
            )?.templates[0].loopPeriodSec,
        ).toBeNull();
    });

    test("参数线载荷仍走原有归一路径", () => {
        const parsed = parseClipboardObject(
            JSON.stringify({
                kind: "param",
                version: 1,
                param: "pitch",
                framePeriodMs: 5,
                values: [60, 62, null],
            }),
        );
        expect(parsed).not.toBeNull();
        expect("segments" in parsed! && parsed!.segments[0]?.values).toEqual([60, 62, 0]);
    });
});
