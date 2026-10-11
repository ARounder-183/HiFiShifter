/**
 * Clip 浮标名称行：**永不**出现路径。
 *
 * 【为什么值得一条测试】用户报障"ARA 里 Clip 名称浮标太长，因为是完整路径字符串"——
 * 根因是浮标取了 `clip.sourcePath`，而插件里那是物化后的私有 PCM 绝对路径
 * （`…\pcm\<namespace>\sources\source-<hash>.wav`）。这条测试把"输出里不含路径"
 * 钉成不变量，免得将来有人"顺手把路径加回来"。
 */
import { expect, test } from "vitest";
import { clipHasNonlinearStretch, clipHasTakeEnvelope, clipTooltipTitle } from "./sessionTypes";

test("uses the clip name, never the source path", () => {
    const title = clipTooltipTitle({
        name: "vocal take 01",
        sourcePath: "C:\\Users\\someone\\Music\\sessions\\vocal take 01.wav",
    });
    expect(title).toBe("vocal take 01");
    expect(title).not.toContain("\\");
    expect(title).not.toContain("/");
});

test("keeps the take index suffix for multi-take clips", () => {
    expect(
        clipTooltipTitle({
            name: "container",
            takes: [
                { id: "t1", name: "Take A" },
                { id: "t2", name: "Take B" },
            ],
            activeTakeId: "t2",
            sourcePath: "D:\\audio\\container.wav",
        }),
    ).toBe("Take B (2 / 2)");
});

test("falls back to the source basename when the name is empty — still not a path", () => {
    const title = clipTooltipTitle({
        name: "   ",
        sourcePath:
            "C:\\Users\\someone\\AppData\\Local\\hifishifter\\pcm\\ns\\sources\\source-abc.wav",
    });
    expect(title).toBe("source-abc.wav");
    expect(title).not.toContain("C:");
    expect(title).not.toContain("\\");
});

test("returns an empty label rather than noise when nothing is available", () => {
    expect(clipTooltipTitle({ name: "" })).toBe("");
    expect(clipTooltipTitle({ name: "", sourcePath: "" })).toBe("");
});

/**
 * 只读提示的判据：**看 active take**，不是看任意 take。
 *
 * 【为什么值得一条测试】多 take item 里 inactive take 的标记与本 take 无关；
 * 若把任一 take 的标记算作"整条 clip 只读"，用户切到干净 take 时仍会被挡住。
 */
test("nonlinear stretch is read from the active take only", () => {
    const clip = {
        name: "container",
        activeTakeId: "t2",
        takes: [
            { id: "t1", stretchMarkers: [{ position: 1 }] },
            { id: "t2", stretchMarkers: [] },
        ],
    };
    expect(clipHasNonlinearStretch(clip)).toBe(false);
    expect(clipHasNonlinearStretch({ ...clip, activeTakeId: "t1" })).toBe(true);
    // 缺 activeTakeId ⇒ 退回第一个 take。
    expect(clipHasNonlinearStretch({ ...clip, activeTakeId: undefined })).toBe(true);
    expect(clipHasNonlinearStretch({})).toBe(false);
});

test("take envelopes are detected from the active take", () => {
    const clip = {
        name: "container",
        activeTakeId: "t2",
        takes: [{ id: "t1", envelopes: { volume: [1, 0.5] } }, { id: "t2" }],
    };
    expect(clipHasTakeEnvelope(clip)).toBe(false);
    expect(clipHasTakeEnvelope({ ...clip, activeTakeId: "t1" })).toBe(true);
    expect(clipHasTakeEnvelope({})).toBe(false);
});
