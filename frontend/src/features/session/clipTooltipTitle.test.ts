/**
 * Clip 浮标名称行：**永不**出现路径。
 *
 * 【为什么值得一条测试】用户报障"ARA 里 Clip 名称浮标太长，因为是完整路径字符串"——
 * 根因是浮标取了 `clip.sourcePath`，而插件里那是物化后的私有 PCM 绝对路径
 * （`…\pcm\<namespace>\sources\source-<hash>.wav`）。这条测试把"输出里不含路径"
 * 钉成不变量，免得将来有人"顺手把路径加回来"。
 */
import { expect, test } from "vitest";
import { clipTooltipTitle } from "./sessionTypes";

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
