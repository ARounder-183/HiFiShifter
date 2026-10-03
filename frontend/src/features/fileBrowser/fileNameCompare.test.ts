import { describe, expect, it } from "vitest";

import { compareFileNames } from "./fileNameCompare";

/** 把一组名字按本比较器排序（不修改入参）。 */
function sorted(names: string[]): string[] {
    return [...names].sort(compareFileNames);
}

/*
 * 用例一律使用合成名字（`song_extra.wav` / `甲.wav` 这类），不引用任何真实目录里的
 * 文件名 —— 排序规则只与字符类别有关，用通用样例同样能钉住每一条。
 */
describe("compareFileNames：与资源管理器对齐的规则", () => {
    it("符号按码点排序：`.` 在 `_` 之前（ICU 的排序表相反）", () => {
        expect(compareFileNames("song.wav", "song_extra.wav")).toBeLessThan(0);
        // `-`(0x2D) 在 `.`(0x2E) 之前，所以 `song-b.wav` 反而排在 `song.wav` 前面。
        expect(compareFileNames("song.wav", "song-b.wav")).toBeGreaterThan(0);
        expect(compareFileNames("song_b.wav", "song-b.wav")).toBeGreaterThan(0); // '_'(0x5F) > '-'(0x2D)
    });

    it("前缀更短的名字排在前面（而不是被标点推到后面）", () => {
        expect(sorted(["song_extra_more.wav", "song.wav", "song_extra.wav"])).toEqual([
            "song.wav",
            "song_extra.wav",
            "song_extra_more.wav",
        ]);
    });

    it("带扩展名与不带扩展名后缀的同类名：短前缀在前", () => {
        expect(sorted(["clip_final.mp4", "clip.mp4"])).toEqual(["clip.mp4", "clip_final.mp4"]);
    });

    it("拉丁排在 CJK 之前（ICU 的 zh 排序相反）", () => {
        expect(compareFileNames("alpha.wav", "汉字.wav")).toBeLessThan(0);
        expect(sorted(["汉字.wav", "alpha.wav", "beta.wav"])).toEqual([
            "alpha.wav",
            "beta.wav",
            "汉字.wav",
        ]);
    });

    it("CJK 之间按语系（拼音）顺序，不是码点顺序", () => {
        // 甲(jiǎ) 在 乙(yǐ) 之前，而码点顺序是 乙(U+4E59) < 甲(U+7532) —— 两者相反，
        // 因此这条能区分"走语系排序器"与"按码点排"。
        expect(compareFileNames("甲.wav", "乙.wav")).toBeLessThan(0);
    });

    it("数字按数值比较（2 在 10 之前）", () => {
        expect(sorted(["take10.wav", "take2.wav", "take1.wav"])).toEqual([
            "take1.wav",
            "take2.wav",
            "take10.wav",
        ]);
    });

    it("数字排在字母之前、字母排在数字之后", () => {
        expect(compareFileNames("1.txt", "a.txt")).toBeLessThan(0);
        expect(compareFileNames("a1.txt", "aa.txt")).toBeLessThan(0);
    });

    it("符号排在数字与字母之前", () => {
        expect(compareFileNames("!.txt", "1.txt")).toBeLessThan(0);
        expect(compareFileNames("!.txt", "a.txt")).toBeLessThan(0);
    });

    it("大小写不参与比较（视为同级）", () => {
        expect(compareFileNames("Readme.md", "readme.md")).toBe(0);
        expect(compareFileNames("ABC", "abc")).toBe(0);
    });

    it("全序：排序结果与两两比较一致，且可重复", () => {
        const names = [
            "song_extra.wav",
            "song.wav",
            "song_extra_more.wav",
            "clip.mp4",
            "clip_final.mp4",
            "take2.wav",
            "take10.wav",
            "a.txt",
            "1.txt",
            "汉字.wav",
        ];
        const once = sorted(names);
        expect(sorted(once)).toEqual(once);
        for (let i = 1; i < once.length; i += 1) {
            expect(compareFileNames(once[i - 1], once[i])).toBeLessThanOrEqual(0);
        }
    });

    it("等价的名字（仅大小写不同）不产生矛盾顺序", () => {
        expect(compareFileNames("A.wav", "a.wav")).toBe(0);
        expect(compareFileNames("a.wav", "A.wav")).toBe(0);
    });
});
