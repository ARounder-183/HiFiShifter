/**
 * 宿主音频读数判定的契约。
 *
 * 【为什么值得单测】这段判定决定界面给用户哪句话。说错方向比不说更坏 —— 用户会照着
 * 一个错误的提示去改工程。因此三条边界都要钉死：未知分类名不得当成已知、计数非法时
 * 不得下溢或变成 NaN、MIDI 片段不得被当成"等待音频"。
 */
import { describe, expect, test } from "vitest";

import { isAwaitingHostAudio, needsFolderTrackNotice, parseHostAudio } from "./hostAudio";

describe("parseHostAudio", () => {
    test("接受已知分类并归一化计数", () => {
        expect(
            parseHostAudio({ state: "folder_parent_without_regions", waiting_clips: 3 }),
        ).toEqual({ state: "folder_parent_without_regions", waiting_clips: 3 });
        expect(parseHostAudio({ state: "ready", waiting_clips: 0 })).toEqual({
            state: "ready",
            waiting_clips: 0,
        });
    });

    /*
     * 【为什么必须拒绝未知分类】后端将来可能加新的分类名。旧前端若把不认识的字符串
     * 当成已知状态渲染，就会显示一条与真实原因不符的指引。返回 null 让调用方沿用
     * 上一次已知值 —— 与"本次没带该字段"同义。
     */
    test("未知分类名一律视为没有读数", () => {
        expect(parseHostAudio({ state: "some_future_state", waiting_clips: 1 })).toBeNull();
        expect(parseHostAudio({ state: "", waiting_clips: 1 })).toBeNull();
        expect(parseHostAudio({ state: 42, waiting_clips: 1 })).toBeNull();
    });

    test("字段缺失或类型不对时视为没有读数", () => {
        expect(parseHostAudio(undefined)).toBeNull();
        expect(parseHostAudio(null)).toBeNull();
        expect(parseHostAudio("folder_parent_without_regions")).toBeNull();
    });

    /* 计数只用于展示，非法值收敛到 0 —— 绝不把 NaN / 负数渲染到界面上。 */
    test("非法计数收敛到 0", () => {
        for (const bad of [undefined, null, "many", Number.NaN, -4, {}]) {
            expect(
                parseHostAudio({ state: "awaiting_regions", waiting_clips: bad }),
                String(bad),
            ).toEqual({ state: "awaiting_regions", waiting_clips: 0 });
        }
        // 小数计数取整（后端给的是整数，这里只是不让 1.7 这类值漏进界面）。
        expect(parseHostAudio({ state: "awaiting_regions", waiting_clips: 1.7 })).toEqual({
            state: "awaiting_regions",
            waiting_clips: 1,
        });
    });
});

describe("needsFolderTrackNotice", () => {
    /*
     * 只有 folder 父轨这一种情形提示：`awaiting_regions` 是打开工程时的正常中间态，
     * 每次都为它弹一条横条等于制造噪音（状态栏的 `等待宿主音频` 已覆盖它）。
     */
    test("只有 folder 父轨需要横条提示", () => {
        expect(
            needsFolderTrackNotice({
                state: "folder_parent_without_regions",
                waiting_clips: 2,
            }),
        ).toBe(true);
        expect(needsFolderTrackNotice({ state: "awaiting_regions", waiting_clips: 2 })).toBe(false);
        expect(needsFolderTrackNotice({ state: "ready", waiting_clips: 0 })).toBe(false);
        expect(needsFolderTrackNotice(null)).toBe(false);
    });
});

describe("isAwaitingHostAudio", () => {
    test("没有源、也没有音符内容的片段才在等待宿主音频", () => {
        expect(isAwaitingHostAudio({ sourcePath: undefined, midiNoteCount: undefined })).toBe(true);
        expect(isAwaitingHostAudio({ sourcePath: "", midiNoteCount: undefined })).toBe(true);
    });

    test("有源文件的片段不是占位", () => {
        expect(isAwaitingHostAudio({ sourcePath: "ara://source", midiNoteCount: undefined })).toBe(
            false,
        );
    });

    /*
     * MIDI / 音高参考片段同样没有 `sourcePath`，但它有音符内容可画 —— 把它标成
     * "等待音频"会让一个功能正常的片段看起来是坏的。
     */
    test("MIDI 片段不是占位", () => {
        expect(isAwaitingHostAudio({ sourcePath: undefined, midiNoteCount: 4 })).toBe(false);
        expect(isAwaitingHostAudio({ sourcePath: undefined, midiNoteCount: 0 })).toBe(false);
    });
});
