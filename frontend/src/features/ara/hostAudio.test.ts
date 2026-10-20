/**
 * 宿主音频读数判定的契约。
 *
 * 【为什么值得单测】这段判定决定界面给用户哪句话。说错方向比不说更坏 —— 用户会照着
 * 一个错误的提示去改工程。因此三条边界都要钉死：未知分类名不得当成已知、计数非法时
 * 不得下溢或变成 NaN、MIDI 片段不得被当成"等待音频"。
 */
import { describe, expect, test } from "vitest";

import {
    hostMediaState,
    isAwaitingHostAudio,
    needsFolderTrackNotice,
    needsRestoreUnresolvedNotice,
    needsSkippedClipsNotice,
    needsUnmatchedRegionsNotice,
    parseHostAudio,
} from "./hostAudio";

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

    /*
     * 【为什么必须保留 reason】它是后端给出的机器可读原因码。此前它被本函数丢掉，
     * 于是界面只剩"等待宿主音频"这一句话 —— 而它同时覆盖"可等"与"用法错了"
     * 两种处置相反的处境。丢了它，用户能看到的只有一句没有信息量的话。
     */
    test("reason 被保留，空串与缺失同义", () => {
        expect(parseHostAudio({ state: "restore_unresolved", reason: "decode failed" })).toEqual({
            state: "restore_unresolved",
            waiting_clips: 0,
            reason: "decode failed",
        });
        // 空串与"没有原因"同义，不带该字段。
        expect(parseHostAudio({ state: "source_read_failed", reason: "" })).toEqual({
            state: "source_read_failed",
            waiting_clips: 0,
        });
        // 非字符串不入库。
        expect(parseHostAudio({ state: "source_read_failed", reason: 7 })).toEqual({
            state: "source_read_failed",
            waiting_clips: 0,
        });
    });

    test("restore 读数被保留", () => {
        const restore = { status: "failed" as const, applied_generation: 0, error: "boom" };
        expect(parseHostAudio({ state: "restore_unresolved", restore })).toEqual({
            state: "restore_unresolved",
            waiting_clips: 0,
            restore,
        });
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

/*
 * 被跳过的片段（MIDI / 零长度 item…）在插件里**根本不存在**：只 log 会让用户看到
 * "我明明有这一段，界面却什么都没有"。数量必须能从读数里取出来，且 0 / 缺省同义。
 */
describe("needsSkippedClipsNotice", () => {
    test("有跳过片段时返回数量", () => {
        expect(
            needsSkippedClipsNotice({
                state: "ready",
                waiting_clips: 0,
                skipped_clips: 3,
            }),
        ).toBe(3);
        expect(
            parseHostAudio({ state: "ready", waiting_clips: 0, skipped_clips: 3 })?.skipped_clips,
        ).toBe(3);
    });

    test("0 或缺省不算需要提示", () => {
        expect(needsSkippedClipsNotice({ state: "ready", waiting_clips: 0 })).toBe(0);
        expect(
            needsSkippedClipsNotice({ state: "ready", waiting_clips: 0, skipped_clips: 0 }),
        ).toBe(0);
        expect(needsSkippedClipsNotice(null)).toBe(0);
        // 0 与"旧后端不带该字段"同义：不把 `skipped_clips: 0` 带进解析结果。
        expect(parseHostAudio({ state: "ready", waiting_clips: 0, skipped_clips: 0 })).toEqual({
            state: "ready",
            waiting_clips: 0,
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
        // 「分配了区域但对不上」是**另一种成因**：提示归 `needsUnmatchedRegionsNotice`。
        expect(needsFolderTrackNotice({ state: "unmatched_regions", waiting_clips: 0 })).toBe(
            false,
        );
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

/*
 * 逐 clip 的成因：一个布尔承载不了四种互不相容的情形。
 *
 * 【为什么值得单测】四种情形此前共用一句"等待 REAPER 提供音频（未分配 ARA 区域）"，
 * 于是"我刚分割了一下"（在途）看起来像"插件坏了"。分类名由后端给，前端只做映射 ——
 * 未知分类名必须沿用旧行为，不能凭空造一个新状态。
 */
describe("hostMediaState", () => {
    test("后端分类优先，且原样透传", () => {
        expect(hostMediaState({ sourcePath: undefined, hostMedia: "pending" })).toBe("pending");
        expect(hostMediaState({ sourcePath: "x", hostMedia: "ready" })).toBe("ready");
        expect(hostMediaState({ sourcePath: undefined, hostMedia: "unavailable" })).toBe(
            "unavailable",
        );
    });

    test("未知分类名不被当成已知状态", () => {
        // 旧前端遇到新后端的新分类时，按"没带该字段"处理（这里回落到 source_path 推断）。
        expect(hostMediaState({ sourcePath: "x", hostMedia: "something-new" })).toBe("ready");
        expect(hostMediaState({ sourcePath: undefined, hostMedia: "something-new" })).toBe(
            "unavailable",
        );
    });

    test("独立 App（无该字段）沿用旧行为", () => {
        expect(hostMediaState({ sourcePath: "C:/a.wav" })).toBe("ready");
        expect(hostMediaState({ sourcePath: undefined })).toBe("unavailable");
        // MIDI 片段有内容可画，不是占位。
        expect(hostMediaState({ sourcePath: undefined, midiNoteCount: 3 })).toBe("ready");
    });
});

/**
 * "宿主分配了区域但没有一条能绑上"必须有自己的分类名与提示。
 *
 * 【为什么不能与 folder 父轨合并】两者的用户可见症状相同（片段是空的、能拖、没内容），
 * 但事实不同：folder 那条是"一个区域都没拿到"，这条是"拿到了但对不上"。提示要说对
 * 原因，否则用户会去改一件无关的事。这个读数此前**只存在于刷屏日志里**，界面上没有
 * 任何对应物。
 */
describe("unmatched_regions", () => {
    test("分类名被接受，且只有它触发该提示", () => {
        const status = parseHostAudio({ state: "unmatched_regions", waiting_clips: 2 });
        expect(status).toEqual({ state: "unmatched_regions", waiting_clips: 2 });
        expect(needsUnmatchedRegionsNotice(status)).toBe(true);
        // 与相邻的两态互不误触发。
        expect(
            needsUnmatchedRegionsNotice(parseHostAudio({ state: "ready", waiting_clips: 0 })),
        ).toBe(false);
        expect(
            needsUnmatchedRegionsNotice(
                parseHostAudio({ state: "folder_parent_without_regions", waiting_clips: 2 }),
            ),
        ).toBe(false);
        expect(needsFolderTrackNotice(status)).toBe(false);
    });
});

/**
 * `restore_unresolved`：音频通路正常，唯独上次保存的参数/曲线没合并进来。
 *
 * 【为什么必须单列一条判定】它与"等音频"完全不同：片段有内容、播放有声，
 * 用户丢的是设置。把它混进 `awaiting_regions` 会让用户去等一个不会来的东西。
 */
describe("restore_unresolved", () => {
    test("分类名被接受，且只有它触发恢复提示", () => {
        const status = parseHostAudio({ state: "restore_unresolved", waiting_clips: 1 });
        expect(status).toEqual({ state: "restore_unresolved", waiting_clips: 1 });
        expect(needsRestoreUnresolvedNotice(status)).toBe(true);
        expect(
            needsRestoreUnresolvedNotice(parseHostAudio({ state: "ready", waiting_clips: 0 })),
        ).toBe(false);
        expect(
            needsRestoreUnresolvedNotice(
                parseHostAudio({ state: "awaiting_regions", waiting_clips: 2 }),
            ),
        ).toBe(false);
    });
});
