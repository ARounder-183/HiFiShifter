import { test } from "vitest";

import { buildTauriArgs } from "./invoke";

test("services/invoke.test.ts scripted checks", async () => {
    /**
     * invoke 位置参数 → Tauri 命名参数映射的回归测试。
     *
     * 背景：set_clip_state 曾因映射表漏加 snapOffsetSec 导致后续参数整体
     * 左移一位 —— 吸附偏移值被写进 fadeInSec（松手瞬间 Clip"变成淡入"）。
     * 本测试锁定：每个位置的键名与值一一对应，新增参数必须同步映射表。
     */

    let checks = 0;
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const e = JSON.stringify(expected);
        if (a !== e) {
            throw new Error(`${label}: expected ${e}, received ${a}`);
        }
        checks += 1;
    }

    // 按后端 set_clip_state 参数顺序构造一份全量位置参数，
    // 每个位置的值都编码自己的名字，任何错位都会立刻暴露。
    const positional = [
        "clip-1", // clipId
        "name-v", // name
        1.5, // startSec
        2.5, // lengthSec
        1.25, // gain
        true, // muted
        0.25, // sourceStartSec
        3.25, // sourceEndSec
        1.5, // playbackRate
        1.25, // clipPlaybackRate
        true, // reversed
        true, // loopEnabled
        0.75, // snapOffsetSec
        0.1, // fadeInSec
        0.2, // fadeOutSec
        3, // fadeInShape
        4, // fadeOutShape
        -0.25, // fadeInDir
        0.5, // fadeOutDir
        0.3, // autoFadeInSec
        0.4, // autoFadeOutSec
        "#112233", // color
        { enabled: true, targetF1Hz: 800, targetF2Hz: 1400, strength: 0.5 }, // formantMorph
        false, // checkpoint
    ];

    const mapped = buildTauriArgs("set_clip_state", positional);
    if (!mapped || "__unwired" in mapped) throw new Error("set_clip_state mapping missing");

    const expectedKeys = [
        "clipId",
        "name",
        "startSec",
        "lengthSec",
        "gain",
        "muted",
        "sourceStartSec",
        "sourceEndSec",
        "playbackRate",
        "clipPlaybackRate",
        "reversed",
        "loopEnabled",
        "snapOffsetSec",
        "fadeInSec",
        "fadeOutSec",
        "fadeInShape",
        "fadeOutShape",
        "fadeInDir",
        "fadeOutDir",
        "autoFadeInSec",
        "autoFadeOutSec",
        "color",
        "formantMorph",
        "checkpoint",
    ];
    assertEqual(Object.keys(mapped), expectedKeys, "set_clip_state key order");

    // 关键断言：snapOffset 与淡入淡出互不串位。
    assertEqual(mapped.snapOffsetSec, 0.75, "snapOffsetSec value");
    assertEqual(mapped.fadeInSec, 0.1, "fadeInSec value");
    assertEqual(mapped.fadeOutSec, 0.2, "fadeOutSec value");
    assertEqual(mapped.checkpoint, false, "checkpoint value");

    // Take 相关命令映射：这些命令曾经漏映射，导致前端乐观更新生效但后端调用
    // 抛 "method not wired yet"，任何后续权威时间轴刷新都会回滚 Take 切换。
    assertEqual(
        buildTauriArgs("set_clip_active_take", ["clip-1", "take-2", true]),
        { clipId: "clip-1", takeId: "take-2", checkpoint: true },
        "set_clip_active_take mapping",
    );
    assertEqual(
        buildTauriArgs("cycle_clip_takes", [["clip-1"], -1, false]),
        { clipIds: ["clip-1"], direction: -1, checkpoint: false },
        "cycle_clip_takes mapping",
    );
    assertEqual(
        buildTauriArgs("duplicate_clip_take", ["clip-1", "take-2", false]),
        { clipId: "clip-1", takeId: "take-2", checkpoint: false },
        "duplicate_clip_take mapping",
    );
    assertEqual(
        buildTauriArgs("remove_clip_take", ["clip-1", "take-2", false]),
        { clipId: "clip-1", takeId: "take-2", checkpoint: false },
        "remove_clip_take mapping",
    );
    assertEqual(
        buildTauriArgs("rename_clip_take", ["clip-1", "take-2", "Lead", false]),
        { clipId: "clip-1", takeId: "take-2", name: "Lead", checkpoint: false },
        "rename_clip_take mapping",
    );
    assertEqual(
        buildTauriArgs("add_clip_take_from_media", ["clip-1", "C:/a.wav", "Take B", false]),
        { clipId: "clip-1", sourcePath: "C:/a.wav", name: "Take B", checkpoint: false },
        "add_clip_take_from_media mapping",
    );
    assertEqual(
        buildTauriArgs("pack_clips_into_takes", [["clip-1", "clip-2"], true]),
        { clipIds: ["clip-1", "clip-2"], checkpoint: true },
        "pack_clips_into_takes mapping",
    );
    assertEqual(
        buildTauriArgs("explode_clip_takes", ["clip-1", true]),
        { clipId: "clip-1", checkpoint: true },
        "explode_clip_takes mapping",
    );
    assertEqual(
        buildTauriArgs("import_media_files_as_takes", [["a.wav", "b.wav"], "track-1", 2.5]),
        { paths: ["a.wav", "b.wav"], trackId: "track-1", startSec: 2.5 },
        "import_media_files_as_takes mapping",
    );
});
