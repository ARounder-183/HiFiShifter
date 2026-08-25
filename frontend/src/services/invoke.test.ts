/**
 * invoke 位置参数 → Tauri 命名参数映射的回归测试。
 *
 * 背景：set_clip_state 曾因映射表漏加 snapOffsetSec 导致后续参数整体
 * 左移一位 —— 吸附偏移值被写进 fadeInSec（松手瞬间 Clip"变成淡入"）。
 * 本测试锁定：每个位置的键名与值一一对应，新增参数必须同步映射表。
 */
import { buildTauriArgs } from "./invoke";

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
    true, // reversed
    true, // loopEnabled
    0.75, // snapOffsetSec
    0.1, // fadeInSec
    0.2, // fadeOutSec
    "sine", // fadeInCurve
    "linear", // fadeOutCurve
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
    "reversed",
    "loopEnabled",
    "snapOffsetSec",
    "fadeInSec",
    "fadeOutSec",
    "fadeInCurve",
    "fadeOutCurve",
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

console.log(`invoke mapping checks passed (${checks})`);
