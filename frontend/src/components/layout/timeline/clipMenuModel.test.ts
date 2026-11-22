/**
 * 片段右键菜单模型的契约（**纯数据**，node 环境即可跑）。
 *
 * 【要钉死什么】三件事都是「改一行判据就会静默坏掉」的：
 *
 * 1. 结构性不可用的项**不出现**，情境性不可用的项**只禁用**。此前两者都靠
 *    gate(cond, reason) 长成「禁用 + 第二行原因」，于是菜单被撑成两倍高。
 * 2. 分隔线收敛：隐藏任意一段之后不留空段、不留连续两条、不以分隔线开场/收尾。
 * 3. 判据只来自能力位 —— 同一份能力位下，能点的项都必须在 caps 里说得清。
 */
import { expect, test } from "vitest";

import { enUS } from "../../../i18n/en-US";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import {
    buildClipMenuModel,
    collapseClipMenuEntries,
    type ClipMenuCaps,
    type ClipMenuEntry,
    type ClipMenuInput,
} from "./clipMenuModel";

const t = (key: keyof typeof enUS): string => enUS[key] as string;

function baseClip(overrides: Partial<ClipInfo> = {}): ClipInfo {
    return {
        id: "clip-1",
        trackId: "track-1",
        name: "vocal",
        startSec: 0,
        lengthSec: 4,
        color: "blue",
        takes: [
            {
                id: "t1",
                name: "Take A",
                gain: 1,
                reversed: false,
                channelMode: 0,
                loopEnabled: false,
                sourcePath: "C:/audio/vocal.wav",
            },
        ],
        activeTakeId: "t1",
        sourcePath: "C:/audio/vocal.wav",
        gain: 1,
        muted: false,
        sourceStartSec: 0,
        sourceEndSec: 4,
        playbackRate: 1,
        reversed: false,
        channelMode: 0,
        loopEnabled: false,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeOutShape: 0,
        ...overrides,
    } as ClipInfo;
}

/** 独立 App 的能力位：除了「宿主专属」的几项，其余都成立。 */
const APP_CAPS: ClipMenuCaps = {
    editClips: true,
    clipboard: true,
    split: true,
    takes: true,
    glue: true,
    silenceDetection: true,
    fakeStereoScan: true,
    reverse: true,
    quickExport: true,
    pitchBridge: false,
    midiAsClip: true,
    fadeShape: true,
};

/** REAPER 插件的能力位：结构写回不成立，倒放与音高桥成立。 */
const REAPER_CAPS: ClipMenuCaps = {
    editClips: true,
    clipboard: true,
    split: true,
    takes: false,
    glue: false,
    silenceDetection: false,
    fakeStereoScan: false,
    reverse: true,
    quickExport: true,
    pitchBridge: true,
    midiAsClip: false,
    fadeShape: true,
};

function build(overrides: Partial<ClipMenuInput> = {}): ClipMenuEntry[] {
    const clip = overrides.clip ?? baseClip();
    return buildClipMenuModel({
        clip,
        selectedClips: overrides.selectedClips ?? [clip],
        playheadInClip: true,
        canSplitSelected: true,
        pluginMode: false,
        caps: APP_CAPS,
        available: {
            replaceMidi: true,
            group: true,
            ungroup: true,
            convertToPitchRef: true,
            updatePitchRef: true,
            exportMidi: true,
            addToParamSelection: true,
            silenceDetection: true,
            scanFakeStereo: true,
            toggleLoop: true,
            channelMode: true,
            editRate: true,
            fadeShape: true,
        },
        shortcuts: {},
        t,
        ...overrides,
    });
}

/** 递归收集所有条目（含子菜单内部）。 */
function flatten(entries: readonly ClipMenuEntry[]): ClipMenuEntry[] {
    const out: ClipMenuEntry[] = [];
    for (const entry of entries) {
        if (entry.kind === "submenu") {
            out.push(entry, ...flatten(entry.children));
        } else {
            out.push(entry);
        }
    }
    return out;
}

const keysOf = (entries: readonly ClipMenuEntry[]): string[] =>
    flatten(entries)
        .filter((entry) => entry.kind === "item")
        .map((entry) => (entry as { key: string }).key);

const labelsOf = (entries: readonly ClipMenuEntry[]): string[] =>
    flatten(entries).map((entry) => {
        if (entry.kind === "item") return entry.label;
        if (entry.kind === "submenu") return entry.label;
        if (entry.kind === "heading") return entry.text;
        if (entry.kind === "takeRow") return entry.label;
        return "-";
    });

/** 没有任何一条条目携带「为什么不能」的说明字段。 */
test("no entry carries a reason string (the model has no such field at all)", () => {
    for (const entry of flatten(build())) {
        expect(Object.prototype.hasOwnProperty.call(entry, "description")).toBe(false);
    }
});

test("standalone keeps the full tree, including take writes and glue", () => {
    const entries = build();
    const keys = keysOf(entries);
    for (const key of ["take-add", "take-duplicate", "take-remove", "scan-fake-stereo"]) {
        expect(keys, key).toContain(key);
    }
    // 独立 App 的逐 take 行带行尾两个按钮（切 take / 翻转方向 / 循环声道）。
    const rows = flatten(entries).filter((entry) => entry.kind === "takeRow");
    expect(rows.length).toBe(1);

    // 胶合只在**多选**下有意义（单条片段没有可胶合的对象）。
    const first = baseClip({ id: "a" });
    const second = baseClip({ id: "b" });
    expect(keysOf(build({ clip: first, selectedClips: [first, second] }))).toContain("glue");
});

test("REAPER plugin drops every structurally impossible entry", () => {
    const entries = build({ pluginMode: true, caps: REAPER_CAPS });
    const keys = keysOf(entries);
    for (const absent of [
        "take-pack",
        "take-add",
        "take-duplicate",
        "take-remove",
        "take-explode",
        "take-cycle-next",
        "take-cycle-prev",
        "glue",
        "silence-detection",
        "scan-fake-stereo",
        "replace",
        "replace-midi",
    ]) {
        expect(keys, absent).not.toContain(absent);
    }
    // 倒放、音高桥、音高参考、快速导出都在。
    for (const present of ["reverse", "host-pitch-import", "host-pitch-export"]) {
        expect(keys, present).toContain(present);
    }
    expect(keys).toContain("pitch-reference-convert");
    expect(keys).toContain("quick-export");
});

/** 两个 take 的片段：插件 take 行的行数断言需要它。 */
function clipWithTwoTakes(): ClipInfo {
    const clip = baseClip();
    const takes = clip.takes ?? [];
    return {
        ...clip,
        takes: [
            takes[0],
            {
                id: "t2",
                name: "Take B",
                gain: 1,
                reversed: false,
                channelMode: 0,
                loopEnabled: false,
                sourcePath: "C:/audio/vocal.wav",
            },
        ],
    } as ClipInfo;
}

test("plugin take submenu renders actionable rows: reverse/channel writable, switching not", () => {
    const clip = clipWithTwoTakes();
    const entries = build({ pluginMode: true, caps: REAPER_CAPS, clip, selectedClips: [clip] });
    const takeSubmenu = entries.find(
        (entry) => entry.kind === "submenu" && entry.key === "takes",
    ) as Extract<ClipMenuEntry, { kind: "submenu" }>;
    expect(takeSubmenu).toBeTruthy();
    // 行渲染（不是只读标题）：逐 take 的方向与声道模式结构上可写回。
    const rows = takeSubmenu.children.filter((child) => child.kind === "takeRow");
    expect(rows.length).toBe(2);
    // 行点击（切换 active take）恒禁用 —— 那是结构写回，会断 ARA region ↔ take
    // 的一一对应；行尾按钮不属于这个边界。
    expect(rows.every((row) => !(row as { switchable: boolean }).switchable)).toBe(true);
    // 重命名仍是最末一项。
    expect((takeSubmenu.children.at(-1) as { key: string }).key).toBe("take-rename");
});

test("plugin take submenu falls back to a read-only list when the host cannot reverse", () => {
    const clip = clipWithTwoTakes();
    const entries = build({
        pluginMode: true,
        caps: { ...REAPER_CAPS, reverse: false },
        clip,
        selectedClips: [clip],
    });
    const takeSubmenu = entries.find(
        (entry) => entry.kind === "submenu" && entry.key === "takes",
    ) as Extract<ClipMenuEntry, { kind: "submenu" }>;
    expect(takeSubmenu.children.map((child) => child.kind)).toEqual(["heading", "heading", "item"]);
});

test("situational unavailability stays visible but disabled", () => {
    const entries = build({ playheadInClip: false });
    const split = flatten(entries).find(
        (entry) => entry.kind === "item" && entry.key === "split",
    ) as Extract<ClipMenuEntry, { kind: "item" }>;
    expect(split).toBeTruthy();
    expect(split.disabled).toBe(true);
});

test("separators never start, end or double up - at any depth", () => {
    const check = (entries: readonly ClipMenuEntry[], where: string) => {
        expect(entries[0]?.kind, where + ": 以分隔线开场").not.toBe("separator");
        expect(entries[entries.length - 1]?.kind, where + ": 以分隔线收尾").not.toBe("separator");
        for (let i = 1; i < entries.length; i += 1) {
            expect(
                entries[i].kind === "separator" && entries[i - 1].kind === "separator",
                where + ": 第 " + String(i) + " 条与前一条同为分隔线",
            ).toBe(false);
        }
        for (const entry of entries) {
            if (entry.kind === "submenu") check(entry.children, where + "/" + entry.key);
        }
    };
    for (const [name, caps] of [
        ["app", APP_CAPS],
        ["reaper", REAPER_CAPS],
    ] as const) {
        for (const pluginMode of [false, true]) {
            check(build({ caps, pluginMode }), name + "/plugin=" + String(pluginMode));
        }
    }
});

test("an empty submenu is dropped instead of rendered as a dead entry", () => {
    const entries = collapseClipMenuEntries([
        { kind: "submenu", key: "empty", label: "Empty", children: [] },
        { kind: "separator", key: "sep" },
        { kind: "item", key: "a", label: "A" },
    ]);
    expect(entries.map((entry) => entry.kind)).toEqual(["item"]);
});

test("multi-selection renders one fade row per direction and a count heading", () => {
    const first = baseClip({ id: "a", fadeInSec: 0.1, fadeOutSec: 0.1 });
    const second = baseClip({ id: "b", fadeInSec: 0.1, fadeOutSec: 0.1 });
    const entries = build({ clip: first, selectedClips: [first, second] });
    const fadeRows = flatten(entries).filter((entry) => entry.kind === "fadeRow");
    expect(fadeRows.map((row) => (row as { direction: string }).direction)).toEqual(["in", "out"]);
    expect(labelsOf(entries)[0]).toContain("2");
});

test("the pitch bridge entry only exists when the capability says so", () => {
    expect(keysOf(build({ caps: { ...REAPER_CAPS, pitchBridge: false } }))).not.toContain(
        "host-pitch-import",
    );
    expect(keysOf(build({ caps: { ...APP_CAPS, pitchBridge: true } }))).toContain(
        "host-pitch-import",
    );
});
