/**
 * 能力矩阵的契约。
 *
 * 【要钉死什么】
 *
 * 1. **每条能力都能回答「为什么不能」**。此前二十条谓词共用一句通用文案，于是
 *    界面上「宿主接管」既可能指「宿主没给接口」、也可能指「只在独立 App 有意义」——
 *    三者对用户意味着完全不同的下一步。
 * 2. **动作映射与能力判定不会各走一套**。代码里已经记过两次因此出的事故（MIDI 导入、
 *    导出：菜单能用、快捷键是死的）。这里穷举动作名，要求 pluginAllowsAction 与
 *    capabilityForAction + capability 给出同一个答案。
 */
// @vitest-environment jsdom
import { beforeEach, expect, test } from "vitest";

import {
    capability,
    capabilityDeniedReason,
    capabilityForAction,
    capabilityDeniedText,
    capabilityReasonText,
    pluginAllowsAction,
    type CapabilityId,
} from "./hostCapabilities";

/** 一张插件形态的 bootstrap（能力位全给，用来把「映射」与「能力」分开验）。 */
function bootstrap(overrides: Record<string, unknown> = {}) {
    return {
        version: 1,
        hostKind: "ara_only",
        transportControl: true,
        trackGrouping: true,
        historyBackend: "local",
        fadeAxes: "continuous",
        hostCapabilities: {
            clipWrite: true,
            undoStack: true,
            split: true,
            itemClipboard: true,
            mediaImport: true,
            exportAudio: true,
            exportData: true,
            trackCreate: true,
            takeImport: true,
            midiImportPitch: true,
            midiImportClip: true,
        },
        ...overrides,
    };
}

beforeEach(() => {
    (globalThis as { window?: unknown }).window = {
        __HFS_PLUGIN_BOOTSTRAP__: bootstrap(),
    } as unknown as typeof globalThis.window;
    // 与 hostCapabilities.test.ts 同一手法：语言由 localStorage 决定，`translateOutsideReact`
    // 每次调用都重读，因此这里改一下就够了。
    localStorage.setItem("hifishifter.locale", "en-US");
});

const ALL: CapabilityId[] = [
    "transport.control",
    "clip.edit",
    "clip.split",
    "clip.clipboard",
    "clip.geometry",
    "clip.normalize",
    "track.create",
    "track.group",
    "track.edit",
    "fade.length",
    "fade.shape",
    "fade.curvature",
    "fade.s",
    "tempo.bpm",
    "media.import",
    "midi.importPitch",
    "midi.importClip",
    "export.audio",
    "export.data",
    "take.import",
    "project.file",
    "recording.local",
    "playback.metronome",
    "surface.pianoRollOnly",
];

test("每条能力在不可用时都给得出理由", () => {
    for (const id of ALL) {
        const decision = capability(id);
        if (decision.allowed) {
            expect(decision.reason, id).toBeNull();
            continue;
        }
        expect(decision.reason, id).toBeTruthy();
        // 文案必须真的翻出来（不是空串、也不是 key 本身）。
        const text = capabilityReasonText(id);
        expect(text.length, id).toBeGreaterThan(0);
        expect(text, id).not.toBe(decision.reason);
    }
});

// 【为什么这里不逐个断言「给了能力位就放行」】那件事已经由 hostCapabilities.test.ts
// 逐条覆盖（每个谓词都有各自的夹具与用例）。重复一遍只会变成「把同一张表抄两遍」，
// 而抄写不会发现漂移 —— 真正会漂的是「动作 → 能力」的映射，那由下面两条用例守。
test("动作映射与能力判定同源（穷举已知动作，两套判据必须一致）", () => {
    const actions = [
        "playback.toggle",
        "playback.stop",
        "playback.metronome",
        "project.importMedia",
        "project.importMidi",
        "project.export",
        "project.open",
        "project.save",
        "project.new",
        "clip.split",
        "clip.delete",
        "clip.group",
        "clip.ungroup",
        "clip.normalize",
        "clip.copy",
        "clip.cut",
        "clip.paste",
        "track.createHost",
        "track.add",
        "track.selectUp",
        "track.toggleMute",
        "recording.start",
        "recording.toggle",
        "transport.play",
        "edit.undo",
        "edit.pasteTracks",
        "edit.smooth",
    ];
    for (const surface of ["timeline", "pianoRoll", "trackHeader", null]) {
        for (const action of actions) {
            const id = capabilityForAction(action, surface);
            // 独立重算一遍：映射到能力 ⇒ 由能力决定；不映射 ⇒ 放行。
            // edit.* 的界面白名单是唯一的例外，单独断言（见下一条用例）。
            const expected =
                id === "surface.pianoRollOnly"
                    ? pluginAllowsAction(action, surface)
                    : id === null
                      ? true
                      : capability(id).allowed;
            expect(pluginAllowsAction(action, surface), `${action} @ ${surface}`).toBe(expected);
        }
    }
});

test("被拒的动作都带得出一条具体理由（不再只有一句通用文案）", () => {
    // 换成一个什么都没给的宿主：这样「被拒」才是被拒。
    (globalThis as { window?: unknown }).window = {
        __HFS_PLUGIN_BOOTSTRAP__: bootstrap({
            transportControl: false,
            trackGrouping: false,
            hostCapabilities: {},
        }),
    } as unknown as typeof globalThis.window;

    // 工程文件与本地录音：独立 App 专属。
    expect(capabilityDeniedReason("project.open", null)).toBe("cap_reason_app_only");
    expect(capabilityDeniedReason("recording.start", null)).toBe("cap_reason_app_only");
    // 宿主剪贴板：宿主没给接口（而不是「宿主接管」）。
    expect(capabilityDeniedReason("clip.delete", "timeline")).toBe(
        "cap_reason_unsupported_by_host",
    );
    // BPM：宿主权威。
    expect(capabilityDeniedReason("playback.toggle", null)).toBe("cap_reason_unsupported_by_host");
    // 不在参数编辑器里的 edit.*：与宿主无关，只是界面不对。
    expect(capabilityDeniedReason("edit.smooth", "timeline")).toBe("cap_reason_piano_roll_only");
    // 放行的动作没有理由。
    expect(capabilityDeniedReason("clip.copy", "timeline")).toBeNull();
});

test("被拒的动作能给出一句可以直接显示的话（快捷键不再静默）", () => {
    // 与 pluginAllowsAction 同源：能做的就返回 null。
    expect(capabilityDeniedText("clip.copy", "timeline")).toBeNull();
    expect(capabilityDeniedText("edit.smooth", "pianoRoll")).toBeNull();

    // 被拒的给一句翻译过的成句文案，而不是 key。
    const reason = capabilityDeniedText("project.open", null);
    expect(reason).toBeTruthy();
    expect(reason).not.toContain("cap_reason_");
    expect(capabilityDeniedText("recording.start", null)).toBe(reason);
});
