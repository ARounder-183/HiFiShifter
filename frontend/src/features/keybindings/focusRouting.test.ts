import { describe, expect, it } from "vitest";
import {
    resolveActionByFocus,
    resolveEditOpRoute,
    resolvePasteRoute,
    ACTION_TO_EDIT_OP,
} from "./focusRouting";
import type { Keybinding } from "./types";
import { IS_MAC } from "../../utils/platform";

/** 构造一个 KeyboardEvent 形状的事件 */
function keyEvent(
    key: string,
    mods: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {},
): KeyboardEvent {
    return {
        key,
        code: key === " " ? "Space" : `Key${key.toUpperCase()}`,
        // 主修饰键按平台映射：macOS = ⌘(metaKey)，Windows/Linux = Ctrl。
        // 与 `isModifierActive` → `isPrimaryModifierDown` 的映射一致；若写死
        // `ctrlKey` + `metaKey: false`，这套测试就只在 Windows/Linux 通过。
        ctrlKey: IS_MAC ? false : Boolean(mods.ctrl),
        metaKey: IS_MAC ? Boolean(mods.ctrl) : false,
        shiftKey: Boolean(mods.shift),
        altKey: Boolean(mods.alt),
    } as unknown as KeyboardEvent;
}

type TestMap = Record<string, Keybinding>;

const CTRL_T: TestMap = {
    // 故意让 track.add 声明在 edit.setPitch 之前 —— 路由结果必须与声明顺序无关
    "track.add": { key: "t", ctrl: true },
    "mode.toggle": { key: "tab" },
};

// 用完整真实元数据（ACTION_META）构造场景键位表
const FULL_CTRL_T: TestMap = {
    "track.add": { key: "t", ctrl: true },
    "edit.setPitch": { key: "t", ctrl: true },
};

const DELETE_SHARED: TestMap = {
    "clip.delete": { key: "delete" },
    // 用户把「删除选中轨道」重绑为裸 Delete：与「删除音频块」共用按键，
    // 由焦点裁决（轨道头 → 删除轨道；时间轴 → 删除音频块）。
    "track.delete": { key: "delete" },
    "track.clone": { key: "d", ctrl: true },
};

describe("resolveActionByFocus — 焦点感知路由", () => {
    it("参数编辑器 + select 工具：paramEditorSelect 操作优先于全局操作", () => {
        // Ctrl+T：音高设置到（paramEditorSelect）应胜过添加轨道（全局）
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            FULL_CTRL_T,
            "pianoRoll",
            "select",
        );
        expect(result).toBe("edit.setPitch");
    });

    it("时间轴焦点：全局操作优先（添加轨道），paramEditorSelect 硬排除", () => {
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            FULL_CTRL_T,
            "timeline",
            "select",
        );
        expect(result).toBe("track.add");
    });

    it("轨道头焦点：全局操作优先（添加轨道）", () => {
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            FULL_CTRL_T,
            "trackHeader",
            "select",
        );
        expect(result).toBe("track.add");
    });

    it("无焦点域：全局操作优先", () => {
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            FULL_CTRL_T,
            null,
            "select",
        );
        expect(result).toBe("track.add");
    });

    it("参数编辑器 + 非 select 工具：paramEditorSelect 硬排除，落到全局操作", () => {
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            FULL_CTRL_T,
            "pianoRoll",
            "draw",
        );
        expect(result).toBe("track.add");
    });

    it("轨道头焦点：trackHeaderFocus 操作优先于同键全局操作（轨道删除 vs 音频块删除）", () => {
        const result = resolveActionByFocus(
            keyEvent("delete"),
            DELETE_SHARED,
            "trackHeader",
            "select",
        );
        expect(result).toBe("track.delete");
    });

    it("时间轴焦点：同键全局操作优先（音频块删除），轨道操作仅兜底", () => {
        const result = resolveActionByFocus(
            keyEvent("delete"),
            DELETE_SHARED,
            "timeline",
            "select",
        );
        expect(result).toBe("clip.delete");
    });

    it("无焦点域：同键全局操作优先（音频块删除）", () => {
        const result = resolveActionByFocus(keyEvent("delete"), DELETE_SHARED, null, "select");
        expect(result).toBe("clip.delete");
    });

    it("参数编辑器焦点：同键全局操作优先（音频块删除）", () => {
        const result = resolveActionByFocus(
            keyEvent("delete"),
            DELETE_SHARED,
            "pianoRoll",
            "select",
        );
        expect(result).toBe("clip.delete");
    });

    it("筛选器：全局唯一匹配仍生效（轨道克隆在任意焦点域）", () => {
        for (const domain of ["pianoRoll", "timeline", "trackHeader", null] as const) {
            const result = resolveActionByFocus(
                keyEvent("d", { ctrl: true }),
                DELETE_SHARED,
                domain,
                "select",
            );
            expect(result).toBe("track.clone");
        }
    });

    it("无匹配时返回 null", () => {
        const result = resolveActionByFocus(
            keyEvent("z", { ctrl: true }),
            FULL_CTRL_T,
            "timeline",
            "select",
        );
        expect(result).toBeNull();
    });

    it("与键位表声明顺序无关（反向声明仍由焦点裁决）", () => {
        // 声明顺序：track.add 在前 vs edit.setPitch 在前 —— 结果一致
        const reversed: TestMap = {
            "edit.setPitch": { key: "t", ctrl: true },
            "track.add": { key: "t", ctrl: true },
        };
        expect(
            resolveActionByFocus(keyEvent("t", { ctrl: true }), reversed, "pianoRoll", "select"),
        ).toBe("edit.setPitch");
        expect(
            resolveActionByFocus(keyEvent("t", { ctrl: true }), reversed, "timeline", "select"),
        ).toBe("track.add");
    });

    it("quickSearch 作用域操作始终硬排除（由弹窗自身处理）", () => {
        const map: TestMap = {
            "quickSearch.navigate.up": { key: "arrowup" },
            "track.selectUp": { key: "arrowup" },
        };
        for (const domain of ["pianoRoll", "timeline", "trackHeader", null] as const) {
            const result = resolveActionByFocus(keyEvent("arrowup"), map, domain, "select");
            expect(result).toBe("track.selectUp");
        }
    });

    it("CTRL_T 裸表（无 ACTION_META 元数据）按全局处理", () => {
        const result = resolveActionByFocus(
            keyEvent("t", { ctrl: true }),
            CTRL_T,
            "pianoRoll",
            "select",
        );
        expect(result).toBe("track.add");
    });
});

describe("resolveEditOpRoute（复制/剪切/粘贴的表面定向派发）", () => {
    it("copy/cut/paste 严格按活动编辑表面路由", () => {
        for (const op of ["copy", "cut", "paste"] as const) {
            expect(resolveEditOpRoute("pianoRoll", op)).toBe("hifi:editOp");
            expect(resolveEditOpRoute("timeline", op)).toBe("hifi:timelineEditOp");
            expect(resolveEditOpRoute("trackHeader", op)).toBe("hifi:timelineEditOp");
        }
    });

    it("copy/cut/paste 在表面未知（启动初期）时不派发", () => {
        for (const op of ["copy", "cut", "paste"] as const) {
            expect(resolveEditOpRoute(null, op)).toBeNull();
        }
    });

    it("selectAll/deselect：参数编辑器仅在 select 工具下接手，其余落时间轴", () => {
        expect(resolveEditOpRoute("pianoRoll", "selectAll", "select")).toBe("hifi:editOp");
        expect(resolveEditOpRoute("pianoRoll", "deselect", "select")).toBe("hifi:editOp");
        expect(resolveEditOpRoute("pianoRoll", "selectAll", "draw")).toBe("hifi:timelineEditOp");
        expect(resolveEditOpRoute("pianoRoll", "selectAll")).toBe("hifi:timelineEditOp");
        expect(resolveEditOpRoute("timeline", "selectAll", "select")).toBe("hifi:timelineEditOp");
        expect(resolveEditOpRoute(null, "deselect")).toBe("hifi:timelineEditOp");
    });

    it("时间轴专有操作固定路由到时间轴通道（与最后点击的表面无关）", () => {
        for (const op of [
            "delete",
            "split",
            "normalize",
            "group",
            "ungroup",
            "cycleTake",
            "cycleTakePrev",
            "pasteTracks",
        ] as const) {
            expect(resolveEditOpRoute("pianoRoll", op)).toBe("hifi:timelineEditOp");
            expect(resolveEditOpRoute("timeline", op)).toBe("hifi:timelineEditOp");
        }
    });

    it("未收录的操作返回 null（调用方走各自固定通道）", () => {
        expect(resolveEditOpRoute("pianoRoll", "pasteVocalShifter")).toBeNull();
        expect(resolveEditOpRoute("timeline", "shiftParamUp")).toBeNull();
    });

    it("ACTION_TO_EDIT_OP：clip.* 与 pianoRoll.* 同义操作归一为同一 op", () => {
        expect(ACTION_TO_EDIT_OP["clip.copy"]).toBe("copy");
        expect(ACTION_TO_EDIT_OP["pianoRoll.copy"]).toBe("copy");
        expect(ACTION_TO_EDIT_OP["clip.cut"]).toBe("cut");
        expect(ACTION_TO_EDIT_OP["pianoRoll.cut"]).toBe("cut");
        expect(ACTION_TO_EDIT_OP["clip.paste"]).toBe("paste");
        expect(ACTION_TO_EDIT_OP["pianoRoll.paste"]).toBe("paste");
        expect(ACTION_TO_EDIT_OP["edit.selectAll"]).toBe("selectAll");
        // 参数编辑器专有操作不在此表（单一归属，无需路由裁决）
        expect(ACTION_TO_EDIT_OP["pianoRoll.shiftParamUp"]).toBeUndefined();
        expect(ACTION_TO_EDIT_OP["edit.pasteVocalShifter"]).toBeUndefined();
    });
});

describe("resolvePasteRoute（粘贴的内容路由，last-copy-wins）", () => {
    it("参数线载荷贴回参数编辑器 —— 与点击表面无关", () => {
        expect(resolvePasteRoute("param", "timeline")).toBe("hifi:editOp");
        expect(resolvePasteRoute("param", "trackHeader")).toBe("hifi:editOp");
        expect(resolvePasteRoute("param", "pianoRoll")).toBe("hifi:editOp");
        expect(resolvePasteRoute("param", null)).toBe("hifi:editOp");
    });

    it("Clip / 整轨 / 工程载荷贴到播放头 —— 与点击表面无关", () => {
        for (const kind of ["clips", "tracks", "project"] as const) {
            expect(resolvePasteRoute(kind, "pianoRoll")).toBe("hifi:timelineEditOp");
            expect(resolvePasteRoute(kind, "timeline")).toBe("hifi:timelineEditOp");
            expect(resolvePasteRoute(kind, null)).toBe("hifi:timelineEditOp");
        }
    });

    it("槽位为空/外来数据时按活动表面兜底", () => {
        // 参数编辑器表面：保留 REAPER/MIDI 兜底流程（含 MIDI 导入对话框）。
        expect(resolvePasteRoute(null, "pianoRoll")).toBe("hifi:editOp");
        // 时间轴/轨道头表面：播放头 REAPER 粘贴。
        expect(resolvePasteRoute(null, "timeline")).toBe("hifi:timelineEditOp");
        expect(resolvePasteRoute(null, "trackHeader")).toBe("hifi:timelineEditOp");
        // 表面未知（启动初期）：不派发。
        expect(resolvePasteRoute(null, null)).toBeNull();
    });

    it("未知 kind 视同外来源（表面兜底）", () => {
        expect(resolvePasteRoute("reaper", "timeline")).toBe("hifi:timelineEditOp");
        expect(resolvePasteRoute("weird", "pianoRoll")).toBe("hifi:editOp");
    });
});
