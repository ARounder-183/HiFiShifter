import { describe, expect, it } from "vitest";
import { resolveActionByFocus } from "./focusRouting";
import type { Keybinding } from "./types";

/** 构造一个 KeyboardEvent 形状的事件 */
function keyEvent(
    key: string,
    mods: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {},
): KeyboardEvent {
    return {
        key,
        code: key === " " ? "Space" : `Key${key.toUpperCase()}`,
        ctrlKey: Boolean(mods.ctrl),
        shiftKey: Boolean(mods.shift),
        altKey: Boolean(mods.alt),
        metaKey: false,
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
