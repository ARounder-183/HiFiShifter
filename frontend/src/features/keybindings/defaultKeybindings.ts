import type { ActionId, ActionMeta, KeybindingMap } from "./types";

/**
 * 默认快捷键映射表
 * 收录了项目中所有硬编码快捷键的默认值
 * key 为 "__none__" 表示无绑定
 */
export const DEFAULT_KEYBINDINGS: KeybindingMap = {
    // 模式切换
    "mode.toggle": { key: "tab" },
    "mode.selectTool": { key: "f7" },
    "mode.drawTool": { key: "f8" },
    "mode.lineTool": { key: "f9" },

    // 播放控制
    "playback.toggle": { key: "space" },
    "playback.stop": { key: "enter" }, // 停止并回到本次播放起点
    "playback.focusCursor": { key: "\\" }, // 聚焦播放光标
    "playback.seekLeft": { key: "arrowleft" },
    "playback.seekRight": { key: "arrowright" },
    "timeline.zoomIn": { key: "arrowup" },
    "timeline.zoomOut": { key: "arrowdown" },

    // 编辑
    "edit.undo": { key: "z", ctrl: true },
    "edit.redo": { key: "y", ctrl: true },
    "edit.selectAll": { key: "a", ctrl: true },
    "edit.deselect": { key: "r", ctrl: true },
    "edit.initialize": { key: "backspace" },
    "edit.transposeCents": { key: "f", ctrl: true },
    "edit.transposeDegrees": { key: "i", ctrl: true },
    "edit.setPitch": { key: "t", ctrl: true },
    "edit.average": { key: "e", ctrl: true },
    "edit.smooth": { key: "m", ctrl: true },
    "edit.addVibrato": { key: "b", ctrl: true },
    "edit.quantize": { key: "p", ctrl: true },
    "edit.meanQuantize": { key: "q", ctrl: true },
    "edit.pasteReaper": { key: "v", ctrl: true, shift: true },
    "edit.pasteVocalShifter": { key: "v", shift: true },

    // 工程
    "project.new": { key: "n", ctrl: true },
    "project.open": { key: "o", ctrl: true, shift: true },
    "project.save": { key: "s", ctrl: true },
    "project.saveAs": { key: "s", ctrl: true, shift: true },
    "project.export": { key: "e", ctrl: true },

    // 轨道
    "track.add": { key: "t", ctrl: true },

    // Clip 操作
    "clip.delete": { key: "delete" },
    "clip.copy": { key: "c", ctrl: true },
    "clip.cut": { key: "x", ctrl: true },
    "clip.paste": { key: "v", ctrl: true },
    "clip.split": { key: "s" },
    "clip.normalize": { key: "n", ctrl: true, shift: true },

    // PianoRoll 操作
    "pianoRoll.copy": { key: "c", ctrl: true },
    "pianoRoll.paste": { key: "v", ctrl: true },
    "pianoRoll.shiftParamUp": { key: "=" },
    "pianoRoll.shiftParamDown": { key: "-" },
    "pianoRoll.shiftParamUpSelection": { key: "]" },
    "pianoRoll.shiftParamDownSelection": { key: "[" },

    // 修饰键行为
    "modifier.clipSlipEdit": { key: "alt", modifierOnly: true, alt: true },
    "modifier.clipStretch": { key: "alt", modifierOnly: true, alt: true },
    "modifier.clipNoSnap": { key: "shift", modifierOnly: true, shift: true },
    "modifier.clipCopyDrag": { key: "control", modifierOnly: true, ctrl: true },
    "modifier.horizontalZoom": { key: "__none__", modifierOnly: true },
    "modifier.pianoRollVerticalZoom": {
        key: "control",
        modifierOnly: true,
        ctrl: true,
    },
    "modifier.scrollHorizontal": {
        key: "shift",
        modifierOnly: true,
        shift: true,
    },
    "modifier.scrollVertical": { key: "alt", modifierOnly: true, alt: true },
    "modifier.pianoKeysVerticalScroll": {
        key: "alt",
        modifierOnly: true,
        alt: true,
    },
    "modifier.pianoKeysVerticalZoom": { key: "__none__", modifierOnly: true },
    "modifier.paramMorph": { key: "alt", modifierOnly: true, alt: true },
    // 目前微调操作有逻辑问题，没法完全实现，暂时禁用快捷键
    // "modifier.paramFineAdjust": { key: "control", modifierOnly: true, ctrl: true },
    "modifier.paramFineAdjust": { key: "__none__", modifierOnly: true },
    "modifier.vibratoAmplitudeAdjust": { key: "__none__", modifierOnly: true },
    "modifier.vibratoFrequencyAdjust": { key: "alt", modifierOnly: true, alt: true },

    // 快速搜索
    "quickSearch.open": { key: "f", ctrl: true },
    "quickSearch.navigate.up": { key: "arrowup" },
    "quickSearch.navigate.down": { key: "arrowdown" },
    "quickSearch.preview": { key: "space" },
    "quickSearch.confirm": { key: "enter" },
    "quickSearch.close": { key: "escape" },
};

/**
 * 操作元信息（用于 UI 分组 & 显示）
 */
export const ACTION_META: Record<ActionId, ActionMeta> = {
    "mode.toggle": { labelKey: "kb_mode_toggle", group: "mode" },
    "mode.selectTool": { labelKey: "kb_mode_select_tool", group: "mode" },
    "mode.drawTool": { labelKey: "kb_mode_draw_tool", group: "mode" },
    "mode.lineTool": { labelKey: "kb_mode_vibrato_tool", group: "mode" },

    "playback.toggle": { labelKey: "kb_playback_toggle", group: "playback" },
    "playback.stop": { labelKey: "kb_playback_stop", group: "playback" },
    "playback.focusCursor": {
        labelKey: "kb_playback_focus_cursor",
        group: "playback",
    },
    "playback.seekLeft": {
        labelKey: "kb_playback_seek_left",
        group: "playback",
    },
    "playback.seekRight": {
        labelKey: "kb_playback_seek_right",
        group: "playback",
    },
    "timeline.zoomIn": {
        labelKey: "kb_timeline_zoom_in",
        group: "playback",
        scopedContext: "timelineFocus",
    },
    "timeline.zoomOut": {
        labelKey: "kb_timeline_zoom_out",
        group: "playback",
        scopedContext: "timelineFocus",
    },

    "edit.undo": { labelKey: "kb_edit_undo", group: "edit" },
    "edit.redo": { labelKey: "kb_edit_redo", group: "edit" },
    "edit.selectAll": { labelKey: "kb_edit_select_all", group: "edit" },
    "edit.deselect": { labelKey: "kb_edit_deselect", group: "edit" },
    "edit.initialize": {
        labelKey: "kb_edit_initialize",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.transposeCents": {
        labelKey: "kb_edit_transpose_cents",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.transposeDegrees": {
        labelKey: "kb_edit_transpose_degrees",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.setPitch": {
        labelKey: "kb_edit_set_pitch",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.average": {
        labelKey: "kb_edit_average",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.smooth": {
        labelKey: "kb_edit_smooth",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.addVibrato": {
        labelKey: "kb_edit_add_vibrato",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.quantize": {
        labelKey: "kb_edit_quantize",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.meanQuantize": {
        labelKey: "kb_edit_mean_quantize",
        group: "edit",
        scopedContext: "paramEditorSelect",
    },
    "edit.pasteReaper": { labelKey: "kb_edit_paste_reaper", group: "edit" },
    "edit.pasteVocalShifter": {
        labelKey: "kb_edit_paste_vocalshifter",
        group: "edit",
    },

    "project.new": { labelKey: "kb_project_new", group: "project" },
    "project.open": { labelKey: "kb_project_open", group: "project" },
    "project.save": { labelKey: "kb_project_save", group: "project" },
    "project.saveAs": { labelKey: "kb_project_save_as", group: "project" },
    "project.export": { labelKey: "kb_project_export", group: "project" },

    "track.add": { labelKey: "kb_track_add", group: "project" },

    "clip.delete": { labelKey: "kb_clip_delete", group: "clip" },
    "clip.copy": { labelKey: "kb_clip_copy", group: "clip" },
    "clip.cut": { labelKey: "kb_clip_cut", group: "clip" },
    "clip.paste": { labelKey: "kb_clip_paste", group: "clip" },
    "clip.split": { labelKey: "kb_clip_split", group: "clip" },
    "clip.normalize": { labelKey: "kb_clip_normalize", group: "clip" },

    "pianoRoll.copy": { labelKey: "kb_pianoroll_copy", group: "pianoRoll" },
    "pianoRoll.paste": { labelKey: "kb_pianoroll_paste", group: "pianoRoll" },
    "pianoRoll.shiftParamUp": {
        labelKey: "kb_pianoroll_shift_param_up",
        group: "pianoRoll",
    },
    "pianoRoll.shiftParamDown": {
        labelKey: "kb_pianoroll_shift_param_down",
        group: "pianoRoll",
    },
    "pianoRoll.shiftParamUpSelection": {
        labelKey: "kb_pianoroll_shift_param_up_selection",
        group: "pianoRoll",
    },
    "pianoRoll.shiftParamDownSelection": {
        labelKey: "kb_pianoroll_shift_param_down_selection",
        group: "pianoRoll",
    },

    "modifier.clipSlipEdit": {
        labelKey: "kb_modifier_slip_edit",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.clipStretch": {
        labelKey: "kb_modifier_stretch",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.clipNoSnap": {
        labelKey: "kb_modifier_no_snap",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.clipCopyDrag": {
        labelKey: "kb_modifier_copy_drag",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.horizontalZoom": {
        labelKey: "kb_modifier_horizontal_zoom",
        group: "modifier",
        modifierOperationType: "wheel",
    },
    "modifier.pianoRollVerticalZoom": {
        labelKey: "kb_modifier_pr_vzoom",
        group: "modifier",
        modifierOperationType: "wheel",
    },
    "modifier.scrollHorizontal": {
        labelKey: "kb_modifier_scroll_h",
        group: "modifier",
        modifierOperationType: "wheel",
    },
    "modifier.scrollVertical": {
        labelKey: "kb_modifier_scroll_v",
        group: "modifier",
        modifierOperationType: "wheel",
    },
    "modifier.pianoKeysVerticalScroll": {
        labelKey: "kb_modifier_piano_keys_scroll_v",
        group: "modifier",
        modifierOperationType: "wheel",
        scopedContext: "pianoKeysWheel",
    },
    "modifier.pianoKeysVerticalZoom": {
        labelKey: "kb_modifier_piano_keys_zoom_v",
        group: "modifier",
        modifierOperationType: "wheel",
        scopedContext: "pianoKeysWheel",
    },
    "modifier.paramMorph": {
        labelKey: "kb_modifier_param_morph",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.paramFineAdjust": {
        labelKey: "kb_modifier_param_fine_adjust",
        group: "modifier",
        modifierOperationType: "drag",
    },
    "modifier.vibratoAmplitudeAdjust": {
        labelKey: "kb_modifier_vibrato_amplitude_adjust",
        group: "modifier",
        modifierOperationType: "wheel",
    },
    "modifier.vibratoFrequencyAdjust": {
        labelKey: "kb_modifier_vibrato_frequency_adjust",
        group: "modifier",
        modifierOperationType: "wheel",
    },

    // 快速搜索
    "quickSearch.open": {
        labelKey: "kb_quick_search_open",
        group: "quickSearch",
    },
    "quickSearch.navigate.up": {
        labelKey: "kb_quick_search_nav_up",
        group: "quickSearch",
        scopedContext: "quickSearch",
    },
    "quickSearch.navigate.down": {
        labelKey: "kb_quick_search_nav_down",
        group: "quickSearch",
        scopedContext: "quickSearch",
    },
    "quickSearch.preview": {
        labelKey: "kb_quick_search_preview",
        group: "quickSearch",
        scopedContext: "quickSearch",
    },
    "quickSearch.confirm": {
        labelKey: "kb_quick_search_confirm",
        group: "quickSearch",
        scopedContext: "quickSearch",
    },
    "quickSearch.close": {
        labelKey: "kb_quick_search_close",
        group: "quickSearch",
        scopedContext: "quickSearch",
    },
};

/**
 * 所有 ActionId 列表（保持顺序一致，方便遍历）
 */
export const ALL_ACTION_IDS: ActionId[] = Object.keys(
    DEFAULT_KEYBINDINGS,
) as ActionId[];

/**
 * 分组标题 i18n key
 */
export const GROUP_LABEL_KEYS: Record<ActionMeta["group"], string> = {
    mode: "kb_group_mode",
    playback: "kb_group_playback",
    edit: "kb_group_edit",
    project: "kb_group_project",
    clip: "kb_group_clip",
    pianoRoll: "kb_group_pianoroll",
    modifier: "kb_group_modifier",
    quickSearch: "kb_group_quick_search",
};
