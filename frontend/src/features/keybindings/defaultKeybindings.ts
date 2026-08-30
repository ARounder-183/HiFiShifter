import type { ActionId, ActionMeta, KeybindingMap } from "./types";
import { IS_MAC } from "../../utils/platform";

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
    "recording.toggle": { key: "r", ctrl: true },
    "playback.focusCursor": { key: "'" }, // 聚焦播放光标
    "playback.seekLeft": { key: "arrowleft" },
    "playback.seekRight": { key: "arrowright" },
    "timeline.zoomIn": { key: "__none__" },
    "timeline.zoomOut": { key: "__none__" },

    // 编辑
    "edit.undo": { key: "z", ctrl: true },
    "edit.redo": IS_MAC ? { key: "z", ctrl: true, shift: true } : { key: "y", ctrl: true },
    "edit.selectAll": { key: "a", ctrl: true },
    "edit.deselect": { key: "__none__" },
    "edit.initialize": { key: "backspace" },
    "edit.transposeCents": { key: "f", ctrl: true },
    "edit.transposeDegrees": { key: "i", ctrl: true },
    "edit.setPitch": { key: "0", ctrl: true },
    "edit.average": { key: "e", ctrl: true },
    "edit.smooth": { key: "m", ctrl: true },
    "edit.addVibrato": { key: "b", ctrl: true },
    "edit.quantize": { key: "p", ctrl: true },
    "edit.meanQuantize": { key: "q", ctrl: true },
    "edit.pasteVocalShifter": { key: "v", shift: true },
    "edit.pasteTracks": { key: "v", ctrl: true, alt: true },

    // 工程
    "project.new": { key: "n", ctrl: true },
    "project.open": { key: "o", ctrl: true, shift: true },
    "project.save": { key: "s", ctrl: true },
    "project.saveAs": { key: "s", ctrl: true, shift: true },
    "project.export": { key: "e", ctrl: true },
    "project.importMedia": { key: "o", ctrl: true },
    "project.importMidi": { key: "__none__" },
    "project.importHifishifter": { key: "__none__" },
    "project.importReaper": { key: "__none__" },
    "project.importVocalShifter": { key: "__none__" },

    // 轨道
    "track.add": { key: "t", ctrl: true },
    "track.clone": { key: "d", ctrl: true },
    "track.delete": { key: "delete", ctrl: true },
    "track.selectUp": { key: "arrowup" },
    "track.selectDown": { key: "arrowdown" },

    // Clip 操作
    "clip.delete": { key: "delete" },
    "clip.copy": { key: "c", ctrl: true },
    "clip.cut": { key: "x", ctrl: true },
    "clip.paste": { key: "v", ctrl: true },
    "clip.split": { key: "s" },
    "clip.normalize": { key: "n", ctrl: true, shift: true },
    "clip.group": { key: "g" },
    "clip.ungroup": { key: "u" },
    "clip.cycleTake": { key: "t" },
    "clip.cycleTakePrev": { key: "t", shift: true },

    // PianoRoll 操作
    "pianoRoll.copy": { key: "c", ctrl: true },
    "pianoRoll.paste": { key: "v", ctrl: true },
    "pianoRoll.shiftParamUp": { key: "=" },
    "pianoRoll.shiftParamDown": { key: "-" },
    "pianoRoll.shiftParamUpSelection": { key: "]" },
    "pianoRoll.shiftParamDownSelection": { key: "[" },
    "pianoRoll.vibratoDragAmplitudeIncrease": { key: "arrowup" },
    "pianoRoll.vibratoDragAmplitudeDecrease": { key: "arrowdown" },
    "pianoRoll.vibratoDragFrequencyIncrease": { key: "arrowleft" },
    "pianoRoll.vibratoDragFrequencyDecrease": { key: "arrowright" },

    // 修饰键行为
    // 多选切换默认为主修饰键（Windows: Ctrl / macOS: ⌘），对齐文件管理器
    // 与 DAW 的"按住主修饰键点击追加选择"惯例；ctrl 字段在 macOS 上自动
    // 映射为 ⌘（isModifierActive 走 isPrimaryModifierDown）。
    "modifier.clipMultiSelectToggle": {
        key: "control",
        modifierOnly: true,
        ctrl: true,
    },
    // 范围选择默认 Shift（按住并点击 = 从上次锚点到点击处范围选择）。
    "modifier.clipRangeSelect": { key: "shift", modifierOnly: true, shift: true },
    // 音高调整默认 Alt+Shift：避免与 Slip/拉伸（Alt）、临时关吸附（Shift）
    // 的单修饰键语义重叠，同时保留 Shift+点击范围选择/⌘+点击多选等点击行为。
    "modifier.clipPitchDrag": {
        key: "alt",
        modifierOnly: true,
        alt: true,
        shift: true,
    },
    "modifier.clipSlipEdit": { key: "alt", modifierOnly: true, alt: true },
    "modifier.clipStretch": { key: "alt", modifierOnly: true, alt: true },
    "modifier.clipNoSnap": { key: "shift", modifierOnly: true, shift: true },
    // macOS 上 ctrl 字段会自动映射为 Command（⌘），因此默认复制拖动为 ⌘+拖动；
    // 避免占用 Option，Option 保留给拉伸/滑动编辑等交替操作。
    "modifier.clipCopyDrag": { key: "control", modifierOnly: true, ctrl: true },
    "modifier.clipCrossfadeGrip": {
        key: "control",
        modifierOnly: true,
        ctrl: true,
    },
    // 淡化包络曲率：对齐 REAPER “Alt 拖动调整张力”惯例；Alt 在
    // 包络线/交叉点目标上无其他绑定，语义干净。
    "modifier.fadeCurvatureDrag": { key: "alt", modifierOnly: true, alt: true },
    // 参数线点击循环切换曲线类型：默认 Ctrl。macOS 上 ctrl 字段自动映射
    // 为 ⌘（与 copyDrag/crossfadeGrip 同一约定）；operationType 用独立
    // "click"，与既有 Ctrl+drag 系键位不构成同类型冲突。
    "modifier.fadeShapeCycleClick": {
        key: "control",
        modifierOnly: true,
        ctrl: true,
    },
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
    "modifier.pianoKeysVerticalScroll": { key: "__none__", modifierOnly: true },
    "modifier.pianoKeysVerticalZoom": { key: "alt", modifierOnly: true, alt: true },
    "modifier.paramMorph": { key: "alt", modifierOnly: true, alt: true },
    "modifier.paramFineAdjust": { key: "control", modifierOnly: true, ctrl: true },
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
    "recording.toggle": { labelKey: "kb_recording_toggle", group: "playback" },
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
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.transposeCents": {
        labelKey: "kb_edit_transpose_cents",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.transposeDegrees": {
        labelKey: "kb_edit_transpose_degrees",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.setPitch": {
        labelKey: "kb_edit_set_pitch",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.average": {
        labelKey: "kb_edit_average",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.smooth": {
        labelKey: "kb_edit_smooth",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.addVibrato": {
        labelKey: "kb_edit_add_vibrato",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.quantize": {
        labelKey: "kb_edit_quantize",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.meanQuantize": {
        labelKey: "kb_edit_mean_quantize",
        group: "paramEditor",
        scopedContext: "paramEditorSelect",
    },
    "edit.pasteVocalShifter": {
        labelKey: "kb_edit_paste_vocalshifter",
        group: "edit",
    },
    "edit.pasteTracks": {
        labelKey: "kb_edit_paste_tracks",
        group: "edit",
    },

    "project.new": { labelKey: "kb_project_new", group: "project" },
    "project.open": { labelKey: "kb_project_open", group: "project" },
    "project.save": { labelKey: "kb_project_save", group: "project" },
    "project.saveAs": { labelKey: "kb_project_save_as", group: "project" },
    "project.export": { labelKey: "kb_project_export", group: "project" },
    "project.importMedia": { labelKey: "kb_project_import_media", group: "project" },
    "project.importMidi": { labelKey: "kb_project_import_midi", group: "project" },
    "project.importHifishifter": {
        labelKey: "kb_project_import_hifishifter",
        group: "project",
    },
    "project.importReaper": {
        labelKey: "kb_project_import_reaper",
        group: "project",
    },
    "project.importVocalShifter": {
        labelKey: "kb_project_import_vocalshifter",
        group: "project",
    },

    "track.add": { labelKey: "kb_track_add", group: "project" },
    // 克隆/删除选中轨道是轨道面板操作：与全局操作（如 clip.delete 的裸
    // Delete）共用按键时由焦点裁决 —— 焦点在轨道头优先轨道操作，否则落
    // 回全局操作（拷贝/粘贴同构的焦点路由，见 focusRouting.ts）。
    "track.clone": {
        labelKey: "kb_track_clone",
        group: "project",
        scopedContext: "trackHeaderFocus",
    },
    "track.delete": {
        labelKey: "kb_track_delete",
        group: "project",
        scopedContext: "trackHeaderFocus",
    },
    "track.selectUp": { labelKey: "kb_track_select_up", group: "project" },
    "track.selectDown": { labelKey: "kb_track_select_down", group: "project" },

    "clip.delete": { labelKey: "kb_clip_delete", group: "clip" },
    "clip.copy": { labelKey: "kb_clip_copy", group: "clip" },
    "clip.cut": { labelKey: "kb_clip_cut", group: "clip" },
    "clip.paste": { labelKey: "kb_clip_paste", group: "clip" },
    "clip.split": { labelKey: "kb_clip_split", group: "clip" },
    "clip.normalize": { labelKey: "kb_clip_normalize", group: "clip" },
    "clip.group": { labelKey: "kb_clip_group", group: "clip" },
    "clip.ungroup": { labelKey: "kb_clip_ungroup", group: "clip" },
    "clip.cycleTake": { labelKey: "kb_clip_cycle_take", group: "clip" },
    "clip.cycleTakePrev": { labelKey: "kb_clip_cycle_take_prev", group: "clip" },

    "pianoRoll.copy": {
        labelKey: "kb_pianoroll_copy",
        group: "pianoRoll",
        scopedContext: "paramEditorSelect",
    },
    "pianoRoll.paste": {
        labelKey: "kb_pianoroll_paste",
        group: "pianoRoll",
        scopedContext: "paramEditorSelect",
    },
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
    "pianoRoll.vibratoDragAmplitudeIncrease": {
        labelKey: "kb_pianoroll_vibrato_drag_amp_increase",
        group: "pianoRoll",
        scopedContext: "pianoRollVibratoDrag",
    },
    "pianoRoll.vibratoDragAmplitudeDecrease": {
        labelKey: "kb_pianoroll_vibrato_drag_amp_decrease",
        group: "pianoRoll",
        scopedContext: "pianoRollVibratoDrag",
    },
    "pianoRoll.vibratoDragFrequencyIncrease": {
        labelKey: "kb_pianoroll_vibrato_drag_freq_increase",
        group: "pianoRoll",
        scopedContext: "pianoRollVibratoDrag",
    },
    "pianoRoll.vibratoDragFrequencyDecrease": {
        labelKey: "kb_pianoroll_vibrato_drag_freq_decrease",
        group: "pianoRoll",
        scopedContext: "pianoRollVibratoDrag",
    },

    // ── 修饰键 · 音频块选择与拖拽（时间轴） ────────────────────
    "modifier.clipMultiSelectToggle": {
        labelKey: "kb_modifier_clip_multi_select_toggle",
        group: "modClip",
        modifierOperationType: "click",
        conflictScenes: ["clip.select"],
    },
    "modifier.clipRangeSelect": {
        labelKey: "kb_modifier_clip_range_select",
        group: "modClip",
        modifierOperationType: "click",
        conflictScenes: ["clip.select"],
    },
    "modifier.clipPitchDrag": {
        labelKey: "kb_modifier_clip_pitch_drag",
        group: "modClip",
        modifierOperationType: "drag",
        conflictScenes: ["clip.move"],
    },
    "modifier.clipSlipEdit": {
        labelKey: "kb_modifier_slip_edit",
        group: "modClip",
        modifierOperationType: "drag",
        conflictScenes: ["clip.move"],
    },
    "modifier.clipStretch": {
        labelKey: "kb_modifier_stretch",
        group: "modClip",
        modifierOperationType: "drag",
        conflictScenes: ["clip.edge", "roll.paramEdge"],
    },
    "modifier.clipNoSnap": {
        labelKey: "kb_modifier_no_snap",
        group: "modClip",
        modifierOperationType: "drag",
        conflictScenes: ["clip.move", "clip.edge", "tempo.ruler", "roll.paramDrag"],
    },
    "modifier.clipCopyDrag": {
        labelKey: "kb_modifier_copy_drag",
        group: "modClip",
        modifierOperationType: "drag",
        conflictScenes: ["clip.move"],
    },
    // ── 修饰键 · 淡化与交叉淡化（时间轴） ──────────────────────
    "modifier.clipCrossfadeGrip": {
        labelKey: "kb_modifier_crossfade_grip",
        group: "modFade",
        modifierOperationType: "drag",
        conflictScenes: ["clip.crossfade"],
    },
    "modifier.fadeCurvatureDrag": {
        labelKey: "kb_modifier_fade_curvature",
        group: "modFade",
        modifierOperationType: "drag",
        conflictScenes: ["clip.crossfade", "clip.fade"],
    },
    "modifier.fadeShapeCycleClick": {
        labelKey: "kb_modifier_fade_shape_cycle",
        group: "modFade",
        modifierOperationType: "click",
        conflictScenes: ["clip.fade"],
    },
    // ── 修饰键 · 参数编辑与颤音（钢琴卷帘） ────────────────────
    "modifier.paramMorph": {
        labelKey: "kb_modifier_param_morph",
        group: "modParam",
        modifierOperationType: "drag",
        conflictScenes: ["roll.morph"],
    },
    "modifier.vibratoAmplitudeAdjust": {
        labelKey: "kb_modifier_vibrato_amplitude_adjust",
        group: "modParam",
        modifierOperationType: "wheel",
        conflictScenes: ["roll.vibratoWheel"],
    },
    "modifier.vibratoFrequencyAdjust": {
        labelKey: "kb_modifier_vibrato_frequency_adjust",
        group: "modParam",
        modifierOperationType: "wheel",
        conflictScenes: ["roll.vibratoWheel"],
    },
    // ── 修饰键 · 滚轮导航 ─────────────────────────────────────
    "modifier.horizontalZoom": {
        labelKey: "kb_modifier_horizontal_zoom",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.timeline", "wheel.pianoRoll", "wheel.pianoKeys"],
    },
    "modifier.pianoRollVerticalZoom": {
        labelKey: "kb_modifier_pr_vzoom",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.timeline", "wheel.pianoRoll"],
    },
    "modifier.scrollHorizontal": {
        labelKey: "kb_modifier_scroll_h",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.timeline", "wheel.pianoRoll", "wheel.pianoKeys"],
    },
    "modifier.scrollVertical": {
        labelKey: "kb_modifier_scroll_v",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.timeline", "wheel.pianoRoll"],
    },
    "modifier.pianoKeysVerticalScroll": {
        labelKey: "kb_modifier_piano_keys_scroll_v",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.pianoKeys"],
    },
    "modifier.pianoKeysVerticalZoom": {
        labelKey: "kb_modifier_piano_keys_zoom_v",
        group: "modWheel",
        modifierOperationType: "wheel",
        conflictScenes: ["wheel.pianoKeys"],
    },
    // ── 修饰键 · 全局微调 ─────────────────────────────────────
    "modifier.paramFineAdjust": {
        labelKey: "kb_modifier_param_fine_adjust",
        group: "modFine",
        modifierOperationType: "hold",
        conflictScenes: ["global.fine", "clip.gain", "roll.vibratoWheel"],
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
export const ALL_ACTION_IDS: ActionId[] = Object.keys(DEFAULT_KEYBINDINGS) as ActionId[];

/**
 * 分组标题 i18n key
 */
export const GROUP_LABEL_KEYS: Record<ActionMeta["group"], string> = {
    playback: "kb_group_playback",
    mode: "kb_group_mode",
    edit: "kb_group_edit",
    project: "kb_group_project",
    clip: "kb_group_clip",
    pianoRoll: "kb_group_pianoroll",
    paramEditor: "kb_group_param_editor",
    quickSearch: "kb_group_quick_search",
    modClip: "kb_group_mod_clip",
    modFade: "kb_group_mod_fade",
    modParam: "kb_group_mod_param",
    modWheel: "kb_group_mod_wheel",
    modFine: "kb_group_mod_fine",
};

/**
 * 设置面板中的分组展示顺序：
 * 先键盘快捷键（按全局 → 时间轴 → 钢琴卷帘 → 快速搜索的场景排列），
 * 再修饰键（按音频块 → 淡化 → 参数编辑 → 滚轮 → 全局微调的场景排列）。
 */
export const ACTION_GROUP_ORDER: ActionMeta["group"][] = [
    "playback",
    "mode",
    "edit",
    "project",
    "clip",
    "pianoRoll",
    "paramEditor",
    "quickSearch",
    "modClip",
    "modFade",
    "modParam",
    "modWheel",
    "modFine",
];
